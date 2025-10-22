from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedNLL, maskedMSETest, maskedNLLTest
from torch.utils.data import DataLoader
import time


def main():
    args = {}
    args['use_cuda'] = True
    args['encoder_size'] = 64
    args['decoder_size'] = 128
    args['in_length'] = 16
    args['out_length'] = 25
    args['grid_size'] = (13, 3)
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16
    args['dyn_embedding_size'] = 32
    args['input_embedding_size'] = 32
    args['num_lat_classes'] = 3
    args['num_lon_classes'] = 2
    args['use_maneuvers'] = True
    args['train_flag'] = False

    metric = 'rmse'  # 或 'nll'

    # 创建模型
    net = highwayNet(args)

    # 加载模型(处理设备兼容性)
    device = torch.device('cuda' if args['use_cuda'] and torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load('trained_models/cslstm_best.tar', map_location=device))
    if args['use_cuda']:
        net = net.cuda()

    net.eval()  # 设置为评估模式

    # 加载测试集
    tsSet = ngsimDataset("data/TestSet1.mat")
    tsDataloader = DataLoader(tsSet, batch_size=128, shuffle=True, num_workers=4, collate_fn=tsSet.collate_fn)

    # 初始化统计变量
    lossVals = torch.zeros(25).cuda() if args['use_cuda'] else torch.zeros(25)
    counts = torch.zeros(25).cuda() if args['use_cuda'] else torch.zeros(25)

    # 机动意图准确率统计
    lat_correct = 0
    lon_correct = 0
    total_samples = 0

    print(f"🚀 Starting evaluation with metric: {metric.upper()}")
    print(f"📊 Total test batches: {len(tsDataloader)}")

    with torch.no_grad():  # 禁用梯度计算
        for i, data in enumerate(tsDataloader):
            st_time = time.time()
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

            if args['use_cuda']:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                mask = mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()

            if metric == 'nll':
                if args['use_maneuvers']:
                    fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l, c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)
                else:
                    fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask, use_maneuvers=False)
            else:
                if args['use_maneuvers']:
                    fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)

                    # 统计机动意图准确率
                    lat_correct += (torch.argmax(lat_pred, dim=1) == torch.argmax(lat_enc, dim=1)).sum().item()
                    lon_correct += (torch.argmax(lon_pred, dim=1) == torch.argmax(lon_enc, dim=1)).sum().item()
                    total_samples += lat_pred.size(0)

                    # 选择最可能的机动组合对应的轨迹
                    fut_pred_max = torch.zeros_like(fut_pred[0])
                    for k in range(lat_pred.shape[0]):
                        lat_man = torch.argmax(lat_pred[k, :]).detach()
                        lon_man = torch.argmax(lon_pred[k, :]).detach()
                        indx = lon_man * 3 + lat_man
                        fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
                    l, c = maskedMSETest(fut_pred_max, fut, op_mask)
                else:
                    fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l, c = maskedMSETest(fut_pred, fut, op_mask)

            lossVals += l.detach()
            counts += c.detach()

            # 显示进度
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(tsDataloader)} batches processed...")

    # 打印结果
    print("\n" + "=" * 70)
    print(f"📊 Evaluation Results ({metric.upper()})")
    print("=" * 70)

    if metric == 'nll':
        nll_values = lossVals / counts
        print("Negative Log-Likelihood per time step:")
        for t in range(25):
            time_sec = (t + 1) * 0.2
            print(f"  t={time_sec:.1f}s: NLL = {nll_values[t]:.4f}")
        print(f"\nAverage NLL: {nll_values.mean():.4f}")
    else:
        rmse = torch.pow(lossVals / counts, 0.5) * 0.3048  # 转换为米
        print("Root Mean Square Error (RMSE) per time step:")
        for t in range(25):
            time_sec = (t + 1) * 0.2
            print(f"  t={time_sec:.1f}s: RMSE = {rmse[t]:.4f} m")
        print(f"\nAverage RMSE: {rmse.mean():.4f} m")
        print(f"Final RMSE (t=5.0s): {rmse[-1]:.4f} m")

    # 打印机动意图准确率
    if args['use_maneuvers'] and total_samples > 0:
        print("\n" + "=" * 70)
        print("🎯 Maneuver Prediction Accuracy")
        print("=" * 70)
        print(f"Lateral Maneuver Accuracy:  {lat_correct / total_samples * 100:.2f}%")
        print(f"Longitudinal Maneuver Accuracy: {lon_correct / total_samples * 100:.2f}%")

    print("=" * 70)


if __name__ == '__main__':
    main()
