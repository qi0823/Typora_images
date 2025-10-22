from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedNLL, maskedMSE, maskedNLLTest
from torch.utils.data import DataLoader
import time
import math


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
    args['train_flag'] = True

    net = highwayNet(args)
    if args['use_cuda']:
        net = net.cuda()

    # 超参数
    pretrainEpochs = 10
    trainEpochs = 20
    batch_size = 128
    learning_rate = 1e-3

    # 优化器和调度器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    crossEnt = torch.nn.BCELoss()

    # 数据加载(使用相对路径)
    trSet = ngsimDataset("data/TrainSet1.mat")
    valSet = ngsimDataset("data/ValSet1.mat")

    # 固定使用4个工作进程
    trDataloader = DataLoader(trSet, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=trSet.collate_fn)
    valDataloader = DataLoader(valSet, batch_size=batch_size, shuffle=True,
                               num_workers=4, collate_fn=valSet.collate_fn)

    # 训练记录
    train_loss = []
    val_loss = []
    prev_val_loss = math.inf
    best_val_loss = math.inf
    patience = 5
    patience_counter = 0

    for epoch_num in range(pretrainEpochs + trainEpochs):
        if epoch_num == 0:
            print('Pre-training with MSE loss')
        elif epoch_num == pretrainEpochs:
            print('Training with NLL loss')

        net.train_flag = True
        avg_tr_loss = 0
        avg_tr_time = 0
        avg_lat_acc = 0
        avg_lon_acc = 0

        for i, data in enumerate(trDataloader):
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

            if args['use_maneuvers']:
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                    avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / \
                                   lat_enc.size()[0]
                    avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / \
                                   lon_enc.size()[0]
            else:
                fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask)

            optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()

            batch_time = time.time() - st_time
            avg_tr_loss += l.item()
            avg_tr_time += batch_time

            if i % 100 == 99:
                eta = avg_tr_time / 100 * (len(trSet) / batch_size - i)
                print(f"Epoch no: {epoch_num + 1} | Epoch progress(%): {i / (len(trSet) / batch_size) * 100:.2f} | "
                      f"Avg train loss: {avg_tr_loss / 100:.4f} | Acc: {avg_lat_acc:.4f} {avg_lon_acc:.4f} | "
                      f"Validation loss prev epoch {prev_val_loss:.4f} | ETA(s): {int(eta)}")
                train_loss.append(avg_tr_loss / 100)
                avg_tr_loss = 0
                avg_lat_acc = 0
                avg_lon_acc = 0
                avg_tr_time = 0

        # 验证阶段
        net.train_flag = False
        print(f"Epoch {epoch_num + 1} complete. Calculating validation loss...")
        avg_val_loss = 0
        avg_val_lat_acc = 0
        avg_val_lon_acc = 0
        val_batch_count = 0

        with torch.no_grad():
            for i, data in enumerate(valDataloader):
                hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

                if args['use_cuda']:
                    hist = hist.cuda()
                    nbrs = nbrs.cuda()
                    mask = mask.cuda()
                    lat_enc = lat_enc.cuda()
                    lon_enc = lon_enc.cuda()
                    fut = fut.cuda()
                    op_mask = op_mask.cuda()

                if args['use_maneuvers']:
                    if epoch_num < pretrainEpochs:
                        # 预训练阶段使用MSE,需要单轨迹
                        net.train_flag = True
                        fut_pred, _, _ = net(hist, nbrs, mask, lat_enc, lon_enc)
                        net.train_flag = False
                        l = maskedMSE(fut_pred, fut, op_mask)
                    else:
                        fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                        l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, avg_along_time=True)
                        avg_val_lat_acc += (torch.sum(
                            torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                        avg_val_lon_acc += (torch.sum(
                            torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
                else:
                    fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    if epoch_num < pretrainEpochs:
                        l = maskedMSE(fut_pred, fut, op_mask)
                    else:
                        l = maskedNLL(fut_pred, fut, op_mask)

                avg_val_loss += l.item()
                val_batch_count += 1

        prev_val_loss = avg_val_loss / val_batch_count
        val_loss.append(prev_val_loss)

        print(
            f'Validation loss: {prev_val_loss:.4f} | Val Acc: {avg_val_lat_acc / val_batch_count * 100:.4f} {avg_val_lon_acc / val_batch_count * 100:.4f}')

        # 学习率调度
        scheduler.step(prev_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 Current learning rate: {current_lr:.6f}")

        # 早停和最佳模型保存
        if prev_val_loss < best_val_loss:
            best_val_loss = prev_val_loss
            patience_counter = 0
            torch.save(net.state_dict(), 'trained_models/cslstm_best.tar')
            print(f"✓ Best model saved with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⚠ No improvement for {patience_counter}/{patience} epochs")

            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # 保存最终模型
    torch.save(net.state_dict(), 'trained_models/cslstm_final.tar')
    print("Training complete!")


if __name__ == '__main__':
    main()
