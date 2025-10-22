import torch
from model import highwayNet
from utils import ngsimDataset
import scipy.io as scp
import matplotlib.pyplot as plt

# 定义参数
args = {
    'use_cuda': torch.cuda.is_available(),
    'encoder_size': 64,
    'decoder_size': 128,
    'in_length': 16,
    'out_length': 25,
    'grid_size': (13, 3),
    'soc_conv_depth': 64,
    'conv_3x1_depth': 16,
    'dyn_embedding_size': 32,
    'input_embedding_size': 32,
    'num_lat_classes': 3,
    'num_lon_classes': 2,
    'use_maneuvers': True,
    'train_flag': False
}

# 加载预训练模型
net = highwayNet(args)

# 尝试在 CPU 上加载模型
try:
    net.load_state_dict(torch.load('trained_models/cslstm_m4.tar', map_location=torch.device('cpu')))
    print("Model loaded successfully on CPU.")
except Exception as e:
    print(f"Error loading model on CPU: {e}")
    raise

# 如果使用 CUDA，将模型移动到 GPU
if args['use_cuda']:
    net = net.cuda()
net.eval()

# 加载数据集并获取指定车辆的数据
dataset = ngsimDataset('E:/ps/conv-social-pooling-fix_preprocessing/conv-social-pooling-fix_preprocessing/TestSet.mat')
vehicleID = 2
hist, fut, nbrs, lat_enc, lon_enc = dataset[vehicleID]

# 数据准备
hist = torch.tensor(hist, dtype=torch.float).unsqueeze(1)  # 添加batch维度
nbrs = torch.tensor(nbrs, dtype=torch.float)
lat_enc = torch.tensor(lat_enc, dtype=torch.float).unsqueeze(0)
lon_enc = torch.tensor(lon_enc, dtype=torch.float).unsqueeze(0)
mask = torch.ones((hist.shape[0], 1, args['grid_size'][0], args['grid_size'][1]), dtype=torch.bool)

if args['use_cuda']:
    hist = hist.cuda()
    nbrs = nbrs.cuda()
    mask = mask.cuda()
    lat_enc = lat_enc.cuda()
    lon_enc = lon_enc.cuda()

# 检查并处理邻居数据
if nbrs.size(1) == 0:
    nbrs = torch.zeros((hist.size(0), 1, 2), dtype=torch.float).cuda() if args['use_cuda'] else torch.zeros((hist.size(0), 1, 2), dtype=torch.float)
    
# 预测
with torch.no_grad():
    fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
    fut_pred_max = torch.zeros_like(fut_pred[0])
    for k in range(lat_pred.shape[0]):
        lat_man = torch.argmax(lat_pred[k, :]).detach()
        lon_man = torch.argmax(lon_pred[k, :]).detach()
        indx = lon_man * 3 + lat_man
        fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]

# 将预测结果转换为numpy格式进行后续处理
predicted_trajectory = fut_pred_max.cpu().numpy()
actual_trajectory = fut.cpu().numpy()

# 可视化实际轨迹和预测轨迹
plt.figure(figsize=(10, 6))
plt.plot(hist[:, 0, 0].cpu().numpy(), hist[:, 0, 1].cpu().numpy(), 'k--', label='History')  # 历史轨迹
plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], 'g-', label='Actual Future')  # 实际未来轨迹
plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'r--', label='Predicted Future')  # 预测未来轨迹
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.title('Trajectory Prediction for Vehicle ID 2')
plt.show()
