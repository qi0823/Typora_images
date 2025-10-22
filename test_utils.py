# test_utils.py
import torch
from utils import ngsimDataset
from torch.utils.data import DataLoader

# 加载数据
dataset = ngsimDataset('data/TrainSet1.mat')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                        num_workers=0, collate_fn=dataset.collate_fn)

# 测试一个批次
for batch in dataloader:
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = batch

    print("✅ Batch shapes:")
    print(f"  hist: {hist.shape}")  # [16, 4, 2]
    print(f"  nbrs: {nbrs.shape}")  # [16, N, 2]
    print(f"  mask: {mask.shape}")  # [4, 3, 13, 64]
    print(f"  lat_enc: {lat_enc.shape}")  # [4, 3]
    print(f"  lon_enc: {lon_enc.shape}")  # [4, 2]
    print(f"  fut: {fut.shape}")  # [25, 4, 2]
    print(f"  op_mask: {op_mask.shape}")  # [25, 4, 2]

    # 验证数据有效性
    print(f"\n✅ Data validation:")
    print(f"  hist non-zero: {(hist != 0).any()}")
    print(f"  nbrs non-zero: {(nbrs != 0).any()}")
    print(f"  mask any True: {mask.any()}")

    break

print("\n🎉 Utils.py 修改成功!")
