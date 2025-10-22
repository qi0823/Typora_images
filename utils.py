# utils.py
from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch


def outputActivation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out


class ngsimDataset(Dataset):
    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
        self.data = scp.loadmat(mat_file)
        self.traj = self.data['traj']
        self.tracks = self.data['tracks']
        self.t_h = t_h
        self.t_f = t_f
        self.d_s = d_s
        self.enc_size = enc_size
        self.grid_size = grid_size

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        row = self.traj[idx]
        dsId, vehId, t = row[:3].astype(int)
        lane_id, lat_man, lon_man, speed, accel, v_class = row[3:9]
        neighbors = row[9:].astype(int)

        hist = self.getHistory(vehId, t, dsId)
        fut = self.getFuture(vehId, t, dsId)

        lat_enc = np.zeros([3])
        if 0 <= lat_man - 1 < lat_enc.size:
            lat_enc[int(lat_man - 1)] = 1

        lon_enc = np.zeros([2])
        if 0 <= lon_man - 1 < lon_enc.size:
            lon_enc[int(lon_man - 1)] = 1

        # 添加速度、加速度和车辆类别
        hist = np.hstack([hist, np.full((hist.shape[0], 1), speed), np.full((hist.shape[0], 1), accel),
                          np.full((hist.shape[0], 1), v_class)])

        return hist, fut, neighbors, lat_enc, lon_enc

    def getHistory(self, vehId, t, dsId):
        if vehId == 0 or dsId > len(self.tracks) or vehId > len(self.tracks[dsId - 1]):
            return np.empty([0, 2])
        else:
            vehTrack = self.tracks[dsId - 1][vehId - 1].T
            time_indices = np.where(vehTrack[:, 0] == t)
            if len(time_indices[0]) == 0:
                return np.empty([0, 2])
            refPos = vehTrack[time_indices[0][0], 1:3]
            stpt = max(0, time_indices[0][0] - self.t_h)
            enpt = time_indices[0][0] + 1
            hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    def getFuture(self, vehId, t, dsId):
        if dsId > len(self.tracks) or vehId > len(self.tracks[dsId - 1]):
            return np.empty([0, 2])
        vehTrack = self.tracks[dsId - 1][vehId - 1].T
        time_indices = np.where(vehTrack[:, 0] == t)
        if len(time_indices[0]) == 0:
            return np.empty([0, 2])
        refPos = vehTrack[time_indices[0][0], 1:3]
        stpt = time_indices[0][0] + self.d_s
        enpt = min(len(vehTrack), time_indices[0][0] + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    def collate_fn(self, samples):
        nbr_batch_size = 0
        for _, _, nbrs, _, _ in samples:
            nbr_batch_size += sum([nbr != 0 for nbr in nbrs])
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 5)  # 改为5以包含速度、加速度和类别特征

        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size, dtype=torch.bool)

        hist_batch = torch.zeros(maxlen, len(samples), 5)  # 改为5以包含速度、加速度和类别特征
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        lat_enc_batch = torch.zeros(len(samples), 3)
        lon_enc_batch = torch.zeros(len(samples), 2)

        count = 0
        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc) in enumerate(samples):
            hist_batch[0:len(hist), sampleId, :] = torch.from_numpy(hist)
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)

            # 获取 t 和 dsId
            t = self.traj[sampleId, 2].astype(int)  # 这里假设 t 是轨迹的第三列
            dsId = self.traj[sampleId, 0].astype(int)  # 这里假设 dsId 是轨迹的第一列

            for id, nbr in enumerate(nbrs):
                if nbr != 0:
                    nbr_hist = self.getHistory(nbr, t, dsId)
                    nbrs_batch[0:len(nbr_hist), count, 0:2] = torch.from_numpy(nbr_hist)
                    count += 1

        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch


def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]
    sigY = y_pred[:, :, 3]
    rho = y_pred[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                                           2) - 2 * rho * torch.pow(
        sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal


def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=2, use_maneuvers=True,
                  avg_along_time=False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:, l] * lon_pred[:, k]
                wts = wts.repeat(len(fut_pred[0]), 1)
                y_pred = fut_pred[k * num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                out = -(torch.pow(ohr, 2) * (
                            torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                                        2) - 2 * rho * torch.pow(
                        sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr))
                acc[:, :, count] = out + torch.log(wts)
                count += 1
        acc = -logsumexp(acc, dim=2)
        acc = acc * op_mask[:, :, 0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc, dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                                               2) - 2 * rho * torch.pow(
            sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:, :, 0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts


def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal


def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:, :, 0], dim=1)
    counts = torch.sum(mask[:, :, 0], dim=1)
    return lossVal, counts


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
