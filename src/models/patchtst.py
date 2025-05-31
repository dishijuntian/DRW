# 文件：models/patchtst_model.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# 参数设置
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
SEQ_LEN = 10
PATCH_LEN = 5
STRIDE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# 数据集定义
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# PatchTST 模型定义
class PatchTST(nn.Module):
    def __init__(self, input_dim, patch_len=PATCH_LEN, stride=STRIDE, d_model=64, nhead=4, num_layers=2):
        super(PatchTST, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.input_fc = nn.Linear(input_dim * patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        patches = []

        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :]  # (batch_size, patch_len, input_dim)
            patch = patch.reshape(batch_size, -1)  # (batch_size, patch_len * input_dim)
            patches.append(patch)

        x_patched = torch.stack(patches, dim=1)  # (batch_size, num_patches, patch_len * input_dim)
        x_embedded = self.input_fc(x_patched)  # (batch_size, num_patches, d_model)
        x_transformed = self.transformer(x_embedded)  # (batch_size, num_patches, d_model)
        x_pooled = x_transformed.mean(dim=1)  # (batch_size, d_model)
        output = self.output_fc(x_pooled).squeeze()  # (batch_size)
        return output

# 滑动窗口构造函数
def create_sliding_windows(X, y, seq_len):
    X_windows = []
    y_windows = []
    for i in range(len(X) - seq_len + 1):
        X_windows.append(X[i:i + seq_len])
        y_windows.append(y[i + seq_len - 1])
    return np.array(X_windows), np.array(y_windows)

def main():
    # 数据加载与预处理
    X_train_df = pd.read_parquet("D:\jacky\kaggle\drw\kaggle\input\X_train.parquet").fillna(0)
    y_train_df = pd.read_parquet("D:\jacky\kaggle\drw\kaggle\input\y_train.parquet").fillna(0)
    X_train = X_train_df.values
    y_train = y_train_df.squeeze().values
    X_train, y_train = create_sliding_windows(X_train, y_train, SEQ_LEN)

    X_test_df = pd.read_parquet("D:\jacky\kaggle\drw\kaggle\input\X_test.parquet").fillna(0)
    X_test_raw = X_test_df.values
    expected_test_rows = len(X_test_raw)
    padding = np.zeros((SEQ_LEN - 1, X_test_raw.shape[1]))
    X_test_padded = np.vstack([padding, X_test_raw])
    X_test = np.array([X_test_padded[i:i + SEQ_LEN] for i in range(len(X_test_raw))])

    # 数据集划分
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.4, shuffle=False)
    train_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # 模型初始化
    model = PatchTST(input_dim=X_train.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best_r2 = -np.inf

    # 训练过程
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        # 验证过程
        model.eval()
        preds_list, y_list = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                preds = model(X_batch).cpu().numpy()
                preds_list.extend(preds)
                y_list.extend(y_batch.numpy())

        r2 = r2_score(y_list, preds_list)
        pearson = pearsonr(y_list, preds_list)[0]
        spearman = spearmanr(y_list, preds_list)[0]
        kendall = kendalltau(y_list, preds_list)[0]
        print(f"R2: {r2:.4f}, Pearson: {pearson:.4f}, Spearman: {spearman:.4f}, Kendall: {kendall:.4f}")

        # 保存最佳模型
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), "models/patchtst_best.pth")
            

if __name__ == "__main__":
    main()
