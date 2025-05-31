# predict_model.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib


# 通用 Dataset 类（与训练时相同）
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx], torch.tensor(0.0)  # dummy label

# 滑动窗口函数
def create_test_windows(X_raw, seq_len):
    padding = np.zeros((seq_len - 1, X_raw.shape[1]))
    padded = np.vstack([padding, X_raw])
    X_seq = np.array([padded[i:i+seq_len] for i in range(len(X_raw))])
    return X_seq

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 加载数据
    X_test_df = pd.read_parquet("D:\jacky\kaggle\drw\kaggle\input\X_test.parquet").fillna(0)
    X_raw = X_test_df.values
    X_seq = create_test_windows(X_raw, args.seq_len)

    # 加载模型类
    module = importlib.import_module(f"src.models.{args.model}")
    ModelClass = getattr(module, args.model_class)

    # 初始化模型
    model = ModelClass(input_dim=X_seq.shape[2]).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 创建 DataLoader
    test_loader = DataLoader(TimeSeriesDataset(X_seq), batch_size=args.batch_size, shuffle=False)

    # 预测
    preds = []
    with torch.no_grad():
        for X_batch, _ in tqdm(test_loader, desc="Predicting"):
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            preds.extend(pred)

    # 生成提交文件
    submission = pd.DataFrame({
        "ID": range(1, len(preds) + 1),
        "prediction": preds
    })
    os.makedirs("result", exist_ok=True)
    submission.to_csv(f"result/sub_{args.model}.csv", index=False)
    print(f"[INFO] Submission saved to result/sub_{args.model}.csv")


if __name__ == "__main__":
    # python src/predict/predict_model.py --model patchtst --model-class PatchTST --model-path models\patchtst_best.pth --seq-len 10 
     
    parser = argparse.ArgumentParser(description="通用模型预测脚本")
    parser.add_argument("--model", type=str, default="tft", help="模型名称（例如 patchtst 或 transformer）")
    parser.add_argument("--model-class", type=str, default="TFT", help="模型类名（默认等于模型名）")
    parser.add_argument("--model-path", type=str, default="models/tft_best.pth", required=True, help="训练好的模型参数文件路径")
    parser.add_argument("--seq-len", type=int, default=10, help="输入序列长度")
    parser.add_argument("--batch-size", type=int, default=64, help="预测批量大小")
    args = parser.parse_args()

    main(args)
