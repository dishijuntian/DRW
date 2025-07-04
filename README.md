# DRW Crypto Market Prediction

> 🏆  **奖金池** : $25,000 (前5名平分)
>
> 📊  **评估指标** : Pearson相关系数
>
> ⏰  **比赛截止** : 2个月后

## 项目简介

参与DRW Trading Group主办的加密货币市场预测比赛，使用机器学习预测加密货币短期价格走势。

 **数据特点** :

* 结合DRW专有特征数据和公开市场数据
* 训练数据：2023年3月1日 - 2024年2月29日
* 高噪声、低信号的市场环境

## 快速开始

### 环境设置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

1. 从Kaggle下载竞赛数据
2. 确保数据放置在正确路径：

```
kaggle/
└── input/
    └── drw-crypto-market-prediction/
        ├── train.parquet
        ├── test.parquet
        └── sample_submission.csv
```

### 运行模型

```bash
# 基线模型
jupyter notebook ipynb/baseline.ipynb

# 查看结果
ls result/  # 提交文件
ls models/  # 模型文件
```

## 项目结构

```
├── README.md
├── requirements.txt
├── .gitignore
├── doc/
│   └── develop.md           # 开发文档
├── ipynb/
│   └── baseline.ipynb       # 基线模型
├── kaggle/
│   └── input/
│       └── drw-crypto-market-prediction/
│           ├── train.parquet
│           ├── test.parquet
│           └── sample_submission.csv
├── models/                  # 保存的模型和预测
│   ├── oof_preds.pkl
│   └── test_preds.pkl
├── result/                  # 提交文件
│   └── sub_ridge_0.116500.csv
├── src/                     # 源代码 (待开发)
└── tests/                   # 测试代码
    └── test_base.py
```

## 核心策略

### 数据分析

* 理解匿名化的DRW专有特征
* 分析时间序列特性和相关性
* 特征工程和选择

### 建模方法

* 传统ML：Ridge, XGBoost, LightGBM
* 时间序列：LSTM, Transformer
* 集成学习：Stacking, Blending

### 验证策略

* 时间序列交叉验证
* 避免数据泄露
* 针对Pearson相关系数优化

## 重要注意事项

⚠️  **数据使用规则** :

* 严禁使用未来数据
* 外部数据必须在预测时间点前可用
* 确保时间序列完整性

⚠️  **提交要求** :

* 按照sample_submission.csv格式
* 提交可复现的Jupyter Notebook
* 代码将被DRW审查

## 当前进展

* ✅ 基线模型 (Ridge回归，Pearson相关系数: 0.0865)
* 🔄 特征工程优化中
* 📋 高级模型开发计划中

## 开发指南

详细的开发指南和技术文档请参考 [`doc/develop.md`](https://claude.xiaoai.shop/chat/doc/develop.md)

## 参考资源

* [Kaggle竞赛页面](https://kaggle.com/competitions/drw-crypto-market-prediction)
* [DRW Trading Group](https://drw.com/)

---

 **免责声明** : 仅用于学习和竞赛目的，不构成投资建议
