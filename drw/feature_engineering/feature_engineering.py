
import sys
import pandas as pd
import numpy as np
# Feature Engineering

def feature_engineering(df):
    # 原始特征
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    
    # 新增交互特征
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
    df['buy_sell_interaction'] = df['buy_qty'] * df['sell_qty']

    # 价差和流动性指标
    df['spread_indicator'] = (df['ask_qty'] - df['bid_qty']) / (df['ask_qty'] + df['bid_qty'] + 1e-8)
    df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
    df['liquidity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_liquidity'] + 1e-8)
    df['relative_spread'] = (df['ask_qty'] - df['bid_qty']) / (df['volume'] + 1e-8)
    df['depth_ratio'] = df['total_liquidity'] / (df['volume'] + 1e-8)
    
    # 成交量加权特征
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['volume_weighted_bid'] = df['bid_qty'] * df['volume']
    df['volume_weighted_ask'] = df['ask_qty'] * df['volume']
    
    # 比率特征
    df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
    df['trade_direction_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    
    # 订单流和市场压力指标
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-8)
    df['net_buy_volume'] = df['buy_qty'] - df['sell_qty']
    
    # 交易强度和市场活动指标
    df['trade_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-8)
    df['avg_trade_size'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['net_trade_flow'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['volume_participation'] = (df['buy_qty'] + df['sell_qty']) / (df['total_liquidity'] + 1e-8)
    df['market_activity'] = df['volume'] * df['total_liquidity']
    
    # 波动性和价差代理指标
    df['realized_volatility_proxy'] = np.abs(df['order_flow_imbalance']) * df['volume']
    
    # 标准化特征
    df['normalized_buy_volume'] = df['buy_qty'] / (df['bid_qty'] + 1e-8)
    df['normalized_sell_volume'] = df['sell_qty'] / (df['ask_qty'] + 1e-8)
    
    # 高级交互特征
    df['liquidity_adjusted_imbalance'] = df['order_flow_imbalance'] * df['depth_ratio']
    df['pressure_spread_interaction'] = df['buying_pressure'] * df['spread_indicator']
    
    # 挂单偏移度
    df['bid_skew'] = df['bid_qty'] / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df['ask_skew'] = df['ask_qty'] / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    
    # 处理无穷大和缺失值
    df = df.replace([np.inf, -np.inf], np.nan)
    #replace函数的语法是replace(to_replace, value, inplace=False, limit=None, regex=False, method='pad')
    #其中，to_replace表示要替换的值，value表示替换后的值，inplace表示是否在原地修改，limit表示替换的最大次数，regex表示是否使用正则表达式，method表示填充方法
    #这里将无穷大和负无穷大替换为NaN
    df = df.fillna(0)
    #而fillna函数的语法是fillna(value, method=None, axis=None, inplace=False, limit=None, downcast=None)
    #其中，value表示填充的值，method表示填充方法，axis表示轴，inplace表示是否在原地修改，limit表示填充的最大次数，downcast表示类型转换
    #这里将NaN替换为0
    #最后，返回处理后的DataFrame
    return df