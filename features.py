# -*- coding: utf-8 -*-
"""因子工程：Alpha158 精简子集，CI 训练端与手机推理端共用同一份代码。
单位无关：1m 与 1s 数据都调同一函数，语义随 bar 粒度缩放。"""
import numpy as np
import pandas as pd

# K线列名（币安 csv 顺序）
KLINES_COLS = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "qav", "trades", "tbb", "tbq", "ignore"]

NUM_COLS = ["open", "high", "low", "close", "volume", "qav", "trades"]


def load_kline_csv(path):
    """读币安 kline csv（无表头），返回 float32 DataFrame。"""
    df = pd.read_csv(path, header=None, names=KLINES_COLS)
    for c in NUM_COLS:
        df[c] = df[c].astype(np.float32)
    df["open_time"] = df["open_time"].astype(np.int64)
    return df


def _safe_div(a, b):
    return (a / b.replace(0, np.nan)).astype(np.float32)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """输入: open/high/low/close/volume/qav/trades 的 DataFrame（按时间升序）。
    输出: 特征矩阵（float32），与输入等长（前部滚动窗为 NaN 行由调用方丢弃）。"""
    c, h, l, v, q = df["close"], df["high"], df["low"], df["volume"], df["qav"]
    o = df["open"]
    f = pd.DataFrame(index=df.index)
    logc = np.log(c.astype(np.float64))

    # 1) 多周期收益率
    for k in (1, 2, 3, 5, 10, 20, 30, 60, 120):
        f[f"ret_{k}"] = logc.diff(k).astype(np.float32)

    # 2) 动量：价格相对滚动均线
    for k in (5, 20, 60, 120):
        f[f"mom_{k}"] = _safe_div(c - c.rolling(k).mean(), c.rolling(k).mean())

    # 3) 波动率与波动结构
    r1 = logc.diff()
    for k in (10, 30, 60):
        f[f"vol_{k}"] = r1.rolling(k).std().astype(np.float32)
    f["vol_ratio_10_60"] = _safe_div(f["vol_10"], f["vol_60"])
    rng = (h - l).clip(lower=1e-12)
    f["hl_range"] = _safe_div(h - l, c)
    f["body_ratio"] = _safe_div((c - o).abs(), rng)
    f["upper_shadow"] = _safe_div(h - c.clip(lower=o), rng)
    f["lower_shadow"] = _safe_div(o.clip(lower=c) - l, rng)

    # 4) 量能特征
    for k in (10, 60):
        f[f"qv_{k}"] = _safe_div(v - v.rolling(k).mean(), v.rolling(k).mean() + 1.0)
    f["qv_ratio_10_60"] = _safe_div(f["qv_10"], f["qv_60"])
    f["trades_10"] = _safe_div(df["trades"] - df["trades"].rolling(10).mean(),
                               df["trades"].rolling(10).mean() + 1.0)
    # 主动买卖压力（tbb=主动买量）
    f["tbb_ratio"] = _safe_div(df["tbb"], v + 1.0)

    # 5) 均线族
    for k in (20, 60):
        m = c.rolling(k).mean()
        s = c.rolling(k).std() + 1e-9
        f[f"ma_pos_{k}"] = _safe_div(c - m, s)          # 类z-score
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    f["macd_cross"] = _safe_div(ema12 - ema26, c)

    # 6) RSI(14)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi_14"] = (100 - 100 / (1 + up / (dn + 1e-9))).astype(np.float32)

    # 7) 区间位置与VWAP
    for k in (10, 30, 120):
        hh, ll = h.rolling(k).max(), l.rolling(k).min()
        f[f"pos_{k}"] = _safe_div(c - ll, (hh - ll).clip(lower=1e-9))
    vwap = _safe_div(q.cumsum(), v.cumsum() + 1.0)
    f["vwap_gap"] = _safe_div(c - vwap, vwap)

    # 8) 时间编码（毫秒时间戳 → 周期正弦）
    t = df["open_time"].astype(np.int64)
    sec_of_day = (t // 1000) % 86400
    f["tod_sin"] = np.sin(2 * np.pi * sec_of_day / 86400).astype(np.float32)
    f["tod_cos"] = np.cos(2 * np.pi * sec_of_day / 86400).astype(np.float32)

    return f


def make_label(df: pd.DataFrame, horizon: int, band: float = 0.0004):
    """未来 horizon 根 bar 的收益方向。|收益|<band 的近中性样本标 -1（训练时剔除）。"""
    fut = (np.log(df["close"].shift(-horizon).astype(np.float64))
           - np.log(df["close"].astype(np.float64)))
    y = np.where(fut > band, 1, np.where(fut < -band, 0, -1)).astype(np.int8)
    return pd.Series(y, index=df.index), fut


def aggregate_bars(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """秒级K线重采样为 k 倍大bar（省内存/降噪，1s→4s 等）。"""
    g = df.groupby(np.arange(len(df)) // k)
    out = pd.DataFrame({
        "open_time": g["open_time"].first(),
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(), "qav": g["qav"].sum(),
        "trades": g["trades"].sum(), "tbb": g["tbb"].sum()})
    return out.reset_index(drop=True)
