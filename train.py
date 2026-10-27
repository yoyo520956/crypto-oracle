#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CI 端训练主程序：TopN 币 1m 全历史 + BTC/ETH 1s 深度模型。
产出 models/*.joblib + metrics.csv + meta.json（Artifacts 交付）。"""
import argparse
import datetime
import json
import os
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import binance_data as bd
import features as ft

TRAIN_YEARS = 4          # 1m 训练矩阵使用的近期年数（更早数据仍下载入库）
S1_FIRST_YEAR = 2025     # 1s 模型数据起点（内存预算）
S1_STRIDE = 4            # 秒线→4秒bar 聚合粒度（省内存/降噪）
LABEL_1M = dict(horizon=5, band=0.0004)     # 预测未来5分钟
LABEL_1S = dict(horizon=1, band=0.0001)     # 预测未来1根聚合bar(4秒)


def log(msg):
    print(f"[{datetime.datetime.utcnow():%H:%M:%S}] {msg}", flush=True)


def build_xy(df, label_kw, stride=1):
    """特征+标签对齐；stride>1 时行采样省内存；中性带(y=-1)剔除。"""
    X = ft.make_features(df)
    y, fut = ft.make_label(df, **label_kw)
    if stride > 1:
        idx = np.arange(0, len(df), stride)
        X, y, fut = X.iloc[idx], y.iloc[idx], fut.iloc[idx]
    keep = y >= 0
    X, y, fut = X[keep], y[keep], fut[keep]
    return X.astype(np.float32), y.astype(np.int32).values, fut.astype(np.float32).values


def walk_forward(X, y, fut):
    """80/20 按时间切分训练+评估，返回(模型, 指标dict, 尾部概率/真实用于回测)。"""
    split = int(len(X) * 0.8)
    m = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=120, early_stopping=True, validation_fraction=0.1,
        random_state=42)
    t0 = time.time()
    m.fit(X.iloc[:split], y[:split])
    prob = m.predict_proba(X.iloc[split:])[:, 1]
    yte = y[split:]
    acc = float(accuracy_score(yte, (prob > 0.5)))
    auc = float(roc_auc_score(yte, prob)) if len(np.unique(yte)) > 1 else 0.5
    # IC：预测概率与真实未来收益相关性
    p, r = pd.Series(prob), pd.Series(fut[split:])
    ic = float(p.corr(r))
    # 朴素策略：p>0.55 做多 / p<0.45 做空，单边费率万五
    pos = np.where(prob > 0.55, 1.0, np.where(prob < 0.45, -1.0, 0.0))
    strat = float(np.clip(np.cumsum(fut[split:] * pos - np.abs(pos) * 5e-4).iloc[-1], -1, 5))
    bh = float(np.clip(np.expm1(fut[split:].sum()), -1, 5))
    hit = float(np.mean(pos != 0))
    metrics = dict(n_train=split, n_test=len(yte), acc=round(acc, 4),
                   auc=round(auc, 4), ic=round(ic, 5),
                   strat_ret=round(strat, 4), buyhold_ret=round(bh, 4),
                   trade_bar_pct=round(hit, 4),
                   train_min=round((time.time() - t0) / 60, 1))
    return m, metrics


def train_symbol(sym, interval, months, label_kw, stride=1, cache="cache"):
    if interval == "1s":
        # 逐月下载+聚合，内存峰值只压单月（runner 7GB 的生存策略）
        frames = []
        for (yy, mm) in months:
            d = bd.fetch_month(sym, interval, yy, mm)
            if d is not None and len(d) > 1000:
                frames.append(ft.aggregate_bars(d, stride))
                log(f"  {sym} 1s {yy}-{mm:02d} -> agg rows={len(frames[-1])}")
            del d
        df = pd.concat(frames, ignore_index=True) if frames else None
        stride = 1
    else:
        df = bd.fetch_range(sym, interval, months, cache_dir=cache, log=log)
        if df is not None:
            cut = int(df["open_time"].iloc[-1]) - TRAIN_YEARS * 365 * 86400_000
            df = df[df["open_time"] >= cut].reset_index(drop=True)
    if df is None or len(df) < 50000:
        return None, None
    X, y, fut = build_xy(df, label_kw, stride)
    if len(X) < 20000:
        return None, None
    m, met = walk_forward(X, y, fut)
    met.update(symbol=sym, interval=interval, rows_total=len(df))
    bundle = dict(model=m, feature_cols=list(X.columns), symbol=sym,
                  interval=interval, label=label_kw,
                  sklearn=__import__("sklearn").__version__)
    return bundle, met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--fast", action="store_true", help="调试：3币/近6个月")
    ap.add_argument("--only", default="", help="逗号分隔调试symbol")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    all_months = bd.month_list()
    months_1m = all_months[-6:] if args.fast else all_months
    months_1s = all_months[-2:] if args.fast else [m for m in all_months
                                                   if m[0] >= S1_FIRST_YEAR]
    if args.only:
        syms = args.only.split(",")
    else:
        syms = bd.top_symbols(args.top, fallback_file="top50.txt")
        if args.fast:
            syms = syms[:3]
    log(f"symbols={len(syms)} 1m_months={len(months_1m)} 1s_months={len(months_1s)}")

    rows = []
    for i, sym in enumerate(syms):
        try:
            bundle, met = train_symbol(sym, "1m", months_1m, LABEL_1M)
            if bundle:
                joblib.dump(bundle, f"models/{sym}_1m.joblib", compress=3)
                rows.append(met)
            else:
                log(f"{sym} skip(数据不足)")
        except Exception as e:
            log(f"{sym} FAILED: {type(e).__name__}: {e}")
        log(f"[{i+1}/{len(syms)}] {sym} done, models={len(rows)}")
        shutil.rmtree("cache", ignore_errors=True)

    if not args.only:
        for sym in ["BTCUSDT", "ETHUSDT"]:
            try:
                bundle, met = train_symbol(sym, "1s", months_1s, LABEL_1S,
                                           stride=S1_STRIDE)
                if bundle:
                    joblib.dump(bundle, f"models/{sym}_1s.joblib", compress=3)
                    rows.append(met)
                    log(f"1s {sym} {met}")
            except Exception as e:
                log(f"1s {sym} FAILED: {type(e).__name__}: {e}")
            shutil.rmtree("cache", ignore_errors=True)

    pd.DataFrame(rows).to_csv("metrics.csv", index=False)
    with open("meta.json", "w") as f:
        json.dump(dict(trained_at=str(datetime.datetime.utcnow()),
                       n_models=len(rows), train_years=TRAIN_YEARS,
                       sklearn=__import__("sklearn").__version__,
                       data_through=f"{all_months[-1][0]}-{all_months[-1][1]:02d}"), f)
    log(f"DONE models={len(rows)}")
    if not rows:
        raise SystemExit("no models trained")


if __name__ == "__main__":
    main()
