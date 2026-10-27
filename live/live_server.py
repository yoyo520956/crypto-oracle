#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""加密神谕·手机端实时推理服务（iSH）。
线程：全市场报价轮询 / K线缓存刷新 / 每秒模型推理 / HTTP仪表盘。
数据源：data-api.binance.vision（REST，国内可达）。"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
import pandas as pd
import requests

import features as ft

API = "https://data-api.binance.vision"
UA = {"User-Agent": "Mozilla/5.0"}
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("ORACLE_MODELS", os.path.join(HERE, "models"))

STATE = {"tick": 0, "updated": 0, "prices": {}, "preds": {}, "chg": {}}
LOCK = threading.Lock()
MODELS = {}   # sym -> {"1m": bundle, "1s": bundle}
CACHE = {}    # (sym, interval) -> DataFrame 滚动K线缓存


def log(m):
    print(f"[live] {m}", flush=True)


def api_get(path, **params):
    r = requests.get(API + path, params=params, headers=UA, timeout=15)
    r.raise_for_status()
    return r.json()


def klines_df(sym, interval, limit):
    raw = api_get("/api/v3/klines", symbol=sym, interval=interval, limit=limit)
    df = pd.DataFrame([r[:12] for r in raw], columns=ft.KLINES_COLS)
    for c in ft.NUM_COLS:
        df[c] = df[c].astype(np.float32)
    df["open_time"] = df["open_time"].astype(np.int64)
    return df


def ticker_loop():
    """每~3秒全市场报价（一次请求拿全部币）。"""
    while True:
        try:
            rows = api_get("/api/v3/ticker/price")
            with LOCK:
                STATE["prices"] = {r["symbol"]: float(r["price"]) for r in rows}
                STATE["updated"] = time.time()
        except Exception as e:
            log(f"ticker: {e}")
        time.sleep(3)


def refresh_cache(sym):
    try:
        if f"{sym}_1m" in MODELS:
            CACHE[(sym, "1m")] = klines_df(sym, "1m", 300)
        if f"{sym}_1s" in MODELS:
            CACHE[(sym, "1s")] = ft.aggregate_bars(klines_df(sym, "1s", 900), 4)
    except Exception as e:
        log(f"cache {sym}: {e}")


def cache_loop():
    syms = list(MODELS.keys())
    while True:
        for s in syms:
            refresh_cache(s)
            time.sleep(0.15)   # 温和限速：50币约8秒刷完一轮
        time.sleep(45)


def infer_loop():
    """每秒：缓存K线+最新价补成partial bar → 特征 → 模型 → 概率。"""
    while True:
        t0 = time.time()
        with LOCK:
            prices = dict(STATE["prices"])
        out = {}
        for sym, msets in MODELS.items():
            px = prices.get(sym)
            if px is None:
                continue
            row = {"price": px}
            for itv, bundle in msets.items():
                df = CACHE.get((sym, itv))
                if df is None or len(df) < 200:
                    continue
                d = df.copy()
                i = len(d) - 1
                d.loc[i, "close"] = px                     # partial bar 贴最新价
                d.loc[i, "high"] = max(d.loc[i, "high"], px)
                d.loc[i, "low"] = min(d.loc[i, "low"], px)
                X = ft.make_features(d)
                p = float(bundle["model"].predict_proba(
                    X.iloc[[-1]].astype(np.float32))[0][1])
                row[f"p_{itv}"] = round(p, 4)
            with LOCK:
                c = STATE["chg"]
            closes = CACHE.get((sym, "1m"))
            if closes is not None and len(closes) > 61:
                cs = closes["close"].values
                row["chg"] = {"1m": round(px / cs[-2] - 1, 5),
                              "5m": round(px / cs[-6] - 1, 5),
                              "1h": round(px / cs[-61] - 1, 5)}
            out[sym] = row
        with LOCK:
            STATE["preds"] = out
            STATE["tick"] += 1
        time.sleep(max(0.15, 1.0 - (time.time() - t0)))


def load_models():
    for fn in sorted(os.listdir(MODELS_DIR)):
        if not fn.endswith(".joblib"):
            continue
        name, itv = fn[:-7].rsplit("_", 1)
        b = joblib.load(os.path.join(MODELS_DIR, fn))
        MODELS.setdefault(name, {})[itv] = b
    log(f"models loaded: {len(MODELS)} symbols")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/state"):
            with LOCK:
                body = json.dumps({"tick": STATE["tick"],
                                   "updated": STATE["updated"],
                                   "preds": STATE["preds"]}).encode()
            self._send(body, "application/json")
        elif self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                self._send(f.read(), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    args = ap.parse_args()
    load_models()
    for th in [ticker_loop, cache_loop, infer_loop]:
        threading.Thread(target=th, daemon=True).start()
    log(f"dashboard: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
