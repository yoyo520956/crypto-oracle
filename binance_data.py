# -*- coding: utf-8 -*-
"""币安历史数据下载器（data.binance.vision，国内可达）。
CI 端全量拉取；手机端只拉最近窗口做实时缓存。"""
import io
import os
import time
import zipfile
import requests

BASE = "https://data.binance.vision/data/spot/monthly/klines"
UA = {"User-Agent": "Mozilla/5.0"}


def month_list(first_year=2017, until=None):
    """生成 [2017-01 .. 上月] 的 (y, m) 列表。"""
    import datetime
    today = until or datetime.date.today()
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    out = []
    cy, cm = first_year, 1
    while (cy, cm) <= (y, m):
        out.append((cy, cm))
        cm += 1
        if cm == 13:
            cy, cm = cy + 1, 1
    return out


def zip_url(symbol, interval, y, m):
    return f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{y:04d}-{m:02d}.zip"


def fetch_month(symbol, interval, y, m, retries=3, timeout=120):
    """下载单月zip并解析为 float32 DataFrame；404返回None。"""
    url = zip_url(symbol, interval, y, m)
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            name = zf.namelist()[0]
            df = __import__("features").load_kline_csv(io.BytesIO(zf.read(name)))
            return df
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def fetch_range(symbol, interval, months, cache_dir=None, log=print):
    """按月拼接下载，可选本地parquet缓存。返回完整 DataFrame。"""
    import pandas as pd
    frames = []
    for (y, m) in months:
        fp = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            fp = os.path.join(cache_dir, f"{symbol}-{interval}-{y:04d}{m:02d}.parquet")
        if fp and os.path.exists(fp):
            frames.append(pd.read_parquet(fp))
            continue
        df = fetch_month(symbol, interval, y, m)
        if df is None:
            continue
        if fp:
            df.to_parquet(fp)
        frames.append(df)
        log(f"  {symbol} {interval} {y}-{m:02d} rows={len(df)}")
    if not frames:
        return None
    import pandas as pd
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return out


def recent_days(symbol, interval, days, limit_months=2):
    """手机端：拉最近 N 天数据（从最新几个月zip里切尾段）。"""
    import datetime
    months = month_list()[-limit_months:]
    df = fetch_range(symbol, interval, months)
    if df is None:
        return None
    cutoff = int((datetime.datetime.now() - datetime.timedelta(days=days)).timestamp() * 1000)
    return df[df["open_time"] >= cutoff].reset_index(drop=True)


STABLE_BASES = {"USDC", "USD1", "RLUSD", "FDUSD", "TUSD", "DAI", "USDP", "USDE",
                "EUR", "EURI", "AEUR", "BUSD", "USDJ", "GUSD", "LUSD", "PYUSD",
                "XUSD", "USDF", "BFUSD", "USDQ", "USDR", "OUSD", "SUSDE", "USDY"}


def top_symbols(n=50, quote="USDT", fallback_file=None):
    """按24h成交额取Top N（剔除稳定币/杠杆票；data-api.binance.vision）。"""
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr",
                         headers=UA, timeout=60)
        r.raise_for_status()
        rows = []
        for x in r.json():
            s = x["symbol"]
            if not s.endswith(quote):
                continue
            base = s[:-len(quote)]
            if base in STABLE_BASES:
                continue
            rows.append(x)
        rows.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        return [x["symbol"] for x in rows[:n]]
    except Exception:
        if fallback_file and os.path.exists(fallback_file):
            return open(fallback_file).read().split()[:n]
        raise
