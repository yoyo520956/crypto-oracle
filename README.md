# Crypto Oracle 加密神谕

币安全市场 AI 量化预测管线：GitHub Actions 免费算力训练，iPhone(iSH) 本地实时推理。

## 架构

```
data.binance.vision(全历史K线) → train.py(CI: 因子→HistGB逐币模型→回测)
    → artifacts(models/) → live/live_server.py(手机: 每秒轮询→特征→推理→仪表盘)
```

- **模型**：每币一个 sklearn HistGradientBoosting 二分类器
  - `*_1m.joblib`：Top50 币，预测未来 5 分钟方向（训练用近 4 年 1m 全历史，数据自 2017 最早月）
  - `*_1s.joblib`：BTC/ETH 秒级深度模型（2025+ 秒线聚合 4s bar，预测未来约 4 秒）
- **因子**：Alpha158 精简 36 列（多周期收益/动量/波动/量能/RSI/VWAP/时间编码），
  训练与推理共用 `features.py` 同一实现
- **重训**：每周一 UTC04:00 自动；也可 Actions → Train Oracle → Run workflow

## 用法（手机端）

```sh
cd live && python3 live_server.py --port 8788
# 浏览器打开 http://127.0.0.1:8788
```

## 诚实声明

秒级预测信噪比极低，模型输出是**统计倾向参考**，扣费后不构成任何盈利保证。
本项目仅研究用途，投资盈亏自负。
