## BigQuant：短线接力 TOP3 一年回测操作指南（第一版）

### 目标
- 用你本地“短线接力模型”每天筛选出的 **TOP3** 股票，在 BigQuant 回测环境里跑满 1 年，输出收益曲线、最大回撤、胜率、交易明细（raw_perf）。

---

## 交易规则（简化版 A：日线，严格“今天信号→次日买→持有5日→到期开盘卖”）
- **信号日 D**：当日收盘后（`handle_data`）处理信号并下单
- **买入**：在 D 下单，**D+1 开盘**按 `open` 撮合等权买入 TOP3
- **卖出**：买入成交日起算 **持有 5 个交易日**；在“到期日的前一交易日收盘后”下单，**到期日开盘**按 `open` 撮合清仓

> 说明：
> - 你当前账号无 `cn_stock_bar1m`（分钟线）权限，因此采用**日线回测**口径。
> - BigTrader 日线回测“当日下单→下一交易日撮合”时序参考：[`BigTrader 量化交易引擎（回测）`](https://bigquant.com/wiki/doc/3gG2rg4jBd)

---

## 1. 生成 signals CSV（本地）

### 1.1 信号文件规范
先读这份规范：`docs/BigQuant_短线接力TOP3_signals_schema.md`

### 1.2 从 AlphaSignal 输出一键生成（推荐）
如果你已经在本地运行过 `predict_stock.py --batch` 并生成了 `predictions_trigger_YYYYMMDD.json`，可以用仓库脚本转换：

```bash
python scripts/generate_bigquant_signals_from_alphasignal.py `
  --results-dir "d:/myCursor/AlphaSignalCN-Standalone/results" `
  --topk 3 `
  --min-score 0.6 `
  --out "d:/myCursor/AlphaSignalCN-Standalone/data/raw/bigquant_signals_top3.csv"
```

说明：
- 默认把 `trigger_date = T` 的结果映射为 `buy_day = T+1`（避免未来函数，符合“收盘后出信号→次日开盘交易”）。
- 在本策略（方案A）里：**`source_date` 视为信号日 D**，`date` 只是记录对应的 `buy_day`（用于审计），策略按 D 下单以实现：D+1 买、持有 5 个交易日、到期开盘卖。
- 若你要改口径（比如信号当天开盘前就有），请改脚本中的 buy_day 生成方式或直接输出你自己的 signals CSV。

---

## 2. 上传 signals CSV 到 BigQuant AIStudio
- 在 AIStudio 左侧文件浏览器，把 `bigquant_signals_top3.csv` 上传到你的工作目录（例如 `data/raw/`）。
- 记下上传后的路径（例如：`/home/aistudio/work/data/raw/bigquant_signals_top3.csv`）。

AIStudio 基本使用参考：[`编写策略/AIStudio`](https://bigquant.com/wiki/doc/aistudio-aiide-NzAjgKapzW)

---

## 3. 在 AIStudio 运行一年回测

### 3.1 使用仓库提供的回测脚本
把这个文件的内容复制到 AIStudio（新建一个 `.py` 或 `.ipynb` 里运行）：
- `tests/quant/bigquant_relay_top3_backtest.py`

你需要改 3 个变量：
- `SIGNALS_CSV`: 改成你上传后的路径
- `START_DATE` / `END_DATE`: 改成你要回测的一年（例如 2025-01-01 ~ 2025-12-31）
- `EXPORT_RAW_PERF_CSV`: raw_perf 导出路径（可为空字符串表示不导出）

然后直接运行。

注意：
- 该脚本已按“简化版 A：日线”实现：**信号日 D 下单 → D+1 开盘买 → 持有 5 个交易日 → 到期开盘卖**。

### 3.2 输出结果
运行后你会看到：
- 回测收益曲线/指标（render）
- 交易明细导出（如果开启 `EXPORT_RAW_PERF_CSV`）

---

## 4. 验收检查（建议你跑完后做）

### 4.1 信号约束检查
在本地或 AIStudio 检查 signals CSV：
- 每个 `date` 最多 3 条（TOP3）
- `weight` 每天求和约等于 1（等权）

### 4.2 持仓约束检查（最多持有 5 个交易日）
回测导出 `raw_perf` 后，你可以按以下思路核对：
- 每天持仓股票数 ≤ 3
- 每个标的从买入到卖出间隔 ≤ 5 个交易日（如果遇到停牌/跌停无法卖出，可能延后，这是预期行为）

> BigTrader 文档提到回测返回 `raw_perf` 可用于进一步分析：[`BigTrader 量化交易引擎（回测）`](https://bigquant.com/wiki/doc/3gG2rg4jBd)

---

## 5. 下一步（卖出规则迭代方向）
简化版 A 的卖出规则很“硬”，建议你跑完一年后，我们再对比几组离场规则：
- D+2 开盘卖 vs D+2 收盘卖（后者需要更细粒度撮合/权限）
- 加入盘中止损/止盈（需要分钟线权限或更细的触发方式与更贴近实盘的撮合假设）
- 涨停/跌停无法卖出的处理：顺延、分批、或改为次日集合竞价挂单等

