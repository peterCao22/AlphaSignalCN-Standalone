## 目标与范围

本文件用于规划：当项目稳定并准备投入长期使用时，将目前“BigQuant → CSV 落盘 → 读取 CSV”的数据流，升级为“BigQuant → Postgres（统一存储）→ 读取 Postgres”的方案。

**当前不执行迁移**，仅作为蓝图与后续实施指南。

## 现状痛点（为什么要迁移）

- **性能瓶颈**：大 CSV（如 `kline_all.csv`、`moneyflow.csv`）每次读全表/多次过滤，I/O 与内存浪费明显。
- **一致性与可追溯性差**：不同脚本各自做去重/合并/清洗，容易出现口径不一致（例如停牌行、坏行覆盖、边界日期判断等）。
- **增量更新不稳定**：仅用 `latest_date` 判定“无需更新”会漏掉“最新日期已到但历史窗口有 NaN/0”的数据质量问题。
- **并发与服务化难**：CSV 不是并发友好存储，难以支撑多进程/多用户服务。

## 总体原则（数据契约）

### 统一主键

除特别说明外，统一使用：

- **主键**：`(instrument, date)`
- **instrument**：BigQuant 标准 `XXXXXX.SH / XXXXXX.SZ / XXXXXX.BJ`
- **date**：交易日日期，存储为 `DATE`（不含时分秒）

### 统一类型与空值语义

- **停牌行**：允许 `open/high/low/close` 为 NULL 且 `volume = 0`（或 NULL），作为“无交易”的合法语义
- **坏行（需要修复）**：例如 `close IS NULL AND volume > 0`、或 `close > 0 AND volume = 0`（视数据源口径），应标记并触发修复策略

### 增量更新口径

增量更新不能只依赖 `MAX(date)`；至少需要：

- `MAX(date)` 检查
- 最近窗口（例如 10 个交易日）关键字段质量检查（`close/open/high/low/volume/amount`）

## 数据集分层（建议）

建议将数据分为 3 层：

- **raw（原始事实表）**：K线、筹码、资金流、龙虎榜、TA 因子等，尽量贴近源数据口径
- **features（特征表/视图）**：训练/预测需要的派生特征（可存表或用 SQL/物化视图）
- **results（结果表）**：预测输出与日志（按 `trigger_date`、`instrument`）

## 建议表结构（初版）

### 1）日K：`raw_cn_stock_bar1d`

- **主键**：`(instrument, date)`
- **核心字段**（建议最小集）：
  - `open, high, low, close` NUMERIC
  - `volume` BIGINT
  - `amount` NUMERIC
  - `pre_close` NUMERIC NULL
  - `change_pct` NUMERIC NULL
  - `turn` NUMERIC NULL
  - `upper_limit, lower_limit` NUMERIC NULL
  - `adjust_factor` NUMERIC NULL
  - `is_limit_up` SMALLINT NULL（可在导入时算，也可后算）
- **停牌语义**：允许 `open/high/low/close` NULL 且 `volume=0`

### 2）TA 因子：`raw_cn_stock_factors_ta`

- **主键**：`(instrument, date)`
- 字段为所选 TA 因子列（保持与 BigQuant 字段名一致，或在入库时统一映射）

### 3）资金流：`raw_cn_stock_moneyflow`

- **主键**：`(instrument, date)`
- 字段为精选字段（例如 `netflow_amount_main`、`net_active_buy_amount_main` 等）

### 4）筹码：`raw_cn_stock_chips`

- **主键**：`(instrument, date)`
- 字段：`win_percent, concentration, avg_cost, ...`

### 5）龙虎榜：`raw_cn_stock_dragon_list`

- 若一天同股多条原因记录：
  - **建议主键**：`(instrument, date, reason_code)` 或 `(instrument, date, reason, rank, ...)`
  - 同时提供一个聚合视图给预测端：按 `instrument+date` 聚合净买入、上榜次数等

### 6）预测结果：`results_second_wave_predictions`

- **主键**：`(instrument, trigger_date)`
- 字段：
  - `final_prob, rule_prob, ml_prob, strength_score`
  - `features_json` JSONB（便于审计与回放）
  - `model_version` TEXT（训练时间戳/特征版本）
  - `created_at` TIMESTAMPTZ

## 分区与索引建议（关键）

### 分区

对大表（K线、TA、资金流）建议按 `date` 做月分区：

- `PARTITION BY RANGE (date)`，每月一个分区
- 优点：写入/查询更快，维护（清理历史）更方便

### 索引

每个分区建议至少有：

- `btree (instrument, date)`：按股票取时间序列（预测/特征计算常用）
- `btree (date)`：按交易日批量取全市场/样本（训练/回测常用）

## 写入策略（UPSERT）

统一使用 `INSERT ... ON CONFLICT (instrument, date) DO UPDATE`：

- 更新时要有“质量优先”策略：
  - 若新记录关键字段为空/为 0（且不属于停牌语义），不应覆盖旧记录的有效值
  - 必要时增加 `quality_score` 或 `updated_at`、`source` 字段便于比较

## 数据质量处理（停牌 vs 坏行）

### 停牌行（允许）

满足以下典型模式可视为停牌：

- `close IS NULL` 且 `volume = 0`（amount 可能 NULL）

### 坏行（需要修复）

需要纳入“强制重拉/修复”的典型模式：

- `close IS NULL AND volume > 0`
- `close > 0 AND volume = 0`（需结合数据源口径确认）
- 最近窗口内出现大量 NULL/0 且非停牌（例如复牌后仍连续多日 NULL）

## 迁移路线（未来实施）

### 阶段0：准备（不改业务）

- 明确各数据集的字段映射、主键、空值语义（本文件）
- 为每张表定义“最小可用字段集”（避免一次性把全字段导入）

### 阶段1：双写（CSV + Postgres）

- 下载脚本仍落 CSV（保留现有稳定性）
- 同时新增入库步骤（Upsert 到 Postgres）
- 关键：入库前做统一清洗/去重/质量策略

### 阶段2：读路径切换（Postgres 为主）

- 预测/训练改为从 Postgres 读取
- CSV 仅作调试导出

### 阶段3：只写 Postgres（可选）

- 彻底移除大 CSV 读写路径
- 增加备份、监控、例行维护（VACUUM/分区滚动）

## 与现有代码的对齐点（后续改造清单）

- 把分散在各脚本里的：
  - `read_csv/concat/drop_duplicates`
  - “latest_date 判定”
  - 停牌/坏行处理
 统一收敛到一个 `DataStore` 层（Postgres 实现），提供：
  - `upsert_*()`
  - `get_*_by_instrument_and_range()`
  - `get_latest_trading_date_available()`

## 备注

- 本方案以“后续稳定投入”为前提；在当前快速迭代阶段，继续使用 CSV 是合理的。
- 等开始实施迁移时，可在此文件基础上补充：具体 SQL DDL、索引/分区创建脚本、以及一次性回灌脚本与校验脚本。

