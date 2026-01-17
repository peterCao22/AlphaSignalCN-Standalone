# 技术分析因子使用指南

本指南说明如何使用 BigQuant 的专业技术分析因子数据 (`cn_stock_factors_ta`)。

## 概述

**为什么使用 BigQuant 技术分析因子？**

1. ✅ **专业计算**: 由 BigQuant 专业团队计算，确保准确性
2. ✅ **数据完整**: 覆盖全市场所有A股
3. ✅ **指标丰富**: 包含20+个常用技术指标
4. ✅ **避免重复计算**: 不需要在本地重复计算MA、RSI等指标

## 数据源信息

**BigQuant 数据表**: `cn_stock_factors_ta`

**数据更新**: 每日更新（T+0）

**覆盖范围**: 全A股市场

**历史数据**: 支持回溯历史数据

## 包含的技术指标

### 移动平均线 (MA)
- `m_avg_price_5`: 5日均价
- `m_avg_price_10`: 10日均价
- `m_avg_price_20`: 20日均价
- `m_avg_price_60`: 60日均价
- `m_avg_price_120`: 120日均价
- `m_avg_amount_5`, `m_avg_amount_10`, `m_avg_amount_60`: 成交额均值

### 相对强弱指标 (RSI)
- `rsi_6`: 6日RSI
- `rsi_12`: 12日RSI
- `rsi_14`: 14日RSI（最常用）
- `rsi_24`: 24日RSI

### MACD
- `macd`: MACD值
- `macd_dea`: DEA线（信号线）
- `macd_dif`: DIF线（快线）

### 布林带 (Bollinger Bands)
- `boll_up`: 上轨
- `boll_mid`: 中轨
- `boll_down`: 下轨

### KDJ 指标
- `kdj_k`: K值
- `kdj_d`: D值
- `kdj_j`: J值

### 其他指标
- `cci`: 顺势指标
- `cci_20`: 20日CCI
- `willr`: 威廉指标
- `willr_14`: 14日威廉指标

## 使用步骤

### 1. 下载技术分析因子数据

首次使用或定期更新时运行：

```bash
# 激活环境
conda activate rqsdk

# 进入目录
cd d:\myCursor\AlphaSignalCN-Standalone

# 下载数据（默认最近90天）
python scripts\download_ta_factors.py
```

**下载后的文件**:
- 保存位置: `data/raw/ta_factors.csv`
- 数据保留: 自动保留最近180天
- 增量下载: 自动检测已有数据，只下载新数据

### 2. 模型训练时自动使用

训练脚本 `train_model.py` 已经集成了技术分析因子的加载逻辑：

```python
# 训练时会自动：
# 1. 检查 data/raw/ta_factors.csv 是否存在
# 2. 如果存在，优先使用 BigQuant 的专业计算数据
# 3. 如果不存在，才会手动计算技术指标
```

**字段映射**:
```python
BigQuant 字段      ->  本地字段
m_avg_price_5     ->  ma5
m_avg_price_10    ->  ma10
m_avg_price_20    ->  ma20
m_avg_price_60    ->  ma60
rsi_14            ->  rsi
```

### 3. 运行训练

```bash
python scripts\train_model.py
```

**日志输出示例**:
```
2026-01-14 19:30:00 - INFO - 加载 BigQuant 技术分析因子数据...
2026-01-14 19:30:01 - INFO - ✓ 技术分析因子加载完成: 450000 条记录
2026-01-14 19:30:02 - INFO - ✓ 技术分析因子已合并到K线数据
```

## 数据更新策略

### 推荐更新频率

| 场景 | 更新频率 |
|------|---------|
| 日常训练 | 每周1次（周末） |
| 数据准备 | 训练前1天 |
| 紧急补数据 | 随时运行 |

### 自动化更新（可选）

可以设置定时任务每周自动更新：

**Windows 任务计划程序**:
```batch
@echo off
call conda activate rqsdk
cd /d D:\myCursor\AlphaSignalCN-Standalone
python scripts\download_ta_factors.py >> logs\ta_factors_download.log 2>&1
```

保存为 `scripts/update_ta_factors.bat`，然后在任务计划程序中设置每周日晚上运行。

## 数据验证

### 检查数据完整性

```python
import pandas as pd

df = pd.read_csv('data/raw/ta_factors.csv')

print(f"总记录数: {len(df)}")
print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
print(f"股票数量: {df['instrument'].nunique()}")

# 检查各字段的有效率
for col in ['ma5', 'ma10', 'ma20', 'ma60', 'rsi']:
    valid_pct = df[col].notna().sum() / len(df) * 100
    print(f"{col}: {valid_pct:.1f}% 有效")
```

### 对比手动计算与BigQuant数据

```python
import pandas as pd

# 读取BigQuant技术分析因子
ta_df = pd.read_csv('data/raw/ta_factors.csv')

# 读取K线数据
kline_df = pd.read_csv('data/raw/kline/kline_all.csv')

# 合并对比
test_stock = '000001.SZ'
test_date = '2026-01-10'

ta_data = ta_df[(ta_df['instrument'] == test_stock) & 
                (ta_df['date'] == test_date)]
kline_data = kline_df[(kline_df['instrument'] == test_stock) & 
                      (kline_df['date'] == test_date)]

print("BigQuant MA5:", ta_data['m_avg_price_5'].values)
print("手动计算 MA5:", "需要从K线数据计算")
```

## 常见问题

### Q1: 下载时提示"Permission denied"

**原因**: 超过200MB数据限制。

**解决**: 已在脚本中优化，只下载最近90天数据，并自动清理超过180天的旧数据。

### Q2: 某些股票的技术指标是NaN

**原因**: 
1. 新股上市不足计算周期（如MA60需要至少60个交易日）
2. 停牌期间没有交易数据

**解决**: 这是正常现象，训练脚本会自动处理NaN值（填充为0或使用手动计算）。

### Q3: 技术指标与其他平台数据不一致

**原因**: 
1. 复权方式不同（前复权 vs 后复权 vs 不复权）
2. 计算方法略有差异

**建议**: BigQuant 的技术分析因子基于专业算法，建议以其为准。

### Q4: 下载很慢

**原因**: 数据量大或网络慢。

**优化**:
1. 使用增量下载（脚本已实现）
2. 减少下载天数（修改 `days=90` 参数）
3. 在网络较好的时间段下载

### Q5: 训练时没有使用BigQuant数据

**检查**:
1. 确认文件存在: `data/raw/ta_factors.csv`
2. 查看训练日志，是否有"加载 BigQuant 技术分析因子数据"
3. 如果报错，查看具体错误信息

## 与增强特征的关系

| 数据类型 | 下载脚本 | 保存位置 | 用途 |
|---------|---------|---------|------|
| 技术分析因子 | `download_ta_factors.py` | `data/raw/ta_factors.csv` | MA、RSI等基础技术指标 |
| 增强特征 | `download_enhanced_features.py` | `data/raw/*.csv` | 热度、概念、竞价等高级特征 |

**两者互补，都需要下载！**

## 最佳实践

1. **首次使用**: 先下载技术分析因子和增强特征，再训练模型
2. **定期更新**: 每周更新一次所有数据
3. **训练前检查**: 确认数据日期是最新的
4. **备份数据**: 定期备份 `data/raw/` 目录
5. **监控日志**: 关注下载和训练日志中的警告信息

## 下一步

下载完技术分析因子后，建议：

1. 下载增强特征: `python scripts\download_enhanced_features.py`
2. 运行模型训练: `python scripts\train_model.py`
3. 测试预测效果: `python predict_stock.py 300346`

更多信息请参考:
- [模型训练指南](MODEL_TRAINING_GUIDE.md)
- [增强特征指南](ENHANCED_FEATURES_GUIDE.md)
- [数据流说明](DATA_FLOW_EXPLANATION.md)
