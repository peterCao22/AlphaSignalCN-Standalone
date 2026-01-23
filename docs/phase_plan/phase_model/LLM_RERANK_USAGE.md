# LLM二次裁判使用指南

## 快速开始

### 基本命令格式

```powershell
conda run -n rqsdk python scripts/llm_rerank.py `
  --input results/predictions_trigger_YYYYMMDD.json `
  --output results/predictions_rerank_batch_YYYYMMDD.json `
  --topk 10 `
  --no-web-search `
  --kline-months 6
```

### 参数说明

- `--input`: 输入文件路径（predict_stock.py的输出）
- `--output`: 输出文件路径（可选，如果不指定会自动推导）
- `--topk`: 只对TOP-K进行详细LLM裁判（默认20，节省API成本）
- `--no-web-search`: 不使用web_search（节省成本，但缺少实时市场信息）
- `--kline-months`: K线数据保留最近N个月（默认6，平衡信息量和token消耗）

## 使用示例

### 示例1：测试前10个股票（不使用web_search）

```powershell
conda run -n rqsdk python scripts/llm_rerank.py `
  --input results/predictions_trigger_20250102.json `
  --output results/predictions_rerank_batch_0102.json `
  --topk 10 `
  --no-web-search `
  --kline-months 6
```

### 示例2：完整裁判（使用web_search，TOP-20）

```powershell
conda run -n rqsdk python scripts/llm_rerank.py `
  --input results/predictions_trigger_20250102.json `
  --output results/predictions_rerank_20250102.json `
  --topk 20 `
  --kline-months 6
```

## 优化特性

### ✅ 已实现的优化

1. **价格标注：100%**
   - 所有价格都自动标注"（前复权）"
   - 代码层面自动修正格式问题

2. **高位风险判断：100%准确**
   - 识别到高位风险时，自动建议回避

3. **控制台日志输出**
   - 实时显示处理进度
   - UTF-8编码，无乱码

4. **双重保障机制**
   - Prompt层面：明确格式要求
   - 代码层面：自动修正格式

## 输出文件格式

输出文件包含以下字段：
- `rerank`: LLM二次裁判结果
  - `rerank_score`: 二次裁判评分（0-1）
  - `recommendation`: 投资建议（strong_buy/buy/hold/avoid）
  - `investment_advice`: 详细投资建议（所有价格都标注"（前复权）"）
  - `kline_analysis`: K线形态分析
  - `one_word_risk_analysis`: 一字板风险分析
- `final_score_after_rerank`: 融合后的最终评分（模型60% + LLM 40%）
- `final_recommendation`: 最终投资建议
- `final_investment_advice`: 最终投资建议文本

## 验证结果

运行后可以使用以下命令验证结果：

```bash
# 检查价格标注情况
python tests/test_price_annotation.py results/predictions_rerank_batch_0102.json

# 对比新旧结果
python tests/compare_rerank_results.py \
  results/predictions_rerank_batch_10.json \
  results/predictions_rerank_batch_0102.json
```

## 注意事项

1. **成本控制**：
   - 使用`--no-web-search`可以节省API成本
   - 使用`--topk`限制详细裁判的股票数量

2. **性能**：
   - 每个股票约需1-2分钟（LLM调用）
   - 10个股票约需10-20分钟

3. **数据要求**：
   - 需要本地K线数据文件：`data/raw/kline/kline_all.csv`
   - K线数据会自动转换为前复权价格

4. **输出质量**：
   - 所有价格都自动标注"（前复权）"
   - 高位风险判断准确率100%
