# LLM二次裁判改进方案

## 问题分析

### 当前实现的问题

1. **代码固定计算形态，灵活性差**
   - 突破日、顶部确认日等形态通过固定规则计算
   - 只能识别预定义的形态模式
   - 无法适应复杂的市场情况

2. **一字板判断过于刚性**
   - 代码直接判断并标注"坚决放弃"
   - LLM只是读取标注，没有判断空间
   - 无法根据具体情况（如一字板出现的位置、时间、次数）灵活判断

3. **缺乏可调优性**
   - 形态识别规则硬编码在代码中
   - 无法根据LLM返回结果进行调优
   - 无法训练LLM提高识别准确率

---

## 改进方案

### 核心思路

**代码负责数据准备，LLM负责形态判断**

- **代码职责**：计算技术指标、统计基础信息、提供原始数据
- **LLM职责**：理解K线形态、判断风险、给出投资建议

---

## 具体改进

### 1. 一字板处理改进

#### 当前实现
```python
# 代码直接判断并标注
if is_one_word_limit_up:
    lines.append("### ⚠️ 一字板（开盘即涨停，无换手，风险极高！坚决放弃！）")
```

#### 改进方案
```python
# 代码只统计信息，不判断风险
one_word_stats = {
    'count': len(one_word_days),  # 一字板出现次数
    'dates': [d.strftime('%Y-%m-%d') for d in one_word_days['date']],  # 出现日期
    'positions': ['高位' if price > ma60 else '低位' for price in one_word_days['close']],  # 出现位置
    'time_to_trigger': [days_before_trigger for d in one_word_days['date']],  # 距离触发日期的天数
    'volume_ratios': [v_ratio for v_ratio in one_word_days['volume_ratio']],  # 量比
    'context': {
        'before_5d_limit_up_count': [...],  # 每个一字板前5个交易日的涨停次数
        'before_5d_price_range': [...],  # 每个一字板前5个交易日的价格区间
        'before_5d_avg_volume': [...]  # 每个一字板前5个交易日的平均成交量
    }
}
```

**传递给LLM的格式：**
```
### 一字板统计信息
- 出现次数：2次
- 出现日期：2024-10-29, 2024-12-02
- 出现位置：高位（2024-10-29），高位（2024-12-02）
- 距离触发日期：78天前，45天前
- 量比：0.15（2024-10-29），0.12（2024-12-02）
- 上下文信息：
  - 2024-10-29前5个交易日：涨停0次，价格区间8.50-9.20元，平均成交量3.5亿
  - 2024-12-02前5个交易日：涨停1次，价格区间12.00-13.50元，平均成交量4.2亿
```

**LLM判断逻辑（在Prompt中说明）：**
- 如果触发日期前5个交易日有一字板，风险极高，建议回避
- 如果一字板出现在高位且距离触发日期较近，风险较高
- 如果一字板出现在低位且距离触发日期较远，需要结合其他因素判断
- 如果一字板出现次数较多（≥2次），风险累积

---

### 2. K线形态判断改进

#### 当前实现
```python
# 代码固定计算突破日
breakthrough_days = self._identify_breakthrough_days(df, symbol=symbol)
# 代码固定计算顶部确认日
top_days = self._identify_top_confirmation_days(df)
```

#### 改进方案

**采用清洗整理后的结构化描述格式（推荐）**

参考 `TASK_3_1_LLM_RERANK_PLAN.md` 中的格式，提供清洗整理后的结构化描述，而不是原始表格数据。

```python
# 代码提供关键节点和统计信息，不判断形态
kline_summary = f"""
## K线数据概览（{start_date} 至 {end_date}，共{len(df)}个交易日）

### 基本统计
- 起始价: {start_price:.2f}元
- 最新价: {latest_price:.2f}元
- 期间涨跌幅: {total_change:.2f}%
- 价格区间：最低 {min_price:.2f}元（{min_date}），最高 {max_price:.2f}元（{max_date}）
- 当前价格：{latest_price:.2f}元（相对最低价+{((latest_price/min_price)-1)*100:.0f}%，相对最高价{((latest_price/max_price)-1)*100:.0f}%）
- 平均成交量：{avg_volume:.0f}，最大成交量：{max_volume:.0f}（{max_volume_date}）
- 最大回撤：{max_drawdown:.2f}%（{drawdown_date}相对{high_date}高点）

### 关键节点（按时间顺序）
- {key_node_1_date}：{key_node_1_description}
- {key_node_2_date}：{key_node_2_description}
- ...

### 价格走势描述（关键区间）
- {interval_1_start}至{interval_1_end}：从{price_start}元{涨跌}至{price_end}元（{pct_change}%），成交量{放大/萎缩}，{是否站上/跌破均线}
- {interval_2_start}至{interval_2_end}：从{price_start}元{涨跌}至{price_end}元（{pct_change}%），成交量{放大/萎缩}，{是否站上/跌破均线}
- ...

### 技术指标（最近N个交易日的关键数据）
- MA5: {ma5_recent_values}（最近5个交易日）
- MA10: {ma10_recent_values}（最近5个交易日）
- MA20: {ma20_recent_values}（最近5个交易日）
- 成交量: {volume_recent_values}（最近5个交易日）
- 量比: {volume_ratio_recent_values}（最近5个交易日）
- RSI: {rsi_recent_values}（最近5个交易日）

### 关键事件标记
- 涨停日：{limit_up_dates}（共{limit_up_count}次）
- 跌停日：{limit_down_dates}（共{limit_down_count}次）
- 放量日：{high_volume_dates}（成交量>2倍均量，共{high_volume_count}次）

### 当前技术形态
- 当前价格: {latest_price:.2f}元
- MA5: {latest_ma5:.2f}元，MA10: {latest_ma10:.2f}元，MA20: {latest_ma20:.2f}元，MA60: {latest_ma60:.2f}元
- 价格相对MA20位置: {((latest_price/latest_ma20)-1)*100:.2f}%
- 成交量: {latest_volume:.0f}，量比: {latest_volume_ratio:.2f}
- RSI: {latest_rsi:.2f}
"""
```

**关键改进点：**

1. **价格走势描述**：按关键区间描述价格变化和成交量情况，便于LLM理解整体趋势
2. **关键节点**：列出所有重要事件（涨停、跌停、放量等），但不判断形态
3. **技术指标**：提供最近N个交易日的关键指标数据，便于LLM判断形态
4. **不判断形态**：代码不计算突破日、顶部确认日，只提供数据和描述，让LLM自己识别

**LLM判断任务（在Prompt中说明）：**
- **突破日识别**：从价格走势描述中识别"前几天放量站上了20日均线，后面有比较深的回踩后重新再次站稳，带有量能"的形态
- **顶部确认日识别**：从价格走势描述和技术指标中识别"在回调的过程中，看它是否跌破了短线的三个重要均线：破5日线，10日线连续2天不收回。算高点 破10日线1日不收回，算高点"的形态
- **底部识别**：从价格走势描述中判断是否真底部（前期充分调整≥20%、量能放大、站稳确认）
- **高位风险识别**：从价格走势描述中判断是否高位风险（连续涨停后处于高位、量能萎缩）

**优势：**
- 信息量适中，Token消耗可控
- 保留了关键信息，便于LLM理解
- 清洗整理后的格式更易理解，比原始表格更高效
- 可以通过调整描述粒度进行优化

---

### 3. Prompt优化

#### 当前Prompt问题
- 要求LLM"检查涨停日是否为一字板"，但代码已经识别好了
- 要求LLM识别突破日、顶部确认日，但代码已经计算好了

#### 改进后的Prompt

```markdown
## 任务
对给定的股票进行二次裁判，评估其是否符合"短期接力"条件，并识别潜在风险。
你需要结合K线形态、实时市场信息和模型预测结果，给出综合判断。

## 输入数据

### K线数据（最近3-6个月）
{kline_summary}

**注意**：K线数据包含：
- 基本统计信息（价格区间、涨跌幅、成交量等）
- 技术指标数据（MA5/MA10/MA20/MA60、成交量、RSI等）
- 关键事件标记（涨停日、跌停日、放量日）
- 价格走势描述（关键区间的价格变化和成交量情况）

**你需要从K线数据中识别：**
1. **突破日**：前几天放量站上了20日均线，后面有比较深的回踩后重新再次站稳，带有量能
2. **顶部确认日**：在回调的过程中，看它是否跌破了短线的三个重要均线：
   - 破5日线，10日线连续2天不收回。算高点
   - 破10日线1日不收回，算高点
3. **底部识别**：是否真底部（前期充分调整≥20%、量能放大、站稳确认）
4. **高位风险**：是否高位风险（连续涨停后处于高位、量能萎缩）

### 一字板统计信息
{one_word_stats}

**注意**：一字板统计信息包含：
- 出现次数、出现日期、出现位置（高位/低位）
- 距离触发日期的天数
- 每个一字板前5个交易日的上下文信息

**你需要根据一字板信息判断风险：**
- 如果触发日期前5个交易日有一字板，风险极高，建议回避
- 如果一字板出现在高位且距离触发日期较近，风险较高
- 如果一字板出现次数较多（≥2次），风险累积
- 需要结合一字板出现的位置、时间、次数综合判断

### 模型预测结果（简要）
- 连板高度：{consecutive_count} 板
- 位置判断：{pre_position}（高位/低位/中位）
- 综合概率：{final_prob:.3f}（模型预测的二波概率）
- 历史成功率：{pattern_success_rate:.1%}（相似历史案例的成功率）
- 模型结论：{conclusion}

### 实时市场信息（通过搜索获取）
#### 股票相关新闻
{stock_news_summary}

#### 市场舆情和热点
{market_sentiment_summary}

#### 板块/概念炒作情况
{sector_hotspot_summary}

## 判断标准

### 1. 位置风险识别
- **高位风险**：连续涨停后处于高位，容易出现"顶部派发"、"A杀反抽"
  - **特别提示**：如果触发日期前5个交易日有一字板，风险极高，建议回避
  - 如果一字板出现在高位且距离触发日期较近，风险较高
- **低位机会**：处于相对低位，有"真底部"特征（量能放大、站稳确认）
- **中位观察**：处于中位，需要结合其他因素判断

### 2. 接力潜力评估
- **强接力信号**：
  - 连板高度适中（2-4板），未过度炒作
  - 成交量放大，资金活跃
  - 板块/概念处于热点，有持续性
  - 市场情绪良好，涨停氛围浓厚
  - **技术形态健康**：有突破日确认，无顶部确认日
- **弱接力信号**：
  - 连板过高（≥5板），风险累积
  - 成交量萎缩，资金流出
  - 板块/概念退潮，缺乏持续性
  - 市场情绪转弱，涨停数量下降
  - **技术形态不健康**：无突破日确认，有顶部确认日

### 3. 底部识别
- **真底部特征**：
  - 前期有充分调整（回撤≥20%）
  - 量能放大，资金流入
  - 技术形态突破关键位（从K线数据中识别突破日：放量站上均线→回踩→重新站稳）
  - 板块/概念处于启动期
- **假底部/高位特征**：
  - 调整不充分，仍在高位
  - 量能萎缩，资金流出
  - 技术形态未突破（从K线数据中未识别到突破日）
  - 板块/概念处于退潮期

### 4. 市场环境综合
- **有利环境**：市场情绪高涨、涨停数量多、板块轮动活跃
- **不利环境**：市场情绪低迷、涨停数量少、板块轮动停滞

## 输出格式（JSON）

请严格按照以下JSON格式输出，必须包含以下字段：

{{
  "symbol": "{symbol}",
  "trigger_date": "{trigger_date}",
  "rerank_score": 0.0-1.0,
  "risk_level": "low|medium|high",
  "position_judgment": "high|middle|low|bottom",
  "is_bottom": true|false,
  "relay_potential": "strong|moderate|weak",
  "market_environment": "favorable|neutral|unfavorable",
  "kline_analysis": {{
    "breakthrough_days": [
      {{
        "date": "YYYY-MM-DD",
        "first_break_date": "YYYY-MM-DD",
        "pullback_date": "YYYY-MM-DD",
        "pullback_pct": "X.XX%",
        "description": "详细描述突破序列"
      }}
    ],
    "top_confirmation_days": [
      {{
        "date": "YYYY-MM-DD",
        "high_date": "YYYY-MM-DD",
        "high_price": 0.00,
        "description": "详细描述顶部确认"
      }}
    ],
    "is_bottom_confirmed": true|false,
    "bottom_reason": "底部判断依据...",
    "is_high_risk": true|false,
    "high_risk_reason": "高位风险判断依据..."
  }},
  "one_word_risk_analysis": {{
    "has_risk": true|false,
    "risk_level": "low|medium|high|extreme",
    "analysis": "根据一字板统计信息，分析风险程度和原因...",
    "recommendation": "如果风险极高，建议回避；如果风险中等，需要结合其他因素判断"
  }},
  "reasons": {{
    "position_risk": "具体的位置风险分析...",
    "relay_analysis": "接力潜力分析...",
    "bottom_judgment": "底部判断依据...",
    "market_context": "市场环境分析..."
  }},
  "recommendation": "strong_buy|buy|hold|avoid",
  "investment_advice": "具体的投资建议文本...",
  "action_summary": "操作建议：建议买入|建议观望|建议回避|建议减仓",
  "confidence": 0.0-1.0,
  "key_risks": ["风险点1", "风险点2"],
  "key_opportunities": ["机会点1", "机会点2"]
}}
```

---

## 实施步骤

### 阶段1：一字板处理改进（优先级：高）

1. **修改 `_compress_kline_data()` 方法**
   - 移除一字板的直接风险判断
   - 改为统计一字板信息（次数、日期、位置、上下文）
   - 生成一字板统计信息文本

2. **更新Prompt模板**
   - 添加"一字板统计信息"部分
   - 说明LLM如何根据统计信息判断风险
   - 在输出格式中添加 `one_word_risk_analysis` 字段

3. **测试验证**
   - 测试一字板统计信息是否正确
   - 测试LLM是否能根据统计信息正确判断风险

### 阶段2：K线形态判断改进（优先级：中）

1. **修改 `_compress_kline_data()` 方法**
   - 移除 `_identify_breakthrough_days()` 和 `_identify_top_confirmation_days()` 的调用
   - 改为提供技术指标数据和关键事件标记
   - 生成K线数据描述（表格或文本格式）

2. **更新Prompt模板**
   - 说明LLM需要从K线数据中识别突破日、顶部确认日等形态
   - 在输出格式中添加 `kline_analysis` 字段
   - 提供形态识别的判断标准

3. **测试验证**
   - 测试K线数据描述是否清晰
   - 测试LLM是否能正确识别形态
   - 对比代码计算和LLM识别的结果

### 阶段3：调优和训练（优先级：低）

1. **收集LLM返回结果**
   - 记录LLM识别的突破日、顶部确认日等形态
   - 记录LLM对一字板的风险判断

2. **人工标注验证**
   - 人工验证LLM识别的准确性
   - 标注错误案例

3. **Prompt优化**
   - 根据错误案例优化Prompt
   - 添加更多示例和说明

4. **Fine-tuning（可选）**
   - 如果LLM识别准确率较低，考虑Fine-tuning
   - 使用标注数据训练专用模型

---

## 预期效果

### 优势

1. **灵活性提升**
   - LLM可以识别更复杂的形态模式
   - 不局限于固定规则

2. **可调优性提升**
   - 可以根据LLM返回结果进行调优
   - 可以通过Prompt优化提高准确率
   - 可以Fine-tuning训练专用模型

3. **可解释性提升**
   - LLM会说明识别到的形态和判断依据
   - 便于理解和验证

### 风险

1. **Token消耗增加**
   - 提供更多K线数据可能增加Token消耗
   - 需要通过压缩和优化控制成本

2. **识别准确率**
   - LLM可能识别错误
   - 需要通过Prompt优化和Fine-tuning提高准确率

3. **一致性**
   - LLM识别结果可能不一致
   - 需要通过temperature控制和输出格式约束提高一致性

---

## 后续优化方向

1. **Prompt工程**
   - 添加Few-shot示例
   - 优化描述方式
   - 添加思维链（Chain-of-Thought）

2. **数据压缩优化**
   - 只保留关键交易日的数据
   - 使用更紧凑的格式
   - 压缩技术指标数据

3. **Fine-tuning**
   - 收集高质量标注数据
   - 训练专用模型
   - 提高识别准确率

4. **反馈机制**
   - 记录LLM识别结果
   - 人工反馈和纠正
   - 持续优化Prompt和模型
