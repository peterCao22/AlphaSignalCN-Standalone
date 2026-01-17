"""
检查BigQuant的cn_stock_bar1d表包含哪些字段
"""
import sys
from pathlib import Path

# 统一以本仓库根目录为准（AlphaSignalCN-Standalone）
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bigquantdai import dai

print("正在查询 cn_stock_bar1d 的字段列表...")

# 查询一条数据，查看所有字段
sql = """
SELECT *
FROM cn_stock_bar1d
WHERE instrument = '000001.SZ'
  AND date = '2024-01-01'
LIMIT 1
"""

try:
    df = dai.query(sql).df()
    print(f"\n✓ cn_stock_bar1d 包含 {len(df.columns)} 个字段：\n")
    
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    # 重点检查turn字段
    if 'turn' in df.columns:
        print(f"\n✓ 发现 'turn' 字段（换手率）！")
        print(f"   示例值: {df['turn'].iloc[0]}")
    else:
        print(f"\n✗ 未发现 'turn' 字段")
        
    # 检查其他可能有用的字段
    useful_fields = ['turn', 'amount', 'volume', 'pre_close', 'change', 
                     'pct_change', 'is_limit_up', 'high_limit', 'low_limit']
    
    print(f"\n有用字段检查:")
    for field in useful_fields:
        status = "✓" if field in df.columns else "✗"
        print(f"  {status} {field}")
        
except Exception as e:
    print(f"✗ 查询失败: {e}")
