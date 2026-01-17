"""
查看 BigQuant cn_stock_moneyflow 资金流数据表结构

作者：AI Assistant
日期：2026-01-14
"""

import os
import sys
from datetime import datetime, timedelta

# 统一以本仓库根目录为准（AlphaSignalCN-Standalone）
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 导入 BigQuant SDK
try:
    from bigquantdai import dai
except ImportError:
    try:
        from bigquant.api import dai
    except ImportError:
        import dai

def preview_moneyflow_fields():
    """查看资金流数据表的字段结构"""
    
    # 计算日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    print("=" * 80)
    print("正在查询 cn_stock_moneyflow 表结构...")
    print("=" * 80)
    print(f"\n📅 查询日期范围: {start_date} ~ {end_date} (最近3天)")
    print(f"💡 提示: 只查询少量数据以节省流量\n")
    
    try:
        # 查询表结构（取少量数据，指定日期范围以节省流量）
        sql = """
        SELECT * FROM cn_stock_moneyflow
        LIMIT 3
        """
        
        # 使用 filters 参数指定日期范围
        df = dai.query(sql, filters={"date": [start_date, end_date]}).df()
        
        print(f"\n✓ 查询成功！")
        print(f"数据行数: {len(df)}")
        print(f"总字段数: {len(df.columns)} 个")
        if 'date' in df.columns:
            print(f"实际数据时间: {df['date'].min()} ~ {df['date'].max()}")
        print(f"查询范围: {start_date} ~ {end_date}")
        
        # 打印所有字段
        print("\n" + "=" * 80)
        print("所有字段列表:")
        print("=" * 80)
        
        for i, col in enumerate(df.columns, 1):
            # 尝试获取该字段的示例值
            try:
                sample_value = df[col].iloc[0]
                if pd.notna(sample_value):
                    print(f"{i:3d}. {col:40s} 示例: {sample_value}")
                else:
                    print(f"{i:3d}. {col:40s} (NaN)")
            except:
                print(f"{i:3d}. {col}")
        
        # 按类别分组显示（根据字段名推测）
        print("\n" + "=" * 80)
        print("字段分类（推测）:")
        print("=" * 80)
        
        # 分类统计
        categories = {
            '基础字段': ['date', 'instrument', 'code', 'name'],
            '主力资金': [col for col in df.columns if 'main' in col.lower() or 'major' in col.lower()],
            '大单': [col for col in df.columns if 'large' in col.lower() or 'big' in col.lower() or 'xlarge' in col.lower()],
            '中单': [col for col in df.columns if 'medium' in col.lower() or 'mid' in col.lower()],
            '小单': [col for col in df.columns if 'small' in col.lower() or 'retail' in col.lower()],
            '净流入': [col for col in df.columns if 'net' in col.lower() or 'inflow' in col.lower()],
            '买入': [col for col in df.columns if 'buy' in col.lower() or 'bid' in col.lower()],
            '卖出': [col for col in df.columns if 'sell' in col.lower() or 'ask' in col.lower()],
            '占比/强度': [col for col in df.columns if 'ratio' in col.lower() or 'pct' in col.lower() or 'strength' in col.lower()],
            '历史统计': [col for col in df.columns if any(d in col.lower() for d in ['3d', '5d', '10d', '20d', '60d', 'ma'])],
        }
        
        for category, fields in categories.items():
            if fields:
                print(f"\n【{category}】({len(fields)} 个):")
                for field in fields[:10]:  # 只显示前10个
                    print(f"  - {field}")
                if len(fields) > 10:
                    print(f"  ... 还有 {len(fields) - 10} 个字段")
        
        # 显示示例数据
        print("\n" + "=" * 80)
        print("示例数据（前3行，前10列）:")
        print("=" * 80)
        print(df.iloc[:3, :10].to_string())
        
        # 保存字段列表到文件（路径统一落在本仓库 data/ 目录下）
        out_dir = REPO_ROOT / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(out_dir / "moneyflow_fields.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"cn_stock_moneyflow 表字段列表\n")
            f.write(f"总计: {len(df.columns)} 个字段\n")
            f.write(f"查询时间: {pd.Timestamp.now()}\n")
            f.write("=" * 80 + "\n\n")
            for i, col in enumerate(df.columns, 1):
                f.write(f"{i:3d}. {col}\n")
        
        print(f"\n✓ 字段列表已保存到: {output_file}")
        
        # 推荐关键字段
        print("\n" + "=" * 80)
        print("💡 推荐用于二波预测的关键字段（需根据实际字段名调整）:")
        print("=" * 80)
        
        recommended = [
            "主力资金净流入相关字段（net_mf_main, main_net_inflow 等）",
            "大单资金流向（large_net_inflow, xlarge_net_inflow 等）",
            "主动买入金额（buy_main, buy_large 等）",
            "资金流入强度/占比（mf_strength, main_ratio 等）",
            "历史累计资金流（net_mf_5d, net_mf_10d, net_mf_20d 等）",
        ]
        
        for i, rec in enumerate(recommended, 1):
            print(f"{i}. {rec}")
        
        return df
        
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        print("\n可能原因:")
        print("1. BigQuant SDK 未正确安装或配置")
        print("2. 网络连接问题")
        print("3. 权限不足或数据表不存在")
        import traceback
        print("\n详细错误:")
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    import pandas as pd
    preview_moneyflow_fields()
