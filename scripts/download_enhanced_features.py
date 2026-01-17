"""
增强特征数据下载脚本

下载以下数据用于提升二波潜力分析准确率：
1. cn_stock_hot_rank - 个股热度排名
2. cn_stock_index_concept_component - 概念归属
3. cn_stock_index_concept_bar1d - 概念行情
4. cn_stock_factors_auction - 竞价因子

作者：AI Assistant
日期：2026-01-13
"""

import os
import sys
import json
import time
import pandas as pd
import logging
from datetime import datetime, timedelta

# 添加项目路径
project_root = os.path.abspath('d:/myCursor/StockAiNews')
sys.path.insert(0, project_root)

# 导入 BigQuant SDK
try:
    from bigquantdai import dai
    from bigquant.api import user
except ImportError:
    try:
        from bigquant.api import dai, user
    except ImportError:
        import dai
        from bigquant.api import user

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class EnhancedFeaturesDownloader:
    """增强特征数据下载器"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 登录
        self._login()
    
    def get_existing_data_info(self, filename):
        """获取已存在数据的信息"""
        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath):
            return None, None
        
        try:
            existing_df = pd.read_csv(filepath)
            if 'date' in existing_df.columns and len(existing_df) > 0:
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                max_date = existing_df['date'].max()
                return existing_df, max_date
        except Exception as e:
            logging.warning(f"读取已存在文件失败 {filename}: {e}")
        
        return None, None
    
    def _login(self):
        """从配置文件读取 AK/SK 并登录"""
        config_path = os.path.expanduser('~/.bigquant/config.json')
        if not os.path.exists(config_path):
            logging.error(f"未找到配置文件: {config_path}")
            return False
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # 处理嵌套结构
            auth = config.get('auth', {})
            ak = auth.get('ak') or config.get('ak') or config.get('access_key')
            sk = auth.get('sk') or config.get('sk') or config.get('secret_key')
            
            if ak and sk:
                keypair = f"{ak}.{sk}"
                logging.info(f"正在使用 AK-SK 登录 BigQuant...")
                res = user.login(keypair=keypair)
                logging.info(f"登录成功")
                return True
            else:
                logging.warning("配置文件中未找到完整的 AK/SK")
                return False
        except Exception as e:
            logging.error(f"登录异常: {e}")
            return False
    
    def safe_query(self, sql, max_retries=3):
        """带重试机制的查询"""
        for attempt in range(max_retries):
            try:
                return dai.query(sql, full_db_scan=True).df()
            except Exception as e:
                if "502" in str(e) or "unavailable" in str(e).lower():
                    wait_time = (attempt + 1) * 10
                    logging.warning(f"服务器繁忙，正在进行第 {attempt + 1} 次重试，等待 {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        
        raise Exception("查询失败，已达到最大重试次数")


def download_hot_rank(downloader, start_date, end_date):
    """
    下载个股热度排名数据（需要历史数据来计算持续热门天数）
    
    核心特征：
    - stock_hot_rank: 个股热度排名
    - concept_hot_rank: 所属概念热度
    - rank_change: 排名变化
    """
    print(f"\n{'='*60}")
    print("1/4 正在下载热度排名数据...")
    print(f"{'='*60}")
    
    output_file = os.path.join(downloader.output_dir, 'hot_rank.csv')
    
    try:
        # 检查已有数据
        existing_df, max_existing_date = downloader.get_existing_data_info('hot_rank.csv')
        
        if existing_df is not None:
            print(f"  ✓ 发现已有数据，最新日期: {max_existing_date.date()}")
            # 只下载新数据
            download_start = (max_existing_date + timedelta(days=1)).strftime('%Y-%m-%d')
            if download_start > end_date:
                print(f"  ✓ 数据已是最新，无需下载")
                return existing_df
            print(f"  - 增量下载: {download_start} 至 {end_date}")
        else:
            download_start = start_date
            print(f"  - 首次下载: {start_date} 至 {end_date} (最近90天)")
        
        sql = f"""
        SELECT * FROM cn_stock_hot_rank 
        WHERE date >= '{download_start}' AND date <= '{end_date}'
        """
        
        new_df = downloader.safe_query(sql)
        
        if new_df.empty:
            print("⚠️ 未获取到新数据")
            return existing_df
        
        # 合并数据
        if existing_df is not None:
            df = pd.concat([existing_df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=['instrument', 'date'], keep='last')
            df = df.sort_values(['date', 'instrument']).reset_index(drop=True)
            print(f"  ✓ 新增 {len(new_df)} 条，合并后共 {len(df)} 条")
        else:
            df = new_df
        
        # 保留最近90天的数据
        cutoff_date = pd.to_datetime(end_date) - timedelta(days=90)
        df = df[df['date'] >= cutoff_date]
        
        # 保存数据
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"✓ 热度排名数据更新完成: {len(df)} 条记录")
        print(f"  - 时间范围: {df['date'].min()} 至 {df['date'].max()}")
        print(f"  - 股票数量: {df['instrument'].nunique()}")
        print(f"  - 保存路径: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"❌ 下载热度排名数据失败: {e}")
        return None


def download_concept_component(downloader, end_date):
    """
    下载概念成分股数据（只需要最新的映射关系，不需要历史数据）
    
    核心特征：
    - concept_codes: 所属概念列表
    - is_concept_leader: 是否概念龙头
    """
    print(f"\n{'='*60}")
    print("2/4 正在下载概念成分股数据...")
    print(f"{'='*60}")
    
    output_file = os.path.join(downloader.output_dir, 'concept_component.csv')
    
    try:
        # 检查已有数据
        existing_df, max_existing_date = downloader.get_existing_data_info('concept_component.csv')
        
        if existing_df is not None:
            print(f"  ✓ 发现已有数据，最新日期: {max_existing_date.date()}")
            # 只下载新数据
            download_start = (max_existing_date + timedelta(days=1)).strftime('%Y-%m-%d')
            if download_start > end_date:
                print(f"  ✓ 数据已是最新，无需下载")
                return existing_df
            print(f"  - 增量下载: {download_start} 至 {end_date}")
        else:
            # 首次下载最近7天
            download_start = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
            print(f"  - 首次下载: {download_start} 至 {end_date} (最近7天)")
        
        sql = f"""
        SELECT * FROM cn_stock_index_concept_component 
        WHERE date >= '{download_start}' AND date <= '{end_date}'
        """
        
        new_df = downloader.safe_query(sql)
        
        if new_df.empty:
            print("⚠️ 未获取到新数据")
            return existing_df
        
        # 合并数据
        if existing_df is not None:
            df = pd.concat([existing_df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=['instrument', 'date'], keep='last')
            df = df.sort_values(['date', 'instrument']).reset_index(drop=True)
            print(f"  ✓ 新增 {len(new_df)} 条，合并后共 {len(df)} 条")
        else:
            df = new_df
        
        # 只保留最近7天的数据（节省空间）
        cutoff_date = pd.to_datetime(end_date) - timedelta(days=7)
        df = df[df['date'] >= cutoff_date]
        
        # 保存数据
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"✓ 概念成分股数据更新完成: {len(df)} 条记录")
        print(f"  - 时间范围: {df['date'].min()} 至 {df['date'].max()}")
        print(f"  - 股票数量: {df['instrument'].nunique()}")
        print(f"  - 保存路径: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"❌ 下载概念成分股数据失败: {e}")
        return None


def download_concept_bar(downloader, start_date, end_date):
    """
    下载概念指数行情数据（需要历史数据来计算3日动量）
    
    核心特征：
    - concept_index_gain: 概念指数涨幅
    - concept_momentum: 概念动量
    - concept_volume_ratio: 概念成交量比
    """
    print(f"\n{'='*60}")
    print("3/4 正在下载概念指数行情数据...")
    print(f"{'='*60}")
    
    output_file = os.path.join(downloader.output_dir, 'concept_bar.csv')
    
    try:
        # 检查已有数据
        existing_df, max_existing_date = downloader.get_existing_data_info('concept_bar.csv')
        
        if existing_df is not None:
            print(f"  ✓ 发现已有数据，最新日期: {max_existing_date.date()}")
            # 只下载新数据
            download_start = (max_existing_date + timedelta(days=1)).strftime('%Y-%m-%d')
            if download_start > end_date:
                print(f"  ✓ 数据已是最新，无需下载")
                return existing_df
            print(f"  - 增量下载: {download_start} 至 {end_date}")
        else:
            download_start = start_date
            print(f"  - 首次下载: {start_date} 至 {end_date} (最近90天)")
        
        sql = f"""
        SELECT * FROM cn_stock_index_concept_bar1d 
        WHERE date >= '{download_start}' AND date <= '{end_date}'
        """
        
        new_df = downloader.safe_query(sql)
        
        if new_df.empty:
            print("⚠️ 未获取到新数据")
            return existing_df
        
        # 合并数据
        if existing_df is not None:
            df = pd.concat([existing_df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=['instrument', 'date'], keep='last')
            df = df.sort_values(['date', 'instrument']).reset_index(drop=True)
            print(f"  ✓ 新增 {len(new_df)} 条，合并后共 {len(df)} 条")
        else:
            df = new_df
        
        # 保留最近90天的数据
        cutoff_date = pd.to_datetime(end_date) - timedelta(days=90)
        df = df[df['date'] >= cutoff_date]
        
        # 保存数据
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"✓ 概念指数行情数据更新完成: {len(df)} 条记录")
        print(f"  - 时间范围: {df['date'].min()} 至 {df['date'].max()}")
        print(f"  - 概念数量: {df['instrument'].nunique()}")
        print(f"  - 保存路径: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"❌ 下载概念指数行情数据失败: {e}")
        return None


def download_auction_factors(downloader, end_date):
    """
    下载集合竞价因子数据（只需要最近的数据）
    
    核心特征：
    - auction_volume_ratio: 竞价成交量比
    - auction_price_gap: 竞价价格偏离
    - auction_large_order: 竞价大单占比
    """
    print(f"\n{'='*60}")
    print("4/4 正在下载集合竞价因子数据...")
    print(f"{'='*60}")
    
    output_file = os.path.join(downloader.output_dir, 'auction_factors.csv')
    
    try:
        # 检查已有数据
        existing_df, max_existing_date = downloader.get_existing_data_info('auction_factors.csv')
        
        if existing_df is not None:
            print(f"  ✓ 发现已有数据，最新日期: {max_existing_date.date()}")
            # 只下载新数据
            download_start = (max_existing_date + timedelta(days=1)).strftime('%Y-%m-%d')
            if download_start > end_date:
                print(f"  ✓ 数据已是最新，无需下载")
                return existing_df
            print(f"  - 增量下载: {download_start} 至 {end_date}")
        else:
            # 首次下载最近30天
            download_start = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
            print(f"  - 首次下载: {download_start} 至 {end_date} (最近30天)")
        
        sql = f"""
        SELECT * FROM cn_stock_factors_auction 
        WHERE date >= '{download_start}' AND date <= '{end_date}'
        """
        
        new_df = downloader.safe_query(sql)
        
        if new_df.empty:
            print("⚠️ 未获取到新数据")
            return existing_df
        
        # 合并数据
        if existing_df is not None:
            df = pd.concat([existing_df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=['instrument', 'date'], keep='last')
            df = df.sort_values(['date', 'instrument']).reset_index(drop=True)
            print(f"  ✓ 新增 {len(new_df)} 条，合并后共 {len(df)} 条")
        else:
            df = new_df
        
        # 只保留最近30天的数据
        cutoff_date = pd.to_datetime(end_date) - timedelta(days=30)
        df = df[df['date'] >= cutoff_date]
        
        # 保存数据
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"✓ 集合竞价因子数据更新完成: {len(df)} 条记录")
        print(f"  - 时间范围: {df['date'].min()} 至 {df['date'].max()}")
        print(f"  - 股票数量: {df['instrument'].nunique()}")
        print(f"  - 保存路径: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"❌ 下载集合竞价因子数据失败: {e}")
        return None


def main():
    """主函数"""
    print("\n" + "="*60)
    print("增强特征数据下载")
    print("="*60)
    
    # 设置日期范围（根据不同数据类型优化下载量）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date_90d = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')  # 热度排名、概念行情
    
    print(f"\n配置信息:")
    print(f"  - 结束日期: {end_date}")
    print(f"  - 热度排名/概念行情: 最近90天")
    print(f"  - 概念成分股: 最近7天（仅需映射关系）")
    print(f"  - 竞价因子: 最近30天")
    
    # 设置输出目录
    base_dir = 'd:/myCursor/StockAiNews/TradingAgents-chinese-market/AlphaSignal-CN'
    output_dir = os.path.join(base_dir, 'data/raw')
    print(f"  - 输出目录: {output_dir}")
    
    # 初始化下载器
    try:
        downloader = EnhancedFeaturesDownloader(output_dir)
    except Exception as e:
        print(f"\n❌ 初始化下载器失败: {e}")
        return
    
    # 下载数据（不同数据使用不同的日期范围）
    results = {}
    results['hot_rank'] = download_hot_rank(downloader, start_date_90d, end_date)
    results['concept_component'] = download_concept_component(downloader, end_date)
    results['concept_bar'] = download_concept_bar(downloader, start_date_90d, end_date)
    results['auction_factors'] = download_auction_factors(downloader, end_date)
    
    # 统计结果
    print(f"\n{'='*60}")
    print("下载完成统计")
    print(f"{'='*60}")
    
    success_count = sum(1 for v in results.values() if v is not None)
    print(f"✓ 成功: {success_count}/4")
    print(f"✗ 失败: {4 - success_count}/4")
    
    if success_count > 0:
        print(f"\n数据已保存到: {output_dir}")
        print("\n下一步:")
        print("  1. 运行特征提取脚本，将这些数据集成到模型中")
        print("  2. 重新训练模型以利用新特征")
        print("  3. 使用 predict_stock.py 进行预测")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
