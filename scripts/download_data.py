import os
import json
import time
import pandas as pd
import logging
from datetime import datetime, timedelta

# 导入旧版 SDK
try:
    from bigquant.api import dai, user
except ImportError:
    import dai
    from bigquant.api import user

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/download.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class BigQuantDataDownloader:
    """BigQuant数据下载器 (旧版 SDK)"""
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        self.ensure_dirs()
        
        # 显式登录
        self._login()
        
        # 检查连接
        self.check_connection()
    
    def ensure_dirs(self):
        """确保目录存在"""
        dirs = [
            f'{self.data_dir}/kline',
            f'{self.data_dir}/limit_up',
            f'{self.data_dir}/chips',
            f'{self.data_dir}/level2',
            'logs'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            
    def _login(self):
        """从配置文件读取 AK/SK 并登录"""
        config_path = os.path.expanduser('~/.bigquant/config.json')
        if not os.path.exists(config_path):
            logging.error(f"未找到配置文件: {config_path}")
            return False
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # 处理嵌套结构 {"auth": {"ak": "...", "sk": "..."}}
            auth = config.get('auth', {})
            ak = auth.get('ak') or config.get('ak') or config.get('access_key')
            sk = auth.get('sk') or config.get('sk') or config.get('secret_key')
            
            if ak and sk:
                keypair = f"{ak}.{sk}"
                logging.info(f"正在使用 AK-SK 登录 BigQuant...")
                res = user.login(keypair=keypair)
                logging.info(f"登录结果: {res}")
                return True
            else:
                logging.warning("配置文件中未找到完整的 AK/SK")
                return False
        except Exception as e:
            logging.error(f"登录异常: {e}")
            return False
    
    def check_connection(self):
        """检查BigQuant连接"""
        logging.info("检查BigQuant连接...")
        try:
            # 简单查询测试
            dai.query("SELECT date FROM cn_stock_bar1d LIMIT 1", full_db_scan=True).df()
            logging.info("BigQuant连接成功!")
        except Exception as e:
            logging.error(f"连接失败: {e}")
            raise
    
    def safe_query(self, sql, full_db_scan=False, max_retries=3):
        """带重试机制的查询"""
        for attempt in range(max_retries):
            try:
                return dai.query(sql, full_db_scan=full_db_scan).df()
            except Exception as e:
                if "502" in str(e) or "unavailable" in str(e).lower():
                    wait_time = (attempt + 1) * 10
                    logging.warning(f"服务器繁忙 (502/Unavailable), 正在进行第 {attempt + 1} 次重试，等待 {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"查询异常: {e}")
                    raise e
        return None

    def download_stock_list(self):
        """下载股票列表"""
        logging.info("下载股票列表...")
        try:
            sql = """
            SELECT instrument, name 
            FROM cn_stock_instruments 
            WHERE (instrument LIKE '000%' OR instrument LIKE '001%' OR instrument LIKE '002%' 
               OR instrument LIKE '003%' OR instrument LIKE '300%' OR instrument LIKE '301%' 
               OR instrument LIKE '600%' OR instrument LIKE '601%' OR instrument LIKE '603%' 
               OR instrument LIKE '605%' OR instrument LIKE '688%')
              AND (instrument LIKE '%.SH' OR instrument LIKE '%.SZ')
            """
            df = self.safe_query(sql, full_db_scan=True)
            
            if df is None or df.empty:
                logging.error("未获取到股票列表数据")
                return None
                
            df = df.drop_duplicates(subset=['instrument'])
            output_file = f"{self.data_dir}/stock_list.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logging.info(f"✓ 股票列表已保存: {len(df)} 只股票 -> {output_file}")
            return df
        except Exception as e:
            logging.error(f"股票列表下载失败: {e}")
            return None
    
    def download_kline_data_batch(self, instruments, start_date, end_date, batch_name="batch"):
        """下载K线数据（批次）"""
        logging.info(f"下载K线数据批次 [{batch_name}]: {len(instruments)} 只股票")
        instruments_str = "'" + "','".join(instruments) + "'"
        try:
            sql = f"""
            SELECT 
                date, instrument, open, high, low, close, volume, amount, adjust_factor
            FROM cn_stock_bar1d
            WHERE date >= '{start_date}' 
              AND date <= '{end_date}'
              AND instrument IN ({instruments_str})
            ORDER BY instrument, date
            """
            df = self.safe_query(sql)
            if df is None or df.empty:
                logging.warning(f"批次 [{batch_name}] 未获取到数据")
                return None
            output_file = f"{self.data_dir}/kline/kline_{batch_name}.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logging.info(f"✓ 批次已保存: {output_file}, {len(df)} 条记录")
            return df
        except Exception as e:
            logging.error(f"批次 [{batch_name}] 下载失败: {e}")
            return None
    
    def download_all_kline_data(self, start_date='2021-01-01', end_date=None, batch_size=1000):
        """下载所有K线数据（分批）"""
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        logging.info(f"开始下载K线数据: {start_date} 到 {end_date}")
        stock_list_file = f"{self.data_dir}/stock_list.csv"
        if os.path.exists(stock_list_file):
            stock_df = pd.read_csv(stock_list_file)
        else:
            stock_df = self.download_stock_list()
        if stock_df is None or stock_df.empty:
            logging.error("无法获取股票列表")
            return
        instruments = stock_df['instrument'].tolist()
        total_stocks = len(instruments)
        for i in range(0, total_stocks, batch_size):
            batch_instruments = instruments[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_stocks + batch_size - 1) // batch_size
            logging.info(f"\n{'='*60}")
            logging.info(f"批次 {batch_num}/{total_batches}")
            logging.info(f"{'='*60}")
            self.download_kline_data_batch(batch_instruments, start_date, end_date, batch_name=f"batch_{batch_num}")
        self.merge_kline_batches()
    
    def merge_kline_batches(self):
        """合并K线批次文件"""
        logging.info("\n合并K线批次文件...")
        kline_dir = f"{self.data_dir}/kline"
        batch_files = [f for f in os.listdir(kline_dir) if f.startswith('kline_batch_')]
        if not batch_files:
            logging.warning("没有找到批次文件")
            return
        all_data = []
        for batch_file in batch_files:
            df = pd.read_csv(f"{kline_dir}/{batch_file}")
            all_data.append(df)
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df = merged_df.sort_values(['instrument', 'date'])
        merged_df = merged_df.drop_duplicates(subset=['date', 'instrument'])
        output_file = f"{kline_dir}/kline_all.csv"
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"✓ 合并完成: {output_file}, {len(merged_df)} 条记录")
    
    def download_limit_up_data(self, start_date='2021-01-01'):
        """下载涨停数据"""
        logging.info(f"下载涨停数据: {start_date} 开始")
        try:
            sql = f"""
            SELECT a.date, a.instrument, a.close, b.upper_limit
            FROM cn_stock_bar1d AS a
            JOIN cn_stock_limit_price AS b 
            ON a.date = b.date AND a.instrument = b.instrument
            WHERE a.date >= '{start_date}'
              AND a.close >= b.upper_limit - 0.01
            ORDER BY a.date DESC
            """
            df = self.safe_query(sql)
            if df is None or df.empty:
                logging.warning("未获取到涨停数据")
                return None
            output_file = f"{self.data_dir}/limit_up/limit_up_history.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logging.info(f"✓ 涨停数据已保存: {output_file}, {len(df)} 条记录")
            return df
        except Exception as e:
            logging.error(f"涨停数据下载失败: {e}")
            return None

    def download_dragon_list_data(self, start_date='2021-01-01'):
        """下载龙虎榜数据"""
        logging.info(f"下载龙虎榜数据: {start_date} 开始")
        try:
            sql = f"""
            SELECT * FROM cn_stock_dragon_list
            WHERE date >= '{start_date}'
            ORDER BY date DESC
            """
            df = self.safe_query(sql)
            if df is None or df.empty:
                logging.warning("未获取到龙虎榜数据")
                return None
            output_file = f"{self.data_dir}/limit_up/dragon_list.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logging.info(f"✓ 龙虎榜数据已保存: {output_file}, {len(df)} 条记录")
        except Exception as e:
            logging.error(f"龙虎榜数据下载失败: {e}")
            return None

    def download_chips_data(self, start_date=None, batch_size=300):
        """
        下载筹码分布摘要数据
        优化策略：仅下载 avg_cost, win_percent, concentration 等摘要指标，
        使用 DISTINCT 过滤掉详细的价格分布行，大幅减少数据量。
        """
        if start_date is None:
            # 修改为从 2021 年开始，以匹配训练需求
            start_date = '2021-01-01'
            
        logging.info(f"开始下载筹码分布摘要数据: {start_date} 开始")
        
        stock_list_file = f"{self.data_dir}/stock_list.csv"
        if os.path.exists(stock_list_file):
            stock_df = pd.read_csv(stock_list_file)
        else:
            stock_df = self.download_stock_list()
        
        if stock_df is None or stock_df.empty:
            return
        
        instruments = stock_df['instrument'].tolist()
        total_stocks = len(instruments)
        chips_dir = f"{self.data_dir}/chips"
        os.makedirs(chips_dir, exist_ok=True)
        
        all_dfs = []
        for i in range(0, total_stocks, batch_size):
            batch_instruments = instruments[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_stocks + batch_size - 1) // batch_size
            
            logging.info(f"下载筹码摘要批次 {batch_num}/{total_batches}...")
            instruments_str = "'" + "','".join(batch_instruments) + "'"
            
            # 使用 DISTINCT 获取摘要指标，避开详细的 price/chips_ratio 冗余
            sql = f"""
            SELECT DISTINCT 
                date, instrument, avg_cost, win_percent, concentration
            FROM cn_stock_chips_distribution
            WHERE date >= '{start_date}' 
              AND instrument IN ({instruments_str})
            ORDER BY date DESC
            """
            try:
                df = self.safe_query(sql)
                if df is not None and not df.empty:
                    output_file = f"{chips_dir}/chips_batch_{batch_num}.csv"
                    df.to_csv(output_file, index=False, encoding='utf-8-sig')
                    all_dfs.append(df)
            except Exception as e:
                logging.error(f"筹码批次 {batch_num} 下载失败: {e}")
            time.sleep(1)
            
        if all_dfs:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            output_file = f"{self.data_dir}/chips_all.csv"
            merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logging.info(f"✓ 筹码分布摘要数据已保存: {output_file}, {len(merged_df)} 条记录")

    def test_level2_data(self, stock_code='000001.SZ', test_date='2024-01-15'):
        """
        测试Level 2逐笔成交数据是否能正常读取
        
        ⚠️ 注意：BigQuant没有免费数据获取权限，且cn_stock_level2_trade_trail表已停用
        此方法仅保留用于历史参考，实际无法使用
        
        Args:
            stock_code: 测试股票代码（默认：000001.SZ）
            test_date: 测试日期（默认：2024-01-15）
        
        Returns:
            DataFrame: Level 2数据，如果失败则返回None
        """
        logging.warning("⚠️ Level 2数据不可用：BigQuant没有免费数据获取权限，且cn_stock_level2_trade_trail表已停用")
        logging.info(f"测试Level 2数据读取: {stock_code}, 日期: {test_date}")
        
        try:
            # 转换日期格式为YYYYMMDD
            trading_day = int(test_date.replace('-', ''))
            
            # 查询Level 2数据（限制1000条，用于测试）
            sql = f"""
            SELECT 
                price, volume, instrument, date, time,
                bs_flag, ask_seq_num, bid_seq_num, seq_num, trading_day
            FROM cn_stock_level2_trade_trail
            WHERE instrument = '{stock_code}'
              AND trading_day = {trading_day}
            ORDER BY date, time, seq_num
            LIMIT 1000
            """
            
            logging.info(f"执行SQL查询: {sql[:100]}...")
            df = self.safe_query(sql, full_db_scan=True)
            
            if df is None or df.empty:
                logging.warning(f"未获取到Level 2数据: {stock_code}, {test_date}")
                logging.warning("可能原因：1) 该日期非交易日 2) 该股票无Level 2数据 3) 试用账号无权限")
                return None
            
            logging.info(f"✓ Level 2数据读取成功: {len(df)} 条记录")
            logging.info(f"数据字段: {list(df.columns)}")
            logging.info(f"数据示例（前5条）:")
            logging.info(f"\n{df.head().to_string()}")
            
            # 统计信息
            if 'price' in df.columns and 'volume' in df.columns:
                logging.info(f"价格范围: {df['price'].min():.2f} - {df['price'].max():.2f}")
                logging.info(f"总成交量: {df['volume'].sum():,}")
                if 'bs_flag' in df.columns:
                    buy_count = (df['bs_flag'] == 66).sum()
                    sell_count = (df['bs_flag'] == 83).sum()
                    logging.info(f"主动买入: {buy_count} 笔, 主动卖出: {sell_count} 笔")
            
            # 保存测试数据
            test_output = f"{self.data_dir}/level2/test_{stock_code}_{test_date}.csv"
            df.to_csv(test_output, index=False, encoding='utf-8-sig')
            logging.info(f"✓ 测试数据已保存: {test_output}")
            
            return df
            
        except Exception as e:
            logging.error(f"Level 2数据测试失败: {e}", exc_info=True)
            return None
    
    def download_level2_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        limit: int = None
    ):
        """
        下载指定股票的Level 2逐笔成交数据
        
        ⚠️ 注意：BigQuant没有免费数据获取权限，且cn_stock_level2_trade_trail表已停用
        此方法仅保留用于历史参考，实际无法使用
        
        Args:
            stock_code: 股票代码（如"000001.SZ"）
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            limit: 限制记录数（可选，用于测试）
        
        Returns:
            DataFrame: Level 2数据，如果失败则返回None
        """
        logging.warning("⚠️ Level 2数据不可用：BigQuant没有免费数据获取权限，且cn_stock_level2_trade_trail表已停用")
        logging.info(f"下载Level 2数据: {stock_code}, {start_date} 到 {end_date}")
        
        try:
            # 转换日期格式为YYYYMMDD
            start_trading_day = int(start_date.replace('-', ''))
            end_trading_day = int(end_date.replace('-', ''))
            
            # 构建SQL查询
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            sql = f"""
            SELECT 
                price, volume, instrument, date, time,
                bs_flag, ask_seq_num, bid_seq_num, seq_num, trading_day
            FROM cn_stock_level2_trade_trail
            WHERE instrument = '{stock_code}'
              AND trading_day >= {start_trading_day}
              AND trading_day <= {end_trading_day}
            ORDER BY date, time, seq_num
            {limit_clause}
            """
            
            logging.info(f"执行SQL查询...")
            df = self.safe_query(sql, full_db_scan=True)
            
            if df is None or df.empty:
                logging.warning(f"未获取到Level 2数据: {stock_code}, {start_date} 到 {end_date}")
                return None
            
            # 保存到CSV
            output_file = f"{self.data_dir}/level2/level2_{stock_code}_{start_date}_{end_date}.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logging.info(f"✓ Level 2数据已保存: {output_file}, {len(df):,} 条记录")
            
            # 显示统计信息
            if 'price' in df.columns and 'volume' in df.columns:
                logging.info(f"价格范围: {df['price'].min():.2f} - {df['price'].max():.2f}")
                logging.info(f"总成交量: {df['volume'].sum():,}")
                if 'bs_flag' in df.columns:
                    buy_count = (df['bs_flag'] == 66).sum()
                    sell_count = (df['bs_flag'] == 83).sum()
                    logging.info(f"主动买入: {buy_count:,} 笔, 主动卖出: {sell_count:,} 笔")
            
            return df
            
        except Exception as e:
            logging.error(f"Level 2数据下载失败: {e}", exc_info=True)
            return None

def main():
    """主函数"""
    print("=" * 60)
    print("BigQuant 旧版 SDK 数据下载工具")
    print("=" * 60)
    
    try:
        downloader = BigQuantDataDownloader()
        
        print("\n请选择下载任务:")
        print("1. 下载股票列表")
        print("2. 下载K线数据 (过去3年)")
        print("3. 筛选涨停数据")
        print("4. 下载龙虎榜数据")
        print("5. 下载筹码分布摘要数据 (2021年至今)")
        print("6. 测试Level 2逐笔成交数据读取")
        print("7. 下载Level 2逐笔成交数据")
        print("8. 全部下载")
        
        choice = input("\n请输入选项 (1-8): ").strip()
        
        if choice == "1":
            downloader.download_stock_list()
        elif choice == "2":
            start_date = (datetime.today() - timedelta(days=3*365)).strftime('%Y-%m-%d')
            downloader.download_all_kline_data(start_date=start_date)
        elif choice == "3":
            downloader.download_limit_up_data()
        elif choice == "4":
            downloader.download_dragon_list_data()
        elif choice == "5":
            downloader.download_chips_data()
        elif choice == "6":
            # 测试Level 2数据读取
            print("\n测试Level 2数据读取...")
            print("默认测试: 000001.SZ, 2024-01-15")
            test_code = input("请输入测试股票代码（直接回车使用默认 000001.SZ）: ").strip() or "000001.SZ"
            test_date = input("请输入测试日期（直接回车使用默认 2024-01-15）: ").strip() or "2024-01-15"
            result = downloader.test_level2_data(stock_code=test_code, test_date=test_date)
            if result is not None:
                print(f"\n✅ Level 2数据测试成功！共读取 {len(result)} 条记录")
            else:
                print("\n❌ Level 2数据测试失败，请检查：")
                print("  1. 该日期是否为交易日")
                print("  2. 该股票是否有Level 2数据")
                print("  3. 试用账号是否有权限访问Level 2数据")
        elif choice == "7":
            # 下载Level 2数据
            print("\n下载Level 2逐笔成交数据...")
            stock_code = input("请输入股票代码（如 000001.SZ）: ").strip()
            start_date = input("请输入开始日期（YYYY-MM-DD，如 2024-01-15）: ").strip()
            end_date = input("请输入结束日期（YYYY-MM-DD，如 2024-01-20）: ").strip()
            limit_input = input("请输入限制记录数（可选，直接回车不限制）: ").strip()
            limit = int(limit_input) if limit_input else None
            
            if stock_code and start_date and end_date:
                result = downloader.download_level2_data(
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit
                )
                if result is not None:
                    print(f"\n✅ Level 2数据下载成功！共 {len(result):,} 条记录")
                else:
                    print("\n❌ Level 2数据下载失败")
            else:
                print("❌ 输入参数不完整")
        elif choice == "8":
            downloader.download_stock_list()
            kline_start = (datetime.today() - timedelta(days=3*365)).strftime('%Y-%m-%d')
            downloader.download_all_kline_data(start_date=kline_start)
            downloader.download_limit_up_data(start_date=kline_start)
            downloader.download_dragon_list_data(start_date=kline_start)
            downloader.download_chips_data()
        else:
            print("无效选项")
        print("\n✅ 任务完成!")
    except Exception as e:
        logging.error(f"执行失败: {e}")

if __name__ == '__main__':
    main()
