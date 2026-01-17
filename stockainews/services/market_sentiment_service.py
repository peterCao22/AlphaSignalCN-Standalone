"""
市场情绪数据服务

管理市场情绪数据（热门板块、赚钱效应）的存储和查询。
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, date, timedelta
import sqlite3
from pathlib import Path
import pandas as pd
import json
from stockainews.core.logger import setup_logger

logger = setup_logger(__name__)


class MarketSentimentService:
    """市场情绪数据服务"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化服务
        
        Args:
            db_path: 数据库路径（如果为None则使用默认路径）
        """
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "TradingAgents-chinese-market" / "AlphaSignal-CN" / "data" / "market_sentiment.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"市场情绪数据服务初始化完成: {self.db_path}")
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建热门板块表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hot_sectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crawl_date DATE NOT NULL,
                rank INTEGER NOT NULL,
                sector_name VARCHAR(50) NOT NULL,
                sector_gain REAL,
                leader_stock_code VARCHAR(10),
                leader_stock_name VARCHAR(50),
                leader_gain REAL,
                stock_count INTEGER,
                capital_inflow REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建市场情绪表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crawl_date DATE NOT NULL,
                market_activity REAL,
                up_count INTEGER,
                down_count INTEGER,
                flat_count INTEGER,
                limit_up_count INTEGER,
                limit_up_real_count INTEGER,
                limit_down_count INTEGER,
                limit_down_real_count INTEGER,
                st_limit_up_count INTEGER,
                st_limit_down_count INTEGER,
                suspended_count INTEGER,
                median_gain_all REAL,
                median_gain_hs300 REAL,
                median_gain_sh REAL,
                median_gain_sz REAL,
                median_gain_cyb REAL,
                sentiment_score REAL,
                gain_distribution_json TEXT,
                update_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(crawl_date)
            )
        ''')

        # 兼容旧库：如果缺少 gain_distribution_json 列则补齐
        try:
            cursor.execute("PRAGMA table_info(market_sentiment)")
            cols = [r[1] for r in cursor.fetchall()]
            if 'gain_distribution_json' not in cols:
                cursor.execute("ALTER TABLE market_sentiment ADD COLUMN gain_distribution_json TEXT")
                logger.info("已为 market_sentiment 增加列 gain_distribution_json")
        except Exception as e:
            logger.warning(f"检查/迁移 market_sentiment 表结构失败: {e}")
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_hot_sectors_date ON hot_sectors(crawl_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_hot_sectors_sector ON hot_sectors(sector_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_hot_sectors_leader ON hot_sectors(leader_stock_code)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_sentiment_date ON market_sentiment(crawl_date)')
        
        conn.commit()
        conn.close()
        logger.info("数据库表初始化完成")
    
    def save_hot_sectors(self, sectors: List[Dict[str, Any]], crawl_date: Optional[date] = None):
        """
        保存热门板块数据
        
        Args:
            sectors: 板块数据列表
            crawl_date: 爬取日期（如果为None则使用今天）
        """
        if not sectors:
            logger.warning("板块数据为空，跳过保存")
            return
        
        if crawl_date is None:
            crawl_date = date.today()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 删除当天的旧数据
            cursor.execute('DELETE FROM hot_sectors WHERE crawl_date = ?', (crawl_date,))
            
            # 插入新数据
            for sector in sectors:
                cursor.execute('''
                    INSERT INTO hot_sectors (
                        crawl_date, rank, sector_name, sector_gain,
                        leader_stock_code, leader_stock_name, leader_gain,
                        stock_count, capital_inflow
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    crawl_date,
                    sector.get('rank', 0),
                    sector.get('sector_name', ''),
                    sector.get('sector_gain', 0),
                    sector.get('leader_stock_code', ''),
                    sector.get('leader_stock_name', ''),
                    sector.get('leader_gain', 0),
                    sector.get('stock_count', 0),
                    sector.get('capital_inflow', 0)
                ))
            
            conn.commit()
            logger.info(f"成功保存 {len(sectors)} 个热门板块数据 ({crawl_date})")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"保存热门板块数据失败: {e}", exc_info=True)
            raise
        finally:
            conn.close()
    
    def save_market_sentiment(self, data: Dict[str, Any], crawl_date: Optional[date] = None):
        """
        保存市场情绪数据
        
        Args:
            data: 市场情绪数据
            crawl_date: 爬取日期（如果为None则使用今天）
        """
        if not data:
            logger.warning("市场情绪数据为空，跳过保存")
            return
        
        if crawl_date is None:
            crawl_date = date.today()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 使用REPLACE INTO实现插入或更新
            cursor.execute('''
                REPLACE INTO market_sentiment (
                    crawl_date, market_activity, up_count, down_count, flat_count,
                    limit_up_count, limit_up_real_count, limit_down_count, limit_down_real_count,
                    st_limit_up_count, st_limit_down_count, suspended_count,
                    median_gain_all, median_gain_hs300, median_gain_sh, median_gain_sz, median_gain_cyb,
                    sentiment_score, gain_distribution_json, update_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                crawl_date,
                data.get('market_activity', 0),
                data.get('up_count', 0),
                data.get('down_count', 0),
                data.get('flat_count', 0),
                data.get('limit_up_count', 0),
                data.get('limit_up_real_count', 0),
                data.get('limit_down_count', 0),
                data.get('limit_down_real_count', 0),
                data.get('st_limit_up_count', 0),
                data.get('st_limit_down_count', 0),
                data.get('suspended_count', 0),
                data.get('median_gain', {}).get('all', 0),
                data.get('median_gain', {}).get('hs300', 0),
                data.get('median_gain', {}).get('sh', 0),
                data.get('median_gain', {}).get('sz', 0),
                data.get('median_gain', {}).get('cyb', 0),
                data.get('sentiment_score', 0),
                json.dumps(data.get('gain_distribution', {}) or {}, ensure_ascii=False),
                data.get('update_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            ))
            
            conn.commit()
            logger.info(f"成功保存市场情绪数据 ({crawl_date})")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"保存市场情绪数据失败: {e}", exc_info=True)
            raise
        finally:
            conn.close()
    
    def get_stock_sector_info(self, stock_code: str, trade_date: date) -> Dict[str, Any]:
        """
        获取股票的板块信息
        
        Args:
            stock_code: 股票代码（6位数字）
            trade_date: 交易日期
        
        Returns:
            {
                'is_hot_sector': True,
                'sector_rank': 3,
                'sector_name': 'AI应用',
                'sector_gain': 5.23,
                'is_sector_leader': False,
                'sector_position_rank': 5,
                'sector_capital_inflow': 2000000000
            }
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 查找该股票是否为某个热门板块的龙头
            cursor.execute('''
                SELECT rank, sector_name, sector_gain, leader_stock_code, 
                       leader_gain, stock_count, capital_inflow
                FROM hot_sectors
                WHERE crawl_date = ? AND leader_stock_code = ?
                ORDER BY rank
                LIMIT 1
            ''', (trade_date, stock_code))
            
            row = cursor.fetchone()
            
            if row:
                # 该股票是某个热门板块的龙头
                return {
                    'is_hot_sector': True,
                    'sector_rank': row[0],
                    'sector_name': row[1],
                    'sector_gain': row[2],
                    'is_sector_leader': True,
                    'sector_position_rank': 1,  # 龙头排名第1
                    'sector_capital_inflow': row[6]
                }
            
            # 如果不是龙头，返回默认值（TODO: 未来可以通过其他方式判断股票所属板块）
            return {
                'is_hot_sector': False,
                'sector_rank': 999,
                'sector_name': '',
                'sector_gain': 0,
                'is_sector_leader': False,
                'sector_position_rank': 999,
                'sector_capital_inflow': 0
            }
            
        except Exception as e:
            logger.error(f"获取股票板块信息失败: {e}", exc_info=True)
            return {
                'is_hot_sector': False,
                'sector_rank': 999,
                'sector_name': '',
                'sector_gain': 0,
                'is_sector_leader': False,
                'sector_position_rank': 999,
                'sector_capital_inflow': 0
            }
        finally:
            conn.close()
    
    def get_market_sentiment(self, trade_date: date) -> Dict[str, Any]:
        """
        获取市场情绪数据
        
        Args:
            trade_date: 交易日期
        
        Returns:
            {
                'market_activity': 71.73,
                'limit_up_real_count': 92,
                'sentiment_score': 75.0,
                'median_gain_all': 0.68,
                ...
            }
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT market_activity, up_count, down_count, flat_count,
                       limit_up_count, limit_up_real_count, limit_down_count,
                       median_gain_all, median_gain_hs300, median_gain_sh,
                       median_gain_sz, median_gain_cyb, sentiment_score,
                       gain_distribution_json
                FROM market_sentiment
                WHERE crawl_date = ?
            ''', (trade_date,))
            
            row = cursor.fetchone()
            
            if row:
                dist = {}
                try:
                    if row[13]:
                        dist = json.loads(row[13])
                except Exception:
                    dist = {}
                return {
                    'market_activity': row[0],
                    'up_count': row[1],
                    'down_count': row[2],
                    'flat_count': row[3],
                    'limit_up_count': row[4],
                    'limit_up_real_count': row[5],
                    'limit_down_count': row[6],
                    'median_gain_all': row[7],
                    'median_gain_hs300': row[8],
                    'median_gain_sh': row[9],
                    'median_gain_sz': row[10],
                    'median_gain_cyb': row[11],
                    'sentiment_score': row[12],
                    'gain_distribution': dist,
                    'ref_date': trade_date,
                    'fallback_days': 0,
                }
            
            # 如果没有数据，尝试查找最近30天内的数据作为近似（向前查找：更符合“用最近历史替代”）
            logger.warning(f"未找到 {trade_date} 的市场情绪数据，尝试向前查找最近数据...")
            
            # 确保trade_date是date对象
            if isinstance(trade_date, str):
                target_date = datetime.strptime(trade_date, '%Y-%m-%d').date()
            elif isinstance(trade_date, datetime):
                target_date = trade_date.date()
            else:
                target_date = trade_date
            
            # 查找最近30天内的数据（向前查找）
            for days_offset in range(1, 31):
                fallback_date = target_date - timedelta(days=days_offset)
                cursor.execute('''
                    SELECT market_activity, up_count, down_count, flat_count,
                           limit_up_count, limit_up_real_count, limit_down_count,
                           median_gain_all, median_gain_hs300, median_gain_sh,
                           median_gain_sz, median_gain_cyb, sentiment_score,
                           gain_distribution_json
                    FROM market_sentiment
                    WHERE crawl_date = ?
                ''', (fallback_date,))
                
                row = cursor.fetchone()
                if row:
                    dist = {}
                    try:
                        if row[13]:
                            dist = json.loads(row[13])
                    except Exception:
                        dist = {}
                    logger.info(f"使用 {fallback_date} 的市场情绪数据作为近似（向前回溯{days_offset}天）")
                    return {
                        'market_activity': row[0],
                        'up_count': row[1],
                        'down_count': row[2],
                        'flat_count': row[3],
                        'limit_up_count': row[4],
                        'limit_up_real_count': row[5],
                        'limit_down_count': row[6],
                        'median_gain_all': row[7],
                        'median_gain_hs300': row[8],
                        'median_gain_sh': row[9],
                        'median_gain_sz': row[10],
                        'median_gain_cyb': row[11],
                        'sentiment_score': row[12],
                        'gain_distribution': dist,
                        'ref_date': fallback_date,
                        'fallback_days': days_offset,
                    }
            
            # 如果还是没有数据，返回默认值
            logger.warning(f"未找到 {trade_date} 及其后30天内的市场情绪数据，返回默认值")
            return {
                'market_activity': 50.0,
                'up_count': 0,
                'down_count': 0,
                'flat_count': 0,
                'limit_up_count': 0,
                'limit_up_real_count': 0,
                'limit_down_count': 0,
                'median_gain_all': 0.0,
                'median_gain_hs300': 0.0,
                'median_gain_sh': 0.0,
                'median_gain_sz': 0.0,
                'median_gain_cyb': 0.0,
                'sentiment_score': 50.0,
                'gain_distribution': {},
                'ref_date': target_date,
                'fallback_days': 999,
            }
            
        except Exception as e:
            logger.error(f"获取市场情绪数据失败: {e}", exc_info=True)
            return {
                'market_activity': 50.0,
                'limit_up_real_count': 0,
                'sentiment_score': 50.0,
                'median_gain_all': 0.0
            }
        finally:
            conn.close()
    
    def get_hot_sectors(self, trade_date: date, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        获取热门板块列表
        
        Args:
            trade_date: 交易日期
            top_n: 返回前N个板块
        
        Returns:
            板块列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT rank, sector_name, sector_gain, leader_stock_code,
                       leader_stock_name, leader_gain, stock_count, capital_inflow
                FROM hot_sectors
                WHERE crawl_date = ?
                ORDER BY rank
                LIMIT ?
            ''', (trade_date, top_n))
            
            rows = cursor.fetchall()
            
            return [
                {
                    'rank': row[0],
                    'sector_name': row[1],
                    'sector_gain': row[2],
                    'leader_stock_code': row[3],
                    'leader_stock_name': row[4],
                    'leader_gain': row[5],
                    'stock_count': row[6],
                    'capital_inflow': row[7]
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"获取热门板块列表失败: {e}", exc_info=True)
            return []
        finally:
            conn.close()

