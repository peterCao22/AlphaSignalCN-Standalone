import os
import pandas as pd
import sqlite3
import logging
import joblib
import json
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PatternMatcher:
    def __init__(self, db_path='data/processed/historical_patterns.db', 
                 processed_kline='data/processed/kline_processed.csv',
                 chips_path='data/raw/chips_all.csv',
                 dragon_list_path='data/raw/limit_up/dragon_list.csv',
                 model_path='models/second_wave_lgb.model',
                 feature_names_path='models/feature_names.json',
                 enhanced_features_dir=None,
                 dragon_seats_db_path=None):
        self.db_path = db_path
        self.processed_kline = processed_kline
        self.chips_path = chips_path
        self.dragon_list_path = dragon_list_path
        self.model_path = model_path
        self.feature_names_path = feature_names_path
        self.enhanced_features_dir = enhanced_features_dir
        # A5：同花顺龙虎榜席位明细（SQLite）
        if dragon_seats_db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dragon_seats_db_path = os.path.join(base_dir, 'data', 'dragon_seats.db')
        self.dragon_seats_db_path = dragon_seats_db_path
        self._dragon_seats_conn = None
        self._dragon_seats_cache = {}
        self._kline_cache = None
        self._chips_cache = None
        self._dragon_cache = None
        self._model_lgb = None
        self._model_xgb = None
        self._feature_names = None
        self._enhanced_service = None
        # 股性/量价(A+B)特征：按“触发日期”做缓存，避免每只股票都扫 1.5GB 大文件
        # cache key: 'YYYY-MM-DD' -> DataFrame(index=instrument, cols=feature_cols)
        self._stock_character_cache_by_date = {}
        self._stock_character_features_file = None

        # 上证指数环境（可选）：来自 data/raw/index_bar1d.csv
        self._index_bar_cache = None
        self._index_bar_path = None

        # 预测端需要补齐的股性/量价/A+B 特征列（与 models/feature_names.json 对齐）
        self._stock_character_feature_cols = [
            # 股性7
            'volatility_60d', 'limit_up_frequency', 'second_wave_history',
            'amplitude_avg_60d', 'hot_stock_days', 'concept_rotation_count', 'rebound_speed',
            # 量价8
            'volume_price_correlation', 'volume_increase_ratio', 'volume_price_divergence',
            'shrink_limit_up', 'turnover_rate', 'volume_continuity',
            'volume_pattern', 'volume_ma_ratio',
            # A+B 6
            'up_day_ratio_60d', 'up_body_sum_ratio_60d', 'net_body_strength_60d',
            'up_volume_ratio_60d', 'up_amount_ratio_60d', 'turnover_trend_20d',
        ]

    def _get_dragon_seats_conn(self):
        if self._dragon_seats_conn is None:
            try:
                self._dragon_seats_conn = sqlite3.connect(self.dragon_seats_db_path)
            except Exception as e:
                logging.warning(f"无法连接龙虎榜席位DB: {e}")
                self._dragon_seats_conn = None
        return self._dragon_seats_conn

    def _load_dragon_seats_one(self, trade_date: str, stock_code: str, window_days: int, category: str = "全部股票"):
        """
        读取单个 (trade_date, window_days, category, stock_code) 的席位 summary + detail。
        返回 dict: {'summary': {...} | None, 'details': [{'side':..,'rank':.., ...}, ...]}
        """
        key = (trade_date, int(window_days), str(category), str(stock_code))
        if key in self._dragon_seats_cache:
            return self._dragon_seats_cache[key]

        conn = self._get_dragon_seats_conn()
        if conn is None:
            self._dragon_seats_cache[key] = {'summary': None, 'details': []}
            return self._dragon_seats_cache[key]

        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT stock_name, reason, total_turnover_yuan, total_buy_yuan, total_sell_yuan, total_net_yuan
                FROM dragon_seat_summary
                WHERE trade_date=? AND window_days=? AND category=? AND stock_code=?
                LIMIT 1
                """,
                (trade_date, int(window_days), str(category), str(stock_code)),
            )
            row = cur.fetchone()
            summary = None
            if row:
                summary = {
                    'stock_name': row[0],
                    'reason': row[1],
                    'total_turnover_yuan': row[2],
                    'total_buy_yuan': row[3],
                    'total_sell_yuan': row[4],
                    'total_net_yuan': row[5],
                }

            cur.execute(
                """
                SELECT side, rank, seat_name, seat_tag, buy_yuan, sell_yuan, net_yuan
                FROM dragon_seat_detail
                WHERE trade_date=? AND window_days=? AND category=? AND stock_code=?
                ORDER BY side ASC, rank ASC
                """,
                (trade_date, int(window_days), str(category), str(stock_code)),
            )
            details = [
                {
                    'side': r[0],
                    'rank': int(r[1] or 0),
                    'seat_name': r[2],
                    'seat_tag': r[3],
                    'buy_yuan': float(r[4]) if r[4] is not None else 0.0,
                    'sell_yuan': float(r[5]) if r[5] is not None else 0.0,
                    'net_yuan': float(r[6]) if r[6] is not None else 0.0,
                }
                for r in cur.fetchall()
            ]

            self._dragon_seats_cache[key] = {'summary': summary, 'details': details}
            return self._dragon_seats_cache[key]
        except Exception as e:
            logging.warning(f"读取龙虎榜席位明细失败: {e}")
            self._dragon_seats_cache[key] = {'summary': None, 'details': []}
            return self._dragon_seats_cache[key]

    def _prev_date_keys(self, target_date, lookback_days: int = 7):
        """生成 target_date 向前回溯的日期键列表（YYYY-MM-DD），用于特征文件未更新到当日时回退。"""
        try:
            d = pd.to_datetime(target_date)
        except Exception:
            return []
        keys = []
        for i in range(1, lookback_days + 1):
            keys.append((d - pd.Timedelta(days=i)).strftime('%Y-%m-%d'))
        return keys

    def _load_stock_character_features_for_dates(self, date_keys):
        """
        一次扫描 features_kline_stock_character_*.csv，返回 {date_key -> DataFrame(index=instrument)}。
        只保留 date_keys 指定的日期集合，避免重复扫 1.5GB 大文件。
        """
        if not date_keys:
            return {}

        feature_file = self._get_stock_character_features_file()
        if not feature_file or not os.path.exists(feature_file):
            return {}

        want = set(date_keys)
        logging.info(f"加载股性/量价特征（按日期缓存）: dates={sorted(want)}, file={feature_file}")

        parts_by_date = {k: [] for k in want}
        usecols = ['instrument', 'date'] + self._stock_character_feature_cols

        try:
            for chunk in pd.read_csv(
                feature_file,
                usecols=lambda c: c in set(usecols),
                chunksize=200_000,
                dtype={'instrument': str},
                low_memory=True,
            ):
                if 'date' not in chunk.columns or 'instrument' not in chunk.columns:
                    continue

                date_str = chunk['date'].astype(str).str.slice(0, 10)
                mask_any = date_str.isin(want)
                if not mask_any.any():
                    continue

                sub = chunk.loc[mask_any].copy()
                sub['__date_key__'] = date_str.loc[mask_any].values

                keep_cols = ['instrument', '__date_key__'] + [
                    c for c in self._stock_character_feature_cols if c in sub.columns
                ]
                sub = sub[keep_cols]

                for dk, g in sub.groupby('__date_key__', sort=False):
                    if dk in parts_by_date:
                        parts_by_date[dk].append(g.drop(columns=['__date_key__']))

            out = {}
            for dk, parts in parts_by_date.items():
                if not parts:
                    continue
                df_day = pd.concat(parts, ignore_index=True)
                df_day = df_day.drop_duplicates(subset=['instrument'], keep='last').set_index('instrument')

                for col in self._stock_character_feature_cols:
                    if col in df_day.columns:
                        df_day[col] = (
                            pd.to_numeric(df_day[col], errors='coerce')
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0.0)
                        )
                out[dk] = df_day
            return out

        except Exception as e:
            logging.warning(f"加载股性/量价特征失败: {e}")
            return {}

    def _get_stock_character_features_file(self):
        """
        获取最新的 features_kline_stock_character_*.csv 文件路径（体积可能很大，需谨慎读取）。
        """
        if self._stock_character_features_file and os.path.exists(self._stock_character_features_file):
            return self._stock_character_features_file

        # enhanced_features_dir 传入时通常是 .../data/raw
        base_raw_dir = self.enhanced_features_dir
        if not base_raw_dir:
            # scripts/pattern_matcher.py -> scripts -> AlphaSignal-CN
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_raw_dir = os.path.join(base_dir, 'data', 'raw')

        features_dir = os.path.join(base_raw_dir, 'features')
        if not os.path.exists(features_dir):
            return None

        prefix = 'features_kline_stock_character_'
        candidates = [
            os.path.join(features_dir, f)
            for f in os.listdir(features_dir)
            if f.startswith(prefix) and f.endswith('.csv')
        ]
        if not candidates:
            return None

        self._stock_character_features_file = max(candidates, key=lambda p: os.path.getmtime(p))
        return self._stock_character_features_file

    def _load_stock_character_features_for_date(self, target_date):
        """
        仅加载 target_date 当天的股性/量价/A+B 特征，并缓存。

        由于 features_kline_stock_character_*.csv 可能 >1GB，不能整表读入内存：
        - 使用 chunksize 分块扫描
        - 只保留目标日期的记录（当天全市场一份）
        """
        try:
            target_date = pd.to_datetime(target_date)
        except Exception:
            return None

        # 统一用 YYYY-MM-DD 作为缓存键，避免时分秒差异
        target_key = target_date.strftime('%Y-%m-%d')
        if target_key in self._stock_character_cache_by_date:
            return self._stock_character_cache_by_date[target_key]

        # 一次扫描：同时尝试 target_key + 向前回溯最多7天（覆盖“特征文件未更新到当日/周末”的情况）
        candidate_keys = [target_key] + self._prev_date_keys(target_date, lookback_days=7)
        to_scan = [k for k in candidate_keys if k not in self._stock_character_cache_by_date]
        scanned = self._load_stock_character_features_for_dates(to_scan) if to_scan else {}

        # 写入缓存（未命中也写 None，避免重复扫描）
        for k in to_scan:
            self._stock_character_cache_by_date[k] = scanned.get(k)

        # 优先返回 target_key；若为空则回退到最近一个有数据的日期
        df_day = self._stock_character_cache_by_date.get(target_key)
        if df_day is not None:
            logging.info(f"[OK] 股性/量价特征已缓存（date={target_key}，stocks={len(df_day)}）")
            return df_day

        for k in candidate_keys[1:]:
            df_day = self._stock_character_cache_by_date.get(k)
            if df_day is not None:
                logging.warning(f"未找到股性/量价特征数据（date={target_key}），回退使用 {k}")
                return df_day

        logging.warning(f"未找到股性/量价特征数据（date={target_key}）")
        return None

    def load_dragon_list(self):
        """延迟加载龙虎榜数据"""
        if self._dragon_cache is None:
            if os.path.exists(self.dragon_list_path):
                logging.info("加载龙虎榜数据...")
                self._dragon_cache = pd.read_csv(self.dragon_list_path)
                self._dragon_cache['date'] = pd.to_datetime(self._dragon_cache['date'], format='ISO8601')
                # 设置索引以加快查询速度
                self._dragon_cache.set_index(['instrument', 'date'], inplace=True)
            else:
                logging.warning("未找到龙虎榜数据文件")
        return self._dragon_cache

    def load_chips(self):
        """延迟加载筹码数据"""
        if self._chips_cache is None:
            if os.path.exists(self.chips_path):
                logging.info("加载筹码分布数据...")
                self._chips_cache = pd.read_csv(self.chips_path)
                self._chips_cache['date'] = pd.to_datetime(self._chips_cache['date'], format='ISO8601')
                # 设置索引以加快查询速度
                self._chips_cache.set_index(['instrument', 'date'], inplace=True)
            else:
                logging.warning("未找到筹码数据文件")
        return self._chips_cache

    def load_models(self):
        """加载机器学习模型集成"""
        if self._model_lgb is None or self._model_xgb is None:
            lgb_path = self.model_path
            xgb_path = self.model_path.replace('_lgb.model', '_xgb.model')
            
            if os.path.exists(lgb_path) and os.path.exists(xgb_path):
                logging.info("加载机器学习模型集成 (LGBM + XGB)...")
                self._model_lgb = joblib.load(lgb_path)
                self._model_xgb = joblib.load(xgb_path)
                with open(self.feature_names_path, 'r') as f:
                    self._feature_names = json.load(f)
            else:
                logging.warning("未找到完整的模型集成，将仅使用规则评分")
        return self._model_lgb, self._model_xgb
    
    def load_enhanced_features_service(self):
        """延迟加载增强特征服务"""
        if self._enhanced_service is None:
            try:
                from enhanced_features_service import EnhancedFeaturesService
                self._enhanced_service = EnhancedFeaturesService(self.enhanced_features_dir)
                logging.info("[OK] 增强特征服务已加载")
            except Exception as e:
                logging.warning(f"无法加载增强特征服务: {e}")
                self._enhanced_service = None
        return self._enhanced_service

    def _get_index_bar1d_path(self):
        """默认读取 data/raw/index_bar1d.csv（上证）"""
        if self._index_bar_path and os.path.exists(self._index_bar_path):
            return self._index_bar_path
        base_raw_dir = self.enhanced_features_dir
        if not base_raw_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_raw_dir = os.path.join(base_dir, 'data', 'raw')
        self._index_bar_path = os.path.join(base_raw_dir, 'index_bar1d.csv')
        return self._index_bar_path

    def _load_index_bar1d(self):
        """延迟加载上证指数日行情"""
        if self._index_bar_cache is not None:
            return self._index_bar_cache
        path = self._get_index_bar1d_path()
        if not path or not os.path.exists(path):
            self._index_bar_cache = None
            return None
        try:
            df = pd.read_csv(path, dtype={'instrument': str})
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='ISO8601', errors='coerce')
            self._index_bar_cache = df
            return df
        except Exception as e:
            logging.warning(f"加载指数数据失败: {e}")
            self._index_bar_cache = None
            return None

    def _get_index_env_features(self, trade_date):
        """
        上证指数环境特征：
        - 连续涨跌、3/5日涨跌
        - MA5/10/20 位置、MA5乖离、是否回踩MA5
        - 量能：成交额/近5日均额
        """
        out = {
            'index_has_data': False,
            'index_ref_date': None,
            'index_fallback_days': 999,
            'index_ret_1d': 0.0,
            'index_ret_3d': 0.0,
            'index_ret_5d': 0.0,
            'index_close': 0.0,
            'index_ma5': 0.0,
            'index_ma10': 0.0,
            'index_ma20': 0.0,
            'index_bias_ma5': 0.0,
            'index_above_ma5': False,
            'index_above_ma10': False,
            'index_above_ma20': False,
            'index_touch_ma5': False,
            'index_amount_ratio_5d': 1.0,
            'index_up_streak': 0,
            'index_down_streak': 0,
        }

        df = self._load_index_bar1d()
        if df is None or df.empty or 'date' not in df.columns:
            return out

        try:
            td = pd.to_datetime(trade_date)
        except Exception:
            return out

        # 找到 ref_date（当日没有则向前回退最多7天）
        ref_date = None
        fallback = 0
        for i in range(0, 8):
            d = (td - pd.Timedelta(days=i)).normalize()
            day = df[df['date'] == d]
            if not day.empty:
                ref_date = d
                fallback = i
                break
        if ref_date is None:
            return out

        # 取同一 instrument（若有 instrument 列，尽量固定用“000001.SH”那条）
        day_df = df[df['date'] == ref_date]
        if 'instrument' in df.columns and not day_df.empty:
            cand = day_df[day_df['instrument'].astype(str).isin(['000001.SH', 'SH000001', '000001.SS', '000001.SH.S'])]
            if not cand.empty:
                day_df = cand
        row = day_df.iloc[-1]

        out['index_has_data'] = True
        out['index_ref_date'] = ref_date.strftime('%Y-%m-%d')
        out['index_fallback_days'] = int(fallback)

        close = float(pd.to_numeric(row.get('close', 0.0), errors='coerce') or 0.0)
        out['index_close'] = close

        hist = df[df['date'] <= ref_date].copy()
        if 'instrument' in df.columns and 'instrument' in day_df.columns:
            inst = str(row.get('instrument') or '')
            if inst:
                hist = hist[hist['instrument'].astype(str) == inst]
        hist = hist.sort_values('date')
        if hist.empty or 'close' not in hist.columns:
            return out

        hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
        hist = hist.dropna(subset=['close'])
        if hist.empty:
            return out

        hist['ret'] = hist['close'].pct_change(fill_method=None) * 100.0
        out['index_ret_1d'] = float(hist['ret'].iloc[-1] or 0.0)
        out['index_ret_3d'] = float((hist['close'].iloc[-1] / hist['close'].shift(3).iloc[-1] - 1.0) * 100.0) if len(hist) >= 4 else 0.0
        out['index_ret_5d'] = float((hist['close'].iloc[-1] / hist['close'].shift(5).iloc[-1] - 1.0) * 100.0) if len(hist) >= 6 else 0.0

        ma5 = float(hist['close'].rolling(5).mean().iloc[-1] or 0.0) if len(hist) >= 5 else 0.0
        ma10 = float(hist['close'].rolling(10).mean().iloc[-1] or 0.0) if len(hist) >= 10 else 0.0
        ma20 = float(hist['close'].rolling(20).mean().iloc[-1] or 0.0) if len(hist) >= 20 else 0.0
        out['index_ma5'] = ma5
        out['index_ma10'] = ma10
        out['index_ma20'] = ma20
        out['index_bias_ma5'] = (close / ma5 - 1.0) if ma5 > 0 else 0.0
        out['index_above_ma5'] = bool(ma5 > 0 and close >= ma5)
        out['index_above_ma10'] = bool(ma10 > 0 and close >= ma10)
        out['index_above_ma20'] = bool(ma20 > 0 and close >= ma20)
        out['index_touch_ma5'] = bool(ma5 > 0 and abs(out['index_bias_ma5']) <= 0.003)

        if 'amount' in hist.columns:
            hist['amount'] = pd.to_numeric(hist['amount'], errors='coerce')
            amt = float(hist['amount'].iloc[-1] or 0.0)
            ma_amt5 = float(hist['amount'].rolling(5).mean().iloc[-1] or 0.0) if len(hist) >= 5 else 0.0
            out['index_amount_ratio_5d'] = (amt / ma_amt5) if ma_amt5 > 0 else 1.0

        # 连续涨跌（按 ret 正负）
        rets = hist['ret'].fillna(0.0).tolist()
        up = 0
        down = 0
        for r in reversed(rets):
            if r > 0:
                if down > 0:
                    break
                up += 1
            elif r < 0:
                if up > 0:
                    break
                down += 1
            else:
                break
        out['index_up_streak'] = int(up)
        out['index_down_streak'] = int(down)

        return out

    def load_kline(self):
        """延迟加载K线数据"""
        if self._kline_cache is None:
            logging.info("加载处理后的K线数据...")
            self._kline_cache = pd.read_csv(self.processed_kline)
            self._kline_cache['date'] = pd.to_datetime(self._kline_cache['date'], format='ISO8601')
        return self._kline_cache

    def get_stock_features(self, symbol, date, extra_context: dict | None = None):
        """提取指定股票在指定日期的特征"""
        df = self.load_kline()
        date = pd.to_datetime(date)
        
        stock_data = df[df['instrument'] == symbol].sort_values('date').reset_index(drop=True)
        if stock_data.empty:
            return None
        
        idx_list = stock_data.index[stock_data['date'] == date].tolist()
        if not idx_list:
            return None
        idx = idx_list[0]
        
        # === A2: 短周期板结构（基于“有效交易日”窗口，跳过停牌/无交易日） ===
        # 有效交易日定义：close_qfq 非空且 volume>0（与 process_kline 逻辑保持一致）
        try:
            vol_series = pd.to_numeric(stock_data['volume'], errors='coerce').fillna(0)
        except Exception:
            vol_series = stock_data.get('volume', 0)
        valid_mask = stock_data['close_qfq'].notna() & (vol_series > 0)

        valid_idx = stock_data.index[valid_mask & (stock_data.index <= idx)].tolist()
        limit_up_count_5d = 0
        limit_up_count_10d = 0
        max_consecutive_limit_up_10d = 0
        days_since_last_limit_up = 999
        gap_days_between_limit_ups = 999

        if valid_idx:
            # 最近 N 个有效交易日（含 trigger 日）
            last5_idx = valid_idx[-5:]
            last10_idx = valid_idx[-10:]

            is_lu_5 = stock_data.loc[last5_idx, 'is_limit_up'].fillna(False).astype(bool).tolist()
            is_lu_10 = stock_data.loc[last10_idx, 'is_limit_up'].fillna(False).astype(bool).tolist()

            limit_up_count_5d = int(sum(is_lu_5))
            limit_up_count_10d = int(sum(is_lu_10))

            # 近10日最长连板（在10个有效交易日子序列内的最长连续 True 段）
            run = 0
            best = 0
            for v in is_lu_10:
                if v:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            max_consecutive_limit_up_10d = int(best)

            # 距离上一次涨停的间隔（按有效交易日计，不含当天）
            if idx in valid_idx:
                pos = len(valid_idx) - 1  # idx 在 valid_idx 的位置（末尾）
                # 向前找上一个涨停
                prev_positions = []
                for j in range(len(valid_idx) - 2, -1, -1):
                    vi = valid_idx[j]
                    if bool(stock_data.loc[vi, 'is_limit_up']):
                        prev_positions.append(j)
                        break
                if prev_positions:
                    days_since_last_limit_up = int((len(valid_idx) - 1) - prev_positions[0])

            # 最近两次涨停间隔（按有效交易日计，不含当天与否都可；这里取截至当天的最近两次）
            lu_pos = [k for k, vi in enumerate(valid_idx) if bool(stock_data.loc[vi, 'is_limit_up'])]
            if len(lu_pos) >= 2:
                gap_days_between_limit_ups = int(lu_pos[-1] - lu_pos[-2])

        # 提取特征
        # 1. 连板数
        consecutive = 1
        check_idx = idx - 1
        while check_idx >= 0 and stock_data.loc[check_idx, 'is_limit_up']:
            consecutive += 1
            check_idx -= 1
            
        # 2. 量比
        volume_ratio = stock_data.loc[idx, 'volume_ratio']
        # 防御：量比可能因分母为0等原因出现 inf/NaN
        try:
            volume_ratio = float(volume_ratio)
        except Exception:
            volume_ratio = np.nan

        # 若异常：用“当天成交量 / 近5个有效交易日均量”即时重算一次（跳过停牌日 volume=0）
        if not np.isfinite(volume_ratio):
            try:
                vol = float(stock_data.loc[idx, 'volume'])
                # 仅使用 volume>0 的历史交易日
                hist_series = pd.to_numeric(stock_data.loc[:idx - 1, 'volume'], errors='coerce')
                hist_valid = hist_series.replace([np.inf, -np.inf], np.nan).dropna()
                hist_valid = hist_valid[hist_valid > 0].tail(5)
                avg_vol = float(hist_valid.mean()) if len(hist_valid) > 0 else 0.0
                if avg_vol > 0:
                    volume_ratio = vol / avg_vol
                    logging.info(f"{symbol} 量比异常已重算: {volume_ratio:.3f}（vol={vol:.0f}, avg5_valid={avg_vol:.0f}）")
                else:
                    volume_ratio = np.nan
                    logging.warning(f"{symbol} 量比异常且无法重算（近5个有效交易日均量=0），将跳过模式匹配")
            except Exception as e:
                volume_ratio = np.nan
                logging.warning(f"{symbol} 量比异常且重算失败: {e}，将跳过模式匹配")
        
        # 3. 位置
        ma60 = stock_data.loc[idx, 'ma60']
        pre_position = "低位" if stock_data.loc[idx, 'close_qfq'] < ma60 else "高位"
        
        # 4. 更多 ML 特征
        rsi = stock_data.loc[idx, 'rsi']
        pct_change = stock_data.loc[idx, 'pct_change']
        close_qfq = stock_data.loc[idx, 'close_qfq']
        
        # 偏离度
        bias_ma5 = (close_qfq / stock_data.loc[idx, 'ma5']) - 1
        bias_ma20 = (close_qfq / stock_data.loc[idx, 'ma20']) - 1
        bias_ma60 = (close_qfq / stock_data.loc[idx, 'ma60']) - 1
        
        # 5. 筹码特征（支持回退查询：找不到当日数据时，向前查找7天内最近数据）
        win_percent = 0.0
        concentration = 0.0
        price_to_cost = 0.0
        
        chips_df = self.load_chips()
        if chips_df is not None:
            try:
                # 尝试获取当日筹码数据
                chip_row = chips_df.loc[(symbol, date)]
                win_percent = float(chip_row['win_percent'])
                concentration = float(chip_row['concentration'])
                avg_cost = float(chip_row['avg_cost'])
                if avg_cost > 0:
                    price_to_cost = (close_qfq / avg_cost) - 1
            except KeyError:
                # 【优化】找不到当日数据，向前查找7天内最近的数据
                try:
                    from datetime import timedelta
                    
                    # 获取该股票的所有筹码数据
                    stock_chips = chips_df.loc[symbol]
                    if not isinstance(stock_chips, pd.DataFrame):
                        stock_chips = pd.DataFrame([stock_chips])
                    
                    # 确保date列是datetime类型
                    if not pd.api.types.is_datetime64_any_dtype(stock_chips.index):
                        stock_chips.index = pd.to_datetime(stock_chips.index)
                    
                    # 查找date之前7天内的最近数据
                    target_date = pd.to_datetime(date) if not isinstance(date, pd.Timestamp) else date
                    lookback_start = target_date - timedelta(days=7)
                    
                    # 筛选时间范围
                    recent_chips = stock_chips[
                        (stock_chips.index >= lookback_start) & 
                        (stock_chips.index <= target_date)
                    ]
                    
                    if not recent_chips.empty:
                        # 找到最近的一条数据
                        chip_row = recent_chips.iloc[-1]
                        win_percent = float(chip_row['win_percent'])
                        concentration = float(chip_row['concentration'])
                        avg_cost = float(chip_row['avg_cost'])
                        if avg_cost > 0:
                            price_to_cost = (close_qfq / avg_cost) - 1
                        
                        actual_date = recent_chips.index[-1].strftime('%Y-%m-%d')
                        days_diff = (target_date - recent_chips.index[-1]).days
                        logging.info(f"使用 {actual_date} 的筹码数据（向前回溯{days_diff}天）")
                    else:
                        logging.warning(f"{symbol} 在 {date} 前7天内无筹码数据")
                except Exception as e:
                    logging.warning(f"筹码数据回退查询失败: {e}")
        
        # 6. 龙虎榜特征
        net_buy_amount = 0.0
        dragon_df = self.load_dragon_list()
        if dragon_df is not None:
            try:
                dragon_row = dragon_df.loc[(symbol, date)]
                # 如果同一天有多条记录（不同原因），取净买入之和
                if isinstance(dragon_row, pd.DataFrame):
                    net_buy_amount = float(dragon_row['net_buy_amount'].sum())
                else:
                    net_buy_amount = float(dragon_row['net_buy_amount'])
            except KeyError:
                pass

        # 6.1 【A5 新增】龙虎榜席位资金结构（同花顺 market/longhu，来自 dragon_seats.db）
        # 说明：席位 DB 使用 6 位 stock_code；symbol 可能带 .SZ/.SH，需去后缀。
        stock_code_6 = str(symbol).split('.')[0]
        trade_date_str = pd.to_datetime(date).strftime('%Y-%m-%d')

        def _is_institution(seat_name: str | None, seat_tag: str | None) -> bool:
            n = str(seat_name or '')
            t = str(seat_tag or '')
            return ('机构' in n) or ('机构' in t)

        def _is_hot_money(seat_name: str | None, seat_tag: str | None) -> bool:
            n = str(seat_name or '')
            t = str(seat_tag or '')
            # 不做人工标注，仅用页面自带标签/关键词粗分
            return ('游资' in t) or ('敢死队' in t) or ('跟风' in t) or ('游资' in n) or ('敢死队' in n) or ('跟风' in n)

        def _calc_seat_features(window_days: int):
            blob = self._load_dragon_seats_one(trade_date_str, stock_code_6, window_days, category="全部股票")
            summary = blob.get('summary')
            details = blob.get('details') or []

            if not summary:
                return {
                    f'seat_has_data_{window_days}d': 0.0,
                    f'seat_total_turnover_yuan_{window_days}d': 0.0,
                    f'seat_total_buy_yuan_{window_days}d': 0.0,
                    f'seat_total_sell_yuan_{window_days}d': 0.0,
                    f'seat_total_net_yuan_{window_days}d': 0.0,
                    f'seat_net_to_turnover_{window_days}d': 0.0,
                    f'seat_buy_top1_ratio_{window_days}d': 0.0,
                    f'seat_buy_top3_ratio_{window_days}d': 0.0,
                    f'seat_buy_top5_ratio_{window_days}d': 0.0,
                    f'seat_buy_inst_ratio_{window_days}d': 0.0,
                    f'seat_buy_hot_money_ratio_{window_days}d': 0.0,
                    f'seat_buy_inst_count_{window_days}d': 0.0,
                    f'seat_buy_hot_money_count_{window_days}d': 0.0,
                    f'seat_buy_sell_overlap_count_{window_days}d': 0.0,
                    f'seat_buy_sell_overlap_ratio_{window_days}d': 0.0,
                    f'seat_structure_score_{window_days}d': 0.0,
                }

            total_turnover = float(summary.get('total_turnover_yuan') or 0.0)
            total_buy = float(summary.get('total_buy_yuan') or 0.0)
            total_sell = float(summary.get('total_sell_yuan') or 0.0)
            total_net = float(summary.get('total_net_yuan') or 0.0)

            buy_rows = [d for d in details if d.get('side') == 'buy']
            buy_rows = sorted(buy_rows, key=lambda x: int(x.get('rank') or 0))
            sell_rows = [d for d in details if d.get('side') == 'sell']
            sell_rows = sorted(sell_rows, key=lambda x: int(x.get('rank') or 0))

            buy_amounts = [float(d.get('buy_yuan') or 0.0) for d in buy_rows]
            buy_top1 = buy_amounts[0] if len(buy_amounts) >= 1 else 0.0
            buy_top3 = sum(buy_amounts[:3]) if buy_amounts else 0.0
            buy_top5 = sum(buy_amounts[:5]) if buy_amounts else 0.0

            inst_buy = 0.0
            hot_buy = 0.0
            inst_cnt = 0
            hot_cnt = 0
            for d in buy_rows[:5]:
                name = d.get('seat_name')
                tag = d.get('seat_tag')
                amt = float(d.get('buy_yuan') or 0.0)
                if _is_institution(name, tag):
                    inst_buy += amt
                    inst_cnt += 1
                if _is_hot_money(name, tag):
                    hot_buy += amt
                    hot_cnt += 1

            denom = total_buy if total_buy > 0 else 0.0

            # 买卖席位重合度（Top5）
            buy_names = {str(d.get('seat_name') or '') for d in buy_rows[:5] if d.get('seat_name')}
            sell_names = {str(d.get('seat_name') or '') for d in sell_rows[:5] if d.get('seat_name')}
            overlap_cnt = float(len(buy_names & sell_names))
            overlap_ratio = overlap_cnt / 5.0

            # 净额/成交额强度
            net_to_turnover = (total_net / total_turnover) if total_turnover > 0 else 0.0

            # 结构分（0~100）：多席位共振 + 机构/游资明确 + 净流入强 + 避免单席位过度集中
            top1_ratio = (buy_top1 / denom) if denom > 0 else 0.0
            top5_ratio = (buy_top5 / denom) if denom > 0 else 0.0
            inst_ratio = (inst_buy / denom) if denom > 0 else 0.0
            hot_ratio = (hot_buy / denom) if denom > 0 else 0.0
            net_strength = max(0.0, min(1.0, net_to_turnover * 5.0))  # 0.2->1.0
            top1_penalty = max(0.0, min(1.0, top1_ratio - 0.60))  # 超过0.6开始扣分
            structure_score = (
                40.0 * max(0.0, min(1.0, top5_ratio)) +
                25.0 * max(0.0, min(1.0, inst_ratio + hot_ratio)) +
                25.0 * net_strength +
                10.0 * (1.0 - top1_penalty)
            )

            return {
                f'seat_has_data_{window_days}d': 1.0,
                f'seat_total_turnover_yuan_{window_days}d': total_turnover,
                f'seat_total_buy_yuan_{window_days}d': total_buy,
                f'seat_total_sell_yuan_{window_days}d': total_sell,
                f'seat_total_net_yuan_{window_days}d': total_net,
                f'seat_net_to_turnover_{window_days}d': net_to_turnover,
                f'seat_buy_top1_ratio_{window_days}d': (buy_top1 / denom) if denom > 0 else 0.0,
                f'seat_buy_top3_ratio_{window_days}d': (buy_top3 / denom) if denom > 0 else 0.0,
                f'seat_buy_top5_ratio_{window_days}d': (buy_top5 / denom) if denom > 0 else 0.0,
                f'seat_buy_inst_ratio_{window_days}d': (inst_buy / denom) if denom > 0 else 0.0,
                f'seat_buy_hot_money_ratio_{window_days}d': (hot_buy / denom) if denom > 0 else 0.0,
                f'seat_buy_inst_count_{window_days}d': float(inst_cnt),
                f'seat_buy_hot_money_count_{window_days}d': float(hot_cnt),
                f'seat_buy_sell_overlap_count_{window_days}d': overlap_cnt,
                f'seat_buy_sell_overlap_ratio_{window_days}d': overlap_ratio,
                f'seat_structure_score_{window_days}d': float(structure_score),
            }

        a5_seat_features = {}
        # 同花顺明细通常同时提供 1日/3日两套口径：都算出来，供报告/打分使用
        a5_seat_features.update(_calc_seat_features(1))
        a5_seat_features.update(_calc_seat_features(3))
        
        # 7. 【Phase 4 新增】市场情绪和板块特征
        is_hot_sector = False
        sector_rank = 999
        sector_gain = 0.0
        is_sector_leader = False
        sector_capital_inflow = 0.0
        market_activity = 50.0
        market_limit_up_real = 0
        market_sentiment_score = 50.0
        relative_strength = 0.0
        
        try:
            # 动态导入，避免循环依赖
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            sys.path.insert(0, str(project_root))
            
            from stockainews.services.market_sentiment_service import MarketSentimentService
            sentiment_service = MarketSentimentService()
            
            # 提取股票代码（去除.SZ/.SH后缀）
            stock_code = symbol.split('.')[0]
            trade_date = date.date() if hasattr(date, 'date') else date
            
            # 获取板块信息
            sector_info = sentiment_service.get_stock_sector_info(stock_code, trade_date)
            is_hot_sector = sector_info.get('is_hot_sector', False)
            sector_rank = sector_info.get('sector_rank', 999)
            sector_gain = sector_info.get('sector_gain', 0.0)
            is_sector_leader = sector_info.get('is_sector_leader', False)
            sector_capital_inflow = sector_info.get('sector_capital_inflow', 0.0)
            
            # 获取市场情绪数据
            market_data = sentiment_service.get_market_sentiment(trade_date)
            market_activity = market_data.get('market_activity', 50.0)
            market_limit_up_real = market_data.get('limit_up_real_count', 0)
            market_sentiment_score = market_data.get('sentiment_score', 50.0)
            sentiment_fallback_days = int(market_data.get('fallback_days', 0) or 0)
            # 市场情绪分级（更适合短线：分档，而不是把 0 当“历史最差”）
            def _sentiment_bucket(score: float):
                try:
                    s = float(score)
                except Exception:
                    s = 50.0
                if s <= 20:
                    return 1, "极弱"
                if s <= 40:
                    return 2, "偏弱"
                if s <= 60:
                    return 3, "中性"
                if s <= 80:
                    return 4, "偏强"
                return 5, "极强"

            market_sentiment_level, market_sentiment_level_label = _sentiment_bucket(market_sentiment_score)

            # A4：情绪趋势（用数据库内 <=trade_date 的最近3条记录，避免跨周末/节假日）
            def _fetch_recent_sentiments(n: int = 3):
                conn = sqlite3.connect(sentiment_service.db_path)
                cur = conn.cursor()
                cur.execute(
                    "SELECT crawl_date, sentiment_score FROM market_sentiment WHERE crawl_date <= ? ORDER BY crawl_date DESC LIMIT ?",
                    (trade_date, n),
                )
                rows = cur.fetchall()
                conn.close()
                return rows

            rows = _fetch_recent_sentiments(3)
            sentiment_slope_2d = 0.0
            sentiment_slope_3d = 0.0
            try:
                if len(rows) >= 2:
                    sentiment_slope_2d = float(rows[0][1] or 0) - float(rows[1][1] or 0)
                if len(rows) >= 3:
                    sentiment_slope_3d = float(rows[0][1] or 0) - float(rows[2][1] or 0)
            except Exception:
                sentiment_slope_2d = 0.0
                sentiment_slope_3d = 0.0
            
            # 计算相对强度（个股涨幅 vs 市场中位数涨幅）
            median_gain_all = market_data.get('median_gain_all', 0.0)
            if median_gain_all != 0:
                relative_strength = pct_change / median_gain_all
            
            logging.info(f"市场情绪特征: 活跃度={market_activity:.2f}%, 情绪得分={market_sentiment_score:.1f}, 相对强度={relative_strength:.2f}")
            if is_hot_sector:
                logging.info(f"板块特征: {sector_info.get('sector_name', '')} (排名{sector_rank}), 龙头={is_sector_leader}")
                
        except Exception as e:
            logging.warning(f"获取市场情绪特征失败: {e}")
            # 使用默认值，不影响主流程
            sentiment_fallback_days = 999
            sentiment_slope_2d = 0.0
            sentiment_slope_3d = 0.0
            market_sentiment_level = 3
            market_sentiment_level_label = "中性"
        
        # 8. 【Phase 4.1 新增】增强特征（热度、概念、竞价）
        hot_rank = 999
        hot_rank_change = 0
        is_hot_stock = False
        hot_duration = 0
        concept_count = 0
        main_concept_gain = 0.0
        main_concept_rank = 999
        is_concept_leader = False
        concept_momentum_3d = 0.0
        concept_volume_ratio = 1.0
        auction_volume_ratio = 1.0
        auction_price_gap = 0.0
        auction_turnover = 0.0
        auction_strength = 0.0
        
        try:
            enhanced_service = self.load_enhanced_features_service()
            if enhanced_service:
                # 获取所有增强特征
                enhanced_features = enhanced_service.get_all_features(symbol, date)
                
                # 热度特征
                hot_rank = enhanced_features.get('hot_rank', 999)
                hot_rank_change = enhanced_features.get('hot_rank_change', 0)
                is_hot_stock = enhanced_features.get('is_hot_stock', False)
                hot_duration = enhanced_features.get('hot_duration', 0)
                
                # 概念特征
                concept_count = enhanced_features.get('concept_count', 0)
                main_concept_gain = enhanced_features.get('main_concept_gain', 0.0)
                main_concept_rank = enhanced_features.get('main_concept_rank', 999)
                is_concept_leader = enhanced_features.get('is_concept_leader', False)
                concept_momentum_3d = enhanced_features.get('concept_momentum_3d', 0.0)
                concept_volume_ratio = enhanced_features.get('concept_volume_ratio', 1.0)
                
                # 竞价特征
                auction_volume_ratio = enhanced_features.get('auction_volume_ratio', 1.0)
                auction_price_gap = enhanced_features.get('auction_price_gap', 0.0)
                auction_turnover = enhanced_features.get('auction_turnover', 0.0)
                auction_strength = enhanced_features.get('auction_strength', 0.0)
                
                if is_hot_stock:
                    logging.info(f"热度特征: 排名{hot_rank} (变化{hot_rank_change:+d}), 持续{hot_duration}天")
                if concept_count > 0:
                    logging.info(f"概念特征: {concept_count}个概念, 主概念涨幅{main_concept_gain:.2f}%, 动量{concept_momentum_3d:.2f}%")
                if auction_strength > 60:
                    logging.info(f"竞价特征: 强度{auction_strength:.1f}, 量比{auction_volume_ratio:.2f}, 高开{auction_price_gap:.2f}%")
                    
        except Exception as e:
            logging.warning(f"获取增强特征失败: {e}")
            # 使用默认值，不影响主流程

        # 8.1 【A3】主线题材匹配（基于概念成分 + 概念日行情，计算“当日TopN概念”命中情况）
        theme_hit_count_topN = 0
        theme_best_rank = 999
        theme_best_gain = 0.0
        is_main_theme = False
        try:
            enhanced_service = self.load_enhanced_features_service()
            if enhanced_service and getattr(enhanced_service, 'concept_component_df', None) is not None and getattr(enhanced_service, 'concept_bar_df', None) is not None:
                comp = enhanced_service.concept_component_df
                bar = enhanced_service.concept_bar_df

                dd = pd.to_datetime(date)
                comp_day = comp[comp['date'] == dd]
                bar_day = bar[bar['date'] == dd]

                # 1) 取“当日 TopN 概念”：按涨幅字段排序（change_pct 优先，其次 pct_change）
                gain_col = None
                if 'change_pct' in bar_day.columns:
                    gain_col = 'change_pct'
                elif 'pct_change' in bar_day.columns:
                    gain_col = 'pct_change'

                if gain_col and not bar_day.empty and 'instrument' in bar_day.columns:
                    bar_day[gain_col] = pd.to_numeric(bar_day[gain_col], errors='coerce')
                    bar_sorted = bar_day.sort_values(gain_col, ascending=False)
                    topN = 20
                    top_concepts = bar_sorted[['instrument', gain_col]].dropna().head(topN).copy()
                    # 排名：1..N
                    top_concepts['rank'] = range(1, len(top_concepts) + 1)
                    rank_map = dict(zip(top_concepts['instrument'], top_concepts['rank']))
                    gain_map = dict(zip(top_concepts['instrument'], top_concepts[gain_col]))
                    top_set = set(rank_map.keys())

                    # 2) 取该股当日所属概念（member_code=股票，instrument=概念）
                    if not comp_day.empty and 'member_code' in comp_day.columns and 'instrument' in comp_day.columns:
                        stock_concepts = comp_day[comp_day['member_code'] == symbol]['instrument'].dropna().unique().tolist()
                        hits = [c for c in stock_concepts if c in top_set]
                        theme_hit_count_topN = int(len(hits))
                        is_main_theme = theme_hit_count_topN > 0
                        if hits:
                            best_c = min(hits, key=lambda c: rank_map.get(c, 999))
                            theme_best_rank = int(rank_map.get(best_c, 999))
                            theme_best_gain = float(gain_map.get(best_c, 0.0) or 0.0)

        except Exception as e:
            logging.warning(f"A3 主线题材匹配计算失败: {e}")

        # 8.1.1 【新增】上证指数环境（可选：本地 index_bar1d.csv）
        index_env = {}
        try:
            index_env = self._get_index_env_features(date)
        except Exception as e:
            logging.warning(f"指数环境特征计算失败: {e}")
            index_env = {}

        # 8.2 【新增】板质量特征（来自涨停池 extra_context）
        board_first_seal_time = ''
        board_last_seal_time = ''
        board_explosion_count = 0
        board_seal_funds = 0.0
        board_turnover_rate = 0.0
        board_consecutive_boards = 0
        board_quality_score = 0.0

        def _parse_hms_to_minutes(hms: str) -> float | None:
            if not hms or not isinstance(hms, str):
                return None
            try:
                parts = hms.strip().split(':')
                if len(parts) < 2:
                    return None
                hh = int(parts[0])
                mm = int(parts[1])
                ss = int(parts[2]) if len(parts) >= 3 else 0
                # 以 09:30 为开盘基准，计算距开盘分钟数
                base = 9 * 60 + 30
                cur = hh * 60 + mm + ss / 60.0
                return cur - base
            except Exception:
                return None

        if extra_context:
            try:
                board_first_seal_time = str(extra_context.get('first_seal_time', '') or '')
                board_last_seal_time = str(extra_context.get('last_seal_time', '') or '')
                board_explosion_count = int(extra_context.get('explosion_count', 0) or 0)
                board_seal_funds = float(extra_context.get('seal_funds', 0.0) or 0.0)
                board_turnover_rate = float(extra_context.get('turnover_rate', 0.0) or 0.0)
                board_consecutive_boards = int(extra_context.get('consecutive_boards', 0) or 0)

                # 计算板质量分（0~100），缺失时保持为0（不影响单股模式）
                score = 50.0
                first_m = _parse_hms_to_minutes(board_first_seal_time)
                last_m = _parse_hms_to_minutes(board_last_seal_time)

                # 越早封板越好
                if first_m is not None:
                    if first_m <= 30:        # 10:00 前
                        score += 20
                    elif first_m <= 60:      # 10:30 前
                        score += 10
                    elif first_m >= 300:     # 14:30 后
                        score -= 10

                # 最后封板过晚略扣分（更像烂板回封）
                if last_m is not None and last_m >= 320:  # 14:50 后
                    score -= 5

                # 炸板次数
                if board_explosion_count == 0:
                    score += 10
                elif board_explosion_count >= 3:
                    score -= 10

                # 封板资金（粗略分段）
                if board_seal_funds >= 1e8:
                    score += 15
                elif board_seal_funds >= 5e7:
                    score += 10
                elif board_seal_funds >= 1e7:
                    score += 5

                # 换手、连板小加分（可选）
                if board_turnover_rate >= 15:
                    score += 5
                if board_consecutive_boards >= 2:
                    score += 5

                board_quality_score = float(max(0.0, min(100.0, score)))
            except Exception as e:
                logging.warning(f"板质量特征解析失败: {e}")

        # 9. 【方案A】补齐股性/量价/A+B 特征（来自 features_kline_stock_character_*.csv）
        # 这部分对 ML 概率影响很大：如果缺失会被模型输入补0，导致 ml_prob 偏低
        #
        # 注意：不要用 locals() 写入局部变量（Python 不保证生效），这里用明确的 dict 承接。
        stock_char = {col: 0.0 for col in self._stock_character_feature_cols}
        try:
            day_df = self._load_stock_character_features_for_date(date)
            # 有些股票在“回退日期”可能仍然没有行（例如 1/15 回退到 1/14，但该股只在 1/13 有记录）
            # 因此这里再做一次“按股票命中回退”：在缓存的日期键里继续向前找最近有该股票的那天。
            picked_key = None
            picked_df = None
            date_key = date.strftime('%Y-%m-%d')
            if day_df is not None and symbol in day_df.index:
                picked_key = date_key
                picked_df = day_df
            else:
                # 向前最多7天找该股票行（这些日期在 _load_stock_character_features_for_date 内已被批量扫描并缓存）
                for dk in self._prev_date_keys(date, lookback_days=7):
                    cached = self._stock_character_cache_by_date.get(dk)
                    if cached is not None and symbol in cached.index:
                        picked_key = dk
                        picked_df = cached
                        break

            if picked_df is not None and picked_key is not None:
                if picked_key != date_key:
                    logging.warning(f"股性/量价特征按股票命中回退：symbol={symbol}, 用 {picked_key} 替代 {date_key}")
                row = picked_df.loc[symbol]
                for col in self._stock_character_feature_cols:
                    if col not in picked_df.columns:
                        continue
                    val = row.get(col, 0.0)
                    if pd.isna(val):
                        val = 0.0
                    fv = float(val)
                    if np.isinf(fv):
                        fv = 0.0
                    stock_char[col] = fv
            else:
                logging.warning(f"未命中股性/量价特征：symbol={symbol}, date={date_key}（将由模型输入补0）")
        except Exception as e:
            logging.warning(f"合并股性/量价特征失败: {e}")
        
        return {
            'symbol': symbol,
            'date': date,
            'consecutive_count': int(consecutive),
            # A2 短周期板结构
            'limit_up_count_5d': int(limit_up_count_5d),
            'limit_up_count_10d': int(limit_up_count_10d),
            'max_consecutive_limit_up_10d': int(max_consecutive_limit_up_10d),
            'days_since_last_limit_up': int(days_since_last_limit_up),
            'gap_days_between_limit_ups': int(gap_days_between_limit_ups),
            'volume_ratio': float(volume_ratio),
            'pre_position': pre_position,
            'rsi': float(rsi),
            'pct_change': float(pct_change),
            'bias_ma5': float(bias_ma5),
            'bias_ma20': float(bias_ma20),
            'bias_ma60': float(bias_ma60),
            'win_percent': win_percent,
            'concentration': concentration,
            'price_to_cost': price_to_cost,
            'net_buy_amount': net_buy_amount,
            # A5 资金结构（同花顺龙虎榜席位明细，1日/3日）
            **a5_seat_features,
            # Phase 4 新增特征（市场情绪）
            'is_hot_sector': is_hot_sector,
            'sector_rank': sector_rank,
            'sector_gain': sector_gain,
            'is_sector_leader': is_sector_leader,
            'sector_capital_inflow': sector_capital_inflow,
            'market_activity': market_activity,
            'market_limit_up_real': market_limit_up_real,
            'market_sentiment_score': market_sentiment_score,
            'market_sentiment_level': int(market_sentiment_level),
            'market_sentiment_level_label': str(market_sentiment_level_label),
            # A4 情绪趋势
            'sentiment_slope_2d': float(sentiment_slope_2d),
            'sentiment_slope_3d': float(sentiment_slope_3d),
            'sentiment_fallback_days': int(sentiment_fallback_days),
            'relative_strength': relative_strength,
            # 上证指数环境（可选）
            **(index_env or {}),
            # Phase 4.1 新增特征（热度、概念、竞价）
            'hot_rank': hot_rank,
            'hot_rank_change': hot_rank_change,
            'is_hot_stock': is_hot_stock,
            'hot_duration': hot_duration,
            'concept_count': concept_count,
            'main_concept_gain': main_concept_gain,
            'main_concept_rank': main_concept_rank,
            'is_concept_leader': is_concept_leader,
            'concept_momentum_3d': concept_momentum_3d,
            'concept_volume_ratio': concept_volume_ratio,
            'auction_volume_ratio': auction_volume_ratio,
            'auction_price_gap': auction_price_gap,
            'auction_turnover': 0.0 if (pd.isna(auction_turnover) or auction_turnover is None) else float(auction_turnover),
            'auction_strength': auction_strength,
            # A3 主线题材匹配
            'theme_hit_count_topN': int(theme_hit_count_topN),
            'theme_best_rank': int(theme_best_rank),
            'theme_best_gain': float(theme_best_gain),
            'is_main_theme': bool(is_main_theme),
            # 板质量（涨停池）
            'board_first_seal_time': board_first_seal_time,
            'board_last_seal_time': board_last_seal_time,
            'board_explosion_count': board_explosion_count,
            'board_seal_funds': board_seal_funds,
            'board_turnover_rate': board_turnover_rate,
            'board_consecutive_boards': board_consecutive_boards,
            'board_quality_score': board_quality_score,
            # 股性/量价/A+B 特征（已默认补0，命中则覆盖）
            **stock_char,
        }

    def find_similar_cases(self, features, limit=10):
        """在模式库中寻找相似案例"""
        if not features:
            logging.warning("特征缺失，无法匹配")
            return pd.DataFrame()

        vr = features.get('volume_ratio', np.nan)
        # 关键防御：inf/NaN 会导致 SQL 拼接失败（例如 BETWEEN inf AND inf）
        if vr is None or (isinstance(vr, float) and not np.isfinite(vr)) or pd.isna(vr):
            logging.warning(f"特征缺失或量比异常（{vr}），无法匹配")
            return pd.DataFrame()
            
        conn = sqlite3.connect(self.db_path)
        
        vr = float(vr)
        query = f"""
        SELECT * FROM stock_patterns
        WHERE consecutive_count = {features['consecutive_count']}
          AND pre_position = '{features['pre_position']}'
          AND volume_ratio BETWEEN {vr * 0.7} AND {vr * 1.3}
        ORDER BY ABS(volume_ratio - {vr}) ASC
        LIMIT {limit}
        """
        
        similar_df = pd.read_sql_query(query, conn)
        conn.close()
        return similar_df

    def calculate_scores(self, features, similar_cases):
        """计算强度分和模式分"""
        # 注意：即使没有相似案例（sample_size=0），也应该照样计算 ML 概率，不能直接全 0 返回
        has_cases = similar_cases is not None and not similar_cases.empty
            
        # 1. 强度分 (Strength Score) - 基于当前特征
        strength = min(features['consecutive_count'] * 20, 100)
        vr = features.get('volume_ratio', np.nan)
        if isinstance(vr, (int, float)) and np.isfinite(vr) and 1.5 <= vr <= 3.0:
            strength += 20
        elif isinstance(vr, (int, float)) and np.isfinite(vr) and vr > 5.0:
            strength += 5
            
        if features['pre_position'] == "低位":
            strength += 10
            
        # 4. 筹码强度加分
        if features.get('win_percent', 0) > 0.9:
            strength += 15  # 获利盘极高，锁仓明显
        if 0 < features.get('concentration', 1) < 0.1:
            strength += 10  # 筹码高度集中
            
        # 5. 龙虎榜加分 (Phase 1 优化：提升资金权重)
        net_buy = features.get('net_buy_amount', 0)
        consecutive_count = features.get('consecutive_count', 0)
        pre_position = features.get('pre_position', '')
        
        # 大幅提升龙虎榜资金权重
        if net_buy > 300000000:  # 净买入 > 3亿 - 超级资金
            strength += 50
        elif net_buy > 200000000:  # 净买入 > 2亿
            strength += 40
        elif net_buy > 100000000:  # 净买入 > 1亿
            strength += 30
        elif net_buy > 50000000:   # 净买入 > 5000万
            strength += 20
        elif net_buy > 10000000:   # 净买入 > 1000万
            strength += 10
        
        # 6. 强势股组合识别加成 (Phase 1 新增)
        # 连续涨停 + 大资金 = 强势确认
        if consecutive_count >= 2 and net_buy > 100000000:
            strength += 30
            logging.info(f"强势股组合：连板{consecutive_count}天 + 资金{net_buy/1e8:.2f}亿，加成+30分")
        
        # 低位 + 大资金 = 启动信号
        if pre_position == "低位" and net_buy > 50000000:
            strength += 25
            logging.info(f"低位启动信号：{pre_position} + 资金{net_buy/1e8:.2f}亿，加成+25分")

        # 6.25 【新增】A5 资金结构（龙虎榜席位明细）
        # 说明：这是对“净买入金额”的补充刻画：资金是否集中、是否偏机构/游资。
        try:
            seat_has = max(float(features.get('seat_has_data_1d', 0.0) or 0.0), float(features.get('seat_has_data_3d', 0.0) or 0.0))
            seat_net = max(float(features.get('seat_total_net_yuan_1d', 0.0) or 0.0), float(features.get('seat_total_net_yuan_3d', 0.0) or 0.0))
            seat_top1 = max(float(features.get('seat_buy_top1_ratio_1d', 0.0) or 0.0), float(features.get('seat_buy_top1_ratio_3d', 0.0) or 0.0))
            seat_top5 = max(float(features.get('seat_buy_top5_ratio_1d', 0.0) or 0.0), float(features.get('seat_buy_top5_ratio_3d', 0.0) or 0.0))
            seat_inst = max(float(features.get('seat_buy_inst_ratio_1d', 0.0) or 0.0), float(features.get('seat_buy_inst_ratio_3d', 0.0) or 0.0))
            seat_hot = max(float(features.get('seat_buy_hot_money_ratio_1d', 0.0) or 0.0), float(features.get('seat_buy_hot_money_ratio_3d', 0.0) or 0.0))
        except Exception:
            seat_has, seat_net, seat_top1, seat_top5, seat_inst, seat_hot = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if seat_has > 0.5:
            # 资金集中且净流入为正：接力更容易形成一致预期
            if seat_net > 0 and seat_top1 >= 0.25:
                strength += 8
                logging.info(f"A5席位集中+净流入（top1={seat_top1:.2f}, net={seat_net/1e8:.2f}亿）: +8分")
            # 机构/游资占比（不做手工标注，仅依据页面标签）
            if seat_inst >= 0.40:
                strength += 6
                logging.info(f"A5机构买入占比高（{seat_inst:.2f}）: +6分")
            if seat_hot >= 0.40:
                strength += 4
                logging.info(f"A5游资买入占比高（{seat_hot:.2f}）: +4分")
            # Top5 覆盖高但不过度单席位，视作“多席位共振”
            if seat_top5 >= 0.70 and seat_top1 <= 0.55:
                strength += 4
                logging.info(f"A5多席位共振（top5={seat_top5:.2f}, top1={seat_top1:.2f}）: +4分")

        # 6.2 【新增】短周期板结构（A2）
        lu5 = int(features.get('limit_up_count_5d', 0) or 0)
        lu10 = int(features.get('limit_up_count_10d', 0) or 0)
        maxc10 = int(features.get('max_consecutive_limit_up_10d', 0) or 0)
        dsl = int(features.get('days_since_last_limit_up', 999) or 999)

        # “5天多板”加分（非连续也算）
        if lu5 >= 2:
            strength += 12
            logging.info(f"短周期强势（近5日{lu5}次涨停）: +12分")
        elif lu10 >= 3:
            strength += 10
            logging.info(f"短周期活跃（近10日{lu10}次涨停）: +10分")

        # 近10日最长连板加分
        if maxc10 >= 3:
            strength += 15
            logging.info(f"近10日最长连板{maxc10}: +15分")
        elif maxc10 == 2:
            strength += 8
            logging.info(f"近10日最长连板2: +8分")

        # 距离上次涨停很近 → 接力热度仍在
        if dsl <= 2:
            strength += 6
            logging.info(f"上次涨停距今{dsl}个有效交易日: +6分")

        # 6.3 【新增】A3 主线题材匹配（当日TopN概念命中）
        thc = int(features.get('theme_hit_count_topN', 0) or 0)
        tbr = int(features.get('theme_best_rank', 999) or 999)
        tbg = float(features.get('theme_best_gain', 0.0) or 0.0)
        if thc > 0:
            # 命中即有接力基础
            strength += min(5 + thc * 3, 18)
            logging.info(f"命中当日Top概念 {thc} 个: +{min(5 + thc * 3, 18)}分")
            # 命中排名靠前的概念额外加分
            if tbr <= 5:
                strength += 18
                logging.info("命中Top5主线概念: +18分")
            elif tbr <= 10:
                strength += 12
                logging.info("命中Top10主线概念: +12分")
            elif tbr <= 20:
                strength += 6
                logging.info("命中Top20主线概念: +6分")
            # 概念涨幅作为微调（避免过度依赖）
            if tbg >= 5:
                strength += 4
            elif tbg >= 2:
                strength += 2

        # 6.1 【新增】板质量加成（A1）
        bqs = features.get('board_quality_score', 0.0)
        try:
            bqs = float(bqs)
        except Exception:
            bqs = 0.0
        if bqs >= 80:
            strength += 20
            logging.info(f"板质量优秀（{bqs:.1f}/100）: +20分")
        elif bqs >= 65:
            strength += 10
            logging.info(f"板质量良好（{bqs:.1f}/100）: +10分")
        elif 0 < bqs < 40:
            strength -= 5
            logging.info(f"板质量偏弱（{bqs:.1f}/100）: -5分")

        exp_cnt = features.get('board_explosion_count', 0)
        try:
            exp_cnt = int(exp_cnt)
        except Exception:
            exp_cnt = 0
        if exp_cnt >= 3:
            strength -= 10
            logging.info(f"炸板次数偏多（{exp_cnt}）: -10分")
            
        # 2. 模式分 (Pattern Score) - 基于历史表现
        if has_cases:
            success_rate = (similar_cases['max_return_20d'] > 0.1).mean()
            avg_max_return = similar_cases['max_return_20d'].mean()
            rule_prob = similar_cases['second_wave_confirmed'].mean()
            sample_size = len(similar_cases)
        else:
            success_rate = 0.0
            avg_max_return = 0.0
            rule_prob = 0.0
            sample_size = 0
        
        # 3. 机器学习预测分 (集成)
        ml_prob = 0.0
        model_lgb, model_xgb = self.load_models()
        if model_lgb and model_xgb and self._feature_names:
            # 准备特征向量
            x = pd.DataFrame([features])
            # 增加缺失特征 (如果有的话，没有则设为 0)
            for col in self._feature_names:
                if col not in x.columns:
                    x[col] = 0.0
            
            # 处理无穷值
            X_input = x[self._feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            prob_lgb = model_lgb.predict(X_input)[0]
            prob_xgb = model_xgb.predict_proba(X_input)[0, 1]
            ml_prob = (prob_lgb + prob_xgb) / 2
            
        # 7. 【Phase 4 新增】板块效应加成
        if features.get('is_hot_sector', False):
            sector_rank = features.get('sector_rank', 999)
            if sector_rank <= 5:
                strength += 30  # TOP5热门板块
                logging.info(f"TOP5热门板块加成: +30分")
            elif sector_rank <= 10:
                strength += 20  # TOP10热门板块
                logging.info(f"TOP10热门板块加成: +20分")
            
            if features.get('is_sector_leader', False):
                strength += 15  # 板块龙头
                logging.info(f"板块龙头加成: +15分")
        
        # 8. 【Phase 4 新增】市场情绪加成
        sentiment_score = features.get('market_sentiment_score', 50)
        sentiment_multiplier = 1.0
        
        if sentiment_score > 70:
            strength += 25  # 强势市场
            sentiment_multiplier = 1.2  # 概率上浮20%
            logging.info(f"强势市场加成: +25分，概率×1.2")
        elif sentiment_score > 60:
            strength += 15
            sentiment_multiplier = 1.1
            logging.info(f"较强市场加成: +15分，概率×1.1")
        elif sentiment_score < 40:
            strength -= 10  # 弱势市场扣分
            sentiment_multiplier = 0.9
            logging.warning(f"弱势市场扣分: -10分，概率×0.9")

        # 8.0.1 【新增】上证指数环境风控（可选：本地 index_bar1d.csv）
        try:
            if bool(features.get('index_has_data', False)):
                idx_ret1 = float(features.get('index_ret_1d', 0.0) or 0.0)
                idx_amt_ratio = float(features.get('index_amount_ratio_5d', 1.0) or 1.0)
                idx_down_streak = int(features.get('index_down_streak', 0) or 0)
                idx_above_ma5 = bool(features.get('index_above_ma5', False))

                # 规则1：大盘下跌放量（系统性风险）
                if idx_ret1 <= -1.0 and idx_amt_ratio >= 1.2:
                    strength -= 10
                    sentiment_multiplier *= 0.85
                    logging.warning(f"指数下跌放量风控: ret1d={idx_ret1:.2f}%, amt_ratio5={idx_amt_ratio:.2f} -> -10分，概率×0.85")
                # 规则2：指数连续下跌且跌破MA5（弱势延续）
                elif (not idx_above_ma5) and idx_down_streak >= 2:
                    strength -= 6
                    sentiment_multiplier *= 0.90
                    logging.info(f"指数弱势延续: down_streak={idx_down_streak}, below_ma5 -> -6分，概率×0.90")
        except Exception as e:
            logging.warning(f"指数环境风控计算失败: {e}")

        # 8.1 【新增】A4 情绪趋势微调（升温/退潮）
        slope2 = float(features.get('sentiment_slope_2d', 0.0) or 0.0)
        slope3 = float(features.get('sentiment_slope_3d', 0.0) or 0.0)
        # 升温：略加分；退潮：略扣分（幅度小，避免过拟合）
        if slope2 >= 5 or slope3 >= 8:
            strength += 6
            sentiment_multiplier *= 1.03
            logging.info(f"情绪升温（slope2={slope2:.1f}, slope3={slope3:.1f}）: +6分，概率×1.03")
        elif slope2 <= -5 or slope3 <= -8:
            strength -= 6
            sentiment_multiplier *= 0.97
            logging.info(f"情绪退潮（slope2={slope2:.1f}, slope3={slope3:.1f}）: -6分，概率×0.97")
        
        # 9. 【Phase 4 新增】相对强度加成
        relative_strength = features.get('relative_strength', 0)
        if relative_strength > 3.0:  # 跑赢市场3倍
            strength += 20
            logging.info(f"相对强度强劲（{relative_strength:.2f}倍市场）: +20分")
        
        # 10. 【Phase 4 新增】资金+板块共振
        if (net_buy > 100000000 and features.get('is_hot_sector', False)):
            strength += 30  # 强势共振
            logging.info(f"资金({net_buy/1e8:.2f}亿)+热门板块共振: +30分")
        
        # 11. 【Phase 4.1 新增】热度特征加成
        if features.get('is_hot_stock', False):
            hot_rank = features.get('hot_rank', 999)
            hot_duration = features.get('hot_duration', 0)
            
            if hot_rank <= 10:
                strength += 25  # TOP10热股
                logging.info(f"TOP10热门股票加成（排名{hot_rank}）: +25分")
            elif hot_rank <= 30:
                strength += 15  # TOP30热股
                logging.info(f"TOP30热门股票加成（排名{hot_rank}）: +15分")
            elif hot_rank <= 100:
                strength += 10  # TOP100热股
                logging.info(f"TOP100热门股票加成（排名{hot_rank}）: +10分")
            
            # 持续热门加成
            if hot_duration >= 5:
                strength += 15
                logging.info(f"持续热门加成（{hot_duration}天）: +15分")
            elif hot_duration >= 3:
                strength += 10
                logging.info(f"持续热门加成（{hot_duration}天）: +10分")
        
        # 12. 【Phase 4.1 新增】概念特征加成
        concept_count = features.get('concept_count', 0)
        main_concept_gain = features.get('main_concept_gain', 0.0)
        concept_momentum_3d = features.get('concept_momentum_3d', 0.0)
        
        if concept_count > 0:
            # 多题材加成
            if concept_count >= 5:
                strength += 15
                logging.info(f"多题材股票（{concept_count}个概念）: +15分")
            elif concept_count >= 3:
                strength += 10
                logging.info(f"多题材股票（{concept_count}个概念）: +10分")
            
            # 概念涨幅加成
            if main_concept_gain > 5.0:
                strength += 20
                logging.info(f"主概念强势（涨幅{main_concept_gain:.2f}%）: +20分")
            elif main_concept_gain > 3.0:
                strength += 15
                logging.info(f"主概念强势（涨幅{main_concept_gain:.2f}%）: +15分")
            
            # 概念动量加成
            if concept_momentum_3d > 10.0:
                strength += 20
                logging.info(f"概念动量强劲（3日{concept_momentum_3d:.2f}%）: +20分")
            elif concept_momentum_3d > 5.0:
                strength += 10
                logging.info(f"概念动量强劲（3日{concept_momentum_3d:.2f}%）: +10分")
        
        # 13. 【Phase 4.1 新增】竞价特征加成
        auction_strength = features.get('auction_strength', 0.0)
        auction_price_gap = features.get('auction_price_gap', 0.0)
        auction_volume_ratio = features.get('auction_volume_ratio', 1.0)
        
        if auction_strength > 80:
            strength += 30  # 竞价极强
            logging.info(f"集合竞价极强（{auction_strength:.1f}）: +30分")
        elif auction_strength > 70:
            strength += 20  # 竞价强
            logging.info(f"集合竞价强势（{auction_strength:.1f}）: +20分")
        elif auction_strength > 60:
            strength += 10  # 竞价良好
            logging.info(f"集合竞价良好（{auction_strength:.1f}）: +10分")
        
        # 竞价量价配合加成
        if auction_price_gap > 3.0 and auction_volume_ratio > 2.0:
            strength += 25
            logging.info(f"竞价量价配合（高开{auction_price_gap:.2f}%+量比{auction_volume_ratio:.2f}）: +25分")
        
        # 14. 【Phase 4.1 新增】热度+概念+竞价三重共振
        if (features.get('is_hot_stock', False) and 
            concept_count >= 3 and 
            auction_strength > 70):
            strength += 40  # 超级共振
            logging.info(f"热度+概念+竞价三重共振: +40分")
        
        # 15. 综合评分 (Phase 1 优化：动态调整权重)
        # 对于资金驱动型（龙虎榜 > 1亿），更信任ML模型
        if net_buy > 100000000:
            final_prob = rule_prob * 0.3 + ml_prob * 0.7  # 更信任ML
            logging.info(f"资金驱动型（{net_buy/1e8:.2f}亿），ML权重提升至70%")
        else:
            final_prob = rule_prob * 0.6 + ml_prob * 0.4  # 保持原有权重
        
        # 16. 【Phase 4 新增】应用市场情绪调整
        final_prob = final_prob * sentiment_multiplier
        
        # 17. 【Phase 4.1 新增】对于高强度股票，提升概率
        # 强度 > 200：极强势股，概率再上浮10%
        if strength > 200:
            final_prob = min(final_prob * 1.1, 1.0)
            logging.info(f"极强势股票（{strength}分），概率×1.1")
        
        return {
            'strength_score': min(strength, 250),  # Phase 4.1: 上限提高到250（增加了很多增强特征）
            'pattern_success_rate': float(success_rate),
            'avg_max_return': float(avg_max_return),
            'rule_prob': float(rule_prob),
            'ml_prob': float(ml_prob),
            'final_prob': float(final_prob),
            'sample_size': int(sample_size)
        }

if __name__ == "__main__":
    # 测试代码
    matcher = PatternMatcher()
    # 使用 2024 年的日期，确保有足够的历史数据计算 MA
    test_symbol = '000001.SZ'
    test_date = '2024-02-21'
    
    logging.info(f"测试匹配: {test_symbol} on {test_date}")
    features = matcher.get_stock_features(test_symbol, test_date)
    if features:
        logging.info(f"特征: {features}")
        similar = matcher.find_similar_cases(features)
        if not similar.empty:
            logging.info(f"找到 {len(similar)} 个相似案例")
            scores = matcher.calculate_scores(features, similar)
            logging.info(f"评分结果: {scores}")
        else:
            logging.info("未找到相似案例")
    else:
        logging.warning("未找到该股票或日期的特征")
