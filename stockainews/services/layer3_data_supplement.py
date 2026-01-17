"""
Layer 3数据补充服务

用于补充Layer 3量化技术分析所需的数据：
1. 日K线数据：通过 API 获取
2. 涨停数据：通过 API 获取
3. 龙虎榜数据：通过 BigQuant API 获取
"""
import pandas as pd
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from stockainews.adapters.zhitu_adapter import ZhituAdapter
from stockainews.adapters.moma_adapter import MomaAdapter
from stockainews.core.logger import setup_logger

logger = setup_logger(__name__)

def _import_bigquant_dai():
    """
    统一导入 BigQuant SDK（dai）。
    - 优先 bigquantdai.dai
    - 兼容 bigquant.api.dai
    - 最后尝试 import dai
    若均不可用：抛出 ImportError（让上层明确感知“补数失败”）。
    """
    try:
        from bigquantdai import dai  # type: ignore
        return dai
    except ImportError:
        try:
            from bigquant.api import dai  # type: ignore
            return dai
        except ImportError:
            try:
                import dai  # type: ignore
                return dai
            except ImportError as e:
                raise ImportError(
                    "BigQuant SDK 不可用：缺少 bigquantdai/bigquant.api/dai 模块，请先安装并配置。"
                ) from e


class Layer3DataSupplement:
    """Layer 3数据补充服务"""
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        zhitu_adapter: Optional[ZhituAdapter] = None,
        moma_adapter: Optional[MomaAdapter] = None,
        use_moma: bool = True  # 默认使用MomaAPI（因为智兔服务器有问题）
    ):
        """
        初始化数据补充服务
        
        Args:
            base_path: Layer 3数据的基础路径
            zhitu_adapter: 智兔API适配器（如果为None则自动创建）
            moma_adapter: 魔码云服API适配器（如果为None则自动创建）
            use_moma: 是否使用MomaAPI（默认True，因为智兔服务器有问题）
        """
        # 设置数据路径
        if base_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.base_path = project_root / "TradingAgents-chinese-market" / "AlphaSignal-CN"
        else:
            self.base_path = Path(base_path)
        
        self.data_dir = self.base_path / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.limit_up_dir = self.raw_dir / "limit_up"
        
        # 确保目录存在
        self.limit_up_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # 【优化3】初始化两个适配器，支持自动切换
        self.use_moma = use_moma
        self.moma_adapter = moma_adapter or MomaAdapter()
        self.zhitu_adapter = zhitu_adapter or ZhituAdapter()
        
        # 设置当前使用的适配器
        if use_moma:
            self.api_adapter = self.moma_adapter  # 使用MomaAPI
            self.current_api = "MomaAPI"
            logger.info("使用魔码云服API（MomaAPI）")
        else:
            self.api_adapter = self.zhitu_adapter  # 使用智兔API
            self.current_api = "ZhituAPI"
            logger.info("使用智兔API（ZhituAPI）")
        
        logger.info(f"Layer3数据补充服务初始化完成，数据路径: {self.base_path}")
    
    def switch_api(self):
        """
        切换API适配器（MomaAPI ↔ ZhituAPI）
        
        当当前API达到限额或出现错误时，自动切换到备用API
        """
        if self.current_api == "MomaAPI":
            self.api_adapter = self.zhitu_adapter
            self.current_api = "ZhituAPI"
            self.use_moma = False
            logger.warning("⚠️ 魔码云服API限额已用完或出错，切换到智兔API")
        else:
            self.api_adapter = self.moma_adapter
            self.current_api = "MomaAPI"
            self.use_moma = True
            logger.warning("⚠️ 智兔API出错，切换到魔码云服API")
        
        return self.current_api
    
    def _is_api_limit_error(self, error_msg: str) -> bool:
        """
        判断是否为API限额错误
        
        Args:
            error_msg: 错误信息
        
        Returns:
            True表示是限额错误
        """
        limit_keywords = [
            '限额', 'limit', 'quota', 'exceeded', 
            '超出', '次数', 'rate limit', '429',
            'too many requests', '用完', '耗尽'
        ]
        error_lower = str(error_msg).lower()
        return any(keyword.lower() in error_lower for keyword in limit_keywords)
    
    def _to_zhitu_symbol(self, stock_code: str) -> str:
        """
        转换股票代码格式（标准格式 → 智兔格式）
        
        Args:
            stock_code: 标准股票代码（如"000001"）
        
        Returns:
            智兔格式（如"000001.SZ"）
        """
        if not stock_code or len(stock_code) != 6:
            return stock_code
        
        # 判断市场
        if stock_code.startswith(("000", "001", "002", "003")):
            return f"{stock_code}.SZ"  # 深市
        elif stock_code.startswith(("300", "301")):
            return f"{stock_code}.SZ"  # 创业板
        elif stock_code.startswith(("688", "689")):
            return f"{stock_code}.SH"  # 科创板
        elif stock_code.startswith(("600", "601", "603", "605")):
            return f"{stock_code}.SH"  # 沪市
        elif stock_code.startswith(("8", "43", "92")):
            return f"{stock_code}.BJ"  # 北交所（30%涨停）
        else:
            return f"{stock_code}.SZ"  # 默认深市
    
    def _normalize_date(self, date_str: str) -> str:
        """
        标准化日期格式为YYYY-MM-DD
        
        Args:
            date_str: 日期字符串（支持YYYYMMDD或YYYY-MM-DD）
        
        Returns:
            YYYY-MM-DD格式的日期字符串
        """
        if len(date_str) == 8:  # YYYYMMDD
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        elif len(date_str) == 10 and '-' in date_str:  # YYYY-MM-DD
            return date_str
        else:
            raise ValueError(f"Invalid date format: {date_str}")
    
    def supplement_kline_data(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        output_file: Optional[str] = None,
        force_refresh_from: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        使用BigQuant SDK补充K线数据
        
        Args:
            stock_codes: 股票代码列表（如["000001", "000002"]）
            start_date: 开始日期（格式：YYYY-MM-DD或YYYYMMDD）
            end_date: 结束日期（格式：YYYY-MM-DD或YYYYMMDD）
            output_file: 输出CSV文件路径（如果为None则使用 data/raw/kline/ 目录）
            force_refresh_from: 强制刷新起始日期（YYYY-MM-DD或YYYYMMDD）。
                - 传入后：忽略“最新日期已满足”的跳过逻辑
                - 并在合并前删除这些股票在 force_refresh_from（含）之后的旧行，再写入新数据实现覆盖
        
        Returns:
            K线数据DataFrame
        """
        logger.info(f"开始补充K线数据: {len(stock_codes)}只股票, {start_date} 至 {end_date}")
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        
        # 【重要】确定输出文件路径
        # 方案A：规定补K线永远只写入 kline_all.csv，避免覆盖特征文件/其他派生文件
        if output_file is None:
            kline_dir = self.raw_dir / "kline"
            kline_dir.mkdir(parents=True, exist_ok=True)
            output_file = kline_dir / "kline_all.csv"
            logger.info(f"将写入K线主文件: {output_file}")
        else:
            output_file = Path(output_file)
        
        # 【优化】检查现有数据，实现增量下载
        existing_df = None
        refresh_dt = None
        # 默认回看窗口：为了修复缺失/坏行/数据源小幅回填，增量下载会额外回看最近几天
        lookback_days = 5
        if force_refresh_from:
            try:
                force_refresh_from = self._normalize_date(force_refresh_from)
                refresh_dt = pd.to_datetime(force_refresh_from)
                logger.warning(f"【强制刷新】force_refresh_from={force_refresh_from}（将覆盖该日期起的旧K线行）")
            except Exception as e:
                logger.warning(f"force_refresh_from 无效（{force_refresh_from}）将忽略: {e}")
                force_refresh_from = None
                refresh_dt = None

        if output_file.exists():
            try:
                existing_df = pd.read_csv(output_file, dtype={'stock_code': str})
                if 'stock_code' in existing_df.columns:
                    existing_df['stock_code'] = existing_df['stock_code'].astype(str).str.zfill(6)
                
                logger.info(f"发现已有K线数据文件: {output_file}, 共 {len(existing_df)} 条记录")
                
                # 检查每只股票的最新日期
                if not existing_df.empty and 'date' in existing_df.columns:
                    existing_df['date'] = pd.to_datetime(existing_df['date'], format='ISO8601')
                    
                    # 过滤出需要更新的股票
                    end_dt = pd.to_datetime(end_date)
                    stocks_to_update = []
                    # 记录每只股票的“建议下载起点”（用于真正的增量下载分组）
                    per_stock_download_start: dict[str, str] = {}
                    # 数据质量检查窗口：即使最新日期已到 end_date，但最近窗口存在 NaN/异常，也要强制重拉修复
                    quality_window_start = end_dt - timedelta(days=10)
                    quality_price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in existing_df.columns]
                    quality_vol_cols = [c for c in ['volume', 'amount'] if c in existing_df.columns]
                    
                    for stock_code in stock_codes:
                        # 转换为 BigQuant instrument 格式
                        instrument = self._to_zhitu_symbol(stock_code)
                        stock_df = existing_df[existing_df['instrument'] == instrument]
                        
                        if stock_df.empty:
                            # 没有该股票的数据，需要下载
                            stocks_to_update.append(stock_code)
                            per_stock_download_start[stock_code] = start_date
                            logger.info(f"{stock_code}: 无历史数据，需要下载")
                        else:
                            latest_date = stock_df['date'].max()
                            # 【新增】数据质量检查：最近窗口存在价格字段为空时，强制重拉修复（避免出现 close/amount 为 NaN、volume 为 0 的“坏行”）
                            needs_quality_fix = False
                            if quality_price_cols:
                                recent = stock_df[stock_df['date'] >= quality_window_start]
                                # 识别“停牌行”（常见表现：价格全空 + volume=0，amount多为NaN）
                                # 这类行在数据上是合理的，不应该触发强制重拉。
                                suspension_mask = None
                                try:
                                    if not recent.empty and 'volume' in recent.columns:
                                        suspension_mask = recent[quality_price_cols].isna().all(axis=1) & (recent['volume'].fillna(0) == 0)
                                except Exception:
                                    suspension_mask = None

                                recent_effective = recent
                                if suspension_mask is not None:
                                    recent_effective = recent.loc[~suspension_mask].copy()

                                # 价格字段缺失 或 价格为0（明显异常）都视为坏行（但排除停牌行）
                                price_missing = (not recent_effective.empty) and recent_effective[quality_price_cols].isna().any().any()
                                price_zero = False
                                try:
                                    if not recent_effective.empty:
                                        price_zero = (recent_effective[quality_price_cols] == 0).any().any()
                                except Exception:
                                    price_zero = False
                                # 成交字段异常：volume/amount 缺失或为0（连续为0通常是写坏/缺字段）
                                vol_bad = False
                                if quality_vol_cols and not recent_effective.empty:
                                    try:
                                        vol_missing = recent_effective[quality_vol_cols].isna().any().any()
                                        vol_zero = (recent_effective[quality_vol_cols] == 0).any().any()
                                        vol_bad = vol_missing or vol_zero
                                    except Exception:
                                        vol_bad = True

                                if price_missing or price_zero or vol_bad:
                                    needs_quality_fix = True
                                    logger.warning(
                                        f"{stock_code}: 最近{(end_dt.date()-quality_window_start.date()).days}天存在坏K线（"
                                        f"price_missing={price_missing}, price_zero={price_zero}, vol_bad={vol_bad}），将强制重拉修复"
                                    )
                            # 如果指定强制刷新：本轮所有股票都视为需要更新（覆盖区间）
                            if force_refresh_from:
                                stocks_to_update.append(stock_code)
                                per_stock_download_start[stock_code] = force_refresh_from
                                logger.info(f"{stock_code}: 强制刷新模式，覆盖 {force_refresh_from} 至 {end_date}")
                            elif latest_date < end_dt or needs_quality_fix:
                                stocks_to_update.append(stock_code)
                                if needs_quality_fix:
                                    # 质量修复：回看最近窗口（至少10天）
                                    per_stock_download_start[stock_code] = max(
                                        start_date,
                                        quality_window_start.strftime('%Y-%m-%d')
                                    )
                                else:
                                    # 正常增量：从最新日期+1开始，但额外回看 lookback_days，避免边界缺失
                                    inc_start_dt = (latest_date + timedelta(days=1)) - timedelta(days=lookback_days)
                                    per_stock_download_start[stock_code] = max(start_date, inc_start_dt.strftime('%Y-%m-%d'))
                                logger.info(f"{stock_code}: 最新数据 {latest_date.strftime('%Y-%m-%d')}, 需要更新到 {end_date}")
                            else:
                                logger.info(f"{stock_code}: 数据已是最新 ({latest_date.strftime('%Y-%m-%d')}), 跳过")
                    
                    # 如果所有股票数据都是最新的，直接返回
                    if not stocks_to_update and not force_refresh_from:
                        logger.info("[OK] 所有股票K线数据已是最新，无需下载")
                        return existing_df
                    
                    # 更新股票列表为需要下载的股票
                    stock_codes = stocks_to_update
                    logger.info(f"需要更新 {len(stock_codes)} 只股票的K线数据")
            except Exception as e:
                logger.warning(f"读取已有K线数据失败: {e}，将进行全量下载")
        
        # 使用 BigQuant SDK 批量下载K线数据
        try:
            dai = _import_bigquant_dai()

            # 【关键优化】按“每只股票的增量起点”分组下载，避免被最早的 start_date 拖成大区间
            if existing_df is not None and 'per_stock_download_start' in locals() and not force_refresh_from:
                # 分组：start_date -> [stock_codes]
                groups: dict[str, list[str]] = {}
                for code in stock_codes:
                    s = per_stock_download_start.get(code, start_date)
                    groups.setdefault(s, []).append(code)
            else:
                groups = {start_date: stock_codes}

            df_parts: list[pd.DataFrame] = []
            for group_start, codes in sorted(groups.items(), key=lambda x: x[0]):
                # 转换股票代码为 BigQuant instrument 格式
                instruments = [self._to_zhitu_symbol(code) for code in codes]
                instruments_str = "','".join(instruments)

                logger.info(f"使用 BigQuant SDK 下载K线数据: {group_start} 至 {end_date}（stocks={len(codes)}）")

                sql = f"""
                SELECT * FROM cn_stock_bar1d
                WHERE instrument IN ('{instruments_str}')
                AND date >= '{group_start}' AND date <= '{end_date}'
                """
                part = dai.query(sql, full_db_scan=True).df()
                if part is None or part.empty:
                    logger.warning(f"未获取到K线数据（{group_start} 至 {end_date}，stocks={len(codes)}）")
                    continue
                df_parts.append(part)

            df = pd.concat(df_parts, ignore_index=True) if df_parts else pd.DataFrame()
            
            if df is None or df.empty:
                logger.warning(f"未获取到K线数据（{start_date} 至 {end_date}）")
                return existing_df if existing_df is not None else pd.DataFrame()
            
            logger.info(f"[OK] BigQuant 返回 {len(df)} 条K线记录")
            logger.info(f"  - 日期范围: {df['date'].min()} 至 {df['date'].max()}")
            logger.info(f"  - 股票数量: {df['instrument'].nunique()}")
            
            # 添加 stock_code 字段（从 instrument 提取）
            df['stock_code'] = df['instrument'].str.extract(r'(\d{6})')[0]
        
        except Exception as e:
            # 这里必须抛错：否则上层会误以为“补数成功”
            logger.error(f"下载K线数据失败: {e}")
            raise
        
        # 确保 stock_code 字段是字符串格式（避免被转换为整数）
        if not df.empty and 'stock_code' in df.columns:
            df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
        
        # 【新增】清洗 BigQuant 返回数据：对关键字段做数值化，并丢弃明显无效行（避免把 NaN/0 写入 kline_all.csv）
        if not df.empty:
            # 日期字段统一为 datetime（便于后续去重/排序/窗口检查）
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='ISO8601')
            
            # 关键数值字段转为 numeric（BigQuant/CSV 可能混合类型）
            numeric_cols = [c for c in [
                'open', 'high', 'low', 'close', 'volume', 'amount',
                'pre_close', 'change_pct', 'change_ratio', 'turn',
                'upper_limit', 'lower_limit', 'adjust_factor'
            ] if c in df.columns]
            for c in numeric_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            
            # 丢弃价格缺失的行（close/open/high/low 任意为空都视为无效K线）
            required_price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
            if required_price_cols:
                before = len(df)
                df = df.dropna(subset=required_price_cols)
                dropped = before - len(df)
                if dropped > 0:
                    logger.warning(f"清洗K线数据：丢弃 {dropped} 条价格缺失记录（避免写入坏行）")
        
        # 计算涨停标志（根据板块不同设置不同阈值）
        if 'change_pct' in df.columns and not df.empty:
            def get_limit_threshold(stock_code):
                """根据股票代码获取涨停阈值"""
                if not stock_code or len(stock_code) < 2:
                    return 9.5
                
                # 北交所：92、43、8开头 -> 30%
                if stock_code.startswith(('92', '43', '8')):
                    return 29.5
                # 创业板、科创板：300, 301, 688, 689开头 -> 20%
                elif stock_code.startswith(('300', '301', '688', '689')):
                    return 19.5
                # 主板：其他 -> 10%
                else:
                    return 9.5
            
            df['limit_threshold'] = df['stock_code'].apply(get_limit_threshold)
            df['is_limit_up'] = (df['change_pct'] >= df['limit_threshold']).astype(int)
            # 删除临时列
            df = df.drop(columns=['limit_threshold'])
        else:
            df['is_limit_up'] = 0
        
        # 合并新旧数据
        if existing_df is not None and not existing_df.empty:
            logger.info(f"合并新旧K线数据...")
            
            # 强制刷新：先删除 existing_df 中“这些股票 + refresh_dt(含)之后”的旧行，再拼接新数据，实现覆盖
            if force_refresh_from and refresh_dt is not None and 'instrument' in existing_df.columns and 'date' in existing_df.columns:
                try:
                    refresh_instruments = {self._to_zhitu_symbol(code) for code in stock_codes}
                    before = len(existing_df)
                    existing_df = existing_df[
                        ~(
                            existing_df['instrument'].isin(refresh_instruments)
                            & (existing_df['date'] >= refresh_dt)
                        )
                    ].copy()
                    removed = before - len(existing_df)
                    logger.warning(
                        f"【强制刷新】已删除旧K线行: {removed} 条（instruments={len(refresh_instruments)}, date>={force_refresh_from}）"
                    )
                except Exception as e:
                    logger.warning(f"【强制刷新】删除旧K线行失败，将回退为普通合并: {e}")

            # 检查列名是否匹配
            existing_cols = set(existing_df.columns)
            new_cols = set(df.columns)
            
            # 如果列名不完全匹配，尝试对齐
            if existing_cols != new_cols:
                # 重要：不能取交集（会把新字段如 turn 丢掉）。使用并集对齐，缺失列补空值。
                logger.warning(
                    f"列名不完全匹配，将使用并集对齐（缺失列补空值）。"
                    f" 原有列数={len(existing_cols)}, 新数据列数={len(new_cols)}"
                )

                # 保持列顺序：优先沿用 existing_df 的列顺序，再追加 df 中新增列
                ordered_cols = list(existing_df.columns)
                for c in df.columns:
                    if c not in existing_cols:
                        ordered_cols.append(c)

                # existing_df 补齐新增列
                for c in ordered_cols:
                    if c not in existing_df.columns:
                        existing_df[c] = np.nan

                # df 补齐旧列
                for c in ordered_cols:
                    if c not in df.columns:
                        df[c] = np.nan

                existing_df = existing_df[ordered_cols]
                df = df[ordered_cols]
                logger.info(
                    f"对齐完成：最终列数={len(ordered_cols)}，新增列={sorted(list(new_cols - existing_cols))}"
                )
            
            # 合并数据
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            
            # 去重：按 date + instrument 去重
            # 重要：不能简单 keep='last'，否则“坏行”可能覆盖/保留。
            if 'date' in combined_df.columns and 'instrument' in combined_df.columns:
                # 质量优先：为每行计算质量分（价格非空且>0、成交非空且>0），保留质量更高的那条
                score_cols = [c for c in ['open', 'high', 'low', 'close', 'volume', 'amount'] if c in combined_df.columns]
                combined_df['_quality_score'] = 0
                for c in score_cols:
                    v = pd.to_numeric(combined_df[c], errors='coerce')
                    combined_df['_quality_score'] += (v.notna() & (v > 0)).astype(int)
                combined_df = combined_df.sort_values(
                    by=['instrument', 'date', '_quality_score'],
                    ascending=[True, True, True]
                )
                combined_df = combined_df.drop_duplicates(subset=['date', 'instrument'], keep='last')
                combined_df = combined_df.drop(columns=['_quality_score'], errors='ignore')
            
            added = len(combined_df) - len(existing_df)
            logger.info(
                f"合并后数据: {len(combined_df)} 条记录（原有 {len(existing_df)} 条，本次下载 {len(df)} 条，实际写入增量 {added} 条）"
            )
            df = combined_df
        
        # 按日期和股票代码排序
        df['date'] = pd.to_datetime(df['date'], format='ISO8601')
        df = df.sort_values(by=['instrument', 'date'])
        
        # 确保保存前 stock_code 是字符串格式
        if 'stock_code' in df.columns:
            df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
        
        # 保存到文件
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"[OK] K线数据已保存: {output_file}, 共 {len(df)} 条记录")
        
        return df
    
    def supplement_limit_up_data(
        self,
        start_date: str,
        end_date: str,
        output_file: Optional[str] = None,
        skip_errors: bool = True
    ) -> pd.DataFrame:
        """
        使用智兔API补充涨停数据
        
        Args:
            start_date: 开始日期（格式：YYYY-MM-DD或YYYYMMDD）
            end_date: 结束日期（格式：YYYY-MM-DD或YYYYMMDD）
            output_file: 输出CSV文件路径（如果为None则使用默认路径）
            skip_errors: 是否跳过错误日期继续处理（默认True）
        
        Returns:
            涨停数据DataFrame
        """
        logger.info(f"开始补充涨停数据: {start_date} 至 {end_date}")
        
        # 生成日期列表（只包含交易日，这里简化处理，实际应该过滤非交易日）
        start = datetime.strptime(self._normalize_date(start_date), "%Y-%m-%d")
        end = datetime.strptime(self._normalize_date(end_date), "%Y-%m-%d")
        
        # 检查日期范围（不能是未来日期）
        today = datetime.now().date()
        if start.date() > today:
            logger.warning(f"开始日期 {start_date} 是未来日期，已跳过")
            return pd.DataFrame()
        if end.date() > today:
            logger.warning(f"结束日期 {end_date} 是未来日期，调整为今天")
            end = datetime.combine(today, datetime.min.time())
        
        all_limit_up_data = []
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        current_date = start
        while current_date <= end:
            # 跳过周末（简化处理，实际应该使用交易日历）
            if current_date.weekday() < 5:  # 0-4为周一到周五
                trade_date = current_date.strftime("%Y-%m-%d")
                trade_date_compact = current_date.strftime("%Y%m%d")
                
                # 检查是否为未来日期
                if current_date.date() > today:
                    skipped_count += 1
                    logger.debug(f"跳过未来日期: {trade_date}")
                    current_date += timedelta(days=1)
                    continue
                
                try:
                    logger.info(f"获取涨停数据: {trade_date}")
                    
                    # 调用API（MomaAPI或智兔API）
                    limit_up_list = self.api_adapter.get_limit_up_pool(trade_date_compact)
                    
                    if not limit_up_list:
                        logger.debug(f"未获取到 {trade_date} 的涨停数据（可能该日期无涨停股票或非交易日）")
                        skipped_count += 1
                    else:
                        # 转换为BigQuant格式
                        for item in limit_up_list:
                            # 提取股票代码
                            stock_code = item.get('stock_code') or item.get('code', '')
                            if not stock_code:
                                continue
                            
                            # 标准化股票代码（确保是6位数字）
                            if len(stock_code) == 6 and stock_code.isdigit():
                                zhitu_symbol = self._to_zhitu_symbol(stock_code)
                                
                                # 提取价格信息
                                current_price = item.get('current_price') or item.get('price', 0)
                                limit_up_price = item.get('limit_up_price') or item.get('upper_limit', current_price)
                                
                                limit_up_data = {
                                    'date': trade_date,
                                    'instrument': zhitu_symbol,
                                    'stock_code': stock_code,
                                    'close': float(current_price) if current_price else 0,
                                    'upper_limit': float(limit_up_price) if limit_up_price else 0,
                                    'price_change_pct': item.get('price_change_pct', 0),
                                }
                                all_limit_up_data.append(limit_up_data)
                        
                        success_count += 1
                        logger.info(f"[OK] {trade_date}: 获取到 {len(limit_up_list)} 只涨停股票")
                
                except Exception as e:
                    error_count += 1
                    error_msg = str(e)
                    
                    # 【优化3】检查是否为API限额错误
                    if self._is_api_limit_error(error_msg):
                        logger.warning(f"⚠️ {trade_date}: 检测到API限额错误，尝试切换API...")
                        self.switch_api()
                        
                        # 使用新的API重试
                        try:
                            limit_up_list = self.api_adapter.get_limit_up_pool(trade_date_compact)
                            if limit_up_list:
                                # 处理数据（复用上面的逻辑）
                                for item in limit_up_list:
                                    stock_code = item.get('stock_code') or item.get('code', '')
                                    if not stock_code:
                                        continue
                                    
                                    if len(stock_code) == 6 and stock_code.isdigit():
                                        zhitu_symbol = self._to_zhitu_symbol(stock_code)
                                        current_price = item.get('current_price') or item.get('price', 0)
                                        limit_up_price = item.get('limit_up_price') or item.get('upper_limit', current_price)
                                        
                                        limit_up_data = {
                                            'date': trade_date,
                                            'instrument': zhitu_symbol,
                                            'stock_code': stock_code,
                                            'close': float(current_price) if current_price else 0,
                                            'upper_limit': float(limit_up_price) if limit_up_price else 0,
                                            'price_change_pct': item.get('price_change_pct', 0),
                                        }
                                        all_limit_up_data.append(limit_up_data)
                                
                                success_count += 1
                                error_count -= 1  # 重试成功，撤销错误计数
                                logger.info(f"[OK] 切换API后重试成功，{trade_date}: 获取到 {len(limit_up_list)} 只涨停股票")
                            else:
                                logger.warning(f"切换API后仍未获取到数据")
                        except Exception as retry_error:
                            logger.error(f"切换API后重试失败: {retry_error}")
                    else:
                        # 其他类型错误
                        if "500" in error_msg or "Internal Server Error" in error_msg:
                            logger.warning(f"⚠️ {trade_date}: API服务器错误（500），可能该日期数据不可用或非交易日")
                        elif "404" in error_msg or "Not Found" in error_msg:
                            logger.warning(f"⚠️ {trade_date}: 数据不存在（404），可能该日期数据不可用")
                        else:
                            logger.error(f"❌ {trade_date}: 获取涨停数据失败: {e}")
                    
                    if not skip_errors and not self._is_api_limit_error(error_msg):
                        raise  # 如果不跳过错误且不是限额错误，则抛出异常
                    # 继续处理下一个日期
            else:
                skipped_count += 1
            
            current_date += timedelta(days=1)
        
        # 输出统计信息
        logger.info(f"\n涨停数据补充统计:")
        logger.info(f"  成功: {success_count} 个交易日")
        logger.info(f"  失败: {error_count} 个交易日")
        logger.info(f"  跳过: {skipped_count} 个日期（周末/未来日期/无数据）")
        logger.info(f"  总计: {len(all_limit_up_data)} 条涨停记录")
        
        
        if not all_limit_up_data:
            if error_count > 0:
                logger.warning(f"未获取到任何涨停数据（{error_count} 个日期失败）")
            else:
                logger.warning("未获取到任何涨停数据（可能日期范围内无涨停股票）")
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(all_limit_up_data)
        
        # 保存到文件（合并已有数据，避免重复）
        if output_file is None:
            output_file = self.limit_up_dir / "limit_up_history_supplement.csv"
        else:
            output_file = Path(output_file)
        
        # 如果文件已存在，合并数据并去重
        if output_file.exists():
            try:
                existing_df = pd.read_csv(output_file)
                logger.info(f"发现已有数据文件: {output_file}, 共 {len(existing_df)} 条记录")
                
                # 合并数据
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                
                # 去重：按 date + instrument 去重，保留最新的数据
                combined_df = combined_df.drop_duplicates(
                    subset=['date', 'instrument'],
                    keep='last'  # 保留最新的数据（补充的数据）
                )
                
                logger.info(f"合并后数据: {len(combined_df)} 条记录（原有 {len(existing_df)} 条，新增 {len(df)} 条，去重后 {len(combined_df) - len(existing_df)} 条）")
                df = combined_df
            except Exception as e:
                logger.warning(f"读取已有数据失败，将覆盖文件: {e}")
        
        # 按日期和股票代码排序
        df = df.sort_values(by=['date', 'instrument'])
        
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"[OK] 涨停数据已保存: {output_file}, 共 {len(df)} 条记录")
        
        return df
    
    def supplement_dragon_list_data(
        self,
        start_date: str,
        end_date: str,
        output_file: Optional[str] = None,
        skip_errors: bool = True
    ) -> pd.DataFrame:
        """
        使用 BigQuant API 补充龙虎榜数据
        
        Args:
            start_date: 开始日期（格式：YYYY-MM-DD或YYYYMMDD）
            end_date: 结束日期（格式：YYYY-MM-DD或YYYYMMDD）
            output_file: 输出CSV文件路径（如果为None则使用默认路径）
            skip_errors: 是否跳过错误（用于兼容性，已弃用）
        
        Returns:
            龙虎榜数据DataFrame
        """
        logger.info(f"开始补充龙虎榜数据（使用 BigQuant API）: {start_date} 至 {end_date}")
        
        # 规范化日期格式
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        
        # 检查日期范围（不能是未来日期）
        today = datetime.now().date()
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start_dt.date() > today:
            logger.warning(f"开始日期 {start_date} 是未来日期，已跳过")
            return pd.DataFrame()
        if end_dt.date() > today:
            logger.warning(f"结束日期 {end_date} 是未来日期，调整为今天")
            end_date = today.strftime("%Y-%m-%d")
        
        # 【优化】检查现有数据，实现增量下载
        if output_file is None:
            output_file = self.limit_up_dir / "dragon_list.csv"
        else:
            output_file = Path(output_file)
        
        # 检查现有数据的最新日期
        existing_df = None
        latest_existing_date = None
        
        if output_file.exists():
            try:
                existing_df = pd.read_csv(output_file)
                if not existing_df.empty and 'date' in existing_df.columns:
                    existing_df['date'] = pd.to_datetime(existing_df['date'], format='ISO8601')
                    latest_existing_date = existing_df['date'].max()
                    logger.info(f"发现已有龙虎榜数据，最新日期: {latest_existing_date.strftime('%Y-%m-%d')}, 共 {len(existing_df)} 条记录")
                    
                    # 如果已有数据已经包含请求的日期范围，直接返回
                    end_dt_check = pd.to_datetime(end_date)
                    if latest_existing_date >= end_dt_check:
                        logger.info(f"[OK] 龙虎榜数据已是最新（已有数据到 {latest_existing_date.strftime('%Y-%m-%d')}），无需下载")
                        return existing_df
                    
                    # 调整起始日期为最新数据的下一天
                    download_start = (latest_existing_date + timedelta(days=1)).strftime('%Y-%m-%d')
                    logger.info(f"执行增量下载: {download_start} 至 {end_date}")
                    start_date = download_start
            except Exception as e:
                logger.warning(f"读取已有龙虎榜数据失败: {e}，将进行全量下载")
        
        # 使用 BigQuant SDK 下载龙虎榜数据
        try:
            logger.info(f"使用 BigQuant SDK 下载龙虎榜数据: {start_date} 至 {end_date}")
            
            dai = _import_bigquant_dai()
            
            # 使用 SQL 查询龙虎榜数据
            sql = f"""
            SELECT * FROM cn_stock_dragon_list 
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            """
            
            df = dai.query(sql, full_db_scan=True).df()
            
            if df is None or df.empty:
                logger.warning(f"未获取到龙虎榜数据（{start_date} 至 {end_date}）")
                return existing_df if existing_df is not None else pd.DataFrame()
            
            logger.info(f"[OK] BigQuant 返回 {len(df)} 条龙虎榜记录")
            logger.info(f"  - 日期范围: {df['date'].min()} 至 {df['date'].max()}")
            logger.info(f"  - 股票数量: {df['instrument'].nunique()}")
        
        except Exception as e:
            logger.error(f"下载龙虎榜数据失败: {e}")
            raise
        
        # 合并新旧数据
        if existing_df is not None and not existing_df.empty:
            logger.info(f"合并新旧数据...")
            
            # 合并数据
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            
            # 去重：按 date + instrument 去重，保留最新的数据
            combined_df = combined_df.drop_duplicates(
                subset=['date', 'instrument'],
                keep='last'  # 保留最新的数据
            )
            
            logger.info(f"合并后数据: {len(combined_df)} 条记录（原有 {len(existing_df)} 条，新增 {len(df)} 条）")
            df = combined_df
        
        # 按日期和股票代码排序
        df['date'] = pd.to_datetime(df['date'], format='ISO8601')
        df = df.sort_values(by=['date', 'instrument'])
        
        # 保存到文件
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"[OK] 龙虎榜数据已保存: {output_file}, 共 {len(df)} 条记录")
        
        return df
    
    def supplement_chips_data(
        self,
        stock_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        使用 BigQuant API 补充筹码数据（仅下载缺失的股票）
        
        Args:
            stock_codes: 股票代码列表（如["000001", "002112"]）
            start_date: 开始日期（默认为90天前）
            end_date: 结束日期（默认为今天）
            output_file: 输出CSV文件路径（如果为None则使用默认路径）
        
        Returns:
            筹码数据DataFrame
        """
        # 确定输出文件路径
        if output_file is None:
            output_file = self.raw_dir / "chips_all.csv"
        else:
            output_file = Path(output_file)
        
        # 确定日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            # 默认下载90天数据
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        logger.info(f"开始补充筹码数据: {len(stock_codes)}只股票, {start_date} 至 {end_date}")
        
        # 【核心优化】检查现有数据，只下载缺失或过期的股票
        existing_df = None
        stocks_to_update = []
        
        if output_file.exists():
            try:
                existing_df = pd.read_csv(output_file)
                logger.info(f"发现已有筹码数据文件: {output_file}, 共 {len(existing_df)} 条记录")
                
                if not existing_df.empty and 'date' in existing_df.columns:
                    existing_df['date'] = pd.to_datetime(existing_df['date'], format='ISO8601')
                    end_dt = pd.to_datetime(end_date)
                    
                    # 检查每只股票的筹码数据状态
                    for stock_code in stock_codes:
                        instrument = self._to_zhitu_symbol(stock_code)
                        stock_df = existing_df[existing_df['instrument'] == instrument]
                        
                        if stock_df.empty:
                            # 没有该股票的筹码数据
                            stocks_to_update.append(stock_code)
                            logger.info(f"  {stock_code}: 无筹码数据，需要下载")
                        else:
                            latest_date = stock_df['date'].max()
                            days_behind = (end_dt - latest_date).days
                            
                            if days_behind > 7:
                                # 数据超过7天未更新
                                stocks_to_update.append(stock_code)
                                logger.info(f"  {stock_code}: 最新数据 {latest_date.strftime('%Y-%m-%d')} (落后{days_behind}天), 需要更新")
                            else:
                                logger.info(f"  {stock_code}: 数据已是最新 ({latest_date.strftime('%Y-%m-%d')}), 跳过")
                    
                    # 如果所有股票数据都是最新的，直接返回
                    if not stocks_to_update:
                        logger.info("[OK] 所有股票筹码数据已是最新，无需下载")
                        return existing_df
                    
                    logger.info(f"需要更新 {len(stocks_to_update)} 只股票的筹码数据")
            except Exception as e:
                logger.warning(f"读取已有筹码数据失败: {e}，将进行全量下载")
                stocks_to_update = stock_codes
        else:
            stocks_to_update = stock_codes
        
        # 使用 BigQuant SDK 下载筹码数据
        try:
            logger.info(f"使用 BigQuant SDK 下载筹码数据: {start_date} 至 {end_date}")
            
            dai = _import_bigquant_dai()
            
            # 转换股票代码为 BigQuant instrument 格式
            instruments = [self._to_zhitu_symbol(code) for code in stocks_to_update]
            instruments_str = "','".join(instruments)
            
            # 使用 SQL 查询筹码分布数据
            sql = f"""
            SELECT * FROM cn_stock_chips_from_level2 
            WHERE instrument IN ('{instruments_str}')
            AND date >= '{start_date}' AND date <= '{end_date}'
            """
            
            df = dai.query(sql, full_db_scan=True).df()
            
            if df is None or df.empty:
                logger.warning(f"未获取到筹码数据（{start_date} 至 {end_date}）")
                return existing_df if existing_df is not None else pd.DataFrame()
            
            logger.info(f"[OK] BigQuant 返回 {len(df)} 条筹码记录")
            logger.info(f"  - 日期范围: {df['date'].min()} 至 {df['date'].max()}")
            logger.info(f"  - 股票数量: {df['instrument'].nunique()}")
            
            # 添加 stock_code 字段
            df['stock_code'] = df['instrument'].str.extract(r'(\d{6})')[0]
            df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
        
        except Exception as e:
            logger.error(f"下载筹码数据失败: {e}")
            raise
        
        # 合并新旧数据
        if existing_df is not None and not existing_df.empty:
            logger.info(f"合并新旧筹码数据...")
            
            # 检查列名是否匹配
            existing_cols = set(existing_df.columns)
            new_cols = set(df.columns)
            
            if existing_cols != new_cols:
                logger.warning(f"列名不完全匹配，原有列: {existing_cols}, 新数据列: {new_cols}")
                # 只保留共同的列
                common_cols = existing_cols & new_cols
                if common_cols:
                    existing_df = existing_df[list(common_cols)]
                    df = df[list(common_cols)]
                    logger.info(f"对齐列名，保留共同列: {common_cols}")
            
            # 合并数据
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            
            # 去重：按 date + instrument 去重，保留最新的数据
            if 'date' in combined_df.columns and 'instrument' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(
                    subset=['date', 'instrument'],
                    keep='last'
                )
            
            logger.info(f"合并后数据: {len(combined_df)} 条记录（原有 {len(existing_df)} 条，新增 {len(df)} 条）")
            df = combined_df
        
        # 按日期和股票代码排序
        df['date'] = pd.to_datetime(df['date'], format='ISO8601')
        df = df.sort_values(by=['instrument', 'date'])
        
        # 保存到文件
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"[OK] 筹码数据已保存: {output_file}, 共 {len(df)} 条记录")
        
        return df

