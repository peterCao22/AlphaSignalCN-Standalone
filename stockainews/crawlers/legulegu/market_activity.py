"""
乐咕乐股赚钱效应爬虫

爬取市场活跃度、涨跌比、涨停家数等市场情绪指标。
数据源：https://www.legulegu.com/stockdata/market-activity
"""
import httpx
from typing import Dict, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import re
from stockainews.core.logger import setup_logger

logger = setup_logger(__name__)


class LeguLeguMarketActivityCrawler:
    """乐咕乐股赚钱效应爬虫"""
    
    BASE_URL = "https://www.legulegu.com/stockdata/market-activity"
    
    def __init__(self, timeout: int = 30):
        """
        初始化爬虫
        
        Args:
            timeout: 请求超时时间（秒）
        """
        self.timeout = timeout
        logger.info("乐咕乐股赚钱效应爬虫初始化完成")
    
    def crawl_market_activity(self) -> Dict[str, Any]:
        """
        爬取赚钱效应数据
        
        Returns:
            {
                'market_activity': 71.73,           # 活跃度（涨跌比）
                'up_count': 3718,                   # 上涨个股数
                'down_count': 1272,                 # 下跌个股数
                'flat_count': 182,                  # 平盘个股数
                'limit_up_count': 111,              # 涨停个股数
                'limit_up_real_count': 92,          # 真实涨停数（非一字板）
                'limit_down_count': 3,              # 跌停个股数
                'limit_down_real_count': 3,         # 真实跌停数
                'st_limit_up_count': 15,            # ST涨停数
                'st_limit_down_count': 2,           # ST跌停数
                'suspended_count': 11,              # 停牌数
                'median_gain': {                    # 中位数涨幅
                    'all': 0.68,
                    'hs300': 0.26,
                    'sh': 0.62,
                    'sz': 0.76,
                    'cyb': 0.62
                },
                'update_time': '2026-01-09 15:00:00',
                'crawl_time': '2026-01-09 15:05:00'
            }
        """
        try:
            logger.info(f"正在访问: {self.BASE_URL}")
            
            # 发送HTTP请求
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            }
            
            response = httpx.get(self.BASE_URL, headers=headers, timeout=self.timeout, follow_redirects=True)
            
            if response.status_code != 200:
                logger.error(f"请求失败: HTTP {response.status_code}")
                return {}
            
            # 解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取数据（优先使用页面“赚钱效应”表格，避免关键词+父节点导致数字错配）
            base_counts = self._extract_effect_table_counts(soup)
            chart_data = self._extract_hidden_chart_data(soup)
            data = {
                'market_activity': self._extract_activity(soup),
                'up_count': base_counts.get('up_count', 0),
                'down_count': base_counts.get('down_count', 0),
                'flat_count': base_counts.get('flat_count', 0),
                'limit_up_count': base_counts.get('limit_up_count', 0),
                'limit_up_real_count': base_counts.get('limit_up_real_count', 0),
                'limit_down_count': base_counts.get('limit_down_count', 0),
                'limit_down_real_count': base_counts.get('limit_down_real_count', 0),
                'st_limit_up_count': base_counts.get('st_limit_up_count', 0),
                'st_limit_down_count': base_counts.get('st_limit_down_count', 0),
                'suspended_count': base_counts.get('suspended_count', 0),
                # 涨跌比分布图（隐藏 div data-chart）
                'gain_distribution': chart_data.get('gain_distribution', {}),
                'chart_counts': chart_data.get('counts', {}),
                'median_gain': self._extract_median_gain(soup),
                'update_time': self._extract_update_time(soup),
                'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 计算情绪得分
            data['sentiment_score'] = self._calculate_sentiment_score(data)
            
            logger.info(f"成功爬取赚钱效应数据: 活跃度={data['market_activity']:.2f}%, 情绪得分={data['sentiment_score']:.1f}")
            return data
            
        except Exception as e:
            logger.error(f"爬取赚钱效应数据失败: {e}", exc_info=True)
            return {}
    
    def _extract_activity(self, soup: BeautifulSoup) -> float:
        """提取涨跌比（活跃度）"""
        try:
            # 页面上有明确标题："涨跌比: xx.xx%"
            h = soup.find(['h1', 'h2', 'h3', 'h4'], string=re.compile(r'涨跌比', re.I))
            if h:
                text = h.get_text(strip=True)
                match = re.search(r'(\d+\.?\d*)%', text)
                if match:
                    return float(match.group(1))
            
            # 备用方案：查找class或id包含activity的元素
            activity_div = soup.find(['div', 'span'], class_=re.compile(r'activity|ratio', re.I))
            if activity_div:
                text = activity_div.get_text()
                match = re.search(r'(\d+\.?\d*)%', text)
                if match:
                    return float(match.group(1))
                    
            logger.warning("未找到活跃度数据")
            return 0.0
        except Exception as e:
            logger.error(f"提取活跃度失败: {e}")
            return 0.0
    
    def _extract_effect_table_counts(self, soup: BeautifulSoup) -> Dict[str, int]:
        """
        解析“赚钱效应”区块的表格，准确提取：上涨/下跌/平盘、涨停/跌停/停牌、真实涨停/真实跌停、ST涨停/ST跌停。

        页面结构（示例）：
        - row1: 上涨 2251 下跌 2808 平盘 113
        - row2: 涨停 67 跌停 61 停牌 12
        - row3: 真实涨停 60 真实跌停 27
        - row4: st st*涨停 9 st st*跌停 11
        """
        out = {
            'up_count': 0,
            'down_count': 0,
            'flat_count': 0,
            'limit_up_count': 0,
            'limit_down_count': 0,
            'suspended_count': 0,
            'limit_up_real_count': 0,
            'limit_down_real_count': 0,
            'st_limit_up_count': 0,
            'st_limit_down_count': 0,
        }
        try:
            # 找到“涨跌比”标题附近的第一张表（就是赚钱效应表）
            h = soup.find(['h1', 'h2', 'h3', 'h4'], string=re.compile(r'涨跌比', re.I))
            table = None
            if h:
                table = h.find_next('table')
            if table is None:
                # 兜底：取页面中第一个包含“上涨/下跌/平盘”的 table
                for t in soup.find_all('table'):
                    txt = t.get_text()
                    if '上涨' in txt and '下跌' in txt and '平盘' in txt:
                        table = t
                        break
            if table is None:
                logger.warning("未找到赚钱效应表格，返回默认0")
                return out

            def parse_row_pairs(cells: list[str]) -> dict[str, int]:
                pairs = {}
                for i in range(0, len(cells) - 1, 2):
                    k = cells[i].strip()
                    v = cells[i + 1].strip().replace(',', '')
                    if not k:
                        continue
                    if re.fullmatch(r'\d+', v):
                        pairs[k] = int(v)
                return pairs

            rows = table.find_all('tr')
            for r in rows:
                cells = [c.get_text(strip=True) for c in r.find_all(['td', 'th'])]
                if not cells:
                    continue
                pairs = parse_row_pairs(cells)
                if not pairs:
                    continue

                # 映射
                if '上涨' in pairs:
                    out['up_count'] = pairs.get('上涨', out['up_count'])
                if '下跌' in pairs:
                    out['down_count'] = pairs.get('下跌', out['down_count'])
                if '平盘' in pairs:
                    out['flat_count'] = pairs.get('平盘', out['flat_count'])

                if '涨停' in pairs:
                    out['limit_up_count'] = pairs.get('涨停', out['limit_up_count'])
                if '跌停' in pairs:
                    out['limit_down_count'] = pairs.get('跌停', out['limit_down_count'])
                if '停牌' in pairs:
                    out['suspended_count'] = pairs.get('停牌', out['suspended_count'])

                if '真实涨停' in pairs:
                    out['limit_up_real_count'] = pairs.get('真实涨停', out['limit_up_real_count'])
                if '真实跌停' in pairs:
                    out['limit_down_real_count'] = pairs.get('真实跌停', out['limit_down_real_count'])

                # ST 标签可能是 "st st*涨停" / "ST涨停" 等
                for k, v in pairs.items():
                    if re.search(r'st.*涨停', k, re.I):
                        out['st_limit_up_count'] = v
                    if re.search(r'st.*跌停', k, re.I):
                        out['st_limit_down_count'] = v

            return out
        except Exception as e:
            logger.error(f"解析赚钱效应表格失败: {e}", exc_info=True)
            return out

    def _extract_hidden_chart_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        解析“涨跌比分布图”对应的隐藏 div（形如：<div id="up0To3" data-chart="1616" style="display:none"></div>）

        Returns:
            {
              "counts": { "total": 5184, "totalUp": 2251, ... },
              "gain_distribution": {
                 "up": {"0_3": 1616, "3_5": 326, "5_7": 143, "7_10": 89, "10_20": 77},
                 "down": {"0_3": 2236, "3_5": 350, "5_7": 104, "7_10": 37, "10_20": 0, "20_10": 0},
                 "price_top": 113,
                 "price_paused": 12
              }
            }
        """
        out: Dict[str, Any] = {"counts": {}, "gain_distribution": {}}
        try:
            divs = soup.find_all('div', attrs={'data-chart': True})
            if not divs:
                return out

            raw: Dict[str, int] = {}
            for d in divs:
                _id = d.get('id')
                val = d.get('data-chart')
                if not _id or val is None:
                    continue
                try:
                    raw[_id] = int(str(val).strip().replace(',', ''))
                except Exception:
                    continue

            # 汇总类
            # 注意：该页面 data-chart 的 id 命名以实际HTML为准（你截图里也展示了）：
            # - limitUp/limitDown/priceStop/pricePaused 等，而不是 totalLimitUp/priceTop
            counts_keys = [
                'total', 'totalUp', 'totalDown',
                'limitUp', 'realLimitUp',
                'limitDown', 'realLimitDown',
                'priceStop', 'pricePaused',
            ]
            out['counts'] = {k: raw.get(k, 0) for k in counts_keys}

            # 分布 bins
            up_bins = {
                '0_3': raw.get('up0To3', 0),
                '3_5': raw.get('up3To5', 0),
                '5_7': raw.get('up5To7', 0),
                '7_10': raw.get('up7To10', 0),
                '10_20': raw.get('up10To20', 0),
            }
            down_bins = {
                '0_3': raw.get('down0To3', 0),
                '3_5': raw.get('down3To5', 0),
                '5_7': raw.get('down5To7', 0),
                '7_10': raw.get('down7To10', 0),
                '10_20': raw.get('down10To20', 0),
                # 页面里有一个 down20To10（-20%~-10%）
                '20_10': raw.get('down20To10', 0),
            }

            out['gain_distribution'] = {
                'up': up_bins,
                'down': down_bins,
                # 兼容：priceStop = 平盘数量；pricePaused = 停牌数量
                'flat': raw.get('priceStop', 0),
                'paused': raw.get('pricePaused', 0),
                # 兼容旧键名（如果外部已有依赖）
                'price_top': raw.get('priceStop', 0),
                'price_paused': raw.get('pricePaused', 0),
            }
            return out
        except Exception as e:
            logger.warning(f"解析涨跌比分布图失败: {e}")
            return out
    
    def _extract_median_gain(self, soup: BeautifulSoup) -> Dict[str, float]:
        """提取中位数涨幅"""
        try:
            median_data = {
                'all': 0.0,
                'hs300': 0.0,
                'sh': 0.0,
                'sz': 0.0,
                'cyb': 0.0
            }
            
            # 查找“中位数涨幅”标题附近的表格
            header = soup.find(text=re.compile(r'中位数涨幅', re.I))
            table = None
            if header:
                p = header.find_parent()
                if p:
                    table = p.find_next('table')
            if table is None:
                # 兜底：取包含“全部A股/沪深300/上证/深证/创业板”的表格
                for t in soup.find_all('table'):
                    txt = t.get_text()
                    if '全部A股' in txt and '沪深300' in txt and '创业板' in txt:
                        table = t
                        break
            
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    text = row.get_text()
                    # 匹配各个指数的中位数涨幅
                    if '全部A股' in text or 'all' in text.lower():
                        match = re.search(r'(\d+\.?\d*)%', text)
                        if match:
                            median_data['all'] = float(match.group(1))
                    elif '沪深300' in text or 'hs300' in text.lower():
                        match = re.search(r'(\d+\.?\d*)%', text)
                        if match:
                            median_data['hs300'] = float(match.group(1))
                    elif '上证' in text or 'sh' in text.lower():
                        match = re.search(r'(\d+\.?\d*)%', text)
                        if match:
                            median_data['sh'] = float(match.group(1))
                    elif '深证' in text or 'sz' in text.lower():
                        match = re.search(r'(\d+\.?\d*)%', text)
                        if match:
                            median_data['sz'] = float(match.group(1))
                    elif '创业板' in text or 'cyb' in text.lower():
                        match = re.search(r'(\d+\.?\d*)%', text)
                        if match:
                            median_data['cyb'] = float(match.group(1))
            
            return median_data
        except Exception as e:
            logger.error(f"提取中位数涨幅失败: {e}")
            return {'all': 0.0, 'hs300': 0.0, 'sh': 0.0, 'sz': 0.0, 'cyb': 0.0}
    
    def _extract_update_time(self, soup: BeautifulSoup) -> str:
        """提取数据更新时间"""
        try:
            # 查找包含时间的元素（页面正文里直接展示）
            time_elem = soup.find(string=re.compile(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'))
            if time_elem:
                match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', time_elem)
                if match:
                    return match.group(1)
            
            # 备用方案：使用当前时间
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.error(f"提取更新时间失败: {e}")
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _calculate_sentiment_score(self, data: Dict[str, Any]) -> float:
        """
        计算市场综合情绪得分（0-100）
        
        逻辑：
        - 涨跌比 > 70%：强势市场（+30分）
        - 真实涨停数 > 80：投机氛围浓厚（+25分）
        - 中位数涨幅 > 0.5%：普涨行情（+20分）
        - 跌停数 < 5：风险可控（+15分）
        - 上涨个股数 > 3000：市场活跃（+10分）
        
        Args:
            data: 市场数据
        
        Returns:
            情绪得分（0-100）
        """
        score = 0
        
        # 涨跌比加分
        activity = data.get('market_activity', 0)
        if activity > 70:
            score += 30
        elif activity > 60:
            score += 20
        elif activity > 50:
            score += 10
        
        # 涨停家数加分
        limit_up_real = data.get('limit_up_real_count', 0)
        if limit_up_real > 80:
            score += 25
        elif limit_up_real > 50:
            score += 15
        elif limit_up_real > 30:
            score += 10
        
        # 中位数涨幅加分
        median_gain_all = data.get('median_gain', {}).get('all', 0)
        if median_gain_all > 0.5:
            score += 20
        elif median_gain_all > 0.3:
            score += 10
        
        # 跌停数扣分/加分
        limit_down = data.get('limit_down_count', 0)
        if limit_down > 10:
            score -= 15
        elif limit_down < 5:
            score += 15
        
        # 上涨个股数加分
        up_count = data.get('up_count', 0)
        if up_count > 3000:
            score += 10
        elif up_count > 2500:
            score += 5
        
        return min(max(score, 0), 100)  # 限制在0-100


# 测试代码
def test_crawler():
    """测试爬虫"""
    logger.info("="*80)
    logger.info("开始测试乐咕乐股赚钱效应爬虫")
    logger.info("="*80)
    
    crawler = LeguLeguMarketActivityCrawler()
    
    try:
        data = crawler.crawl_market_activity()
        
        if data:
            logger.info("\n成功爬取赚钱效应数据:")
            logger.info("-"*80)
            logger.info(f"活跃度（涨跌比）: {data['market_activity']:.2f}%")
            logger.info(f"上涨个股: {data['up_count']}")
            logger.info(f"下跌个股: {data['down_count']}")
            logger.info(f"涨停个股: {data['limit_up_count']} (真实: {data['limit_up_real_count']})")
            logger.info(f"跌停个股: {data['limit_down_count']} (真实: {data['limit_down_real_count']})")
            logger.info(f"中位数涨幅:")
            logger.info(f"  全部A股: {data['median_gain']['all']:.2f}%")
            logger.info(f"  沪深300: {data['median_gain']['hs300']:.2f}%")
            logger.info(f"  上证: {data['median_gain']['sh']:.2f}%")
            logger.info(f"  深证: {data['median_gain']['sz']:.2f}%")
            logger.info(f"  创业板: {data['median_gain']['cyb']:.2f}%")
            logger.info(f"情绪得分: {data['sentiment_score']:.1f}/100")
            logger.info(f"更新时间: {data['update_time']}")
            logger.info("-"*80)
        else:
            logger.warning("未获取到数据")
            
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
    
    logger.info("="*80)
    logger.info("测试完成")
    logger.info("="*80)


if __name__ == "__main__":
    test_crawler()

