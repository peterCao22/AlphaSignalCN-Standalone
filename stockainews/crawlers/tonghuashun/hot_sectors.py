"""
同花顺热门板块爬虫

爬取同花顺市场热点数据，包括热门板块排名、涨幅、龙头股等信息。
数据源：https://eq.10jqka.com.cn/frontend/thsTopRank/index.html#/
"""
import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from typing import List, Dict, Any, Optional
from datetime import datetime
from stockainews.core.logger import setup_logger
from stockainews.crawlers.base_crawler import BaseCrawler, CrawlerError

logger = setup_logger(__name__)


class TonghuashunHotSectorsCrawler(BaseCrawler):
    """同花顺热门板块爬虫"""
    
    BASE_URL = "https://eq.10jqka.com.cn/frontend/thsTopRank/index.html#/"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("同花顺热门板块爬虫初始化完成")
    
    async def crawl(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        爬取热门板块数据
        
        Returns:
            热门板块列表
        """
        return await self.crawl_hot_sectors()
    
    async def crawl_hot_sectors(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        爬取热门板块数据
        
        Args:
            top_n: 获取前N个热门板块（默认20）
        
        Returns:
            [
                {
                    'rank': 1,
                    'sector_name': 'AI应用',
                    'sector_gain': 5.23,
                    'leader_stock_code': '002202',
                    'leader_stock_name': '金风科技',
                    'leader_gain': 10.01,
                    'stock_count': 85,
                    'capital_inflow': 2000000000,  # 20亿（单位：元）
                    'crawl_time': '2025-12-29 15:00:00'
                },
                ...
            ]
        """
        try:
            await self._init_browser()
            await self._get_page()
            
            if not self.page:
                raise CrawlerError("页面对象未初始化")
            
            logger.info(f"正在访问: {self.BASE_URL}")
            
            # 访问页面
            await self.page.goto(self.BASE_URL, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(2)

            # === 关键：必须切到「板块」页，且再切到「概念板块」 ===
            await self._ensure_board_concept_view()
            
            # 提取板块数据
            sectors = await self._extract_sector_data(top_n)
            
            logger.info(f"成功爬取 {len(sectors)} 个热门板块")
            return sectors
            
        except Exception as e:
            logger.error(f"爬取热门板块数据失败: {e}", exc_info=True)
            raise CrawlerError(f"爬取热门板块数据失败: {e}")
        finally:
            await self.cleanup()

    async def _ensure_board_concept_view(self) -> None:
        """
        确保页面位于「板块」页，并切换到「概念板块」。

        说明：同花顺页面是多层 swiper，单纯用通用选择器容易误抓到“热股卡片”。
        这里用两段式保证：
        - 先点击顶部 tab「板块」，等待板块 swiper `#plate` 出现
        - 再通过 `#plate.swiper.slideTo(0)` 强制切换到概念容器 `#plate-container-concept`
        """
        if not self.page:
            raise CrawlerError("页面对象未初始化")

        # 1) 点击顶部「板块」tab（JS 点击更稳，避免遮罩层拦截）
        try:
            await self.page.evaluate(
                """
                () => {
                  const tabs = Array.from(document.querySelectorAll('div, a, button, span'));
                  const el = tabs.find(e => (e.innerText || '').trim() === '板块');
                  if (el) el.click();
                }
                """
            )
        except Exception as e:
            logger.debug(f"JS 点击板块失败（将继续等待 #plate）: {e}")

        try:
            await self.page.wait_for_selector('#plate', timeout=15000)
            logger.info("✓ 已进入「板块」页（检测到 #plate）")
        except Exception:
            # 再兜底一次：用 Playwright click 尝试
            try:
                await self.page.click('text=板块', timeout=5000)
                await self.page.wait_for_selector('#plate', timeout=15000)
                logger.info("✓ 已进入「板块」页（click 兜底成功）")
            except Exception as e:
                raise CrawlerError(f"无法进入「板块」页（未检测到 #plate）: {e}")

        await asyncio.sleep(1)

        # 2) 强制切到「概念板块」：#plate 是内层 swiper，slideTo(0) 对应 concept
        try:
            await self.page.evaluate(
                """
                () => {
                  const plate = document.querySelector('#plate');
                  if (plate && plate.swiper && typeof plate.swiper.slideTo === 'function') {
                    plate.swiper.slideTo(0, 0);
                  }
                }
                """
            )
            await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"切换到概念板块失败（将继续尝试直接提取）: {e}")
    
    async def _extract_sector_data(self, top_n: int) -> List[Dict[str, Any]]:
        """
        从页面提取板块数据（基于div布局）
        
        Args:
            top_n: 提取前N个板块
        
        Returns:
            板块数据列表
        """
        try:
            # 等待「板块」页概念容器出现（避免抓到热股）
            try:
                await self.page.wait_for_selector('.swiper-slide-active[title="板块"] #plate-container-concept', timeout=15000)
            except Exception:
                # 允许板块 slide 存在但未被 aria 识别为 active（DOM 仍可提取）
                await self.page.wait_for_selector('#plate-container-concept', timeout=15000)

            await asyncio.sleep(2)  # 等待数据加载
            
            # 使用JavaScript提取数据（基于div布局）
            sectors = await self.page.evaluate(f'''
                () => {{
                    const results = [];

                    // === 核心：限定作用域到「板块」页（概念榜）===
                    const boardSlide = document.querySelector('.swiper-slide-active[title="板块"]') || document.querySelector('.swiper-slide[title="板块"]');
                    if (!boardSlide) {{
                        console.log('未找到 title=板块 的 slide，放弃提取');
                        return results;
                    }}
                    const conceptContainer = boardSlide.querySelector('#plate-container-concept');
                    if (!conceptContainer) {{
                        console.log('未找到 #plate-container-concept，放弃提取');
                        return results;
                    }}

                    // 仅选择“列表直接子元素”（最稳）：pl-32 容器下的 20 个卡片（pt-16/pt-20...）
                    const listWrapper = conceptContainer.querySelector('.pl-32.pr-32.pb-24');
                    let cards = [];
                    if (listWrapper) {{
                        cards = Array.from(listWrapper.children).filter(el =>
                            el.classList && el.classList.contains('bgc-white') && el.classList.contains('border')
                        );
                    }}

                    // 兜底：如果结构变化，退回到“包含 rank 节点 + ellipsis 的 bgc-white.border”元素
                    if (cards.length === 0) {{
                        cards = Array.from(conceptContainer.querySelectorAll('div.bgc-white.border'))
                            .filter(el => !!el.querySelector('.THSMF-M.bold') && !!el.querySelector('.ellipsis'));
                    }}

                    // 提取并按页面显示 rank 排序（DOM 顺序可能不稳定）
                    const parsed = cards.map((card) => {{
                        try {{
                            const rankEl = card.querySelector('.THSMF-M.bold');
                            const rank = rankEl ? parseInt((rankEl.innerText || '').trim(), 10) : NaN;

                            // 名称：ellipsis 第一行（避免重复文本/标签干扰）
                            const ell = card.querySelector('.ellipsis');
                            let name = ell ? (ell.innerText || '').trim().split('\\n')[0].trim() : '';
                            if (!name) {{
                              const lines = (card.innerText || '').split('\\n').map(x=>x.trim()).filter(Boolean);
                              name = lines.length > 1 ? lines[1] : (lines[0] || '');
                            }}

                            // 板块涨幅：优先取 range 元素
                            let gain = 0;
                            const rangeEl = card.querySelector('.range');
                            const gainTxt = rangeEl ? (rangeEl.innerText || '') : (card.innerText || '');
                            const mGain = gainTxt.match(/([+-]?\\d+\\.\\d+)%/);
                            if (mGain) gain = parseFloat(mGain[1]);

                            // 涨停家数
                            let limitUpCount = 0;
                            const mLU = (card.innerText || '').match(/(\\d+)家涨停/);
                            if (mLU) limitUpCount = parseInt(mLU[1], 10);

                            // 资金流入（页面未必有该字段，有则解析，否则 0）
                            let capitalInflow = 0;
                            const mMoney = (card.innerText || '').match(/(\\d+\\.?\\d*)(亿|万)流入/);
                            if (mMoney) {{
                              const v = parseFloat(mMoney[1]);
                              capitalInflow = mMoney[2] === '亿' ? v * 1e8 : v * 1e4;
                            }}

                            if (!Number.isFinite(rank) || !name || name.length < 2) return null;
                            return {{
                              rank,
                              sector_name: name,
                              sector_gain: gain,
                              leader_stock_code: '',
                              leader_stock_name: '',
                              leader_gain: 0,
                              stock_count: limitUpCount,
                              capital_inflow: capitalInflow
                            }};
                        }} catch (err) {{
                            console.error('解析板块数据失败:', err);
                            return null;
                        }}
                    }}).filter(x => x && Number.isFinite(x.rank));

                    parsed.sort((a, b) => a.rank - b.rank);
                    parsed.slice(0, {top_n}).forEach(x => results.push(x));
                    
                    return results;
                }}
            ''')
            
            # 添加爬取时间
            crawl_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for sector in sectors:
                sector['crawl_time'] = crawl_time
            
            if len(sectors) < 5:  # 如果少于5个板块，保存调试信息
                logger.warning(f"只提取到 {len(sectors)} 个板块，可能需要调整选择器")
                # 保存页面截图和HTML用于调试
                debug_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                screenshot_path = f"debug_hot_sectors_{debug_time}.png"
                html_path = f"debug_hot_sectors_{debug_time}.html"
                
                await self.page.screenshot(path=screenshot_path)
                html_content = await self.page.content()
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                logger.info(f"已保存调试文件: {screenshot_path}, {html_path}")
            
            return sectors
            
        except Exception as e:
            logger.error(f"提取板块数据失败: {e}", exc_info=True)
            return []


# 测试代码
async def test_crawler():
    """测试爬虫"""
    logger.info("="*80)
    logger.info("开始测试同花顺热门板块爬虫")
    logger.info("="*80)
    
    crawler = TonghuashunHotSectorsCrawler(headless=False)  # 使用有头模式便于调试
    
    try:
        sectors = await crawler.crawl_hot_sectors(top_n=10)
        
        if sectors:
            logger.info(f"\n成功爬取 {len(sectors)} 个热门板块:")
            logger.info("-"*80)
            for sector in sectors:
                logger.info(f"[{sector['rank']}] {sector['sector_name']}")
                logger.info(f"    涨幅: {sector['sector_gain']:.2f}%")
                logger.info(f"    龙头: {sector['leader_stock_name']}({sector['leader_stock_code']}) {sector['leader_gain']:.2f}%")
                logger.info(f"    个股数: {sector['stock_count']}")
                logger.info(f"    资金流入: {sector['capital_inflow']/1e8:.2f}亿")
                logger.info("-"*80)
        else:
            logger.warning("未获取到板块数据")
            
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
    
    logger.info("="*80)
    logger.info("测试完成")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(test_crawler())

