"""
同花顺热点榜单爬虫

爬取同花顺网站的市场热点榜单数据，包括热点名称、热度指数、变化趋势和关联股票。
"""
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from stockainews.crawlers.base_crawler import BaseCrawler
from stockainews.core.logger import setup_logger
from stockainews.core.exceptions import CrawlerError
from stockainews.crawlers.utils.popup_handler import PopupHandler

logger = setup_logger(__name__)


class TonghuashunHotTopicsCrawler(BaseCrawler):
    """同花顺热点榜单爬虫"""
    
    # 同花顺热点榜单URL
    URL = "https://eq.10jqka.com.cn/frontend/thsTopRank/index.html#/"
    
    def __init__(self, **kwargs):
        """
        初始化同花顺热点爬虫
        
        Args:
            **kwargs: 传递给BaseCrawler的参数
        """
        super().__init__(**kwargs)
        logger.info("同花顺热点爬虫初始化完成")
    
    async def crawl(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        实现抽象方法crawl（委托给crawl_hot_topics）
        
        Args:
            *args: 位置参数（未使用）
            **kwargs: 关键字参数（未使用）
        
        Returns:
            List[Dict]: 热点数据列表
        """
        return await self.crawl_hot_topics()
    
    async def crawl_hot_topics(self) -> List[Dict[str, Any]]:
        """
        爬取市场热点榜单
        
        Returns:
            List[Dict]: 热点数据列表，每个热点包含：
                - topic: 热点名称
                - rank: 排名（从1开始）
                - heat_index: 热度指数（0-100）
                - change: 变化趋势（如"+5%"）
                - leading_stocks: 关联的龙头股票代码列表
                - crawl_time: 爬取时间（YYYY-MM-DD HH:MM:SS）
        
        Raises:
            CrawlerError: 爬取失败时抛出异常
        """
        try:
            # 初始化浏览器和页面
            await self._init_browser()
            await self._get_page()  # 确保创建page对象
            
            if not self.page:
                raise CrawlerError("页面对象未初始化")
            
            logger.info(f"开始爬取同花顺热点榜单: {self.URL}")
            
            # 访问页面
            await self.page.goto(self.URL, wait_until="networkidle", timeout=30000)
            await self.page.wait_for_timeout(2000)  # 等待页面加载
            
            # 关闭弹窗（重要：不关闭弹窗无法正常定位元素）
            await PopupHandler.close_popups(self.page)
            await self.page.wait_for_timeout(1000)
            
            # 等待热点列表加载
            # 同花顺热点榜单可能使用动态加载，需要等待数据渲染
            try:
                # 尝试等待热点列表容器
                await self.page.wait_for_selector(
                    ".topic-list, .hot-list, [class*='topic'], [class*='hot']",
                    timeout=10000
                )
            except Exception as e:
                logger.warning(f"等待热点列表超时: {e}，尝试直接提取数据")
            
            # 再次关闭可能出现的弹窗
            await PopupHandler.close_popups(self.page)
            await self.page.wait_for_timeout(1000)
            
            # 使用JavaScript提取热点数据
            # 同花顺热点榜单的具体DOM结构需要通过实际页面分析确定
            # 这里提供一个通用的提取逻辑
            topics = await self.page.evaluate("""
                () => {
                    const topics = [];
                    
                    // 方法1: 尝试查找常见的热点列表容器
                    const containers = [
                        '.topic-list',
                        '.hot-list',
                        '[class*="topic"]',
                        '[class*="hot"]',
                        '[class*="rank"]',
                        'table tbody tr',
                        '.list-item',
                        '[data-type="topic"]'
                    ];
                    
                    let items = [];
                    for (const selector of containers) {
                        try {
                            const found = document.querySelectorAll(selector);
                            if (found.length > 0) {
                                items = Array.from(found);
                                break;
                            }
                        } catch (e) {
                            // 忽略选择器错误
                        }
                    }
                    
                    // 如果没找到，尝试查找所有可能包含热点信息的元素
                    if (items.length === 0) {
                        // 查找包含"热点"、"概念"、"题材"等关键词的元素
                        const allDivs = document.querySelectorAll('div, li, tr');
                        items = Array.from(allDivs).filter(el => {
                            const text = el.innerText || '';
                            return text.includes('热点') || 
                                   text.includes('概念') || 
                                   text.includes('题材') ||
                                   (el.querySelector && el.querySelector('[class*="topic"], [class*="hot"]'));
                        });
                    }
                    
                    // 提取热点数据
                    items.forEach((item, index) => {
                        try {
                            const text = item.innerText || '';
                            
                            // 跳过空元素或明显不是热点的元素
                            if (!text || text.length < 2) return;
                            
                            // 尝试提取热点名称（通常在第一个文本节点或特定元素中）
                            const topicName = item.querySelector('.topic-name, .name, [class*="name"]')?.innerText?.trim() ||
                                            item.querySelector('td:first-child, .first')?.innerText?.trim() ||
                                            text.split('\\n')[0]?.trim() ||
                                            text.split('\\t')[0]?.trim();
                            
                            if (!topicName || topicName.length < 1) return;
                            
                            // 尝试提取热度指数（数字，通常在0-100之间）
                            const heatText = item.querySelector('.heat-value, .heat, [class*="heat"], [class*="index"]')?.innerText ||
                                           text.match(/\\d+(\\.\\d+)?/)?.[0];
                            const heatIndex = heatText ? parseFloat(heatText) : null;
                            
                            // 尝试提取变化趋势（包含+或-的文本）
                            const changeText = item.querySelector('.change, [class*="change"], [class*="trend"]')?.innerText ||
                                             text.match(/[+-]\\d+(\\.\\d+)?%?/)?.[0] ||
                                             null;
                            
                            // 尝试提取关联股票代码（6位数字）
                            const stockCodes = [];
                            const stockElements = item.querySelectorAll('.stock-code, [class*="stock"], [class*="code"]');
                            stockElements.forEach(el => {
                                const code = el.innerText?.trim();
                                if (code && /^\\d{6}$/.test(code)) {
                                    stockCodes.push(code);
                                }
                            });
                            
                            // 如果没有找到股票代码元素，尝试从文本中提取
                            if (stockCodes.length === 0) {
                                const codeMatches = text.match(/\\b\\d{6}\\b/g);
                                if (codeMatches) {
                                    stockCodes.push(...codeMatches);
                                }
                            }
                            
                            topics.push({
                                rank: index + 1,
                                topic: topicName,
                                heat_index: heatIndex,
                                change: changeText,
                                leading_stocks: stockCodes
                            });
                        } catch (e) {
                            // 忽略单个元素的提取错误
                            console.error('Error extracting topic item:', e);
                        }
                    });
                    
                    return topics;
                }
            """)
            
            # 数据清洗和验证
            cleaned_topics = []
            for topic in topics:
                # 验证热点名称
                if not topic.get("topic") or len(topic["topic"]) < 1:
                    continue
                
                # 标准化数据格式
                cleaned_topic = {
                    "topic": topic.get("topic", "").strip(),
                    "rank": topic.get("rank", 0),
                    "heat_index": topic.get("heat_index") if topic.get("heat_index") is not None else None,
                    "change": topic.get("change", "").strip() if topic.get("change") else None,
                    "leading_stocks": topic.get("leading_stocks", []),
                    "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # 验证股票代码格式（6位数字）
                cleaned_topic["leading_stocks"] = [
                    code for code in cleaned_topic["leading_stocks"]
                    if re.match(r"^\d{6}$", str(code))
                ]
                
                cleaned_topics.append(cleaned_topic)
            
            logger.info(f"成功爬取{len(cleaned_topics)}个热点")
            
            if len(cleaned_topics) == 0:
                logger.warning("未提取到热点数据，可能需要调整选择器")
                # 尝试获取页面HTML用于调试
                try:
                    page_content = await self.page.content()
                    logger.debug(f"页面内容长度: {len(page_content)}")
                    # 可以保存页面内容用于调试
                    # with open("tonghuashun_debug.html", "w", encoding="utf-8") as f:
                    #     f.write(page_content)
                except Exception as e:
                    logger.debug(f"无法获取页面内容: {e}")
            
            return cleaned_topics
            
        except Exception as e:
            logger.error(f"爬取同花顺热点榜单失败: {e}", exc_info=True)
            raise CrawlerError(f"爬取同花顺热点榜单失败: {str(e)}")
        
        finally:
            # 清理资源
            await self.cleanup()
    
    async def get_hot_topics_with_stocks(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取与指定股票相关的热点
        
        Args:
            stock_code: 股票代码（6位数字）
        
        Returns:
            List[Dict]: 与股票相关的热点列表
        """
        # 先爬取所有热点
        all_topics = await self.crawl_hot_topics()
        
        # 筛选与指定股票相关的热点
        related_topics = [
            topic for topic in all_topics
            if stock_code in topic.get("leading_stocks", [])
        ]
        
        return related_topics

