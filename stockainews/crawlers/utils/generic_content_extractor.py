"""
通用内容提取器

提供基于配置的通用内容提取能力，不依赖硬编码的选择器或文本。
支持使用 EnhancedDOMLocator 进行智能定位。
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from playwright.sync_api import Page, ElementHandle

from stockainews.crawlers.utils.enhanced_dom_locator import EnhancedDOMLocator, EnhancedElementInfo
from stockainews.core.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ExtractionRule:
    """内容提取规则"""
    
    # 定位规则
    search_texts: List[str] = field(default_factory=list)  # 用于定位区域的搜索文本列表
    context_text: Optional[str] = None  # 上下文文本，用于更精确的定位
    
    # 内容解析规则
    date_patterns: List[str] = field(default_factory=lambda: [
        r'(\d{4}-\d{2}-\d{2})',  # 标准格式：2024-01-01
        r'(\d{4}/\d{2}/\d{2})',  # 斜杠格式：2024/01/01
        r'(\d{4}\.\d{2}\.\d{2})',  # 点格式：2024.01.01
    ])
    date_format: str = "%Y-%m-%d"  # 日期解析格式
    
    # 内容分割规则
    question_markers: List[str] = field(default_factory=lambda: ["问：", "Q:", "提问："])
    answer_markers: List[str] = field(default_factory=lambda: ["答：", "A:", "回答：", "尊敬的投资者"])
    
    # 块分割规则（用于无明确问答标记的情况）
    block_start_markers: List[str] = field(default_factory=list)  # 块开始标记
    block_end_markers: List[str] = field(default_factory=list)  # 块结束标记
    
    # 过滤规则
    min_text_length: int = 10  # 最小文本长度
    min_line_length: int = 5  # 最小行长度（用于过滤短行）
    exclude_keywords: List[str] = field(default_factory=lambda: [
        "加载更多", "热度排行", "查看更多"
    ])  # 排除包含这些关键词的内容
    exclude_patterns: List[str] = field(default_factory=list)  # 排除匹配这些正则的内容
    
    # 导航规则（用于点击"更多"链接等）
    more_link_texts: List[str] = field(default_factory=lambda: ["更多", "查看更多", ">>"])
    
    # 自定义解析函数
    custom_parser: Optional[Callable] = None  # 自定义内容解析函数


@dataclass
class ExtractedItem:
    """提取的内容项"""
    title: str
    question: str = ""
    answer: str = ""
    published_at: Optional[datetime] = None
    url: Optional[str] = None
    full_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外的元数据


class GenericContentExtractor:
    """通用内容提取器"""
    
    def __init__(self, page: Page, rule: ExtractionRule, debug: bool = False):
        """
        初始化提取器
        
        Args:
            page: Playwright页面对象
            rule: 提取规则
            debug: 是否开启调试模式
        """
        self.page = page
        self.rule = rule
        self.debug = debug
        self.locator = EnhancedDOMLocator(page)
    
    # 通用内容提取器
    def extract(
        self,
        days: int = 30,
        navigate_to_detail: bool = True
    ) -> List[ExtractedItem]:
        """
        执行内容提取
        
        Args:
            days: 提取最近N天的内容
            navigate_to_detail: 是否导航到详情页（如果找到"更多"链接）
            
        Returns:
            提取的内容列表
        """
        try:
            # 1. 定位内容区域
            content_element = self._locate_content_area()
            if not content_element:
                logger.warning("无法定位内容区域")
                return []
            
            # 2. 如果需要，导航到详情页
            if navigate_to_detail:
                navigated = self._navigate_to_detail_if_needed()
                if navigated:
                    # 重新初始化 locator 和定位内容区域
                    self.locator.close()
                    self.locator = EnhancedDOMLocator(self.page)
                    content_element = self._locate_content_area()
                    if not content_element:
                        logger.warning("导航后无法定位内容区域")
                        return []
            
            # 3. 提取内容
            section_text = content_element.inner_text() if content_element else ""
            if self.debug:
                self._save_debug_text(section_text)
            
            # 4. 解析内容
            items = self._parse_content(section_text, days)
            
            logger.info(f"成功提取 {len(items)} 条内容")
            return items
            
        except Exception as e:
            logger.error(f"内容提取失败: {e}", exc_info=True)
            return []
        finally:
            try:
                self.locator.close()
            except:
                pass
    
    # 定位内容区域
    def _locate_content_area(self) -> Optional[ElementHandle]:
        """
        定位内容区域
        
        Returns:
            内容区域的元素，如果找不到返回None
        """
        # 尝试每个搜索文本
        for search_text in self.rule.search_texts:
            elements = self.locator.find_elements_by_text(search_text, exact_match=False)
            
            if not elements:
                continue
            
            # 过滤掉顶层元素和文本过长的元素
            filtered = [
                e for e in elements
                if (e.tag_name not in ['html', 'body'] and
                    e.text_content and
                    len(e.text_content) < 100000)
            ]
            
            if not filtered:
                continue
            
            # 选择最合适的元素（包含搜索文本且文本长度适中）
            def score_element(elem):
                text = elem.text_content or ''
                score = len(text)
                
                # 包含搜索文本加分
                if search_text in text:
                    score += 1000
                
                # 包含问答标记加分
                if any(marker in text for marker in self.rule.question_markers):
                    score += 500
                if any(marker in text for marker in self.rule.answer_markers):
                    score += 500
                
                # 包含日期模式加分
                if any(re.search(pattern, text) for pattern in self.rule.date_patterns):
                    score += 300
                
                # 文本过长减分
                if len(text) > 50000:
                    score -= 2000
                
                return score
            
            best_element = max(filtered, key=score_element)
            
            # 使用 XPath 定位元素
            if best_element.xpath:
                element = self.locator.get_element_by_xpath(best_element.xpath)
                if element:
                    logger.info(f"成功定位内容区域: {search_text} (XPath: {best_element.xpath})")
                    return element
        
        logger.warning("无法通过搜索文本定位内容区域，尝试回退方法")
        
        # 回退方法：使用 Playwright 文本搜索
        for search_text in self.rule.search_texts:
            try:
                title_element = self.page.get_by_text(search_text, exact=False).first
                if title_element:
                    # 向上查找包含内容的父容器
                    container_handle = title_element.evaluate_handle("""
                        el => {
                            let current = el.parentElement;
                            for (let i = 0; i < 5 && current; i++) {
                                const text = current.innerText || '';
                                // 查找包含足够内容的容器
                                if (text.length > 500) {
                                    return current;
                                }
                                current = current.parentElement;
                            }
                            return el.parentElement;
                        }
                    """)
                    element = container_handle.as_element() if container_handle else None
                    if element:
                        logger.info(f"通过回退方法定位内容区域: {search_text}")
                        return element
            except Exception as e:
                logger.debug(f"回退方法失败: {e}")
        
        return None
    
    # 如果需要，导航到详情页
    def _navigate_to_detail_if_needed(self) -> bool:
        """
        如果需要，导航到详情页
        
        Returns:
            是否成功导航
        """
        for link_text in self.rule.more_link_texts:
            more_link = self.locator.find_more_link(
                link_text,
                context_text=self.rule.context_text
            )
            
            if not more_link:
                continue
            
            logger.info(f"找到'{link_text}'链接，准备导航到详情页")
            
            # 获取链接的 href 属性
            href = more_link.attributes.get('href', '')
            if not href or href.startswith('#'):
                logger.debug(f"链接没有有效的 href: {href}")
                continue
            
            # 构建完整 URL
            before_url = self.page.url
            target_url = self._build_full_url(href, before_url)
            
            if not target_url:
                continue
            
            try:
                logger.info(f"导航到: {target_url}")
                self.page.goto(target_url, wait_until="domcontentloaded", timeout=10000)
                self.page.wait_for_timeout(2000)  # 等待动态内容加载
                
                after_url = self.page.url
                if after_url != before_url:
                    logger.info(f"成功导航到新页面: {after_url}")
                    return True
                else:
                    logger.warning("URL 没有改变，可能没有真正导航")
            except Exception as e:
                logger.warning(f"导航失败: {e}")
        
        return False
    
    # 构建完整 URL
    def _build_full_url(self, href: str, current_url: str) -> Optional[str]:
        """
        构建完整 URL
        
        Args:
            href: 链接的 href 属性
            current_url: 当前页面 URL
            
        Returns:
            完整 URL，失败返回 None
        """
        from urllib.parse import urljoin, urlparse
        
        try:
            if href.startswith('http'):
                return href
            elif href.startswith('//'):
                # 相对协议 URL
                parsed = urlparse(current_url)
                return f"{parsed.scheme}:{href}"
            elif href.startswith('/'):
                # 相对路径
                return urljoin(current_url, href)
            else:
                # 相对路径（无斜杠）
                return urljoin(current_url, href)
        except Exception as e:
            logger.debug(f"构建 URL 失败: {e}")
            return None
    
    # 解析内容文本
    def _parse_content(self, section_text: str, days: int) -> List[ExtractedItem]:
        """
        解析内容文本
        
        Args:
            section_text: 区域文本
            days: 时间范围（天）
            
        Returns:
            提取的内容列表
        """
        # 如果有自定义解析函数，使用它
        if self.rule.custom_parser:
            try:
                return self.rule.custom_parser(section_text, days, self.rule)
            except Exception as e:
                logger.warning(f"自定义解析函数失败: {e}，使用默认解析")
        
        # 使用默认解析逻辑
        cutoff_date = datetime.now() - timedelta(days=days)
        items = []
        
        # 1. 分割内容块
        blocks = self._split_into_blocks(section_text)
        logger.info(f"分割出 {len(blocks)} 个内容块")
        
        # 2. 解析每个块
        for block in blocks:
            try:
                item = self._parse_block(block, cutoff_date)
                if item:
                    items.append(item)
            except Exception as e:
                logger.debug(f"解析块失败: {e}")
                continue
        
        # 3. 按时间排序
        items.sort(key=lambda x: x.published_at if x.published_at else datetime.min, reverse=True)
        
        return items
    
    # 将文本分割成内容块
    def _split_into_blocks(self, text: str) -> List[str]:
        """
        将文本分割成内容块
        
        Args:
            text: 原始文本
            
        Returns:
            内容块列表
        """
        blocks = []
        lines = text.split('\n')
        current_block = []
        in_block = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否遇到块结束标记
            if any(marker in line for marker in self.rule.block_end_markers):
                if current_block and in_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                    in_block = False
                break  # 停止处理后续行
            
            # 检查是否是块开始标记
            if any(marker in line for marker in self.rule.block_start_markers):
                # 保存前一个块
                if current_block and in_block:
                    blocks.append('\n'.join(current_block))
                # 开始新块
                current_block = [line]
                in_block = True
            elif in_block:
                current_block.append(line)
        
        # 保存最后一个块
        if current_block and in_block:
            blocks.append('\n'.join(current_block))
        
        return blocks
    
    # 解析单个内容块
    def _parse_block(self, block_text: str, cutoff_date: datetime) -> Optional[ExtractedItem]:
        """
        解析单个内容块
        
        Args:
            block_text: 块文本
            cutoff_date: 截止日期
            
        Returns:
            提取的内容，失败返回 None
        """
        # 1. 检查最小长度
        if len(block_text) < self.rule.min_text_length:
            return None
        
        # 2. 检查排除关键词
        if any(keyword in block_text for keyword in self.rule.exclude_keywords):
            return None
        
        # 3. 检查排除模式
        for pattern in self.rule.exclude_patterns:
            if re.search(pattern, block_text):
                return None
        
        # 4. 提取日期
        published_at = self._extract_date(block_text)
        if not published_at:
            return None
        
        # 5. 检查日期范围
        if published_at < cutoff_date:
            return None
        
        # 6. 提取问题和答案
        question, answer = self._extract_qa(block_text)
        
        if not question and not answer:
            return None
        
        # 7. 构建标题
        title = question[:50] + "..." if len(question) > 50 else question
        if not title:
            title = block_text[:50] + "..."
        
        return ExtractedItem(
            title=title,
            question=question,
            answer=answer,
            published_at=published_at,
            url=self.page.url,
            full_text=block_text
        )
    
    # 从文本中提取日期
    def _extract_date(self, text: str) -> Optional[datetime]:
        """
        从文本中提取日期
        
        Args:
            text: 文本
            
        Returns:
            日期对象，失败返回 None
        """
        for pattern in self.rule.date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                # 尝试解析日期
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"]:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except:
                        continue
        return None
    
    # 从文本中提取问题和答案
    def _extract_qa(self, text: str) -> tuple[str, str]:
        """
        从文本中提取问题和答案
        
        Args:
            text: 文本
            
        Returns:
            (问题, 答案) 元组
        """
        question = ""
        answer = ""
        
        # 方法1: 使用问答标记分割
        for q_marker in self.rule.question_markers:
            if q_marker in text:
                for a_marker in self.rule.answer_markers:
                    if a_marker in text:
                        parts = text.split(a_marker, 1)
                        if len(parts) == 2:
                            question_part = parts[0]
                            answer = parts[1].strip()
                            if q_marker in question_part:
                                question = question_part.split(q_marker, 1)[1].strip()
                            else:
                                question = question_part.strip()
                            return question, answer
        
        # 方法2: 逐行解析（无明确标记）
        lines = text.split('\n')
        question_lines = []
        answer_lines = []
        in_question = False
        in_answer = False
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < self.rule.min_line_length:
                continue
            
            # 跳过块开始标记行
            if any(marker in line for marker in self.rule.block_start_markers):
                continue
            
            # 跳过日期行
            if any(re.search(pattern, line) for pattern in self.rule.date_patterns):
                in_question = True  # 日期后开始问题
                continue
            
            # 检测答案开始
            if any(marker in line for marker in self.rule.answer_markers):
                in_answer = True
                in_question = False
            
            # 收集内容（只收集足够长的行）
            if len(line) >= 10:
                if in_question:
                    question_lines.append(line)
                elif in_answer:
                    answer_lines.append(line)
        
        question = '\n'.join(question_lines).strip()
        answer = '\n'.join(answer_lines).strip()
        
        return question, answer
    
    # 保存调试文本
    def _save_debug_text(self, text: str):
        """保存调试文本"""
        try:
            debug_dir = Path("debug_outputs")
            debug_dir.mkdir(exist_ok=True)
            debug_file = debug_dir / f"extracted_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"调试文本已保存到: {debug_file}")
        except Exception as e:
            logger.debug(f"保存调试文本失败: {e}")

