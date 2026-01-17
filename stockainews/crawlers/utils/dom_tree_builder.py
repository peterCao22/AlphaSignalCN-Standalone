"""
DOM树构建工具

基于 browser-use 项目的 buildDomTree.js，提供DOM树构建和元素标注功能。
用于帮助爬虫更好地理解和定位页面元素。
"""
from typing import Optional, Dict, Any, List
from pathlib import Path
from playwright.sync_api import Page

from stockainews.core.logger import setup_logger

logger = setup_logger(__name__)


# 读取完整的 buildDomTree.js 文件
def _load_build_dom_tree_js() -> str:
    """加载 buildDomTree.js 文件内容"""
    js_file = Path(__file__).parent / "buildDomTree.js"
    if js_file.exists():
        return js_file.read_text(encoding='utf-8')
    else:
        logger.warning(f"buildDomTree.js not found at {js_file}, using fallback")
        return _get_fallback_js()


def _get_fallback_js() -> str:
    """降级方案：简化的 JS 代码"""
    return """
(args = {}) => {
  const { doHighlightElements = false, focusHighlightIndex = -1, viewportExpansion = -1, debugMode = false } = args;
  return { rootId: null, map: {} };
};
"""


# 加载 JS 代码
BUILD_DOM_TREE_JS = _load_build_dom_tree_js()


class DomTreeBuilder:
    """DOM树构建器，基于 browser-use 的 buildDomTree.js"""
    
    def __init__(self):
        """初始化DOM树构建器"""
        self.logger = setup_logger(__name__)
    
    def build_dom_tree(
        self,
        page: Page,
        highlight_elements: bool = False,
        focus_highlight_index: Optional[int] = None,
        viewport_expansion: int = -1,
        debug_mode: bool = False
    ) -> Dict[str, Any]:
        """
        构建页面的简化DOM树
        
        Args:
            page: Playwright页面对象
            highlight_elements: 是否高亮显示可交互元素
            focus_highlight_index: 只高亮特定索引的元素（可选，-1表示全部）
            viewport_expansion: 视口扩展范围（像素），-1表示禁用视口检查
            debug_mode: 是否启用调试模式（返回性能指标）
            
        Returns:
            DOM树结构：
            {
                "rootId": "0",
                "map": {
                    "0": {
                        "tagName": "body",
                        "children": ["1", "2"],
                        "isVisible": true,
                        "isInteractive": false,
                        "isInViewport": true,
                        "highlightIndex": 0,  # 如果可交互且启用了高亮
                        "xpath": "/body",
                        "attributes": {...}
                    },
                    ...
                },
                "perfMetrics": {...}  # 如果debug_mode=True
            }
        """
        try:
            args = {
                "doHighlightElements": highlight_elements,
                "focusHighlightIndex": focus_highlight_index if focus_highlight_index is not None else -1,
                "viewportExpansion": viewport_expansion,
                "debugMode": debug_mode
            }
            
            result = page.evaluate(BUILD_DOM_TREE_JS, args)
            
            if debug_mode and "perfMetrics" in result:
                self.logger.debug(f"DOM树构建性能指标: {result['perfMetrics']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"构建DOM树失败: {e}", exc_info=True)
            return {"rootId": None, "map": {}}
    
    def find_interactive_elements(
        self,
        page: Page,
        highlight: bool = True
    ) -> List[Dict[str, Any]]:
        """
        查找页面中所有可交互的元素
        
        Args:
            page: Playwright页面对象
            highlight: 是否高亮显示元素
            
        Returns:
            可交互元素列表，每个元素包含：
            {
                "id": "5",
                "tagName": "button",
                "attributes": {...},
                "highlightIndex": 0,
                "xpath": "/body/div[1]/button[1]",
                ...
            }
        """
        dom_tree = self.build_dom_tree(
            page,
            highlight_elements=highlight,
            viewport_expansion=-1  # 获取所有元素，不限制视口
        )
        
        interactive_elements = []
        for node_id, node_data in dom_tree.get("map", {}).items():
            if node_data.get("isInteractive", False):
                interactive_elements.append({
                    "id": node_id,
                    **node_data
                })
        
        # 按 highlightIndex 排序
        interactive_elements.sort(key=lambda x: x.get("highlightIndex", -1))
        
        self.logger.info(f"找到 {len(interactive_elements)} 个可交互元素")
        return interactive_elements
    
    def find_elements_by_text(
        self,
        page: Page,
        search_text: str,
        case_sensitive: bool = False,
        highlight: bool = False
    ) -> List[Dict[str, Any]]:
        """
        在DOM树中查找包含特定文本的元素
        
        Args:
            page: Playwright页面对象
            search_text: 要搜索的文本
            case_sensitive: 是否区分大小写
            highlight: 是否高亮显示匹配的元素
            
        Returns:
            匹配的元素列表
        """
        dom_tree = self.build_dom_tree(page, highlight_elements=highlight)
        
        matching_elements = []
        search_text_lower = search_text.lower() if not case_sensitive else search_text
        
        for node_id, node_data in dom_tree.get("map", {}).items():
            # 检查文本内容
            text = node_data.get("text", "")
            if text:
                text_to_check = text.lower() if not case_sensitive else text
                if search_text_lower in text_to_check:
                    matching_elements.append({
                        "id": node_id,
                        "text": text,
                        **node_data
                    })
            
            # 检查属性值
            attributes = node_data.get("attributes", {})
            for attr_name, attr_value in attributes.items():
                if attr_value:
                    value_to_check = str(attr_value).lower() if not case_sensitive else str(attr_value)
                    if search_text_lower in value_to_check:
                        matching_elements.append({
                            "id": node_id,
                            "matchedAttribute": attr_name,
                            "matchedValue": attr_value,
                            **node_data
                        })
                        break
        
        self.logger.info(f"找到 {len(matching_elements)} 个包含文本 '{search_text}' 的元素")
        return matching_elements
    
    def find_element_by_highlight_index(
        self,
        page: Page,
        highlight_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        根据高亮索引查找元素
        
        Args:
            page: Playwright页面对象
            highlight_index: 高亮索引
            
        Returns:
            元素信息，如果找不到返回None
        """
        # 使用 data-testid 属性定位元素
        try:
            element = page.query_selector(f'[data-testid="zp-{highlight_index}"]')
            if element:
                # 获取元素的详细信息
                dom_tree = self.build_dom_tree(page, highlight_elements=False)
                for node_id, node_data in dom_tree.get("map", {}).items():
                    if node_data.get("highlightIndex") == highlight_index:
                        return {
                            "id": node_id,
                            "element": element,
                            **node_data
                        }
        except Exception as e:
            self.logger.warning(f"根据高亮索引查找元素失败: {e}")
        
        return None
    
    def clear_highlights(self, page: Page):
        """清除页面上的所有高亮标记"""
        try:
            page.evaluate("""
                const container = document.getElementById('playwright-highlight-container');
                if (container) {
                    container.remove();
                }
            """)
            self.logger.debug("已清除所有高亮标记")
        except Exception as e:
            self.logger.warning(f"清除高亮标记失败: {e}")


def build_dom_tree_simple(
    page: Page,
    highlight_elements: bool = False
) -> Dict[str, Any]:
    """
    简化版DOM树构建函数
    
    Args:
        page: Playwright页面对象
        highlight_elements: 是否高亮显示可交互元素
        
    Returns:
        DOM树结构
    """
    builder = DomTreeBuilder()
    return builder.build_dom_tree(page, highlight_elements=highlight_elements)
