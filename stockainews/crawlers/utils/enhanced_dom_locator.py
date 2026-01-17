"""
增强的 DOM 定位工具，参考 browser-use 的方法

使用 CDP 获取详细的 DOM 信息，增强元素定位能力
"""
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from playwright.sync_api import Page, CDPSession

logger = logging.getLogger(__name__)


@dataclass
class EnhancedElementInfo:
    """增强的元素信息"""
    backend_node_id: int
    node_id: Optional[int] = None
    tag_name: Optional[str] = None
    text_content: Optional[str] = None
    is_visible: bool = False
    is_clickable: bool = False
    bounds: Optional[Dict[str, float]] = None
    attributes: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    xpath: Optional[str] = None
    css_selector: Optional[str] = None


class EnhancedDOMLocator:
    """增强的 DOM 定位器，使用 CDP 获取详细的 DOM 信息"""
    
    def __init__(self, page: Page):
        self.page = page
        self.cdp_session: Optional[CDPSession] = None
        self._init_cdp_session()
    
    def _init_cdp_session(self):
        """初始化 CDP 会话"""
        try:
            self.cdp_session = self.page.context.new_cdp_session(self.page)
            logger.debug("CDP session initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize CDP session: {e}")
            self.cdp_session = None
    
    def _get_dom_snapshot(self) -> Optional[Dict]:
        """获取 DOM 快照"""
        if not self.cdp_session:
            return None
        
        try:
            snapshot = self.cdp_session.send('DOMSnapshot.captureSnapshot', {
                'computedStyles': [
                    'display',
                    'visibility',
                    'opacity',
                    'overflow',
                    'overflow-x',
                    'overflow-y',
                    'cursor',
                    'pointer-events',
                    'position',
                ],
                'includePaintOrder': True,
                'includeDOMRects': True,
            })
            return snapshot
        except Exception as e:
            logger.warning(f"Failed to get DOM snapshot: {e}")
            return None
    
    def _get_dom_tree(self) -> Optional[Dict]:
        """获取 DOM 树"""
        if not self.cdp_session:
            return None
        
        try:
            dom_tree = self.cdp_session.send('DOM.getDocument', {
                'depth': -1,
                'pierce': True,
            })
            return dom_tree
        except Exception as e:
            logger.warning(f"Failed to get DOM tree: {e}")
            return None
    
    def _get_ax_tree(self) -> Optional[Dict]:
        """获取可访问性树"""
        if not self.cdp_session:
            return None
        
        try:
            ax_tree = self.cdp_session.send('Accessibility.getFullAXTree', {})
            return ax_tree
        except Exception as e:
            logger.warning(f"Failed to get AX tree: {e}")
            return None
    
    def _build_snapshot_lookup(self, snapshot: Dict) -> Dict[int, Dict]:
        """构建快照查找表：backend_node_id -> snapshot_data"""
        lookup = {}
        
        if not snapshot or 'documents' not in snapshot:
            return lookup
        
        strings = snapshot.get('strings', [])
        
        for document in snapshot.get('documents', []):
            nodes = document.get('nodes', {})
            layout = document.get('layout', {})
            
            # 构建 backend_node_id 到快照索引的映射
            backend_node_to_index = {}
            if 'backendNodeId' in nodes:
                for i, backend_node_id in enumerate(nodes['backendNodeId']):
                    backend_node_to_index[backend_node_id] = i
            
            # 构建布局索引映射
            layout_index_map = {}
            if layout and 'nodeIndex' in layout:
                for layout_idx, node_index in enumerate(layout['nodeIndex']):
                    if node_index not in layout_index_map:
                        layout_index_map[node_index] = layout_idx
            
            # 为每个 backend_node_id 构建快照数据
            for backend_node_id, snapshot_index in backend_node_to_index.items():
                snapshot_data = {
                    'backend_node_id': backend_node_id,
                    'snapshot_index': snapshot_index,
                    'is_clickable': None,
                    'bounds': None,
                    'computed_styles': {},
                }
                
                # 解析可点击性
                if 'isClickable' in nodes and snapshot_index in nodes['isClickable'].get('index', []):
                    snapshot_data['is_clickable'] = True
                
                # 解析边界框
                if snapshot_index in layout_index_map:
                    layout_idx = layout_index_map[snapshot_index]
                    if layout_idx < len(layout.get('bounds', [])):
                        bounds = layout['bounds'][layout_idx]
                        if len(bounds) >= 4:
                            snapshot_data['bounds'] = {
                                'x': bounds[0],
                                'y': bounds[1],
                                'width': bounds[2],
                                'height': bounds[3],
                            }
                    
                    # 解析计算样式
                    if layout_idx < len(layout.get('styles', [])):
                        style_indices = layout['styles'][layout_idx]
                        computed_styles = {}
                        required_styles = [
                            'display', 'visibility', 'opacity', 'overflow',
                            'overflow-x', 'overflow-y', 'cursor', 'pointer-events', 'position'
                        ]
                        for i, style_index in enumerate(style_indices):
                            if i < len(required_styles) and 0 <= style_index < len(strings):
                                computed_styles[required_styles[i]] = strings[style_index]
                        snapshot_data['computed_styles'] = computed_styles
                
                lookup[backend_node_id] = snapshot_data
        
        return lookup
    
    def _is_element_visible(self, computed_styles: Dict[str, str]) -> bool:
        """检查元素是否可见"""
        display = computed_styles.get('display', '').lower()
        visibility = computed_styles.get('visibility', '').lower()
        opacity = computed_styles.get('opacity', '1')
        
        if display == 'none' or visibility == 'hidden':
            return False
        
        try:
            if float(opacity) <= 0:
                return False
        except (ValueError, TypeError):
            pass
        
        return True
    
    def _is_element_clickable(
        self,
        tag_name: str,
        attributes: Dict[str, str],
        computed_styles: Dict[str, str],
        snapshot_data: Optional[Dict] = None
    ) -> bool:
        """检查元素是否可点击（参考 browser-use 的检测逻辑）"""
        # 交互式标签
        interactive_tags = {
            'button', 'a', 'input', 'select', 'textarea',
            'details', 'summary', 'label'
        }
        
        if tag_name and tag_name.lower() in interactive_tags:
            return True
        
        # 检查 role 属性
        role = attributes.get('role', '').lower()
        interactive_roles = {
            'button', 'link', 'checkbox', 'radio', 'tab',
            'menuitem', 'option', 'combobox', 'textbox'
        }
        
        if role in interactive_roles:
            return True
        
        # 检查光标样式
        cursor = computed_styles.get('cursor', '').lower()
        if cursor in ['pointer', 'grab', 'grabbing']:
            return True
        
        # 检查 pointer-events
        pointer_events = computed_styles.get('pointer-events', '').lower()
        if pointer_events == 'none':
            return False
        
        # 检查快照数据中的可点击性
        if snapshot_data and snapshot_data.get('is_clickable'):
            return True
        
        # 检查是否有 onclick 或事件处理器
        if 'onclick' in attributes or 'data-onclick' in attributes:
            return True
        
        return False
    
    def _get_node_text(self, node: Dict, dom_tree: Dict) -> str:
        """获取节点的文本内容"""
        text_parts = []
        
        def collect_text(n):
            if n.get('nodeType') == 3:  # TEXT_NODE
                text = n.get('nodeValue', '').strip()
                if text:
                    text_parts.append(text)
            elif n.get('nodeType') == 1:  # ELEMENT_NODE
                children = n.get('children', [])
                for child in children:
                    collect_text(child)
        
        collect_text(node)
        return ' '.join(text_parts).strip()
    
    def _generate_xpath(self, node: Dict, dom_tree: Dict) -> str:
        """生成元素的 XPath 路径（相对路径，参考 browser-use）"""
        xpath_parts = []
        current_node = node
        
        while current_node:
            node_type = current_node.get('nodeType')
            # 只处理元素节点
            if node_type != 1:  # ELEMENT_NODE
                break
                
            node_name = current_node.get('nodeName', '').lower()
            
            # 停止条件：到达 iframe
            parent_id = current_node.get('parentId')
            if parent_id:
                parent_node = self._find_node_by_id(dom_tree, parent_id)
                if parent_node and parent_node.get('nodeName', '').lower() == 'iframe':
                    break
            
            # 计算同类型兄弟节点的位置
            position = self._get_element_position(current_node, dom_tree)
            
            # 只在有多个相同标签的兄弟时才添加索引
            if position > 0:
                xpath_parts.insert(0, f"{node_name}[{position}]")
            else:
                xpath_parts.insert(0, node_name)
            
            # 移动到父节点
            if parent_id:
                current_node = self._find_node_by_id(dom_tree, parent_id)
            else:
                break
        
        # 返回相对路径（不带开头的 /）
        return '/'.join(xpath_parts)
    
    def _get_element_position(self, element: Dict, dom_tree: Dict) -> int:
        """获取元素在同类型兄弟节点中的位置（1-based，如果是唯一的则返回0表示不需要索引）"""
        parent_id = element.get('parentId')
        if not parent_id:
            return 0
        
        parent_node = self._find_node_by_id(dom_tree, parent_id)
        if not parent_node or 'children' not in parent_node:
            return 0
        
        node_name = element.get('nodeName', '').lower()
        same_tag_siblings = [
            child for child in parent_node.get('children', [])
            if child.get('nodeType') == 1 and  # ELEMENT_NODE
               child.get('nodeName', '').lower() == node_name
        ]
        
        # 如果是唯一的同类型节点，不需要索引
        if len(same_tag_siblings) <= 1:
            return 0
        
        try:
            # XPath 是 1-indexed
            position = same_tag_siblings.index(element) + 1
            return position
        except ValueError:
            return 0
    
    def _find_node_by_id(self, dom_tree: Dict, node_id: int) -> Optional[Dict]:
        """通过 nodeId 查找节点"""
        def search_node(node):
            if node.get('nodeId') == node_id:
                return node
            for child in node.get('children', []):
                result = search_node(child)
                if result:
                    return result
            return None
        
        return search_node(dom_tree.get('root', {}))
    
    def find_elements_by_text(
        self,
        search_text: str,
        exact_match: bool = False,
        case_sensitive: bool = False
    ) -> List[EnhancedElementInfo]:
        """通过文本查找元素"""
        if not self.cdp_session:
            logger.warning("CDP session not available, falling back to Playwright")
            return []
        
        try:
            # 获取 DOM 树和快照
            dom_tree = self._get_dom_tree()
            snapshot = self._get_dom_snapshot()
            
            if not dom_tree:
                return []
            
            snapshot_lookup = {}
            if snapshot:
                snapshot_lookup = self._build_snapshot_lookup(snapshot)
            
            # 遍历 DOM 树查找匹配的元素
            matching_elements = []
            
            def traverse_node(node: Dict, parent_xpath: str = ''):
                node_type = node.get('nodeType')
                node_name = node.get('nodeName', '').lower()
                
                # 跳过脚本和样式节点
                if node_name in ['script', 'style', 'head', 'meta', 'link']:
                    return
                
                # 构建 XPath（使用改进的生成方法）
                if node_type == 1:  # ELEMENT_NODE
                    try:
                        xpath = self._generate_xpath(node, dom_tree)
                    except Exception as e:
                        logger.debug(f"Failed to generate XPath for node: {e}")
                        # 如果生成失败，使用简化版本
                        if parent_xpath:
                            xpath = f"{parent_xpath}/{node_name}"
                        else:
                            xpath = f"/{node_name}"
                else:
                    xpath = parent_xpath
                
                # 获取节点文本
                node_text = self._get_node_text(node, dom_tree)
                
                # 检查是否匹配
                if node_text:
                    search_lower = search_text.lower() if not case_sensitive else search_text
                    text_lower = node_text.lower() if not case_sensitive else node_text
                    
                    if exact_match:
                        matches = search_lower == text_lower
                    else:
                        matches = search_lower in text_lower
                    
                    if matches:
                        # 获取 backend_node_id
                        backend_node_id = node.get('backendNodeId')
                        if backend_node_id:
                            snapshot_data = snapshot_lookup.get(backend_node_id, {})
                            
                            # 获取属性
                            attributes = {}
                            attrs = node.get('attributes', [])
                            for i in range(0, len(attrs), 2):
                                if i + 1 < len(attrs):
                                    attributes[attrs[i]] = attrs[i + 1]
                            
                            # 构建元素信息
                            computed_styles = snapshot_data.get('computed_styles', {})
                            is_visible = self._is_element_visible(computed_styles)
                            is_clickable = self._is_element_clickable(
                                node_name, attributes, computed_styles, snapshot_data
                            )
                            
                            # 生成 XPath
                            xpath = self._generate_xpath(node, dom_tree)
                            
                            element_info = EnhancedElementInfo(
                                backend_node_id=backend_node_id,
                                node_id=node.get('nodeId'),
                                tag_name=node_name if node_type == 1 else None,
                                text_content=node_text,
                                is_visible=is_visible,
                                is_clickable=is_clickable,
                                bounds=snapshot_data.get('bounds'),
                                attributes=attributes,
                                xpath=xpath,
                            )
                            matching_elements.append(element_info)
                
                # 递归处理子节点
                children = node.get('children', [])
                for child in children:
                    traverse_node(child, xpath)
            
            # 从根节点开始遍历
            root = dom_tree.get('root', {})
            traverse_node(root)
            
            logger.info(f"Found {len(matching_elements)} elements matching text '{search_text}'")
            return matching_elements
            
        except Exception as e:
            logger.error(f"Error finding elements by text: {e}", exc_info=True)
            return []
    
    def get_element_by_xpath(self, xpath: str) -> Optional[Any]:
        """通过 XPath 获取 Playwright 元素（使用相对路径 XPath）"""
        try:
            # browser-use 生成的是相对路径 XPath（如 "html/body/div[1]"）
            # Playwright 需要绝对路径（以 // 开头）或完整路径（以 / 开头）
            
            # 方法1: 转换为绝对路径并使用 Playwright locator
            try:
                # 如果是相对路径，转换为绝对路径
                if not xpath.startswith('/'):
                    xpath = '//' + xpath
                
                # 使用 Playwright 的 XPath 定位器
                element = self.page.locator(f"xpath={xpath}").first
                if element:
                    return element.element_handle()
            except Exception as e:
                logger.debug(f"XPath locator failed: {e}")
            
            # 方法2: 使用 JavaScript evaluate（回退方案）
            try:
                # 移除开头的 // 或 /，使用 document.evaluate
                clean_xpath = xpath.lstrip('/').replace("'", "\\'")
                js_code = f"""
                    () => {{
                        const xpath = '//{clean_xpath}';
                        const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                        return result.singleNodeValue;
                    }}
                """
                element_handle = self.page.evaluate_handle(js_code)
                if element_handle:
                    element = element_handle.as_element()
                    if element:
                        return element
            except Exception as e:
                logger.debug(f"XPath evaluation failed: {e}")
            
            return None
        except Exception as e:
            logger.debug(f"Failed to get element by XPath {xpath}: {e}")
            return None
    
    def get_element_by_backend_node_id(self, backend_node_id: int) -> Optional[Any]:
        """通过 backend_node_id 获取 Playwright 元素（使用 CDP）"""
        if not self.cdp_session:
            return None
        
        try:
            # 使用 CDP 解析节点获取 object ID
            resolve_result = self.cdp_session.send('DOM.resolveNode', {
                'backendNodeId': backend_node_id
            })
            
            if 'object' not in resolve_result or 'objectId' not in resolve_result['object']:
                return None
            
            object_id = resolve_result['object']['objectId']
            
            # 使用 object ID 通过 evaluate_handle 获取元素
            element_handle = self.page.evaluate_handle(f"""
                () => {{
                    // 这是一个占位符，实际上我们需要通过 CDP 来操作
                    return null;
                }}
            """)
            
            # 直接返回 None，因为 Playwright 和 CDP 之间的桥接比较复杂
            # 对于点击操作，我们直接使用 CDP
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get element by backend_node_id {backend_node_id}: {e}")
            return None
    
    def find_more_link(
        self, 
        search_text: str = "更多",
        context_text: str = "投资者互动"
    ) -> Optional[EnhancedElementInfo]:
        """
        查找"更多"链接，优先选择与上下文相关的链接
        
        Args:
            search_text: 要查找的文本（默认"更多"）
            context_text: 上下文文本，用于判断相关性（默认"投资者互动"）
        """
        try:
            # 首先查找包含上下文文本的元素（如"投资者互动"区域）
            context_elements = self.find_elements_by_text(context_text, exact_match=False)
            
            # 查找所有包含"更多"的可点击元素
            more_elements = self.find_elements_by_text(search_text, exact_match=False)
            
            # 优先选择在上下文元素附近的"更多"链接
            best_more_link = None
            min_distance = float('inf')
            
            for more_elem in more_elements:
                if not more_elem.is_clickable:
                    continue
                
                # 检查是否是链接
                if more_elem.tag_name != 'a' and more_elem.attributes.get('role') != 'link':
                    continue
                
                # 检查文本是否包含"更多"
                more_text = more_elem.text_content or ""
                if search_text not in more_text:
                    continue
                
                # 优先选择文本较短的元素（"更多>>"这样的链接文本通常很短）
                # 如果文本过长，可能是包含"更多"的容器元素，不是链接本身
                if len(more_text) > 50:
                    continue
                
                # 计算与上下文元素的距离（通过位置）
                more_bounds = more_elem.bounds
                if not more_bounds:
                    continue
                
                for ctx_elem in context_elements:
                    ctx_bounds = ctx_elem.bounds
                    if not ctx_bounds:
                        continue
                    
                    # 计算距离（使用中心点距离）
                    more_center_x = more_bounds.get('x', 0) + more_bounds.get('width', 0) / 2
                    more_center_y = more_bounds.get('y', 0) + more_bounds.get('height', 0) / 2
                    ctx_center_x = ctx_bounds.get('x', 0) + ctx_bounds.get('width', 0) / 2
                    ctx_center_y = ctx_bounds.get('y', 0) + ctx_bounds.get('height', 0) / 2
                    
                    distance = ((more_center_x - ctx_center_x) ** 2 + 
                              (more_center_y - ctx_center_y) ** 2) ** 0.5
                    
                    # 如果"更多"链接在上下文元素的边界框内，优先选择
                    if (more_bounds.get('x', 0) >= ctx_bounds.get('x', 0) and
                        more_bounds.get('x', 0) + more_bounds.get('width', 0) <= 
                        ctx_bounds.get('x', 0) + ctx_bounds.get('width', 0) and
                        more_bounds.get('y', 0) >= ctx_bounds.get('y', 0) and
                        more_bounds.get('y', 0) + more_bounds.get('height', 0) <= 
                        ctx_bounds.get('y', 0) + ctx_bounds.get('height', 0)):
                        # 在上下文元素内部，距离为0（最高优先级）
                        distance = 0
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_more_link = more_elem
            
            if best_more_link:
                logger.info(f"找到'更多'链接（与'{context_text}'相关）: {best_more_link.text_content[:50]}")
                logger.info(f"  距离上下文元素: {min_distance:.1f}px")
                return best_more_link
            
            # 如果没有找到相关的，返回第一个找到的"更多"链接
            for more_elem in more_elements:
                if more_elem.is_clickable and more_elem.tag_name == 'a':
                    logger.info(f"找到'更多'链接（通用）: {more_elem.text_content[:50]}")
                    return more_elem
            
            return None
        except Exception as e:
            logger.error(f"Error finding 'more' link: {e}", exc_info=True)
            return None
            return None
    
    def click_more_link(self, more_element: EnhancedElementInfo) -> bool:
        """点击"更多"链接（使用 backend_node_id 而不是 XPath）"""
        try:
            if not more_element.backend_node_id:
                logger.warning("'更多'链接没有 backend_node_id，无法点击")
                return False
            
            # 使用 CDP 通过 backend_node_id 定位元素并点击
            if not self.cdp_session:
                logger.warning("CDP session 不可用")
                return False
            
            try:
                # 方法1: 使用 CDP 的 DOM.scrollIntoViewIfNeeded + JavaScript 点击
                # 先滚动到视图中
                try:
                    self.cdp_session.send('DOM.scrollIntoViewIfNeeded', {
                        'backendNodeId': more_element.backend_node_id
                    })
                    time.sleep(0.1)
                except:
                    pass
                
                # 解析节点以获取 object ID
                resolve_result = self.cdp_session.send('DOM.resolveNode', {
                    'backendNodeId': more_element.backend_node_id
                })
                
                if 'object' not in resolve_result or 'objectId' not in resolve_result['object']:
                    logger.warning("无法获取元素的 object ID")
                    return False
                
                object_id = resolve_result['object']['objectId']
                
                # 使用 JavaScript 点击
                self.cdp_session.send('Runtime.callFunctionOn', {
                    'functionDeclaration': 'function() { this.click(); }',
                    'objectId': object_id
                })
                
                logger.info("成功点击'更多'链接")
                
                # 等待可能的页面跳转或动态内容加载
                try:
                    # 尝试等待导航（如果是跳转）
                    self.page.wait_for_load_state("domcontentloaded", timeout=3000)
                except:
                    # 如果不是跳转，可能是动态内容加载
                    pass
                
                # 额外等待动态内容加载（AJAX请求等）
                time.sleep(3)  # 增加等待时间，等待可能的AJAX请求
                
                return True
                
            except Exception as e:
                logger.error(f"CDP 点击失败: {e}")
                return False
                
        except Exception as e:
            logger.error(f"点击'更多'链接失败: {e}", exc_info=True)
            return False
    
    def find_interactive_elements(self) -> List[EnhancedElementInfo]:
        """查找所有可交互元素"""
        if not self.cdp_session:
            return []
        
        try:
            dom_tree = self._get_dom_tree()
            snapshot = self._get_dom_snapshot()
            
            if not dom_tree:
                return []
            
            snapshot_lookup = {}
            if snapshot:
                snapshot_lookup = self._build_snapshot_lookup(snapshot)
            
            interactive_elements = []
            
            def traverse_node(node: Dict):
                node_type = node.get('nodeType')
                if node_type != 1:  # 只处理元素节点
                    return
                
                backend_node_id = node.get('backendNodeId')
                if not backend_node_id:
                    return
                
                snapshot_data = snapshot_lookup.get(backend_node_id, {})
                node_name = node.get('nodeName', '').lower()
                
                # 获取属性
                attributes = {}
                attrs = node.get('attributes', [])
                for i in range(0, len(attrs), 2):
                    if i + 1 < len(attrs):
                        attributes[attrs[i]] = attrs[i + 1]
                
                computed_styles = snapshot_data.get('computed_styles', {})
                is_visible = self._is_element_visible(computed_styles)
                is_clickable = self._is_element_clickable(
                    node_name, attributes, computed_styles, snapshot_data
                )
                
                if is_clickable and is_visible:
                    node_text = self._get_node_text(node, dom_tree)
                    
                    element_info = EnhancedElementInfo(
                        backend_node_id=backend_node_id,
                        node_id=node.get('nodeId'),
                        tag_name=node_name,
                        text_content=node_text,
                        is_visible=is_visible,
                        is_clickable=is_clickable,
                        bounds=snapshot_data.get('bounds'),
                        attributes=attributes,
                    )
                    interactive_elements.append(element_info)
                
                # 递归处理子节点
                children = node.get('children', [])
                for child in children:
                    traverse_node(child)
            
            root = dom_tree.get('root', {})
            traverse_node(root)
            
            logger.info(f"Found {len(interactive_elements)} interactive elements")
            return interactive_elements
            
        except Exception as e:
            logger.error(f"Error finding interactive elements: {e}", exc_info=True)
            return []
    
    def close(self):
        """关闭 CDP 会话"""
        if self.cdp_session:
            try:
                self.cdp_session.detach()
            except:
                pass
            self.cdp_session = None

