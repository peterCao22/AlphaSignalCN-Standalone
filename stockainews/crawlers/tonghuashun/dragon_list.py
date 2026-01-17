"""
同花顺龙虎榜爬虫

说明：
- 龙虎榜“汇总数据”已改用 BigQuant（见 data/raw/limit_up/dragon_list.csv）
- 本文件仅保留同花顺 `market/longhu` 页的“营业部席位明细”爬取，用于 A5 资金结构特征
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
import ssl
import urllib.request

from stockainews.crawlers.base_crawler import BaseCrawler
from stockainews.core.exceptions import CrawlerError
from stockainews.core.logger import setup_logger

logger = setup_logger(__name__)


class TonghuashunLonghuSeatsCrawler(BaseCrawler):
    """
    同花顺龙虎榜（market/longhu）席位明细爬虫

    目标：把“买入/卖出金额最大的前5名营业部”明细抓出来，并区分 1日 / 3日。

    页面特性：同一只股票同一天，列表里会同时出现“1日行（无标签）”和“3日行（左侧第一列显示 3日）”。
    我们逐行点击（点击代码列避免跳转链接），右侧明细区会更新为该行对应的口径，然后解析两张席位表。
    """

    BASE_URL = "https://data.10jqka.com.cn/market/longhu/"
    REPORT_URL_TPL = "https://data.10jqka.com.cn/ifmarket/lhbggxq/report/{trade_date}/"

    @staticmethod
    def _parse_amount_to_yuan(txt: str) -> float | None:
        """解析金额文本（亿/万/元） -> 元"""
        if not txt:
            return None
        t = re.sub(r"[,\s\xa0]+", "", str(txt))
        m = re.search(r"([+-]?\d+(?:\.\d+)?)(亿|万|元)?", t)
        if not m:
            return None
        v = float(m.group(1))
        u = m.group(2) or "元"
        if u == "亿":
            return v * 1e8
        if u == "万":
            return v * 1e4
        return v

    @staticmethod
    def _infer_window_days_from_reason(reason: str) -> int:
        """从原因文本推断口径（通常为 1日 或 3日；也兼容 2日）。"""
        r = str(reason or "")
        if "连续三个交易日" in r or "3个交易日" in r or "三个交易日" in r:
            return 3
        if "连续两个交易日" in r or "2个交易日" in r or "两个交易日" in r:
            return 2
        return 1

    async def _crawl_report_page(
        self,
        trade_date: str,
        category: str,
        limit_rows: int,
    ) -> List[Dict[str, Any]]:
        """
        抓取 ifmarket/lhbggxq/report/YYYY-MM-DD/ 的“整页明细”。

        该页面会预渲染大量 div.stockcont[stockcode]，每个包含买/卖前5席位表。
        用于“指定日期”补历史席位明细，避免日期控件/iframe。
        """
        # 延迟导入，避免在未安装依赖的环境里直接 import 失败
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except Exception as e:
            raise CrawlerError(f"缺少依赖 bs4，无法解析 report 页面: {e}")

        if not trade_date or not re.match(r"^\d{4}-\d{2}-\d{2}$", trade_date):
            raise CrawlerError(f"trade_date 格式不正确: {trade_date}（应为 YYYY-MM-DD）")

        url = self.REPORT_URL_TPL.format(trade_date=trade_date)
        logger.info(f"使用 report URL 抓取指定日期席位明细: {url}")

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
                body = r.read()
        except Exception as e:
            raise CrawlerError(f"请求 report 页面失败: {e}")

        # 页面通常为 gbk
        html = body.decode("gbk", "ignore")
        soup = BeautifulSoup(html, "html.parser")

        # 每个明细块：<div class="stockcont" stockcode="000620" ...>
        blocks = soup.select("div.stockcont[stockcode]")
        if not blocks:
            raise CrawlerError("report 页面未找到 div.stockcont[stockcode]，可能页面结构变化或被反爬")

        # 以“key 去重 + 质量优先”方式合并：同一 (trade_date, window_days, category, stock_code) 可能出现多条原因
        # 这里优先保留 total_buy_yuan 最大的那条（资金更强/信息更完整）
        best: dict[tuple[str, int, str, str], Dict[str, Any]] = {}

        def _extract_top5(tb) -> List[Dict[str, Any]]:
            rows = tb.select("tbody tr")
            out: List[Dict[str, Any]] = []
            for idx, tr in enumerate(rows[:5], start=1):
                tds = tr.find_all("td")
                if len(tds) < 4:
                    continue
                seat_td = tds[0]
                a = seat_td.find("a")
                seat_name = (a.get_text(strip=True) if a else seat_td.get_text(" ", strip=True)).strip()
                # 标签在 label.label 上（如 一线游资/知名游资/机构专用）
                lab = seat_td.find("label")
                seat_tag = lab.get_text(strip=True) if lab else None

                def _to_wan_float(td):
                    t = td.get_text(strip=True).replace(",", "").replace("\xa0", "")
                    try:
                        return float(t)
                    except Exception:
                        return 0.0

                buy_wan = _to_wan_float(tds[1])
                sell_wan = _to_wan_float(tds[2])
                net_wan = _to_wan_float(tds[3])

                out.append(
                    {
                        "rank": idx,
                        "seat_name": seat_name,
                        "seat_tag": seat_tag,
                        "buy_yuan": buy_wan * 1e4,
                        "sell_yuan": sell_wan * 1e4,
                        "net_yuan": net_wan * 1e4,
                    }
                )
            return out

        for b in blocks[: max(1, int(limit_rows))]:
            try:
                stockcode = (b.get("stockcode") or "").strip()
                if not stockcode:
                    continue

                # 过滤转债（方案A）：report 页里转债 stockcode 通常以 11/12 开头或名称含“转债”
                if stockcode.startswith(("11", "12")):
                    continue

                # 标题：如 “盈新发展(000620)明细：日涨幅偏离值达7%的证券”
                title_p = None
                for p in b.find_all("p"):
                    t = p.get_text(" ", strip=True).replace("\xa0", " ")
                    if "明细" in t and "(" in t and ")" in t:
                        title_p = p
                        break
                title_text = (
                    title_p.get_text(" ", strip=True).replace("\xa0", " ") if title_p else ""
                )
                m = re.match(r"^(?P<name>.+?)\((?P<code>\d{6})\)明细[:：](?P<reason>.+)$", title_text)
                if not m:
                    # 少数情况下标题不规范，跳过
                    continue

                stock_name = m.group("name").strip()
                reason = m.group("reason").strip()
                if "转债" in stock_name:
                    continue

                window_days = self._infer_window_days_from_reason(reason)

                # 汇总：包含 成交额/合计买入/合计卖出/净额
                total_turnover_yuan = None
                total_buy_yuan = None
                total_sell_yuan = None
                total_net_yuan = None
                for p in b.find_all("p"):
                    # 注意：同花顺会把“数值”和“单位(亿/万/元)”拆开（例如 10.98 + 亿元），
                    # 所以必须把单位一起抓出来，否则会被当成“元”，造成数量级错误。
                    t_raw = p.get_text(" ", strip=True).replace("\xa0", " ")
                    if "成交额" in t_raw and "合计买入" in t_raw and "合计卖出" in t_raw and "净额" in t_raw:
                        t = re.sub(r"\s+", "", t_raw)
                        m_turn = re.search(r"成交额[:：]([+-]?\d+(?:\.\d+)?)(亿|万|元)", t)
                        m_buy = re.search(r"合计买入[:：]([+-]?\d+(?:\.\d+)?)(亿|万|元)", t)
                        m_sell = re.search(r"合计卖出[:：]([+-]?\d+(?:\.\d+)?)(亿|万|元)", t)
                        m_net = re.search(r"净额[:：]([+-]?\d+(?:\.\d+)?)(亿|万|元)", t)

                        total_turnover_yuan = (
                            self._parse_amount_to_yuan(m_turn.group(1) + m_turn.group(2)) if m_turn else None
                        )
                        total_buy_yuan = (
                            self._parse_amount_to_yuan(m_buy.group(1) + m_buy.group(2)) if m_buy else None
                        )
                        total_sell_yuan = (
                            self._parse_amount_to_yuan(m_sell.group(1) + m_sell.group(2)) if m_sell else None
                        )
                        total_net_yuan = (
                            self._parse_amount_to_yuan(m_net.group(1) + m_net.group(2)) if m_net else None
                        )
                        break

                buy_top5: List[Dict[str, Any]] = []
                sell_top5: List[Dict[str, Any]] = []
                for tb in b.select("table.m-table.m-table-nosort.mt10"):
                    thead = tb.select_one("thead")
                    head = (thead.get_text(" ", strip=True) if thead else "").replace("\xa0", " ")
                    if "买入金额最大的前5名营业部" in head:
                        buy_top5 = _extract_top5(tb)
                    elif "卖出金额最大的前5名营业部" in head:
                        sell_top5 = _extract_top5(tb)

                # 如果只拿到一张表（极少数），按“表数量兜底”
                if (not buy_top5 or not sell_top5) and len(b.select("table.m-table.m-table-nosort.mt10")) >= 2:
                    tbs = b.select("table.m-table.m-table-nosort.mt10")[:2]
                    if not buy_top5:
                        buy_top5 = _extract_top5(tbs[0])
                    if not sell_top5:
                        sell_top5 = _extract_top5(tbs[1])

                rec: Dict[str, Any] = {
                    "trade_date": trade_date,
                    "window_days": int(window_days),
                    "category": category,
                    "stock_code": stockcode,
                    "stock_name": stock_name,
                    "reason": reason,
                    "detail_title": title_text,
                    "summary_text": "",
                    "total_turnover_yuan": total_turnover_yuan,
                    "total_buy_yuan": total_buy_yuan,
                    "total_sell_yuan": total_sell_yuan,
                    "total_net_yuan": total_net_yuan,
                    "buy_top5": buy_top5,
                    "sell_top5": sell_top5,
                }

                key = (trade_date, int(window_days), category, stockcode)
                prev = best.get(key)
                prev_buy = float(prev.get("total_buy_yuan") or 0.0) if prev else 0.0
                cur_buy = float(total_buy_yuan or 0.0)
                if (prev is None) or (cur_buy >= prev_buy):
                    best[key] = rec

            except Exception:
                continue

        records = list(best.values())
        logger.info(f"成功抓取 report 指定日期席位明细: {len(records)} 条（去重后）")
        return records

    async def crawl(
        self,
        trade_date: Optional[str] = None,
        category: str = "全部股票",
        limit_rows: int = 80,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            trade_date: YYYY-MM-DD；None 则默认页面当前日期
            category: 标签页：全部股票/机构参与/敢死队上榜/游资上榜/跟风高手上榜
            limit_rows: 最多抓取列表前 N 行（页面通常会包含 1日+3日两种行）

        Returns:
            每条记录包含：trade_date, window_days, stock_code, stock_name, reason, totals, buy_top5, sell_top5 等
        """
        # 指定日期：优先走 report-page（整页明细），避免日期控件/iframe
        if trade_date:
            return await self._crawl_report_page(trade_date=trade_date, category=category, limit_rows=limit_rows)

        try:
            await self._init_browser()
            await self._get_page()
            if not self.page:
                raise CrawlerError("页面对象未初始化")

            await self.page.goto(self.BASE_URL, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(1.5)

            # 选择类别 tab（用 JS click 稳一点）
            await self.page.evaluate(
                """
                (cat) => {
                  const links = Array.from(document.querySelectorAll('a'));
                  const el = links.find(a => (a.textContent || '').trim() === cat);
                  if (el) el.click();
                }
                """,
                category,
            )
            await asyncio.sleep(1.0)

            # 设置日期（页面输入框 class：m_text_date startday）
            if trade_date:
                await self.page.fill("input.m_text_date.startday", trade_date)
                # 一些页面需要回车触发刷新
                await self.page.keyboard.press("Enter")
                await asyncio.sleep(1.2)

            # 等待列表 table 出现
            await self.page.wait_for_selector("table.m-table", timeout=15000)
            await asyncio.sleep(0.8)

            records: List[Dict[str, Any]] = []

            # 逐行抓取：点击“代码列”触发右侧明细更新，然后解析右侧两张席位表
            # 注意：页面里左侧列表（leftcol）与右侧明细（rightcol）都有 m-table。
            # 必须严格限定到左侧列表范围，避免误点右侧明细表导致解析错乱。
            list_row_count = await self.page.evaluate(
                """
                () => {
                  const scope = document.querySelector('div.ggmx.clearfix div.leftcol');
                  if (!scope) return 0;
                  const tables = Array.from(scope.querySelectorAll('table.m-table'));
                  const scores = tables.map(t => ({rows: t.querySelectorAll('tbody tr').length, t}));
                  scores.sort((a,b)=>b.rows-a.rows);
                  return scores.length ? scores[0].rows : 0;
                }
                """
            )

            if not list_row_count:
                logger.warning("未找到龙虎榜列表行（table.m-table tbody tr）")
                return []

            max_i = min(int(list_row_count), int(limit_rows))

            for i in range(1, max_i + 1):
                try:
                    # 点击左侧列表：优先点第2列（兼容“1日/3日口径列”在第1列的情况），不行再点第1列。
                    # 不能用 page.evaluate(...).click()，因为部分站点会依赖 isTrusted；必须用 Playwright 的真实点击。
                    detail = None
                    for code_td_index in (2, 1):
                        try:
                            await self.page.click(
                                f"div.ggmx.clearfix div.leftcol table.m-table tbody tr:nth-child({i}) td:nth-child({code_td_index})",
                                timeout=5000,
                            )
                            await asyncio.sleep(0.6)
                        except Exception:
                            continue

                        detail = await self.page.evaluate(
                            """
                            () => {
                          const result = {};

                          // trade_date：取左上日期输入框
                          const dateInput = document.querySelector('input.m_text_date.startday');
                          result.trade_date = dateInput ? (dateInput.value || '').trim() : '';

                          // 右侧明细区域：div.ggmx.clearfix div.rightcol
                          const rightCol = document.querySelector('div.ggmx.clearfix div.rightcol');
                          if (!rightCol) return null;

                          // active detail container（同一页面会预渲染多只标的明细 div.stockcode，未选中的为 display:none）
                          // 明细块一般是：<div class="stockcont" stockcode="000620" style="display:none|block">...</div>
                          const candidates = Array.from(rightCol.querySelectorAll('.stockcont[stockcode]'));
                          const active = candidates.find(el => {
                            const st = window.getComputedStyle(el);
                            return st && st.display !== 'none' && st.visibility !== 'hidden';
                          }) || rightCol.querySelector('.stockcont[stockcode]') || rightCol;

                          // 右侧明细标题：如 “盈新发展(000620)明细：日涨幅偏离值达7%的证券”
                          const titleP = Array.from(active.querySelectorAll('p')).find(p => (p.innerText || '').includes('明细：'));
                          const titleText = titleP ? (titleP.innerText || '').trim() : '';
                          result.detail_title = titleText;

                          const m = titleText.match(/\\((\\d{6})\\)明细：(.+)$/);
                          result.stock_code = m ? m[1] : '';
                          result.reason = m ? m[2].trim() : '';
                          result.stock_name = titleText ? titleText.split('(')[0].trim() : '';

                          // 右侧汇总：包含 “成交额：xx 合计买入：xx 合计卖出：xx 净额：xx”
                          const sumP = Array.from(active.querySelectorAll('p')).find(
                            p => (p.innerText || '').includes('成交额：') && (p.innerText || '').includes('合计买入：')
                          );
                          result.summary_text = sumP ? (sumP.innerText || '').trim() : '';

                          // 解析金额文本（亿/万/元） -> 元
                          function parseAmountToYuan(txt) {
                            if (!txt) return null;
                            const t = txt.replace(/[,\\s]/g, '');
                            const mm = t.match(/([+-]?[0-9]+(?:\\.[0-9]+)?)(亿|万|元)?/);
                            if (!mm) return null;
                            const v = parseFloat(mm[1]);
                            const u = mm[2] || '元';
                            if (u === '亿') return v * 1e8;
                            if (u === '万') return v * 1e4;
                            return v;
                          }

                          // 汇总金额
                          const sumLine = result.summary_text;
                          const mTurn = sumLine.match(/成交额：([^\\s]+)\\s/);
                          const mBuy = sumLine.match(/合计买入：([^\\s]+)\\s/);
                          const mSell = sumLine.match(/合计卖出：([^\\s]+)\\s/);
                          const mNet = sumLine.match(/净额：([^\\s]+)/);
                          result.total_turnover_yuan = mTurn ? parseAmountToYuan(mTurn[1]) : null;
                          result.total_buy_yuan = mBuy ? parseAmountToYuan(mBuy[1]) : null;
                          result.total_sell_yuan = mSell ? parseAmountToYuan(mSell[1]) : null;
                          result.total_net_yuan = mNet ? parseAmountToYuan(mNet[1]) : null;

                          // 明细表：买入/卖出前5（右侧明细里通常有两张 table.m-table-nosort.mt10）
                          // 注意：标题“买入金额最大的前5名营业部/卖出金额最大的前5名营业部”不一定在 table 内部，
                          // 因此必须限定在右侧明细区域，并通过“标题节点 -> 后续表格”来定位。
                          function parseTop5FromRightCol(side) {
                            const tables = Array.from(active.querySelectorAll('table.m-table.m-table-nosort.mt10'));
                            if (!tables.length) return [];

                            function findSideLabelForTable(tb) {
                              // 优先读 thead：同花顺会把“买入/卖出金额最大的前5名营业部”放在表头里
                              const theadText = (tb.querySelector('thead')?.innerText || '').replace(/\\s+/g, ' ').trim();
                              if (theadText.includes('买入金额最大的前5名营业部')) return '买入金额最大的前5名营业部';
                              if (theadText.includes('卖出金额最大的前5名营业部')) return '卖出金额最大的前5名营业部';

                              // 标题通常在 table 前面的兄弟节点（或父级前序兄弟）
                              let cur = tb.previousElementSibling;
                              for (let i=0; i<12 && cur; i++) {
                                const txt = (cur.innerText || '').replace(/\\s+/g,' ').trim();
                                if (txt && (txt.includes('买入金额最大的前5名营业部') || txt.includes('卖出金额最大的前5名营业部'))) {
                                  return txt;
                                }
                                cur = cur.previousElementSibling;
                              }

                              let p = tb.parentElement;
                              for (let j=0; j<8 && p; j++) {
                                let sib = p.previousElementSibling;
                                for (let k=0; k<6 && sib; k++) {
                                  const txt = (sib.innerText || '').replace(/\\s+/g,' ').trim();
                                  if (txt && (txt.includes('买入金额最大的前5名营业部') || txt.includes('卖出金额最大的前5名营业部'))) {
                                    return txt;
                                  }
                                  sib = sib.previousElementSibling;
                                }
                                p = p.parentElement;
                              }
                              return '';
                            }

                            const tagged = tables.map(tb => ({ tb, label: findSideLabelForTable(tb) }));
                            const buyTb = tagged.find(x => x.label.includes('买入金额最大的前5名营业部'))?.tb || null;
                            const sellTb = tagged.find(x => x.label.includes('卖出金额最大的前5名营业部'))?.tb || null;

                            let target = null;
                            if (side === 'buy') target = buyTb;
                            if (side === 'sell') target = sellTb;

                            if (!target) {
                              // 兜底：如果只找到一个 side，则另一张表认为是另一个 side；否则按顺序取
                              if (side === 'buy') {
                                target = buyTb || (sellTb ? tagged.find(x => x.tb !== sellTb)?.tb : null) || tagged[0]?.tb || null;
                              } else {
                                target = sellTb || (buyTb ? tagged.find(x => x.tb !== buyTb)?.tb : null) || tagged[1]?.tb || tagged[0]?.tb || null;
                              }
                            }

                            if (!target) return [];

                            const bodyRows = Array.from(target.querySelectorAll('tbody tr')).slice(0, 5);
                            return bodyRows
                              .map((tr, idx) => {
                                const tds = tr.querySelectorAll('td');
                                if (!tds || tds.length < 4) return null;

                                const seatCell = tds[0];
                                const a = seatCell.querySelector('a');
                                const rawLines = (seatCell.innerText || '').split('\\n').map(s => s.trim()).filter(Boolean);
                                const seatName = a ? (a.innerText || '').trim() : (rawLines[0] || '').trim();
                                let seatTag = rawLines.length >= 2 ? (rawLines[1] || null) : null;
                                if (!seatTag && rawLines.length === 1 && seatName) {
                                  // 兼容“席位名 + 标签”在同一行的情况（如：深股通专用 一线游资）
                                  const oneLine = rawLines[0] || '';
                                  const rest = oneLine.replace(seatName, '').trim();
                                  seatTag = rest || null;
                                }

                                const buyWan = parseFloat((tds[1].innerText || '0').replace(/,/g, '')) || 0;
                                const sellWan = parseFloat((tds[2].innerText || '0').replace(/,/g, '')) || 0;
                                const netWan = parseFloat((tds[3].innerText || '0').replace(/,/g, '')) || 0;

                                return {
                                  rank: idx + 1,
                                  seat_name: seatName,
                                  seat_tag: seatTag,
                                  buy_yuan: buyWan * 1e4,
                                  sell_yuan: sellWan * 1e4,
                                  net_yuan: netWan * 1e4,
                                };
                              })
                              .filter(x => x && x.seat_name);
                          }

                          result.buy_top5 = parseTop5FromRightCol('buy');
                          result.sell_top5 = parseTop5FromRightCol('sell');

                          return result;
                        }
                        """
                        )
                        if detail and detail.get("stock_code"):
                            break

                    # window_days：从列表行第一列判断（3日/空）
                    window_days = await self.page.evaluate(
                        f"""
                        () => {{
                          const tr = document.querySelector('div.ggmx.clearfix div.leftcol table.m-table tbody tr:nth-child({i})');
                          if (!tr) return 1;
                          const td1 = tr.querySelector('td:nth-child(1)');
                          const t = td1 ? (td1.innerText || '').trim() : '';
                          const m = t.match(/(\\d+)日/);
                          return m ? parseInt(m[1], 10) : 1;
                        }}
                        """
                    )

                    if not detail or not detail.get("stock_code"):
                        continue

                    # 过滤转债：同花顺右侧明细标题里会出现 11xxxx / 12xxxx 等转债代码（如 123xxx）
                    stock_code = str(detail.get("stock_code"))
                    stock_name = str(detail.get("stock_name") or "")
                    if stock_code.startswith(("11", "12")) or ("转债" in stock_name):
                        continue

                    detail["window_days"] = int(window_days)
                    detail["category"] = category
                    records.append(detail)

                except Exception as e:
                    logger.warning(f"解析第{i}行失败: {e}")
                    continue

            logger.info(f"成功爬取龙虎榜席位明细: {len(records)} 条（含1日/3日行）")
            return records

        except Exception as e:
            error_msg = f"爬取龙虎榜席位明细失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise CrawlerError(error_msg)
        finally:
            await self.cleanup()
