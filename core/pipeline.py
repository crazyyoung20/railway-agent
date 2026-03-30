"""
Pipeline层：简单查询硬编码流程
特点：
- 不走LLM，直接代码调用Skills
- 延迟稳定（P99 < 300ms）
- 只处理格式标准的直达查询
"""
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from config import settings

logger = logging.getLogger(__name__)


class SimpleQueryParams:
    """Pipeline查询参数"""
    def __init__(
        self,
        origin: str,
        destination: str,
        date: Optional[str] = None,
    ):
        self.origin = origin
        self.destination = destination
        self.date = date or self._default_date()

    @staticmethod
    def _default_date() -> str:
        """默认明天"""
        tomorrow = datetime.now() + timedelta(days=1)
        return tomorrow.strftime("%Y-%m-%d")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "date": self.date,
        }


class SimpleQueryPipeline:
    """
    简单查询Pipeline
    流程：提取参数 → 标准化站名 → 查询余票 → 格式化输出
    """
    def __init__(self, tools: Dict[str, Any]):
        """
        初始化Pipeline
        :param tools: Skills字典，key是tool name
        """
        self._tools = tools
        self._normalize = tools.get("normalize_station")
        self._query_tickets = tools.get("query_tickets")
        self._search_knowledge = tools.get("search_railway_knowledge")

        if not self._normalize or not self._query_tickets:
            raise RuntimeError("Pipeline需要normalize_station和query_tickets两个Skill")

    def _parse_date(self, date_str: Optional[str]) -> str:
        """解析日期字符串"""
        if not date_str:
            return SimpleQueryParams._default_date()

        date_str = date_str.strip()

        if date_str == "明天":
            return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        if date_str == "后天":
            return (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        if date_str == "大后天":
            return (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")

        # 已有的 YYYY-MM-DD 格式
        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            return date_str

        # 解析 MM月DD日
        match = re.match(r"(\d{1,2})月(\d{1,2})日", date_str)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            year = datetime.now().year
            # 如果月份已经过了，默认明年
            if month < datetime.now().month:
                year += 1
            return f"{year}-{month:02d}-{day:02d}"

        return SimpleQueryParams._default_date()

    def execute(self, params: SimpleQueryParams) -> Dict[str, Any]:
        """
        执行Pipeline
        :param params: 查询参数
        :return: 结构化结果
        """
        logger.info(f"[Pipeline] 执行简单查询: {params.origin} → {params.destination} @ {params.date}")
        start_time = datetime.now()

        try:
            # Step 1: 标准化站名
            origin_result = self._call_tool(
                self._normalize,
                {"station_name": params.origin}
            )
            origin_data = json.loads(origin_result) if isinstance(origin_result, str) else origin_result
            origin_std = origin_data.get("primary", params.origin)

            dest_result = self._call_tool(
                self._normalize,
                {"station_name": params.destination}
            )
            dest_data = json.loads(dest_result) if isinstance(dest_result, str) else dest_result
            dest_std = dest_data.get("primary", params.destination)

            logger.info(f"[Pipeline] 站名标准化: {params.origin}→{origin_std}, {params.destination}→{dest_std}")

            # Step 2: 查询余票
            tickets_result = self._call_tool(
                self._query_tickets,
                {
                    "from_station": origin_std,
                    "to_station": dest_std,
                    "date": params.date,
                }
            )
            tickets_data = json.loads(tickets_result) if isinstance(tickets_result, str) else tickets_result

            # Step 3: 格式化输出
            result = self._format_response(
                params=params,
                origin_std=origin_std,
                dest_std=dest_std,
                tickets=tickets_data
            )

            elapsed = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.info(f"[Pipeline] 执行完成，耗时 {elapsed}ms")

            return {
                "success": True,
                "route_info": {"layer": "pipeline", "latency_ms": elapsed},
                **result
            }

        except Exception as e:
            logger.error(f"[Pipeline] 执行失败: {e}", exc_info=True)
            raise

    def execute_from_query(self, query: str, extracted_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        从用户查询和路由层提取的参数执行
        :param query: 用户原始查询
        :param extracted_params: 路由层提取的参数
        :return: 结构化结果
        """
        params = SimpleQueryParams(
            origin=extracted_params["origin"],
            destination=extracted_params["destination"],
            date=self._parse_date(extracted_params.get("date")),
        )
        result = self.execute(params)
        result["user_query"] = query
        return result

    def _call_tool(self, tool, params: Dict[str, Any]) -> Any:
        """统一调用Skill"""
        if hasattr(tool, "invoke"):
            return tool.invoke(params)
        return tool(**params)

    def _format_response(
        self,
        params: SimpleQueryParams,
        origin_std: str,
        dest_std: str,
        tickets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """格式化Pipeline输出，和Agent输出保持一致"""
        # 整理车次信息
        trains = tickets.get("trains", []) if isinstance(tickets, dict) else []
        train_summary = []

        for i, train in enumerate(trains[:5]):  # 只显示前5个
            if isinstance(train, dict):
                train_no = train.get("train_no", "")
                start_time = train.get("start_time", "")
                end_time = train.get("end_time", "")
                duration = train.get("duration", "")
                second_class = train.get("second_class", "")
                first_class = train.get("first_class", "")
                business_class = train.get("business_class", "")

                summary = f"{train_no}: {start_time}→{end_time} ({duration})"
                prices = []
                if second_class:
                    prices.append(f"二等座{second_class}")
                if first_class:
                    prices.append(f"一等座{first_class}")
                if business_class:
                    prices.append(f"商务座{business_class}")
                if prices:
                    summary += f" - {', '.join(prices)}"
                train_summary.append(summary)

        # 构建最终回答
        final_answer = (
            f"为您查询到 {params.date} {origin_std} 到 {dest_std} 的直达列车信息：\n\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(train_summary))
        )

        if not train_summary:
            final_answer = f"未查询到 {params.date} {origin_std} 到 {dest_std} 的直达列车，建议您尝试中转方案。"

        return {
            "final_answer": final_answer,
            "query": f"{origin_std}→{dest_std} {params.date}",
            "origin": origin_std,
            "destination": dest_std,
            "date": params.date,
            "tickets": tickets,
        }


# 辅助：需要导入re
import re
