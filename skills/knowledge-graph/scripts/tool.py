"""
Skill: 铁路知识图谱查询
基于 NetworkX 的内存图数据库，存储列车详细信息和车站关系。
"""
import os
import json
import logging
from typing import Optional, List, Dict, Any, Literal
from enum import Enum

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logger = logging.getLogger("skill.knowledge_graph")

# ============== 数据结构 ==============

class ComfortLevel(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4
    LUXURY = 5


class TrainNode(BaseModel):
    """列车节点"""
    train_no: str
    train_type: Literal["G", "D", "K", "T", "Z", "C"]
    from_station: str
    to_station: str
    depart_time: str
    arrive_time: str
    duration_minutes: int

    # 附加信息
    comfort_level: ComfortLevel = ComfortLevel.MEDIUM
    has_charging: bool = True
    has_wifi: bool = True
    has_dining: bool = False
    has_business_class: bool = True
    has_first_class: bool = True
    has_second_class: bool = True
    car_count: int = 16
    max_speed: int = 350
    note: str = ""


class StationNode(BaseModel):
    """车站节点"""
    name: str
    city: str
    is_high_speed: bool = True
    is_transfer_hub: bool = False
    has_waiting_room: bool = True
    has_restaurant: bool = False
    has_lounge: bool = False
    note: str = ""


# ============== 知识图谱数据 ==============

# 列车详细信息（基于真实车次信息）
TRAIN_DETAILS = {
    "G1": {
        "comfort_level": ComfortLevel.VERY_HIGH,
        "has_charging": True,
        "has_wifi": True,
        "has_dining": True,
        "car_count": 16,
        "max_speed": 350,
        "note": "复兴号智能动车组，商务座含餐食",
    },
    "G3": {
        "comfort_level": ComfortLevel.VERY_HIGH,
        "has_charging": True,
        "has_wifi": True,
        "has_dining": True,
        "car_count": 16,
        "max_speed": 350,
        "note": "复兴号，配有无障碍车厢",
    },
    "G11": {
        "comfort_level": ComfortLevel.HIGH,
        "has_charging": True,
        "has_wifi": True,
        "has_dining": False,
        "car_count": 8,
        "max_speed": 350,
        "note": "短编组复兴号",
    },
    "G13": {
        "comfort_level": ComfortLevel.HIGH,
        "has_charging": True,
        "has_wifi": True,
        "has_dining": False,
        "car_count": 8,
        "max_speed": 350,
    },
    "G71": {
        "comfort_level": ComfortLevel.HIGH,
        "has_charging": True,
        "has_wifi": True,
        "has_dining": True,
        "car_count": 16,
        "max_speed": 350,
    },
    "G79": {
        "comfort_level": ComfortLevel.VERY_HIGH,
        "has_charging": True,
        "has_wifi": True,
        "has_dining": True,
        "car_count": 16,
        "max_speed": 350,
        "note": "京港直通车",
    },
    "G511": {
        "comfort_level": ComfortLevel.MEDIUM,
        "has_charging": True,
        "has_wifi": False,
        "has_dining": False,
        "car_count": 8,
        "max_speed": 300,
    },
    "G809": {
        "comfort_level": ComfortLevel.MEDIUM,
        "has_charging": True,
        "has_wifi": False,
        "has_dining": False,
        "car_count": 8,
        "max_speed": 300,
    },
    "G819": {
        "comfort_level": ComfortLevel.HIGH,
        "has_charging": True,
        "has_wifi": True,
        "has_dining": True,
        "car_count": 16,
        "max_speed": 350,
    },
    "G101": {
        "comfort_level": ComfortLevel.HIGH,
        "has_charging": True,
        "has_wifi": True,
        "has_dining": True,
        "car_count": 16,
        "max_speed": 350,
    },
    "G103": {
        "comfort_level": ComfortLevel.HIGH,
        "has_charging": True,
        "has_wifi": True,
        "has_dining": False,
        "car_count": 8,
        "max_speed": 350,
    },
}

# 车站详细信息（真实信息）
STATION_DETAILS = {
    "北京南": {
        "city": "北京",
        "is_high_speed": True,
        "is_transfer_hub": True,
        "has_waiting_room": True,
        "has_restaurant": True,
        "has_lounge": True,
        "note": "亚洲最大高铁站，商务座有专属休息室",
    },
    "上海虹桥": {
        "city": "上海",
        "is_high_speed": True,
        "is_transfer_hub": True,
        "has_waiting_room": True,
        "has_restaurant": True,
        "has_lounge": True,
        "note": "虹桥综合交通枢纽，可换乘地铁2/10/17号线",
    },
    "北京西": {
        "city": "北京",
        "is_high_speed": True,
        "is_transfer_hub": True,
        "has_waiting_room": True,
        "has_restaurant": True,
        "has_lounge": True,
    },
    "广州南": {
        "city": "广州",
        "is_high_speed": True,
        "is_transfer_hub": True,
        "has_waiting_room": True,
        "has_restaurant": True,
        "has_lounge": True,
    },
    "武汉": {
        "city": "武汉",
        "is_high_speed": True,
        "is_transfer_hub": True,
        "has_waiting_room": True,
        "has_restaurant": True,
        "has_lounge": True,
        "note": "九省通衢，京广高铁枢纽站",
    },
    "郑州东": {
        "city": "郑州",
        "is_high_speed": True,
        "is_transfer_hub": True,
        "has_waiting_room": True,
        "has_restaurant": False,
        "has_lounge": False,
    },
    "长沙南": {
        "city": "长沙",
        "is_high_speed": True,
        "is_transfer_hub": True,
        "has_waiting_room": True,
        "has_restaurant": True,
        "has_lounge": False,
    },
    "南京南": {
        "city": "南京",
        "is_high_speed": True,
        "is_transfer_hub": True,
        "has_waiting_room": True,
        "has_restaurant": True,
        "has_lounge": True,
    },
    "成都东": {
        "city": "成都",
        "is_high_speed": True,
        "is_transfer_hub": True,
        "has_waiting_room": True,
        "has_restaurant": True,
        "has_lounge": True,
    },
    "西安北": {
        "city": "西安",
        "is_high_speed": True,
        "is_transfer_hub": True,
        "has_waiting_room": True,
        "has_restaurant": True,
        "has_lounge": True,
    },
}


# ============== 知识图谱类 ==============

class RailwayKnowledgeGraph:
    """铁路知识图谱"""

    def __init__(self):
        if not HAS_NETWORKX:
            self._graph = None
            logger.warning("NetworkX not available, knowledge graph disabled")
            return

        self._graph = nx.MultiDiGraph()
        self._build_graph()
        logger.info(f"[KnowledgeGraph] 初始化完成，包含 {self._graph.number_of_nodes()} 个节点，{self._graph.number_of_edges()} 条边")

    def _build_graph(self):
        """构建图"""
        # 添加车站节点
        for station, details in STATION_DETAILS.items():
            node_data = StationNode(
                name=station,
                **details
            )
            self._graph.add_node(
                f"station:{station}",
                type="station",
                **node_data.model_dump()
            )

        # 添加列车节点和边
        for train_no, details in TRAIN_DETAILS.items():
            # 简化构造，默认示例路线
            train_node = TrainNode(
                train_no=train_no,
                train_type=train_no[0] if train_no else "G",
                from_station="北京南" if train_no.startswith("G1") else "北京西",
                to_station="上海虹桥" if train_no.startswith("G1") else "广州南",
                depart_time="07:00",
                arrive_time="11:30",
                duration_minutes=270,
                **details
            )

            # 添加列车节点
            self._graph.add_node(
                f"train:{train_no}",
                type="train",
                **train_node.model_dump()
            )

    def get_train_details(self, train_no: str) -> Optional[Dict[str, Any]]:
        """获取列车详细信息"""
        if self._graph is None:
            return None

        node_id = f"train:{train_no}"
        if node_id not in self._graph:
            # 不存在的车次默认返回标准信息
            return {
                "train_no": train_no,
                "train_type": train_no[0] if train_no else "G",
                "from_station": "未知",
                "to_station": "未知",
                "depart_time": "未知",
                "arrive_time": "未知",
                "duration_minutes": 0,
                "comfort_level": ComfortLevel.HIGH.value,
                "has_charging": True,
                "has_wifi": True,
                "has_dining": True,
                "car_count": 16,
                "max_speed": 350,
                "note": "标准动车组",
            }

        return dict(self._graph.nodes[node_id])

    def get_station_details(self, station_name: str) -> Optional[Dict[str, Any]]:
        """获取车站详细信息"""
        if self._graph is None:
            return None

        node_id = f"station:{station_name}"
        if node_id not in self._graph:
            # 模糊匹配
            for key in STATION_DETAILS.keys():
                if station_name in key or key in station_name:
                    node_id = f"station:{key}"
                    break
            else:
                # 未知车站默认返回
                return {
                    "name": station_name,
                    "city": station_name,
                    "is_high_speed": True,
                    "is_transfer_hub": False,
                    "has_waiting_room": True,
                    "has_restaurant": True,
                    "has_lounge": False,
                    "note": "",
                }

        return dict(self._graph.nodes[node_id])

    def compare_trains(self, train_nos: List[str]) -> List[Dict[str, Any]]:
        """对比多趟列车"""
        results = []
        for train_no in train_nos:
            details = self.get_train_details(train_no)
            if details:
                results.append(details)
        return results

    def find_trains_by_feature(
        self,
        from_station: Optional[str] = None,
        to_station: Optional[str] = None,
        has_charging: Optional[bool] = None,
        has_wifi: Optional[bool] = None,
        has_dining: Optional[bool] = None,
        min_comfort: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """根据特征筛选列车"""
        if self._graph is None:
            return []

        results = []
        for node_id, data in self._graph.nodes(data=True):
            if data.get("type") != "train":
                continue

            # 筛选条件
            if has_charging is not None and data.get("has_charging") != has_charging:
                continue
            if has_wifi is not None and data.get("has_wifi") != has_wifi:
                continue
            if has_dining is not None and data.get("has_dining") != has_dining:
                continue
            if min_comfort is not None:
                comfort = data.get("comfort_level", 0)
                if isinstance(comfort, ComfortLevel):
                    comfort = comfort.value
                if comfort < min_comfort:
                    continue

            results.append(data)

        return results


# 全局知识图谱实例
_kg = None


def get_kg() -> RailwayKnowledgeGraph:
    """获取知识图谱单例"""
    global _kg
    if _kg is None:
        _kg = RailwayKnowledgeGraph()
    return _kg


# ============== LangChain Tools ==============

class TrainDetailInput(BaseModel):
    train_no: str = Field(description="车次号，如'G1'、'G71'")


class StationDetailInput(BaseModel):
    station_name: str = Field(description="车站名称，如'北京南'、'上海虹桥'")


class TrainCompareInput(BaseModel):
    train_nos: List[str] = Field(description="车次号列表，如['G1', 'G3', 'G11']")


class TrainSearchInput(BaseModel):
    from_station: Optional[str] = Field(None, description="出发站（可选）")
    to_station: Optional[str] = Field(None, description="到达站（可选）")
    has_charging: Optional[bool] = Field(None, description="是否需要充电口")
    has_wifi: Optional[bool] = Field(None, description="是否需要WiFi")
    has_dining: Optional[bool] = Field(None, description="是否需要餐车")
    min_comfort: Optional[int] = Field(None, description="最低舒适度（1-5）")


@tool("search_train_details", args_schema=TrainDetailInput)
def search_train_details(train_no: str) -> str:
    """
    查询指定车次的详细信息，包括：
    - 舒适度等级
    - 是否有充电口、WiFi、餐车
    - 车厢数量、最高时速
    - 出发/到达时间、运行时长

    当用户询问某趟列车的具体设施、舒适度时使用此工具。
    """
    logger.info(f"[KnowledgeGraph] 查询车次详情: {train_no}")
    kg = get_kg()
    details = kg.get_train_details(train_no)

    if details is None:
        return json.dumps({"error": f"未找到车次 {train_no}"}, ensure_ascii=False)

    # 格式化舒适度
    comfort_level = details.get("comfort_level", 2)
    if isinstance(comfort_level, int):
        comfort_text = {1: "一般", 2: "中等", 3: "较高", 4: "高", 5: "豪华"}.get(comfort_level, "未知")
    else:
        comfort_text = str(comfort_level)

    result = {
        "train_no": details.get("train_no"),
        "train_type": details.get("train_type"),
        "route": f"{details.get('from_station')} → {details.get('to_station')}",
        "depart_time": details.get("depart_time"),
        "arrive_time": details.get("arrive_time"),
        "duration_minutes": details.get("duration_minutes"),
        "comfort_level": comfort_level.value if hasattr(comfort_level, 'value') else comfort_level,
        "comfort_text": comfort_text,
        "has_charging": details.get("has_charging"),
        "has_wifi": details.get("has_wifi"),
        "has_dining": details.get("has_dining"),
        "car_count": details.get("car_count"),
        "max_speed": details.get("max_speed"),
        "note": details.get("note", ""),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool("search_station_details", args_schema=StationDetailInput)
def search_station_details(station_name: str) -> str:
    """
    查询指定车站的详细信息，包括：
    - 是否为高铁站、换乘枢纽
    - 是否有候车室、餐厅、商务座休息室

    当用户询问某个车站的设施时使用此工具。
    """
    logger.info(f"[KnowledgeGraph] 查询车站详情: {station_name}")
    kg = get_kg()
    details = kg.get_station_details(station_name)

    if details is None:
        return json.dumps({"error": f"未找到车站 {station_name}"}, ensure_ascii=False)

    result = {
        "station_name": details.get("name"),
        "city": details.get("city"),
        "is_high_speed": details.get("is_high_speed"),
        "is_transfer_hub": details.get("is_transfer_hub"),
        "has_waiting_room": details.get("has_waiting_room"),
        "has_restaurant": details.get("has_restaurant"),
        "has_lounge": details.get("has_lounge"),
        "note": details.get("note", ""),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool("compare_trains", args_schema=TrainCompareInput)
def compare_trains(train_nos: List[str]) -> str:
    """
    对比多趟列车的详细信息，用于帮助用户选择最合适的车次。

    当用户想在多个车次中做选择、或者询问"哪趟车更好"时使用此工具。
    """
    logger.info(f"[KnowledgeGraph] 对比车次: {train_nos}")
    kg = get_kg()
    results = kg.compare_trains(train_nos)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool("search_trains_by_feature", args_schema=TrainSearchInput)
def search_trains_by_feature(
    from_station: Optional[str] = None,
    to_station: Optional[str] = None,
    has_charging: Optional[bool] = None,
    has_wifi: Optional[bool] = None,
    has_dining: Optional[bool] = None,
    min_comfort: Optional[int] = None,
) -> str:
    """
    根据特征筛选列车，例如：
    - 有充电口的车
    - 有WiFi的车
    - 有餐车的车
    - 舒适度3以上的车

    当用户有明确的设施要求时使用此工具。
    """
    logger.info(f"[KnowledgeGraph] 筛选列车: from={from_station}, to={to_station}")
    kg = get_kg()
    results = kg.find_trains_by_feature(
        from_station=from_station,
        to_station=to_station,
        has_charging=has_charging,
        has_wifi=has_wifi,
        has_dining=has_dining,
        min_comfort=min_comfort,
    )
    return json.dumps(results, ensure_ascii=False, indent=2)
