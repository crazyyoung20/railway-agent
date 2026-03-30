"""
Skill: 余票查询（半真实Mock数据）
标准化为 LangChain @tool，供 ReAct Agent 调用。
输入：from_station, to_station, date, train_filter(可选)
输出：JSON 字符串（列车列表）
"""

import os
import re
import json
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from enum import Enum

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger("skill.ticket_query")

# ══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════════════

class TicketStatus(str, Enum):
    AVAILABLE = "有票"
    SOLD_OUT  = "无票"
    FEW_LEFT  = "紧张"
    NOT_OPEN  = "未开售"

class SeatInfo(BaseModel):
    seat_type: str
    status: TicketStatus
    count: Optional[int] = None
    price: Optional[float] = None

class TrainTicket(BaseModel):
    train_no: str
    train_type: str
    from_station: str
    to_station: str
    depart_time: str
    arrive_time: str
    duration: str
    duration_minutes: int
    is_cross_day: bool = False
    seats: List[SeatInfo] = Field(default_factory=list)
    data_source: str = "mock"

    def available_seats(self) -> List[str]:
        return [s.seat_type for s in self.seats
                if s.status in (TicketStatus.AVAILABLE, TicketStatus.FEW_LEFT)]

# ══════════════════════════════════════════════════════════════════════════════
# 站名 → 电报码 映射（精简版）
# ══════════════════════════════════════════════════════════════════════════════

STATION_CODE_MAP: Dict[str, tuple] = {
    "北京南": ("VNP", "beijingnan"),   "北京": ("VAP", "beijing"),
    "北京西": ("VBP", "beijingxi"),    "上海虹桥": ("AOH", "shanghaihongqiao"),
    "上海": ("SHH", "shanghai"),        "广州南": ("IZQ", "guangzhounan"),
    "广州": ("GZQ", "guangzhou"),       "深圳北": ("IOQ", "shenzhenBei"),
    "深圳": ("SZQ", "shenzhen"),        "武汉": ("WHN", "wuhan"),
    "郑州东": ("ZAF", "zhengzhoudong"), "长沙南": ("CWQ", "changshaHan"),
    "成都东": ("ICW", "chengdudong"),   "重庆北": ("CQW", "chongqingbei"),
    "西安北": ("EAY", "xianBei"),       "杭州东": ("HGH", "hangzhoudong"),
    "南京南": ("NJH", "nanjingnan"),    "济南西": ("JNK", "jinanxi"),
    "沈阳北": ("SYT", "shenyangbei"),   "哈尔滨西": ("HBB", "haerbinxi"),
    "贵阳北": ("GIW", "guiyangbei"),    "昆明南": ("KMJ", "kunmingnan"),
    "太原南": ("TYV", "taiyuannan"),    "石家庄": ("SJP", "shijiazhuang"),
    "天津南": ("TJP", "tianjinnan"),    "天津": ("TJT", "tianjin"),
}

def get_station_code(name: str) -> Optional[str]:
    if name in STATION_CODE_MAP:
        return STATION_CODE_MAP[name][0]
    for k, (code, _) in STATION_CODE_MAP.items():
        if k in name or name in k:
            return code
    return None

# ══════════════════════════════════════════════════════════════════════════════
# 真实线路和票价数据
# ══════════════════════════════════════════════════════════════════════════════

# 真实票价数据（元/公里：G字头0.46元/公里，D字头0.31元/公里）
ROUTE_DATA = {
    ("北京南", "上海虹桥"): {"distance": 1318, "duration": 270, "type": "G",
                         "price": {"二等座": 553, "一等座": 933, "商务座": 1748}},
    ("北京西", "广州南"): {"distance": 2298, "duration": 480, "type": "G",
                         "price": {"二等座": 862, "一等座": 1380, "商务座": 2724}},
    ("北京西", "武汉"): {"distance": 1229, "duration": 240, "type": "G",
                       "price": {"二等座": 520, "一等座": 832, "商务座": 1640}},
    ("武汉", "广州南"): {"distance": 1069, "duration": 220, "type": "G",
                       "price": {"二等座": 463, "一等座": 740, "商务座": 1460}},
    ("北京西", "郑州东"): {"distance": 693, "duration": 130, "type": "G",
                         "price": {"二等座": 309, "一等座": 495, "商务座": 977}},
    ("郑州东", "广州南"): {"distance": 1605, "duration": 350, "type": "G",
                         "price": {"二等座": 727, "一等座": 1163, "商务座": 2296}},
}

# 热门线路模板
HOT_ROUTES = list(ROUTE_DATA.keys())

def _get_route_info(from_s: str, to_s: str) -> dict:
    """获取线路真实信息，没有的话自动生成相似数据"""
    key = (from_s, to_s)
    if key in ROUTE_DATA:
        return ROUTE_DATA[key].copy()
    # 反向线路
    reverse_key = (to_s, from_s)
    if reverse_key in ROUTE_DATA:
        info = ROUTE_DATA[reverse_key].copy()
        return info
    # 自动生成合理数据
    distance = random.randint(500, 2500)
    duration = int(distance / 300 * 60)  # 平均300km/h
    price_2nd = int(distance * 0.46)
    return {
        "distance": distance,
        "duration": duration,
        "type": "G",
        "price": {"二等座": price_2nd, "一等座": int(price_2nd * 1.8), "商务座": int(price_2nd * 3.2)}
    }

def _make_mock_seats(route_info: dict) -> List[SeatInfo]:
    """生成真实的座位余票状态"""
    statuses = [TicketStatus.AVAILABLE, TicketStatus.AVAILABLE, TicketStatus.FEW_LEFT, TicketStatus.SOLD_OUT]
    seats = []
    price = route_info["price"]
    for seat_type, base_price in price.items():
        # 高峰时段（早8晚6）、节假日余票紧张
        now = datetime.now()
        is_peak = (7 <= now.hour <= 9) or (17 <= now.hour <= 19) or now.weekday() >=5
        st = random.choice(statuses) if not is_peak else random.choice([TicketStatus.FEW_LEFT, TicketStatus.SOLD_OUT, TicketStatus.AVAILABLE])

        seats.append(SeatInfo(
            seat_type=seat_type,
            status=st,
            count=random.randint(1, 20) if st != TicketStatus.SOLD_OUT else 0,
            price=round(base_price * random.uniform(0.98, 1.02), 0)  # 票价小幅度波动
        ))
    return seats

def _generate_trains(from_s: str, to_s: str, date: str, count: int = 5) -> List[TrainTicket]:
    """生成真实的车次列表"""
    route_info = _get_route_info(from_s, to_s)
    trains = []

    start_hour = 6  # 首班车6点
    interval = random.randint(30, 90)  # 发车间隔30-90分钟

    for i in range(count):
        dep_h = start_hour + (i * interval // 60)
        dep_m = (i * interval) % 60
        depart_time = f"{dep_h:02d}:{dep_m:02d}"

        # 计算到达时间
        dur_min = route_info["duration"] + random.randint(-10, 20)  # 运行时间±10-20分钟
        total_m = dep_h * 60 + dep_m + dur_min
        arr_h, arr_m = total_m // 60 % 24, total_m % 60
        arrive_time = f"{arr_h:02d}:{arr_m:02d}"
        is_cross_day = total_m // 60 >= 24

        train_no = f"{route_info['type']}{random.randint(1, 999)}"

        trains.append(TrainTicket(
            train_no=train_no,
            train_type=route_info["type"],
            from_station=from_s,
            to_station=to_s,
            depart_time=depart_time,
            arrive_time=arrive_time,
            duration=f"{dur_min//60}时{dur_min%60}分",
            duration_minutes=dur_min,
            is_cross_day=is_cross_day,
            seats=_make_mock_seats(route_info),
            data_source="mock"
        ))

    return trains


# ══════════════════════════════════════════════════════════════════════════════
# LangChain Tool 定义
# ══════════════════════════════════════════════════════════════════════════════

class TicketQueryInput(BaseModel):
    from_station: str = Field(description="出发站标准站名，如'北京南'")
    to_station:   str = Field(description="到达站标准站名，如'广州南'")
    date:         str = Field(description="出行日期 YYYY-MM-DD")
    train_filter: Optional[str] = Field(None, description="车次类型过滤：G/D/K，不填则全部")

@tool("query_tickets", args_schema=TicketQueryInput)
def query_tickets(from_station: str, to_station: str, date: str,
                  train_filter: Optional[str] = None) -> str:
    """
    查询指定区间和日期的列车余票信息。
    返回 JSON 格式的列车列表，包含车次、时刻、座位余票和票价信息。
    当需要查询直达票或中转票的某一段时，调用此工具。
    """
    logger.info(f"[Skill:TicketQuery] {from_station}→{to_station} {date}")

    # 标准化站名匹配
    from_std = from_station
    to_std = to_station
    for station in STATION_CODE_MAP.keys():
        if from_station in station or station in from_station:
            from_std = station
        if to_station in station or station in to_station:
            to_std = station

    # 生成车次
    trains = _generate_trains(from_std, to_std, date)

    if train_filter:
        trains = [t for t in trains if t.train_no.startswith(train_filter.upper())]

    output = []
    for t in trains:
        avail = t.available_seats()
        seat_info = [{"seat": s.seat_type, "status": s.status.value,
                      "count": s.count, "price": s.price}
                     for s in t.seats]
        output.append({
            "train_no": t.train_no,
            "train_type": t.train_type,
            "from": t.from_station,
            "to": t.to_station,
            "depart": t.depart_time,
            "arrive": t.arrive_time,
            "duration_min": t.duration_minutes,
            "is_cross_day": t.is_cross_day,
            "available_seats": avail,
            "seats": seat_info,
            "source": t.data_source,
        })

    result = {
        "from_station": from_std,
        "to_station": to_std,
        "date": date,
        "total": len(output),
        "trains": output,
    }
    logger.info(f"[Skill:TicketQuery] 返回 {len(output)} 趟列车")
    return json.dumps(result, ensure_ascii=False, indent=2)
