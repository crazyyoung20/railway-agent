"""
Skill: 余票查询
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
import requests
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
# Mock 数据生成器
# ══════════════════════════════════════════════════════════════════════════════

MOCK_TRAINS = {
    ("北京南", "上海虹桥"): [
        {"no": "G1", "dur": 268, "dep": "06:48", "arr": "11:16", "type": "G"},
        {"no": "G3", "dur": 282, "dep": "07:00", "arr": "11:42", "type": "G"},
        {"no": "G11", "dur": 290, "dep": "08:00", "arr": "12:50", "type": "G"},
        {"no": "G13", "dur": 298, "dep": "09:00", "arr": "13:58", "type": "G"},
    ],
    ("北京西", "广州南"): [
        {"no": "G71", "dur": 480, "dep": "08:00", "arr": "16:00", "type": "G"},
        {"no": "G79", "dur": 508, "dep": "09:28", "arr": "17:56", "type": "G"},
    ],
    ("北京西", "武汉"): [
        {"no": "G511", "dur": 230, "dep": "07:00", "arr": "10:50", "type": "G"},
        {"no": "G515", "dur": 238, "dep": "08:30", "arr": "12:28", "type": "G"},
        {"no": "G517", "dur": 245, "dep": "10:00", "arr": "14:05", "type": "G"},
        {"no": "G521", "dur": 240, "dep": "13:00", "arr": "17:00", "type": "G"},
    ],
    ("武汉", "广州南"): [
        {"no": "G809", "dur": 220, "dep": "09:00", "arr": "12:40", "type": "G"},
        {"no": "G811", "dur": 228, "dep": "11:00", "arr": "14:48", "type": "G"},
        {"no": "G819", "dur": 235, "dep": "13:30", "arr": "17:25", "type": "G"},
        {"no": "G821", "dur": 240, "dep": "15:00", "arr": "19:00", "type": "G"},
    ],
    ("北京西", "郑州东"): [
        {"no": "G551", "dur": 130, "dep": "07:30", "arr": "09:40", "type": "G"},
        {"no": "G553", "dur": 135, "dep": "08:00", "arr": "10:15", "type": "G"},
        {"no": "G555", "dur": 132, "dep": "10:00", "arr": "12:12", "type": "G"},
        {"no": "G557", "dur": 130, "dep": "12:00", "arr": "14:10", "type": "G"},
        {"no": "G559", "dur": 128, "dep": "15:00", "arr": "17:08", "type": "G"},
    ],
    ("郑州东", "广州南"): [
        {"no": "G821", "dur": 350, "dep": "10:30", "arr": "16:20", "type": "G"},
        {"no": "G823", "dur": 360, "dep": "11:00", "arr": "17:00", "type": "G"},
        {"no": "G825", "dur": 355, "dep": "12:30", "arr": "18:25", "type": "G"},
        {"no": "G829", "dur": 340, "dep": "14:30", "arr": "20:10", "type": "G"},
    ],
    ("北京南", "南京南"): [
        {"no": "G101", "dur": 130, "dep": "07:00", "arr": "09:10", "type": "G"},
        {"no": "G103", "dur": 125, "dep": "08:30", "arr": "10:35", "type": "G"},
        {"no": "G105", "dur": 128, "dep": "10:00", "arr": "12:08", "type": "G"},
    ],
    ("南京南", "上海虹桥"): [
        {"no": "G7001", "dur": 70, "dep": "09:30", "arr": "10:40", "type": "G"},
        {"no": "G7003", "dur": 68, "dep": "11:00", "arr": "12:08", "type": "G"},
        {"no": "G7005", "dur": 72, "dep": "13:00", "arr": "14:12", "type": "G"},
    ],
    ("上海虹桥", "广州南"): [
        {"no": "G1701", "dur": 508, "dep": "07:00", "arr": "15:28", "type": "G"},
        {"no": "G1703", "dur": 520, "dep": "09:00", "arr": "17:40", "type": "G"},
    ],
    ("长沙南", "广州南"): [
        {"no": "G1001", "dur": 120, "dep": "08:00", "arr": "10:00", "type": "G"},
        {"no": "G1003", "dur": 115, "dep": "10:30", "arr": "12:25", "type": "G"},
        {"no": "G1005", "dur": 118, "dep": "12:00", "arr": "14:00", "type": "G"},
        {"no": "G1007", "dur": 122, "dep": "15:00", "arr": "17:02", "type": "G"},
    ],
    ("北京西", "长沙南"): [
        {"no": "G801", "dur": 350, "dep": "07:00", "arr": "12:50", "type": "G"},
        {"no": "G803", "dur": 360, "dep": "09:00", "arr": "15:00", "type": "G"},
        {"no": "G805", "dur": 355, "dep": "11:00", "arr": "16:55", "type": "G"},
    ],
}

SEAT_PRICE_BASE = {
    ("北京南", "上海虹桥"): {"商务座": 1748, "一等座": 933, "二等座": 553},
    ("北京西", "广州南"):   {"商务座": 2630, "一等座": 1283, "二等座": 864},
    ("北京西", "武汉"):     {"商务座": 1180, "一等座": 580, "二等座": 349},
    ("武汉", "广州南"):     {"商务座": 1180, "一等座": 590, "二等座": 354},
    ("北京西", "郑州东"):   {"商务座": 690, "一等座": 339, "二等座": 204},
    ("郑州东", "广州南"):   {"商务座": 1630, "一等座": 798, "二等座": 479},
    ("北京南", "南京南"):   {"商务座": 720, "一等座": 352, "二等座": 212},
    ("南京南", "上海虹桥"): {"商务座": 260, "一等座": 128, "二等座": 77},
    ("长沙南", "广州南"):   {"商务座": 650, "一等座": 318, "二等座": 191},
    ("北京西", "长沙南"):   {"商务座": 1630, "一等座": 798, "二等座": 479},
}

def _make_mock_seats(from_s: str, to_s: str) -> List[SeatInfo]:
    base = SEAT_PRICE_BASE.get((from_s, to_s), {
        "商务座": 1200, "一等座": 600, "二等座": 360
    })
    statuses = [TicketStatus.AVAILABLE, TicketStatus.AVAILABLE, TicketStatus.FEW_LEFT, TicketStatus.SOLD_OUT]
    seats = []
    for seat, price in base.items():
        st = random.choice(statuses)
        seats.append(SeatInfo(
            seat_type=seat,
            status=st,
            count=random.randint(1, 20) if st != TicketStatus.SOLD_OUT else 0,
            price=round(price * random.uniform(0.95, 1.05), 0)
        ))
    return seats

def _get_mock_trains(from_s: str, to_s: str, date: str) -> List[TrainTicket]:
    key = (from_s, to_s)
    templates = MOCK_TRAINS.get(key, [])
    if not templates:
        # 反向查找再尝试
        for (a, b), trains in MOCK_TRAINS.items():
            if (a in from_s or from_s in a) and (b in to_s or to_s in b):
                templates = trains
                break
    if not templates:
        # 生成通用 mock
        templates = [
            {"no": f"G{random.randint(100,999)}", "dur": random.randint(120, 600),
             "dep": f"{random.randint(6,18):02d}:00", "arr": "", "type": "G"}
        ]

    result = []
    for t in templates:
        dep_h, dep_m = map(int, t["dep"].split(":"))
        total_m = dep_h * 60 + dep_m + t["dur"]
        arr_h, arr_m = total_m // 60 % 24, total_m % 60
        cross = total_m // 60 >= 24
        arr = t.get("arr") or f"{arr_h:02d}:{arr_m:02d}"

        result.append(TrainTicket(
            train_no=t["no"],
            train_type=t["type"],
            from_station=from_s,
            to_station=to_s,
            depart_time=t["dep"],
            arrive_time=arr,
            duration=f"{t['dur']//60:02d}:{t['dur']%60:02d}",
            duration_minutes=t["dur"],
            is_cross_day=cross,
            seats=_make_mock_seats(from_s, to_s),
            data_source="mock"
        ))
    return result

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
    trains = _get_mock_trains(from_station, to_station, date)

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
        "from_station": from_station,
        "to_station": to_station,
        "date": date,
        "total": len(output),
        "trains": output,
    }
    logger.info(f"[Skill:TicketQuery] 返回 {len(output)} 趟列车")
    return json.dumps(result, ensure_ascii=False, indent=2)
