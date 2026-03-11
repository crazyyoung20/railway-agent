"""
模块二：12306 余票查询工具层
功能：真实 API 查询 → 超时/封禁 → 模拟数据 fallback
依赖：requests, pydantic, python-dateutil（pip install python-dateutil --break-system-packages）
运行方式：python ticket_query.py
"""

import os
import re
import time
import random
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger("ticket_query")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

# ══════════════════════════════════════════════════════════════════════════════
# 1. 数据结构定义
# ══════════════════════════════════════════════════════════════════════════════

class SeatType(str, Enum):
    BUSINESS    = "商务座"
    FIRST_CLASS = "一等座"
    SECOND_CLASS = "二等座"
    HARD_SLEEPER = "硬卧"
    SOFT_SLEEPER = "软卧"
    HARD_SEAT    = "硬座"
    NO_SEAT      = "无座"


class TicketStatus(str, Enum):
    AVAILABLE = "有票"
    SOLD_OUT  = "无票"
    FEW_LEFT  = "紧张"    # ≤3 张
    NOT_OPEN  = "未开售"
    CHECKING  = "候补"


class SeatInfo(BaseModel):
    seat_type: str
    status: TicketStatus
    count: Optional[int] = Field(None, description="剩余张数，None 表示未知")
    price: Optional[float] = Field(None, description="票价（元）")


class TrainTicket(BaseModel):
    """单个车次的票务信息"""
    train_no: str               = Field(..., description="车次号，如 G101")
    train_type: str             = Field(..., description="列车类型：G/D/C/Z/T/K")
    origin_station: str         = Field(..., description="始发站名")
    destination_station: str    = Field(..., description="终到站名")
    from_station: str           = Field(..., description="上车站名")
    to_station: str             = Field(..., description="下车站名")
    depart_time: str            = Field(..., description="出发时间 HH:MM")
    arrive_time: str            = Field(..., description="到达时间 HH:MM")
    duration: str               = Field(..., description="历时，如 04:38")
    duration_minutes: int       = Field(..., description="历时（分钟）")
    date: str                   = Field(..., description="出发日期 YYYY-MM-DD")
    seats: List[SeatInfo]       = Field(default_factory=list)
    is_cross_day: bool          = Field(False, description="是否跨天到达")
    from_data_source: str       = Field("real", description="real / mock / cache")

    def has_available_seat(self, seat_type: Optional[str] = None) -> bool:
        for s in self.seats:
            if seat_type and s.seat_type != seat_type:
                continue
            if s.status in (TicketStatus.AVAILABLE, TicketStatus.FEW_LEFT):
                return True
        return False

    def get_seat(self, seat_type: str) -> Optional[SeatInfo]:
        for s in self.seats:
            if s.seat_type == seat_type:
                return s
        return None


class QueryResult(BaseModel):
    """查询结果容器"""
    success: bool
    trains: List[TrainTicket]   = Field(default_factory=list)
    from_station: str           = ""
    to_station: str             = ""
    date: str                   = ""
    data_source: str            = "real"   # real / mock / error
    error_msg: Optional[str]    = None
    query_time: str             = Field(default_factory=lambda: datetime.now().isoformat())
    total_count: int            = 0

    def model_post_init(self, __context):
        self.total_count = len(self.trains)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 站名 ↔ 12306 电报码映射
# ══════════════════════════════════════════════════════════════════════════════

# 格式：站名 → (电报码, 拼音首字母)
STATION_CODE_MAP: Dict[str, Tuple[str, str]] = {
    # 北京
    "北京南": ("VNP", "bjn"),
    "北京":   ("VAP", "bj"),
    "北京西": ("VBP", "bjx"),
    "北京北": ("VOP", "bjb"),
    # 上海
    "上海虹桥": ("AOH", "shhq"),
    "上海":     ("SHH", "sh"),
    "上海南":   ("SNH", "shn"),
    # 广州
    "广州南":   ("IZQ", "gzn"),
    "广州":     ("GZQ", "gz"),
    "广州东":   ("GGQ", "gzd"),
    # 深圳
    "深圳北":   ("IOQ", "szb"),
    "深圳":     ("SZQ", "sz"),
    "深圳东":   ("EIQ", "szd"),
    # 杭州
    "杭州东":   ("HGH", "hzd"),
    "杭州":     ("HZH", "hz"),
    # 成都
    "成都东":   ("ICW", "cdd"),
    "成都":     ("CDW", "cd"),
    "成都西":   ("WCW", "cdx"),
    # 武汉
    "武汉":     ("WHN", "wh"),
    "武昌":     ("WCN", "wc"),
    "汉口":     ("HKN", "hk"),
    # 西安
    "西安北":   ("EAY", "xab"),
    "西安":     ("XAY", "xa"),
    # 南京
    "南京南":   ("NKH", "njn"),
    "南京":     ("NJH", "nj"),
    # 重庆
    "重庆北":   ("CQW", "cqb"),
    "重庆":     ("CQW", "cq"),
    "重庆西":   ("WIW", "cqx"),
    # 郑州
    "郑州东":   ("ZHF", "zzd"),
    "郑州":     ("ZZF", "zz"),
    # 长沙
    "长沙南":   ("CSQ", "csn"),
    "长沙":     ("CSQ", "cs"),
    # 天津
    "天津南":   ("TJP", "tjn"),
    "天津":     ("TJP", "tj"),
    # 济南
    "济南西":   ("JNK", "jnx"),
    "济南":     ("JNK", "jn"),
    # 哈尔滨
    "哈尔滨西": ("HBB", "hebx"),
    "哈尔滨":   ("HRB", "heb"),
    # 沈阳
    "沈阳北":   ("SNB", "syb"),
    "沈阳":     ("SYB", "sy"),
    # 昆明
    "昆明南":   ("KMM", "kmn"),
    "昆明":     ("KMM", "km"),
    # 其他主要站
    "合肥南":   ("HFH", "hfn"),
    "合肥":     ("HFH", "hf"),
    "苏州北":   ("SZH", "szb2"),
    "苏州":     ("SUH", "su"),
    "南昌西":   ("NCG", "ncx"),
    "南昌":     ("NCG", "nc"),
    "贵阳北":   ("GYW", "gyb"),
    "贵阳":     ("GYW", "gy"),
    "福州":     ("FZS", "fz"),
    "福州南":   ("FNS", "fzn"),
    "厦门北":   ("XMH", "xmb"),
    "厦门":     ("XMH", "xm"),
    "青岛":     ("QDK", "qd"),
    "青岛北":   ("QBK", "qdb"),
    "太原南":   ("TYV", "tyn"),
    "石家庄":   ("SJP", "sjz"),
    "石家庄北": ("SKP", "sjzb"),
    "兰州西":   ("LZJ", "lzx"),
    "兰州":     ("LZJ", "lz"),
    "乌鲁木齐": ("WMR", "wlmq"),
    "拉萨":     ("LSO", "ls"),
}

# 反向映射：电报码 → 站名
CODE_STATION_MAP: Dict[str, str] = {v[0]: k for k, v in STATION_CODE_MAP.items()}


def get_station_code(station_name: str) -> Optional[str]:
    """站名 → 电报码"""
    info = STATION_CODE_MAP.get(station_name)
    if info:
        return info[0]
    # 模糊匹配
    for name, (code, _) in STATION_CODE_MAP.items():
        if station_name in name or name in station_name:
            logger.warning(f"[StationCode] 模糊匹配: '{station_name}' → '{name}' ({code})")
            return code
    logger.error(f"[StationCode] 未找到站名对应电报码: '{station_name}'")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. 模拟数据生成器
# ══════════════════════════════════════════════════════════════════════════════

# 主要线路的真实车次模板（用于生成仿真数据）
ROUTE_TEMPLATES = {
    ("VNP", "AOH"): {  # 北京南→上海虹桥
        "trains": [
            {"no": "G1",   "dep": "09:00", "arr": "13:48", "dur": 288},
            {"no": "G5",   "dep": "10:00", "arr": "14:28", "dur": 268},
            {"no": "G11",  "dep": "11:00", "arr": "15:28", "dur": 268},
            {"no": "G13",  "dep": "12:00", "arr": "16:38", "dur": 278},
            {"no": "G15",  "dep": "13:00", "arr": "17:28", "dur": 268},
            {"no": "G21",  "dep": "15:00", "arr": "19:28", "dur": 268},
            {"no": "G31",  "dep": "17:00", "arr": "21:38", "dur": 278},
            {"no": "G41",  "dep": "18:00", "arr": "22:38", "dur": 278},
        ],
        "prices": {"商务座": 1748, "一等座": 933, "二等座": 553},
    },
    ("AOH", "VNP"): {  # 上海虹桥→北京南
        "trains": [
            {"no": "G2",   "dep": "08:00", "arr": "12:38", "dur": 278},
            {"no": "G6",   "dep": "09:00", "arr": "13:28", "dur": 268},
            {"no": "G12",  "dep": "10:00", "arr": "14:28", "dur": 268},
            {"no": "G22",  "dep": "14:00", "arr": "18:28", "dur": 268},
            {"no": "G32",  "dep": "16:00", "arr": "20:38", "dur": 278},
            {"no": "G42",  "dep": "18:00", "arr": "22:38", "dur": 278},
        ],
        "prices": {"商务座": 1748, "一等座": 933, "二等座": 553},
    },
    ("VNP", "IZQ"): {  # 北京南→广州南
        "trains": [
            {"no": "G71",  "dep": "08:00", "arr": "16:18", "dur": 498},
            {"no": "G79",  "dep": "09:00", "arr": "17:28", "dur": 508},
            {"no": "G811", "dep": "10:00", "arr": "18:28", "dur": 508},
            {"no": "G821", "dep": "13:00", "arr": "21:28", "dur": 508},
        ],
        "prices": {"商务座": 3283, "一等座": 1953, "二等座": 862},
    },
    ("IZQ", "VNP"): {  # 广州南→北京南
        "trains": [
            {"no": "G72",  "dep": "08:00", "arr": "16:18", "dur": 498},
            {"no": "G80",  "dep": "10:00", "arr": "18:28", "dur": 508},
            {"no": "G812", "dep": "13:00", "arr": "21:28", "dur": 508},
        ],
        "prices": {"商务座": 3283, "一等座": 1953, "二等座": 862},
    },
    ("VNP", "WHN"): {  # 北京南→武汉
        "trains": [
            {"no": "G507", "dep": "07:48", "arr": "11:37", "dur": 229},
            {"no": "G511", "dep": "09:08", "arr": "13:07", "dur": 239},
            {"no": "G517", "dep": "11:08", "arr": "15:07", "dur": 239},
            {"no": "G521", "dep": "13:08", "arr": "17:07", "dur": 239},
            {"no": "G527", "dep": "15:08", "arr": "19:07", "dur": 239},
        ],
        "prices": {"商务座": 1473, "一等座": 670, "二等座": 418},
    },
    ("WHN", "IZQ"): {  # 武汉→广州南
        "trains": [
            {"no": "G841", "dep": "08:15", "arr": "11:58", "dur": 223},
            {"no": "G845", "dep": "10:15", "arr": "13:58", "dur": 223},
            {"no": "G851", "dep": "12:15", "arr": "15:58", "dur": 223},
            {"no": "G857", "dep": "14:15", "arr": "17:58", "dur": 223},
            {"no": "G861", "dep": "16:15", "arr": "19:58", "dur": 223},
            {"no": "G865", "dep": "18:15", "arr": "21:58", "dur": 223},
        ],
        "prices": {"商务座": 1236, "一等座": 618, "二等座": 346},
    },
    ("IZQ", "WHN"): {  # 广州南→武汉
        "trains": [
            {"no": "G842", "dep": "08:00", "arr": "11:43", "dur": 223},
            {"no": "G846", "dep": "10:00", "arr": "13:43", "dur": 223},
            {"no": "G852", "dep": "12:00", "arr": "15:43", "dur": 223},
            {"no": "G858", "dep": "14:00", "arr": "17:43", "dur": 223},
        ],
        "prices": {"商务座": 1236, "一等座": 618, "二等座": 346},
    },
    ("ICW", "VNP"): {  # 成都东→北京
        "trains": [
            {"no": "G308", "dep": "08:00", "arr": "14:28", "dur": 388},
            {"no": "G310", "dep": "10:00", "arr": "16:28", "dur": 388},
            {"no": "G316", "dep": "13:00", "arr": "19:28", "dur": 388},
        ],
        "prices": {"商务座": 2804, "一等座": 1482, "二等座": 928},
    },
    ("VNP", "ICW"): {  # 北京南→成都东
        "trains": [
            {"no": "G307", "dep": "08:00", "arr": "14:28", "dur": 388},
            {"no": "G309", "dep": "10:00", "arr": "16:28", "dur": 388},
            {"no": "G315", "dep": "13:00", "arr": "19:28", "dur": 388},
        ],
        "prices": {"商务座": 2804, "一等座": 1482, "二等座": 928},
    },
}

# 睡铺车次模板（无高铁时用普速）
SLEEPER_TEMPLATES = {
    ("VAP", "CDW"): {  # 北京→成都（普速/动卧）
        "trains": [
            {"no": "Z7",  "dep": "21:12", "arr": "14:57+1", "dur": 1065, "cross_day": True},
            {"no": "T7",  "dep": "16:44", "arr": "20:00+1", "dur": 1636, "cross_day": True},
            {"no": "K17", "dep": "18:06", "arr": "07:44+2", "dur": 2258, "cross_day": True},
        ],
        "prices": {"软卧": 410, "硬卧": 248, "硬座": 138},
    },
}


def _get_train_type(train_no: str) -> str:
    if not train_no:
        return "未知"
    prefix = train_no[0].upper()
    type_map = {"G": "高铁", "D": "动车", "C": "城际", "Z": "直达特快",
                "T": "特快", "K": "快速", "Y": "旅游"}
    return type_map.get(prefix, "普通")


def _generate_seat_availability(prices: Dict[str, float], force_sold_out: bool = False) -> List[SeatInfo]:
    """随机生成座位余票状态（模拟真实购票分布）"""
    seats = []
    for seat_type, price in prices.items():
        if force_sold_out:
            status = TicketStatus.SOLD_OUT
            count = 0
        else:
            r = random.random()
            if r < 0.25:
                status = TicketStatus.SOLD_OUT
                count = 0
            elif r < 0.40:
                count = random.randint(1, 3)
                status = TicketStatus.FEW_LEFT
            else:
                count = random.randint(5, 50)
                status = TicketStatus.AVAILABLE
        seats.append(SeatInfo(
            seat_type=seat_type,
            status=status,
            count=count if not force_sold_out else 0,
            price=price
        ))
    return seats


def generate_mock_tickets(
    from_station: str,
    to_station: str,
    date: str,
    from_code: str,
    to_code: str,
) -> List[TrainTicket]:
    """
    生成模拟票务数据
    优先查路由模板，找不到则生成通用模拟数据
    """
    key = (from_code, to_code)
    rev_key = (to_code, from_code)
    template = ROUTE_TEMPLATES.get(key) or SLEEPER_TEMPLATES.get(key)

    tickets = []

    if template:
        logger.info(f"[MockData] 使用路由模板 {key}，共 {len(template['trains'])} 个车次")
        for t in template["trains"]:
            cross_day = t.get("cross_day", False)
            # 随机让部分车次无票（模拟高峰期）
            is_peak = random.random() < 0.3
            seats = _generate_seat_availability(
                template["prices"],
                force_sold_out=False
            )
            if is_peak:
                # 高峰期：随机让某些座位卖完
                for s in seats:
                    if random.random() < 0.5:
                        s.status = TicketStatus.SOLD_OUT
                        s.count = 0

            tickets.append(TrainTicket(
                train_no=t["no"],
                train_type=_get_train_type(t["no"]),
                origin_station=from_station,
                destination_station=to_station,
                from_station=from_station,
                to_station=to_station,
                depart_time=t["dep"].replace("+1", ""),
                arrive_time=t["arr"].replace("+1", "").replace("+2", ""),
                duration=f"{t['dur']//60:02d}:{t['dur']%60:02d}",
                duration_minutes=t["dur"],
                date=date,
                seats=seats,
                is_cross_day=cross_day,
                from_data_source="mock",
            ))
    else:
        # 通用生成：根据距离估算时间
        logger.info(f"[MockData] 无路由模板 {key}，生成通用模拟数据")
        base_hour = random.randint(7, 9)
        for i in range(random.randint(3, 6)):
            dep_hour = base_hour + i * 2
            dep_min = random.choice([0, 8, 15, 30, 45])
            dur_min = random.randint(120, 480)
            arr_total = dep_hour * 60 + dep_min + dur_min
            arr_hour = (arr_total // 60) % 24
            arr_min = arr_total % 60
            cross = arr_total >= 24 * 60

            train_prefix = random.choice(["G", "G", "G", "D", "D"])
            train_no = f"{train_prefix}{random.randint(100, 999)}"
            prices = {"一等座": round(dur_min * 0.8, 0), "二等座": round(dur_min * 0.5, 0)}
            if train_prefix == "G" and dur_min > 180:
                prices["商务座"] = round(dur_min * 1.5, 0)

            tickets.append(TrainTicket(
                train_no=train_no,
                train_type=_get_train_type(train_no),
                origin_station=from_station,
                destination_station=to_station,
                from_station=from_station,
                to_station=to_station,
                depart_time=f"{dep_hour:02d}:{dep_min:02d}",
                arrive_time=f"{arr_hour:02d}:{arr_min:02d}",
                duration=f"{dur_min//60:02d}:{dur_min%60:02d}",
                duration_minutes=dur_min,
                date=date,
                seats=_generate_seat_availability(prices),
                is_cross_day=cross,
                from_data_source="mock",
            ))

    logger.info(f"[MockData] 生成 {len(tickets)} 个车次")
    return tickets


# ══════════════════════════════════════════════════════════════════════════════
# 4. 12306 真实 API 查询
# ══════════════════════════════════════════════════════════════════════════════

# 12306 座位类型编码（API 返回字段顺序固定）
SEAT_INDEX_MAP = {
    # index in the split array → seat_type
    # 参考：https://kyfw.12306.cn 返回字段定义
    3:  "商务座",
    4:  "特等座",
    5:  "一等座",
    6:  "二等座",
    7:  "高级软卧",
    10: "软卧",
    11: "动卧",
    28: "硬卧",
    29: "软座",
    30: "硬座",
    26: "无座",
    21: "餐车",
}

# 12306 API 请求头（模拟浏览器）
HEADERS_12306 = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://kyfw.12306.cn/otn/leftTicket/init",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cookie": "",  # 生产环境需填入有效 cookie
}


def _parse_12306_response(raw_data: dict, from_station: str, to_station: str, date: str) -> List[TrainTicket]:
    """解析 12306 API 原始响应"""
    tickets = []
    try:
        data_list = raw_data.get("data", {}).get("result", [])
        for row in data_list:
            fields = row.split("|")
            if len(fields) < 35:
                continue

            # 关键字段提取
            train_no     = fields[3]
            from_code    = fields[6]
            to_code      = fields[7]
            depart_time  = fields[8]
            arrive_time  = fields[9]
            duration     = fields[10]

            # 解析历时
            dur_parts = duration.split(":")
            dur_min = int(dur_parts[0]) * 60 + int(dur_parts[1]) if len(dur_parts) == 2 else 0

            # 解析座位余票
            seats = []
            for idx, seat_name in SEAT_INDEX_MAP.items():
                if idx < len(fields):
                    val = fields[idx].strip()
                    if val == "" or val == "无":
                        continue
                    if val == "有":
                        status = TicketStatus.AVAILABLE
                        count = None
                    elif val == "*":
                        status = TicketStatus.NOT_OPEN
                        count = None
                    elif val == "候补":
                        status = TicketStatus.CHECKING
                        count = None
                    elif val.isdigit():
                        count = int(val)
                        status = TicketStatus.FEW_LEFT if count <= 3 else TicketStatus.AVAILABLE
                    else:
                        status = TicketStatus.SOLD_OUT
                        count = 0

                    seats.append(SeatInfo(seat_type=seat_name, status=status, count=count))

            cross_day = "1" in fields[11] if len(fields) > 11 else False

            from_name = CODE_STATION_MAP.get(from_code, from_station)
            to_name   = CODE_STATION_MAP.get(to_code, to_station)

            tickets.append(TrainTicket(
                train_no=train_no,
                train_type=_get_train_type(train_no),
                origin_station=from_name,
                destination_station=to_name,
                from_station=from_station,
                to_station=to_station,
                depart_time=depart_time,
                arrive_time=arrive_time,
                duration=duration,
                duration_minutes=dur_min,
                date=date,
                seats=seats,
                is_cross_day=cross_day,
                from_data_source="real",
            ))
    except Exception as e:
        logger.error(f"[12306Parser] 解析响应异常: {e}")

    return tickets


# ══════════════════════════════════════════════════════════════════════════════
# 5. 主查询类（含降级逻辑）
# ══════════════════════════════════════════════════════════════════════════════

class TicketQueryTool:
    """
    12306 余票查询工具
    降级策略：真实 API → 超时/封禁 → 模拟数据
    """

    API_URL = "https://kyfw.12306.cn/otn/leftTicket/query"
    TIMEOUT = 8         # 秒
    MAX_RETRIES = 2

    def __init__(self, use_real_api: bool = True):
        self.use_real_api = use_real_api
        try:
            import requests
            self._requests = requests
            logger.info("[TicketQuery] requests 库加载成功")
        except ImportError:
            self._requests = None
            logger.warning("[TicketQuery] requests 未安装，将使用模拟数据")

    def query(
        self,
        from_station: str,
        to_station: str,
        date: str,
        train_filter: Optional[str] = None,   # 筛选车次类型，如 "G" 只看高铁
    ) -> QueryResult:
        """
        查询指定区间、日期的余票
        
        Args:
            from_station: 出发站名（标准站名，如"北京南"）
            to_station:   到达站名
            date:         出发日期 YYYY-MM-DD
            train_filter: 可选，车次前缀过滤（G/D/C/Z/T/K）
        """
        logger.info(f"[TicketQuery] 查询: {from_station} → {to_station}  日期: {date}")

        # 获取电报码
        from_code = get_station_code(from_station)
        to_code   = get_station_code(to_station)

        if not from_code or not to_code:
            return QueryResult(
                success=False,
                from_station=from_station,
                to_station=to_station,
                date=date,
                data_source="error",
                error_msg=f"站名无法转换为电报码: from={from_station}({from_code}), to={to_station}({to_code})"
            )

        # 尝试真实 API
        if self.use_real_api and self._requests:
            result = self._query_real_api(from_station, to_station, date, from_code, to_code)
            if result.success:
                if train_filter:
                    result.trains = [t for t in result.trains if t.train_no.startswith(train_filter)]
                return result
            else:
                logger.warning(f"[TicketQuery] 真实API失败({result.error_msg})，切换模拟数据")

        # Fallback：模拟数据
        return self._query_mock(from_station, to_station, date, from_code, to_code, train_filter)

    def _query_real_api(
        self,
        from_station: str,
        to_station: str,
        date: str,
        from_code: str,
        to_code: str,
    ) -> QueryResult:
        """调用 12306 真实接口"""
        params = {
            "leftTicketDTO.train_date":     date,
            "leftTicketDTO.from_station":   from_code,
            "leftTicketDTO.to_station":     to_code,
            "purpose_codes":                "ADULT",
        }

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(f"[12306API] 第{attempt}次请求: {from_code}→{to_code} {date}")
                resp = self._requests.get(
                    self.API_URL,
                    params=params,
                    headers=HEADERS_12306,
                    timeout=self.TIMEOUT,
                    verify=True,
                )
                logger.info(f"[12306API] HTTP状态: {resp.status_code}")

                if resp.status_code == 200:
                    data = resp.json()
                    # 检查是否被反爬拦截
                    if "verifySMS" in resp.text or "verify" in str(data).lower():
                        logger.warning("[12306API] 触发验证码拦截")
                        return QueryResult(
                            success=False, data_source="error",
                            error_msg="12306 验证码拦截",
                            from_station=from_station, to_station=to_station, date=date
                        )

                    tickets = _parse_12306_response(data, from_station, to_station, date)
                    logger.info(f"[12306API] 解析到 {len(tickets)} 个车次")
                    return QueryResult(
                        success=True,
                        trains=tickets,
                        from_station=from_station,
                        to_station=to_station,
                        date=date,
                        data_source="real",
                    )

                elif resp.status_code in (429, 503):
                    logger.warning(f"[12306API] 被限流 HTTP {resp.status_code}，等待重试")
                    time.sleep(2 ** attempt)
                else:
                    return QueryResult(
                        success=False, data_source="error",
                        error_msg=f"HTTP {resp.status_code}",
                        from_station=from_station, to_station=to_station, date=date
                    )

            except Exception as e:
                err_type = type(e).__name__
                logger.warning(f"[12306API] 第{attempt}次请求异常 [{err_type}]: {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(1.5)

        return QueryResult(
            success=False, data_source="error",
            error_msg="12306 接口连接失败（超时/网络不通）",
            from_station=from_station, to_station=to_station, date=date
        )

    def _query_mock(
        self,
        from_station: str,
        to_station: str,
        date: str,
        from_code: str,
        to_code: str,
        train_filter: Optional[str] = None,
    ) -> QueryResult:
        """生成模拟票务数据"""
        logger.info(f"[TicketQuery] 使用模拟数据: {from_station}({from_code}) → {to_station}({to_code})")
        tickets = generate_mock_tickets(from_station, to_station, date, from_code, to_code)

        if train_filter:
            tickets = [t for t in tickets if t.train_no.startswith(train_filter)]

        return QueryResult(
            success=True,
            trains=tickets,
            from_station=from_station,
            to_station=to_station,
            date=date,
            data_source="mock",
        )

    def query_transfer(
        self,
        from_station: str,
        mid_station: str,
        to_station: str,
        date: str,
        min_transfer_minutes: int = 30,
    ) -> List[Tuple[TrainTicket, TrainTicket]]:
        """
        查询中转方案（第一程+第二程的组合）
        
        Args:
            min_transfer_minutes: 中转最短等待时间（分钟），避免衔接过紧
        """
        logger.info(f"[TicketQuery] 中转查询: {from_station} → {mid_station} → {to_station}  {date}")

        # 查第一程
        leg1_result = self.query(from_station, mid_station, date)
        if not leg1_result.success or not leg1_result.trains:
            logger.warning(f"[TicketQuery] 第一程无车次: {from_station}→{mid_station}")
            return []

        valid_pairs = []
        for train1 in leg1_result.trains:
            if not train1.has_available_seat():
                continue

            # 计算最早可接驳时间
            arr_parts = train1.arrive_time.split(":")
            if len(arr_parts) != 2:
                continue
            arr_minutes = int(arr_parts[0]) * 60 + int(arr_parts[1])
            if train1.is_cross_day:
                # 跨天到达，第二天查询
                next_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                leg2_result = self.query(mid_station, to_station, next_date)
            else:
                leg2_result = self.query(mid_station, to_station, date)

            for train2 in leg2_result.trains:
                if not train2.has_available_seat():
                    continue

                dep2_parts = train2.depart_time.split(":")
                if len(dep2_parts) != 2:
                    continue
                dep2_minutes = int(dep2_parts[0]) * 60 + int(dep2_parts[1])

                # 检查换乘时间是否充足
                wait_minutes = dep2_minutes - arr_minutes
                if train1.is_cross_day:
                    wait_minutes += 24 * 60

                if wait_minutes >= min_transfer_minutes:
                    valid_pairs.append((train1, train2))
                    logger.debug(
                        f"[TicketQuery] 有效中转: {train1.train_no}({train1.arrive_time}) "
                        f"→ 等{wait_minutes}分 → {train2.train_no}({train2.depart_time})"
                    )

        logger.info(f"[TicketQuery] 找到 {len(valid_pairs)} 个有效中转组合")
        return valid_pairs


# ══════════════════════════════════════════════════════════════════════════════
# 6. 格式化输出
# ══════════════════════════════════════════════════════════════════════════════

def format_query_result(result: QueryResult) -> str:
    lines = []
    lines.append(f"\n{'='*65}")
    lines.append(f"  {result.from_station} → {result.to_station}   {result.date}")
    lines.append(f"  数据来源: {'⚠️  模拟数据' if result.data_source == 'mock' else '✅ 真实数据'}")
    lines.append(f"  共 {result.total_count} 个车次")
    lines.append(f"{'='*65}")

    if not result.success:
        lines.append(f"  ❌ 查询失败: {result.error_msg}")
        return "\n".join(lines)

    for t in result.trains:
        cross = " (+1天)" if t.is_cross_day else ""
        lines.append(
            f"\n  {t.train_no:<8} {t.train_type:<4}  "
            f"{t.depart_time} → {t.arrive_time}{cross}  历时{t.duration}"
        )
        for s in t.seats:
            icon = {"有票": "✅", "紧张": "⚡", "无票": "❌", "未开售": "⏳", "候补": "🔄"}.get(s.status.value, "?")
            count_str = f"({s.count}张)" if s.count is not None and s.count > 0 else ""
            price_str = f"¥{s.price:.0f}" if s.price else ""
            lines.append(f"    {icon} {s.seat_type:<6} {s.status.value:<4} {count_str:<6} {price_str}")

    lines.append(f"\n{'='*65}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 7. 测试入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from datetime import datetime, timedelta
    tool = TicketQueryTool(use_real_api=True)  # 真实API失败会自动fallback

    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

    print("\n🚄 模块二：12306 余票查询工具 - 测试")

    # 测试1：直接查询（有路由模板）
    print("\n【测试1】北京南 → 上海虹桥")
    r1 = tool.query("北京南", "上海虹桥", tomorrow)
    print(format_query_result(r1))

    # 测试2：有路由模板的另一条线
    print("\n【测试2】武汉 → 广州南")
    r2 = tool.query("武汉", "广州南", tomorrow)
    print(format_query_result(r2))

    # 测试3：无路由模板（通用生成）
    print("\n【测试3】南京南 → 西安北（通用生成）")
    r3 = tool.query("南京南", "西安北", next_week)
    print(format_query_result(r3))

    # 测试4：中转组合查询
    print("\n【测试4】中转查询: 北京南 → 武汉 → 广州南")
    pairs = tool.query_transfer("北京南", "武汉", "广州南", tomorrow, min_transfer_minutes=45)
    print(f"\n  找到 {len(pairs)} 个有效中转组合，展示前3个：")
    for i, (t1, t2) in enumerate(pairs[:3]):
        t1_seat = next((s for s in t1.seats if s.status != TicketStatus.SOLD_OUT), None)
        t2_seat = next((s for s in t2.seats if s.status != TicketStatus.SOLD_OUT), None)
        seat1_str = f"{t1_seat.seat_type}" if t1_seat else "无票"
        seat2_str = f"{t2_seat.seat_type}" if t2_seat else "无票"
        print(
            f"\n  方案{i+1}: {t1.train_no}({t1.depart_time}→{t1.arrive_time}) "
            f"[{seat1_str}] + {t2.train_no}({t2.depart_time}→{t2.arrive_time}) [{seat2_str}]"
        )

    # 测试5：站名不存在（错误处理）
    print("\n【测试5】站名不存在（错误处理）")
    r5 = tool.query("火星站", "月球站", tomorrow)
    print(f"  success={r5.success}, error={r5.error_msg}")