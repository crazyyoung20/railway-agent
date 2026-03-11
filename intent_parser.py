""" 
模块一：站名标准化 + 日期解析 + 意图槽位抽取
技术栈：ZhipuAI GLM-4-flash + Pydantic + Python 3.9
运行方式：python intent_parser.py
"""

import os
import re
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, Field, validator

# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("intent_parser")

# ── 加载环境变量（兼容 .env 文件）────────────────────────────────────────────
def load_dotenv(path=".env"):
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

load_dotenv()

# ── ZhipuAI 初始化 ────────────────────────────────────────────────────────────
try:
    from zhipuai import ZhipuAI
    zhipu_client = ZhipuAI(api_key=os.environ.get("ZHIPUAI_API_KEY", ""))
    logger.info("ZhipuAI 客户端初始化成功")
except Exception as e:
    zhipu_client = None
    logger.warning(f"ZhipuAI 初始化失败，将使用 Mock 模式：{e}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Pydantic 数据结构定义
# ══════════════════════════════════════════════════════════════════════════════

class StationCandidate(BaseModel):
    """单个候选站点"""
    station_name: str = Field(..., description="标准站名，如 北京南")
    station_code: str = Field(..., description="12306 站点代码，如 VNP")
    city: str = Field(..., description="所属城市")
    priority: int = Field(..., description="优先级，1最高")


class ParsedStation(BaseModel):
    """站名解析结果"""
    raw_input: str = Field(..., description="用户原始输入")
    candidates: List[StationCandidate] = Field(default_factory=list)
    is_ambiguous: bool = Field(False, description="是否有歧义（多个候选）")
    primary: Optional[StationCandidate] = Field(None, description="首选站点")


class ParsedDate(BaseModel):
    """日期解析结果"""
    raw_input: str = Field(..., description="用户原始输入")
    date_str: str = Field(..., description="标准化日期 YYYY-MM-DD")
    is_relative: bool = Field(False, description="是否为相对日期表达")
    confidence: float = Field(1.0, description="解析置信度 0~1")


class PassengerType(BaseModel):
    """乘客信息"""
    type: str = Field("adult", description="adult/student/child/senior/disability")
    count: int = Field(1, description="人数")


class TravelPreference(BaseModel):
    """出行偏好"""
    seat_type: Optional[str] = Field(None, description="座位类型：商务座/一等座/二等座/硬卧/软卧")
    priority: Optional[str] = Field(None, description="优先目标：最快/最稳/最便宜/最舒适")
    max_transfer: int = Field(2, description="最大中转次数")
    earliest_depart: Optional[str] = Field(None, description="最早出发时间 HH:MM")
    latest_arrive: Optional[str] = Field(None, description="最晚到达时间 HH:MM")


class IntentSlots(BaseModel):
    """完整意图槽位"""
    origin: Optional[ParsedStation] = Field(None, description="出发站解析结果")
    destination: Optional[ParsedStation] = Field(None, description="到达站解析结果")
    travel_date: Optional[ParsedDate] = Field(None, description="出行日期")
    return_date: Optional[ParsedDate] = Field(None, description="返程日期（可选）")
    passengers: List[PassengerType] = Field(default_factory=lambda: [PassengerType()])
    preference: TravelPreference = Field(default_factory=TravelPreference)
    raw_query: str = Field("", description="原始用户输入")
    parse_success: bool = Field(True)
    error_msg: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# 2. 站名标准化模块
# ══════════════════════════════════════════════════════════════════════════════

# 主要城市站点知识库（覆盖高频歧义城市）
STATION_DB = {
    # 北京
    "北京": [
        StationCandidate(station_name="北京南", station_code="VNP", city="北京", priority=1),
        StationCandidate(station_name="北京", station_code="VAP", city="北京", priority=2),
        StationCandidate(station_name="北京西", station_code="VBP", city="北京", priority=3),
        StationCandidate(station_name="北京北", station_code="VOP", city="北京", priority=4),
    ],
    "北京南": [StationCandidate(station_name="北京南", station_code="VNP", city="北京", priority=1)],
    "北京西": [StationCandidate(station_name="北京西", station_code="VBP", city="北京", priority=1)],
    "北京北": [StationCandidate(station_name="北京北", station_code="VOP", city="北京", priority=1)],

    # 上海
    "上海": [
        StationCandidate(station_name="上海虹桥", station_code="AOH", city="上海", priority=1),
        StationCandidate(station_name="上海", station_code="SHH", city="上海", priority=2),
    ],
    "上海虹桥": [StationCandidate(station_name="上海虹桥", station_code="AOH", city="上海", priority=1)],
    "上海站": [StationCandidate(station_name="上海", station_code="SHH", city="上海", priority=1)],

    # 广州
    "广州": [
        StationCandidate(station_name="广州南", station_code="IZQ", city="广州", priority=1),
        StationCandidate(station_name="广州", station_code="GZQ", city="广州", priority=2),
        StationCandidate(station_name="广州东", station_code="GGQ", city="广州", priority=3),
    ],
    "广州南": [StationCandidate(station_name="广州南", station_code="IZQ", city="广州", priority=1)],
    "广州东": [StationCandidate(station_name="广州东", station_code="GGQ", city="广州", priority=1)],

    # 深圳
    "深圳": [
        StationCandidate(station_name="深圳北", station_code="IOQ", city="深圳", priority=1),
        StationCandidate(station_name="深圳", station_code="SZQ", city="深圳", priority=2),
    ],
    "深圳北": [StationCandidate(station_name="深圳北", station_code="IOQ", city="深圳", priority=1)],

    # 杭州
    "杭州": [
        StationCandidate(station_name="杭州东", station_code="HGH", city="杭州", priority=1),
        StationCandidate(station_name="杭州", station_code="HZH", city="杭州", priority=2),
    ],
    "杭州东": [StationCandidate(station_name="杭州东", station_code="HGH", city="杭州", priority=1)],

    # 成都
    "成都": [
        StationCandidate(station_name="成都东", station_code="ICW", city="成都", priority=1),
        StationCandidate(station_name="成都", station_code="CDW", city="成都", priority=2),
        StationCandidate(station_name="成都西", station_code="WCW", city="成都", priority=3),
    ],
    "成都东": [StationCandidate(station_name="成都东", station_code="ICW", city="成都", priority=1)],

    # 武汉
    "武汉": [
        StationCandidate(station_name="武汉", station_code="WHN", city="武汉", priority=1),
        StationCandidate(station_name="武昌", station_code="WCN", city="武汉", priority=2),
        StationCandidate(station_name="汉口", station_code="HKN", city="武汉", priority=3),
    ],

    # 西安
    "西安": [
        StationCandidate(station_name="西安北", station_code="EAY", city="西安", priority=1),
        StationCandidate(station_name="西安", station_code="XAY", city="西安", priority=2),
    ],
    "西安北": [StationCandidate(station_name="西安北", station_code="EAY", city="西安", priority=1)],

    # 南京
    "南京": [
        StationCandidate(station_name="南京南", station_code="NKH", city="南京", priority=1),
        StationCandidate(station_name="南京", station_code="NJH", city="南京", priority=2),
    ],
    "南京南": [StationCandidate(station_name="南京南", station_code="NKH", city="南京", priority=1)],

    # 重庆
    "重庆": [
        StationCandidate(station_name="重庆北", station_code="CQW", city="重庆", priority=1),
        StationCandidate(station_name="重庆", station_code="CQW", city="重庆", priority=2),
        StationCandidate(station_name="重庆西", station_code="WIW", city="重庆", priority=3),
    ],

    # 郑州
    "郑州": [
        StationCandidate(station_name="郑州东", station_code="ZHF", city="郑州", priority=1),
        StationCandidate(station_name="郑州", station_code="ZZF", city="郑州", priority=2),
    ],

    # 长沙
    "长沙": [
        StationCandidate(station_name="长沙南", station_code="CSQ", city="长沙", priority=1),
        StationCandidate(station_name="长沙", station_code="CSQ", city="长沙", priority=2),
    ],

    # 哈尔滨
    "哈尔滨": [
        StationCandidate(station_name="哈尔滨西", station_code="HBB", city="哈尔滨", priority=1),
        StationCandidate(station_name="哈尔滨", station_code="HRB", city="哈尔滨", priority=2),
    ],

    # 沈阳
    "沈阳": [
        StationCandidate(station_name="沈阳北", station_code="SNB", city="沈阳", priority=1),
        StationCandidate(station_name="沈阳", station_code="SYB", city="沈阳", priority=2),
    ],

    # 天津
    "天津": [
        StationCandidate(station_name="天津南", station_code="TJP", city="天津", priority=1),
        StationCandidate(station_name="天津", station_code="TJP", city="天津", priority=2),
    ],

    # 济南
    "济南": [
        StationCandidate(station_name="济南西", station_code="JNK", city="济南", priority=1),
        StationCandidate(station_name="济南", station_code="JNK", city="济南", priority=2),
    ],

    # 昆明
    "昆明": [
        StationCandidate(station_name="昆明南", station_code="KMM", city="昆明", priority=1),
        StationCandidate(station_name="昆明", station_code="KMM", city="昆明", priority=2),
    ],
}

# 别名映射
STATION_ALIAS = {
    "魔都": "上海",
    "帝都": "北京",
    "京": "北京",
    "沪": "上海",
    "穗": "广州",
    "蓉": "成都",
    "渝": "重庆",
    "汉": "武汉",
    "宁": "南京",
    "杭": "杭州",
    "深": "深圳",
    "津": "天津",
    "蓉城": "成都",
    "羊城": "广州",
    "花城": "广州",
    "春城": "昆明",
}


class StationNormalizer:
    """站名标准化器"""

    def normalize(self, raw: str) -> ParsedStation:
        """
        将用户输入的站名标准化为候选站点列表
        """
        raw = raw.strip()
        logger.info(f"[StationNormalizer] 输入: '{raw}'")

        # 1. 别名替换
        normalized_input = STATION_ALIAS.get(raw, raw)
        if normalized_input != raw:
            logger.info(f"[StationNormalizer] 别名替换: '{raw}' → '{normalized_input}'")

        # 2. 精确匹配
        if normalized_input in STATION_DB:
            candidates = STATION_DB[normalized_input]
            result = ParsedStation(
                raw_input=raw,
                candidates=candidates,
                is_ambiguous=len(candidates) > 1,
                primary=candidates[0]
            )
            logger.info(f"[StationNormalizer] 精确匹配: 候选={[c.station_name for c in candidates]}")
            return result

        # 3. 模糊匹配（后缀去除）
        for suffix in ["站", "火车站", "高铁站", "动车站"]:
            cleaned = normalized_input.replace(suffix, "")
            if cleaned in STATION_DB:
                candidates = STATION_DB[cleaned]
                result = ParsedStation(
                    raw_input=raw,
                    candidates=candidates,
                    is_ambiguous=len(candidates) > 1,
                    primary=candidates[0]
                )
                logger.info(f"[StationNormalizer] 去后缀匹配 '{suffix}': 候选={[c.station_name for c in candidates]}")
                return result

        # 4. 包含匹配（如输入"北京南站"）
        for key, candidates in STATION_DB.items():
            if key in normalized_input or normalized_input in key:
                result = ParsedStation(
                    raw_input=raw,
                    candidates=candidates,
                    is_ambiguous=len(candidates) > 1,
                    primary=candidates[0]
                )
                logger.info(f"[StationNormalizer] 包含匹配 key='{key}': 候选={[c.station_name for c in candidates]}")
                return result

        # 5. 未识别，返回空结果
        logger.warning(f"[StationNormalizer] 未识别站名: '{raw}'")
        return ParsedStation(
            raw_input=raw,
            candidates=[],
            is_ambiguous=False,
            primary=None
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. 日期解析模块
# ══════════════════════════════════════════════════════════════════════════════

WEEKDAY_MAP = {
    "周一": 0, "星期一": 0, "礼拜一": 0,
    "周二": 1, "星期二": 1, "礼拜二": 1,
    "周三": 2, "星期三": 2, "礼拜三": 2,
    "周四": 3, "星期四": 3, "礼拜四": 3,
    "周五": 4, "星期五": 4, "礼拜五": 4,
    "周六": 5, "星期六": 5, "礼拜六": 5,
    "周日": 6, "星期日": 6, "礼拜日": 6,
    "周天": 6, "星期天": 6,
}


class DateParser:
    """日期解析器，支持相对/绝对/模糊表达"""

    def __init__(self, base_date: Optional[datetime] = None):
        self.base_date = base_date or datetime.now()

    def parse(self, raw: str) -> ParsedDate:
        raw = raw.strip()
        logger.info(f"[DateParser] 输入: '{raw}'")

        # 1. 相对日期
        result = self._parse_relative(raw)
        if result:
            return result

        # 2. 绝对日期（多种格式）
        result = self._parse_absolute(raw)
        if result:
            return result

        # 3. 月日表达（"3月15日"、"3.15"）
        result = self._parse_month_day(raw)
        if result:
            return result

        # 4. 兜底：返回今天，置信度低
        logger.warning(f"[DateParser] 无法解析日期 '{raw}'，使用今天作为默认值")
        return ParsedDate(
            raw_input=raw,
            date_str=self.base_date.strftime("%Y-%m-%d"),
            is_relative=False,
            confidence=0.3
        )

    def _parse_relative(self, raw: str) -> Optional[ParsedDate]:
        today = self.base_date.replace(hour=0, minute=0, second=0, microsecond=0)

        relative_map = {
            "今天": 0, "今日": 0,
            "明天": 1, "明日": 1,
            "后天": 2, "大后天": 3,
            "昨天": -1, "昨日": -1,
        }

        for keyword, delta in relative_map.items():
            if keyword in raw:
                target = today + timedelta(days=delta)
                logger.info(f"[DateParser] 相对日期 '{keyword}' → {target.strftime('%Y-%m-%d')}")
                return ParsedDate(
                    raw_input=raw,
                    date_str=target.strftime("%Y-%m-%d"),
                    is_relative=True,
                    confidence=1.0
                )

        # 下周X / 这周X
        for prefix in ["下周", "下个星期", "下星期"]:
            for weekday_str, weekday_num in WEEKDAY_MAP.items():
                if prefix in raw and weekday_str in raw:
                    days_ahead = weekday_num - today.weekday() + 7
                    target = today + timedelta(days=days_ahead)
                    logger.info(f"[DateParser] 下周匹配 '{prefix}{weekday_str}' → {target.strftime('%Y-%m-%d')}")
                    return ParsedDate(
                        raw_input=raw,
                        date_str=target.strftime("%Y-%m-%d"),
                        is_relative=True,
                        confidence=0.95
                    )

        for prefix in ["这周", "本周", "这个星期"]:
            for weekday_str, weekday_num in WEEKDAY_MAP.items():
                if prefix in raw and weekday_str in raw:
                    days_ahead = weekday_num - today.weekday()
                    if days_ahead < 0:
                        days_ahead += 7
                    target = today + timedelta(days=days_ahead)
                    logger.info(f"[DateParser] 本周匹配 → {target.strftime('%Y-%m-%d')}")
                    return ParsedDate(
                        raw_input=raw,
                        date_str=target.strftime("%Y-%m-%d"),
                        is_relative=True,
                        confidence=0.9
                    )

        # 纯星期（"周五"→下一个周五）
        for weekday_str, weekday_num in WEEKDAY_MAP.items():
            if raw == weekday_str or raw == weekday_str + "的":
                days_ahead = weekday_num - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                target = today + timedelta(days=days_ahead)
                logger.info(f"[DateParser] 纯星期 '{weekday_str}' → {target.strftime('%Y-%m-%d')}")
                return ParsedDate(
                    raw_input=raw,
                    date_str=target.strftime("%Y-%m-%d"),
                    is_relative=True,
                    confidence=0.85
                )

        # N天后
        match = re.search(r"(\d+)\s*天后", raw)
        if match:
            delta = int(match.group(1))
            target = today + timedelta(days=delta)
            return ParsedDate(raw_input=raw, date_str=target.strftime("%Y-%m-%d"), is_relative=True, confidence=1.0)

        # N周后
        match = re.search(r"(\d+)\s*周后", raw)
        if match:
            delta = int(match.group(1)) * 7
            target = today + timedelta(days=delta)
            return ParsedDate(raw_input=raw, date_str=target.strftime("%Y-%m-%d"), is_relative=True, confidence=1.0)

        return None

    def _parse_absolute(self, raw: str) -> Optional[ParsedDate]:
        year = self.base_date.year
        formats = [
            ("%Y-%m-%d", raw),
            ("%Y/%m/%d", raw),
            ("%Y.%m.%d", raw),
            ("%Y年%m月%d日", raw),
            ("%m-%d", raw),
            ("%m/%d", raw),
        ]
        for fmt, text in formats:
            try:
                if "%Y" not in fmt:
                    dt = datetime.strptime(text, fmt).replace(year=year)
                    # 如果解析出的日期已经过去，自动+1年
                    if dt.date() < self.base_date.date():
                        dt = dt.replace(year=year + 1)
                else:
                    dt = datetime.strptime(text, fmt)
                logger.info(f"[DateParser] 绝对日期格式 '{fmt}' → {dt.strftime('%Y-%m-%d')}")
                return ParsedDate(raw_input=raw, date_str=dt.strftime("%Y-%m-%d"), is_relative=False, confidence=1.0)
            except ValueError:
                continue
        return None

    def _parse_month_day(self, raw: str) -> Optional[ParsedDate]:
        year = self.base_date.year
        # "3月15日" / "3月15号"
        match = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?", raw)
        if match:
            month, day = int(match.group(1)), int(match.group(2))
            try:
                dt = datetime(year, month, day)
                if dt.date() < self.base_date.date():
                    dt = dt.replace(year=year + 1)
                logger.info(f"[DateParser] 月日匹配 → {dt.strftime('%Y-%m-%d')}")
                return ParsedDate(raw_input=raw, date_str=dt.strftime("%Y-%m-%d"), is_relative=False, confidence=0.95)
            except ValueError:
                pass
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 4. 意图槽位抽取（LLM + 结构化输出）
# ══════════════════════════════════════════════════════════════════════════════

SLOT_EXTRACTION_PROMPT_TEMPLATE = """你是一个铁路出行意图解析助手。请从用户输入中精确抽取出行槽位信息，以JSON格式返回。

【规则】
1. 只返回JSON，不要任何解释或额外文字
2. 无法确定的字段填 null
3. 日期保持原始表达（如"明天"、"下周一"、"3月15日"），不要自己计算日期
4. 座位类型候选：商务座/一等座/二等座/硬卧/软卧/硬座/无座
5. 优先目标候选：最快/最稳/最便宜/最舒适
6. 人群类型候选：adult(成人)/student(学生)/child(儿童)/senior(老人)/disability(残障)

【输出JSON格式】
{{
  "origin_raw": "出发地原文，如'北京'",
  "destination_raw": "目的地原文，如'上海'",
  "travel_date_raw": "出行日期原文，如'明天'",
  "return_date_raw": "返程日期原文，null表示单程",
  "passenger_type": "adult/student/child/senior/disability",
  "passenger_count": 1,
  "seat_type": "座位类型或null",
  "priority": "最快/最稳/最便宜/最舒适 或 null",
  "max_transfer": 2,
  "earliest_depart": "最早出发时间HH:MM或null",
  "latest_arrive": "最晚到达时间HH:MM或null"
}}

用户输入：{user_input}"""


class IntentParser:
    """
    完整意图解析器
    整合：StationNormalizer + DateParser + LLM槽位抽取
    """

    def __init__(self):
        self.station_normalizer = StationNormalizer()
        self.date_parser = DateParser()

    def parse(self, user_input: str) -> IntentSlots:
        logger.info(f"[IntentParser] ===== 开始解析 =====")
        logger.info(f"[IntentParser] 原始输入: '{user_input}'")

        # Step 1: LLM 槽位抽取
        raw_slots = self._extract_slots_llm(user_input)
        if raw_slots is None:
            logger.error("[IntentParser] LLM 槽位抽取失败")
            return IntentSlots(
                raw_query=user_input,
                parse_success=False,
                error_msg="LLM 槽位抽取失败"
            )

        logger.info(f"[IntentParser] LLM 抽取结果: {json.dumps(raw_slots, ensure_ascii=False)}")

        # Step 2: 站名标准化
        origin = None
        if raw_slots.get("origin_raw"):
            origin = self.station_normalizer.normalize(raw_slots["origin_raw"])

        destination = None
        if raw_slots.get("destination_raw"):
            destination = self.station_normalizer.normalize(raw_slots["destination_raw"])

        # Step 3: 日期解析
        travel_date = None
        if raw_slots.get("travel_date_raw"):
            travel_date = self.date_parser.parse(raw_slots["travel_date_raw"])

        return_date = None
        if raw_slots.get("return_date_raw"):
            return_date = self.date_parser.parse(raw_slots["return_date_raw"])

        # Step 4: 乘客信息
        passengers = [PassengerType(
            type=raw_slots.get("passenger_type", "adult"),
            count=raw_slots.get("passenger_count", 1) or 1
        )]

        # Step 5: 偏好
        preference = TravelPreference(
            seat_type=raw_slots.get("seat_type"),
            priority=raw_slots.get("priority"),
            max_transfer=raw_slots.get("max_transfer", 2) or 2,
            earliest_depart=raw_slots.get("earliest_depart"),
            latest_arrive=raw_slots.get("latest_arrive"),
        )

        result = IntentSlots(
            origin=origin,
            destination=destination,
            travel_date=travel_date,
            return_date=return_date,
            passengers=passengers,
            preference=preference,
            raw_query=user_input,
            parse_success=True
        )

        logger.info(f"[IntentParser] ===== 解析完成 =====")
        return result

    def _extract_slots_llm(self, user_input: str) -> Optional[dict]:
        """调用 GLM-4-flash 进行槽位抽取，失败时 fallback 到规则解析"""
        prompt = SLOT_EXTRACTION_PROMPT_TEMPLATE.format(user_input=user_input)

        # 尝试调用 LLM
        if zhipu_client:
            try:
                logger.info("[IntentParser] 调用 GLM-4-flash...")
                response = zhipu_client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512,
                )
                content = response.choices[0].message.content.strip()
                logger.info(f"[IntentParser] LLM 原始响应: {content}")

                # 清理 markdown 代码块
                content = re.sub(r"```json\s*", "", content)
                content = re.sub(r"```\s*", "", content)
                content = content.strip()

                slots = json.loads(content)
                return slots
            except json.JSONDecodeError as e:
                logger.error(f"[IntentParser] JSON 解析失败: {e}，响应: {content}")
            except Exception as e:
                logger.error(f"[IntentParser] LLM 调用异常: {e}")

        # Fallback：基于规则的简单槽位抽取
        logger.warning("[IntentParser] 使用规则 fallback 进行槽位抽取")
        return self._extract_slots_rule(user_input)

    def _extract_slots_rule(self, user_input: str) -> dict:
        """
        规则 fallback：用正则/关键词粗粒度抽取槽位
        仅在 LLM 不可用时使用
        """
        slots = {
            "origin_raw": None,
            "destination_raw": None,
            "travel_date_raw": None,
            "return_date_raw": None,
            "passenger_type": "adult",
            "passenger_count": 1,
            "seat_type": None,
            "priority": None,
            "max_transfer": 2,
            "earliest_depart": None,
            "latest_arrive": None,
        }

        # 抽取出发地/目的地
        slots["origin_raw"], slots["destination_raw"] = self._extract_stations_rule(user_input)

        # 抽取日期关键词
        date_keywords = ["明天", "后天", "今天", "大后天", "下周", "本周", "这周"]
        for kw in date_keywords:
            if kw in user_input:
                # 尝试获取更完整的表达，如"下周五"
                idx = user_input.find(kw)
                slots["travel_date_raw"] = user_input[idx:idx+4].strip()
                break

        # 绝对日期
        date_match = re.search(r"(\d{1,2}月\d{1,2}[日号]?|\d{4}-\d{2}-\d{2}|\d{2}/\d{2})", user_input)
        if date_match and not slots["travel_date_raw"]:
            slots["travel_date_raw"] = date_match.group(1)

        # 座位偏好
        for seat in ["商务座", "一等座", "二等座", "硬卧", "软卧", "硬座"]:
            if seat in user_input:
                slots["seat_type"] = seat
                break

        # 优先目标
        for priority in ["最快", "最便宜", "最舒适", "最稳"]:
            if priority in user_input:
                slots["priority"] = priority
                break

        # 学生票
        if "学生" in user_input:
            slots["passenger_type"] = "student"
        elif "儿童" in user_input or "小孩" in user_input:
            slots["passenger_type"] = "child"
        elif "老人" in user_input or "老年" in user_input:
            slots["passenger_type"] = "senior"

        logger.info(f"[IntentParser] 规则抽取结果: {slots}")
        return slots

    def _extract_stations_rule(self, text: str) -> tuple:
        """
        规则层站名提取：
        策略1：用站名库中的所有名称直接在文本中匹配，找到出现位置，
               按"从...到..."语序区分出发/到达
        策略2：正则匹配"X到Y"/"X去Y"/"从X到Y"结构
        """
        # 构建候选站名列表（按名称长度降序，避免"北京南"被"北京"短匹配覆盖）
        all_station_names = sorted(STATION_DB.keys(), key=len, reverse=True)
        # 同时加入别名
        alias_keys = sorted(STATION_ALIAS.keys(), key=len, reverse=True)

        found: list = []  # [(pos, name)]

        for name in all_station_names:
            idx = text.find(name)
            if idx != -1:
                # 避免重叠：检查该位置是否已被更长名称占用
                overlap = any(
                    abs(idx - prev_pos) < len(prev_name)
                    for prev_pos, prev_name in found
                )
                if not overlap:
                    found.append((idx, name))

        # 别名匹配（帝都/魔都等）
        for alias in alias_keys:
            idx = text.find(alias)
            if idx != -1:
                real_name = STATION_ALIAS[alias]
                overlap = any(abs(idx - p) < max(len(alias), len(n)) for p, n in found)
                if not overlap:
                    found.append((idx, real_name))

        # 按出现位置排序
        found.sort(key=lambda x: x[0])

        if len(found) >= 2:
            origin_name = found[0][1]
            dest_name   = found[1][1]
            logger.info(f"[IntentParser] 规则站名匹配: '{origin_name}' → '{dest_name}'")
            return origin_name, dest_name
        elif len(found) == 1:
            logger.warning(f"[IntentParser] 规则站名只找到1个: '{found[0][1]}'，目的地未识别")
            return found[0][1], None
        else:
            # 最后兜底：正则匹配城市名（2-4字）
            sep_pattern = r"(.{2,5}?)(?:到|去|前往|开往)(.{2,5?})(?:[，,。\s]|$)"
            m = re.search(sep_pattern, text)
            if m:
                logger.info(f"[IntentParser] 正则兜底站名: '{m.group(1)}' → '{m.group(2)}'")
                return m.group(1).strip(), m.group(2).strip()
            logger.warning(f"[IntentParser] 规则无法提取站名: '{text}'")
            return None, None


# ══════════════════════════════════════════════════════════════════════════════
# 5. 测试入口
# ══════════════════════════════════════════════════════════════════════════════

def print_result(result: IntentSlots):
    """格式化输出解析结果"""
    print("\n" + "="*60)
    print(f"原始输入: {result.raw_query}")
    print(f"解析成功: {result.parse_success}")

    if result.origin:
        print(f"\n【出发站】")
        print(f"  原始输入: {result.origin.raw_input}")
        print(f"  是否歧义: {result.origin.is_ambiguous}")
        if result.origin.primary:
            print(f"  首选站点: {result.origin.primary.station_name} ({result.origin.primary.station_code})")
        if result.origin.is_ambiguous:
            print(f"  全部候选: {[c.station_name for c in result.origin.candidates]}")

    if result.destination:
        print(f"\n【到达站】")
        print(f"  原始输入: {result.destination.raw_input}")
        print(f"  是否歧义: {result.destination.is_ambiguous}")
        if result.destination.primary:
            print(f"  首选站点: {result.destination.primary.station_name} ({result.destination.primary.station_code})")
        if result.destination.is_ambiguous:
            print(f"  全部候选: {[c.station_name for c in result.destination.candidates]}")

    if result.travel_date:
        print(f"\n【出行日期】")
        print(f"  原始输入: {result.travel_date.raw_input}")
        print(f"  标准日期: {result.travel_date.date_str}")
        print(f"  相对表达: {result.travel_date.is_relative}")
        print(f"  置信度: {result.travel_date.confidence}")

    if result.return_date:
        print(f"\n【返程日期】")
        print(f"  标准日期: {result.return_date.date_str}")

    print(f"\n【乘客信息】")
    for p in result.passengers:
        print(f"  类型: {p.type}, 人数: {p.count}")

    print(f"\n【出行偏好】")
    pref = result.preference
    print(f"  座位类型: {pref.seat_type}")
    print(f"  优先目标: {pref.priority}")
    print(f"  最大中转: {pref.max_transfer}次")
    print(f"  最早出发: {pref.earliest_depart}")
    print(f"  最晚到达: {pref.latest_arrive}")
    print("="*60)


if __name__ == "__main__":
    parser = IntentParser()

    test_cases = [
        "我想从北京去上海，明天出发，要二等座",
        "帝都到魔都下周五的票，学生票，越便宜越好",
        "广州到成都3月15日出发，要硬卧，最多中转一次",
        "从北京西出发去深圳北，后天，商务座",
        "上海到西安这周三，2个人，最快的",
        "北京到武汉明天出发，想要最稳的方案",
    ]

    print("\n🚄 铁路出行意图解析系统 - 模块一测试\n")
    for query in test_cases:
        result = parser.parse(query)
        print_result(result)