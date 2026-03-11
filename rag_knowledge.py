"""
模块三：RAG 知识库层
功能：FAISS 向量检索 + BM25 关键词检索混合，存储铁路领域知识
依赖：langchain, faiss-cpu, rank-bm25, zhipuai, jieba
运行方式：python rag_knowledge.py
"""

import os
import re
import math
import json
import logging
import jieba
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger("rag_knowledge")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

# ── 加载 .env ─────────────────────────────────────────────────────────────────
def load_dotenv(path=".env"):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# 1. 知识库文档定义
# ══════════════════════════════════════════════════════════════════════════════

class KnowledgeDoc(BaseModel):
    doc_id: str
    category: str       # route / hub / policy / tip
    title: str
    content: str
    tags: List[str]     = Field(default_factory=list)
    metadata: Dict      = Field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 知识库内容（结构化文本，模拟真实运营经验）
# ══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_DOCS: List[KnowledgeDoc] = [

    # ── 线路知识 ──────────────────────────────────────────────────────────────
    KnowledgeDoc(
        doc_id="route_001",
        category="route",
        title="京沪高铁线路知识",
        content="""京沪高铁全长1318公里，连接北京南站与上海虹桥站，途经天津南、济南西、南京南、苏州北等主要城市。
全程最快约4小时28分，高铁G字头列车全程运营，票价：商务座约1748元，一等座约933元，二等座约553元。
京沪高铁是中国最繁忙的高铁线路之一，节假日和周末票源紧张，建议提前15天购票。
主要中转枢纽：南京南（可转合肥、杭州方向），济南西（可转青岛、太原方向），天津南（可转天津主城）。
注意：北京南站是京沪高铁始发站，北京站和北京西站不发京沪高铁。""",
        tags=["京沪", "北京南", "上海虹桥", "高铁", "G字头"],
        metadata={"distance_km": 1318, "min_duration_min": 268}
    ),

    KnowledgeDoc(
        doc_id="route_002",
        category="route",
        title="京广高铁线路知识",
        content="""京广高铁全长2298公里，连接北京西站与广州南站，途经石家庄、郑州东、武汉、长沙南等城市。
全程最快约8小时，是世界上运营里程最长的高铁线路之一。
北京至武汉约4小时，武汉至广州约3小时43分。
主要换乘节点：郑州东（京广与徐兰高铁交汇，可换乘西安、兰州方向）；武汉（可换乘武九、沪汉蓉方向）。
注意：北京发往广州南的高铁从北京西站始发，不是北京南站。""",
        tags=["京广", "北京西", "广州南", "武汉", "郑州东", "高铁"],
        metadata={"distance_km": 2298}
    ),

    KnowledgeDoc(
        doc_id="route_003",
        category="route",
        title="沪昆高铁（沪杭甬、杭长、长昆段）",
        content="""沪昆高铁东起上海虹桥，经杭州东、金华、南昌西、长沙南，西至昆明南，全程约2252公里。
上海虹桥至杭州东约45分钟，上海至长沙约4.5小时，上海至昆明约11小时。
主要中转点：杭州东（可换乘宁波、温州、福州方向），南昌西（可换乘福州、厦门），长沙南（京广与沪昆交汇大枢纽）。
长沙南是京广高铁与沪昆高铁的重要交汇枢纽，换乘东西方向、南北方向均在此实现。""",
        tags=["沪昆", "上海", "杭州", "长沙南", "昆明", "南昌"],
        metadata={"distance_km": 2252}
    ),

    KnowledgeDoc(
        doc_id="route_004",
        category="route",
        title="成渝高铁与川渝地区铁路",
        content="""成渝高铁连接成都东与重庆北，全程308公里，约1小时20分，是中国最繁忙的城际高铁之一。
成都至北京高铁：经郑州东中转，全程约8-10小时，从成都东或成都西出发。
成都至上海高铁：经武汉中转，全程约10-12小时。
重庆至北京：经郑州东或武汉中转，全程约8-10小时。
注意：成都有成都站（普速）、成都东（高铁）、成都西（高铁）三个主要车站，购票时需注意区分。
成都东主要发京沪方向车次，成都西主要发兰州、西安方向车次。""",
        tags=["成渝", "成都东", "成都西", "重庆", "川渝"],
        metadata={}
    ),

    KnowledgeDoc(
        doc_id="route_005",
        category="route",
        title="徐兰高铁（郑西段）",
        content="""徐兰高铁连接徐州与兰州，其中郑州东至西安北段（郑西高铁）全程约2小时，是连接华中与西北的重要通道。
主要站点：郑州东—新郑机场—许昌东—漯河南—驻马店西—信阳东—南阳—西安北。
西安北是西北地区最重要的高铁枢纽，可换乘兰新高铁（至兰州、乌鲁木齐）、宝兰高铁。
郑州东是全国重要的高铁十字路口，京广高铁（南北）与徐兰高铁（东西）在此交汇。
由华东前往西安，可先乘高铁到郑州东再换乘，是效率较高的方案。""",
        tags=["徐兰", "郑州东", "西安北", "郑西高铁", "换乘"],
        metadata={}
    ),

    # ── 枢纽换乘经验 ──────────────────────────────────────────────────────────
    KnowledgeDoc(
        doc_id="hub_001",
        category="hub",
        title="郑州东站换乘经验",
        content="""郑州东站是京广高铁与徐兰高铁的十字交汇点，日均旅客量超过10万人次。
换乘时间建议：同站台换乘最少30分钟，跨站台换乘建议45分钟以上，高峰期（节假日/春运）建议留足60分钟。
郑州东站有南北两个候车厅，京广高铁使用北候车厅，徐兰高铁使用南候车厅，两厅之间步行约10-15分钟。
换乘风险：节假日人流量极大，安检排队可能需要20-30分钟，建议留足90分钟以上缓冲。
出站换乘无需出站，站内换乘凭有效车票通行即可。
附近酒店：郑州东站周边酒店集中，若错过接驳车次，当晚住宿方便。""",
        tags=["郑州东", "换乘", "京广", "徐兰", "枢纽经验"],
        metadata={"min_transfer_min": 30, "recommend_transfer_min": 60, "peak_transfer_min": 90}
    ),

    KnowledgeDoc(
        doc_id="hub_002",
        category="hub",
        title="武汉站换乘经验",
        content="""武汉是华中最重要的铁路枢纽，武汉站（高铁）、汉口站（普速+动车）、武昌站（普速）三站鼎立。
武汉站：京广高铁、武九高铁、沪汉蓉高铁在此交汇，高铁换乘请在武汉站进行。
汉口站至武汉站距离约15公里，乘地铁2号线约30分钟，打车约40分钟（高峰可能更长）。
武昌站至武汉站距离约12公里，乘地铁4号线至武汉站约25分钟。
重要提醒：跨站换乘（如汉口出发、武汉站接驳）需要出站乘地铁，时间成本高，建议预留2小时以上。
武汉站内换乘时间：同站换乘建议45分钟，高峰期60分钟。""",
        tags=["武汉", "汉口", "武昌", "换乘", "跨站", "华中枢纽"],
        metadata={"min_transfer_min": 45, "cross_station_min": 120}
    ),

    KnowledgeDoc(
        doc_id="hub_003",
        category="hub",
        title="上海虹桥枢纽换乘经验",
        content="""上海虹桥枢纽集高铁、地铁、机场于一体，是全国最复杂的综合交通枢纽之一。
虹桥火车站（高铁）与虹桥机场2号航站楼相连，步行约10-15分钟可换乘飞机。
高铁站内换乘：虹桥高铁站有东西两个候车厅，同站台换乘25分钟即可，跨候车厅建议40分钟。
上海虹桥至上海站（普速）：乘地铁2号/10号线约40-50分钟，节假日可能需要1小时以上。
上海虹桥至上海南站：乘地铁约60分钟，不建议当天换乘。
注意：上海方向的高铁大部分在虹桥，少量停上海站，购票前务必确认始发/终到站。""",
        tags=["上海虹桥", "换乘", "机场联运", "上海站", "上海南"],
        metadata={"min_transfer_min": 25, "recommend_transfer_min": 40}
    ),

    KnowledgeDoc(
        doc_id="hub_004",
        category="hub",
        title="北京各站区别与换乘",
        content="""北京有北京南、北京、北京西、北京北四个主要火车站，功能各异。
北京南站：京沪高铁、京津城际、京雄城际，主要发往华东、天津方向。
北京站：主要普速列车，发往东北、华北、西南方向，部分动车。
北京西站：京广高铁、京九线，主要发往华南、华中、西南方向；部分西北方向普速。
北京北站：京张高铁（去张家口、呼和浩特），部分内蒙古方向。
各站之间相距较远（10-30公里），换乘需乘地铁，最少需要60-90分钟，节假日请预留2小时。
常见误区：很多人以为去广州坐北京南，实际应该坐北京西；去成都坐北京西；去上海坐北京南。""",
        tags=["北京南", "北京西", "北京站", "北京北", "换乘", "跨站"],
        metadata={"cross_station_min": 90}
    ),

    KnowledgeDoc(
        doc_id="hub_005",
        category="hub",
        title="长沙南站换乘经验",
        content="""长沙南站是京广高铁与沪昆高铁的交汇枢纽，可以在此实现南北（北京/广州）与东西（上海/昆明）方向的换乘。
站内换乘建议时间：35-45分钟，高峰期60分钟。
长沙南至长沙站（普速）：乘地铁6号线约50分钟，不建议当天换乘。
中转便利度：长沙南是南方最重要的换乘枢纽之一，北京→广州→昆明、上海→广州等路线可在此中转。
餐饮补给：长沙南站内有丰富餐饮，换乘等候期间可就餐，但高峰期等位时间较长。""",
        tags=["长沙南", "换乘", "京广", "沪昆", "华南枢纽"],
        metadata={"min_transfer_min": 35, "recommend_transfer_min": 45}
    ),

    KnowledgeDoc(
        doc_id="hub_006",
        category="hub",
        title="广州南站换乘经验",
        content="""广州南站是华南最大的高铁枢纽，京广高铁、广深港高铁、贵广高铁、南广高铁在此交汇。
站内换乘建议时间：40分钟，高峰期60分钟以上。
广州南至广州站（普速/地铁枢纽）：乘地铁2号线约30分钟，可换乘更多方向。
广州南至深圳北：广深港高铁约30分钟，是往来深港的主要通道。
注意：广深高铁、城际铁路停广州东站，不是广州南，确认票面站名。
贵广高铁从广州南出发，前往贵阳约4小时，是进出贵州的主要高铁通道。""",
        tags=["广州南", "换乘", "华南枢纽", "深圳北", "贵广"],
        metadata={"min_transfer_min": 40}
    ),

    # ── 12306 规则与政策 ──────────────────────────────────────────────────────
    KnowledgeDoc(
        doc_id="policy_001",
        category="policy",
        title="12306 退改签规则",
        content="""高铁/动车退票规则（2024年版）：
开车前15天以上：不收退票费，全额退款。
开车前48小时至15天：收票价5%退票费。
开车前24至48小时：收票价10%退票费。
开车前8至24小时：收票价20%退票费。
开车前8小时内（含开车后）：收票价50%退票费，但仅限开车后8小时内的终到站在本线的车票。
改签规则：票价相同或更高车次免费改签；改签至更低票价车次差额不退；每张票限改签1次。
候补购票：若车票已售完，可申请候补，候补成功率因热门程度而异。春运期间候补成功率约30-50%。
注意：实名制火车票，退票须原证件办理，网上购票可在12306 App直接操作。""",
        tags=["退票", "改签", "退款", "手续费", "政策"],
        metadata={"policy_version": "2024"}
    ),

    KnowledgeDoc(
        doc_id="policy_002",
        category="policy",
        title="学生票优惠规则",
        content="""学生票（火车票学生优惠）规则：
享受对象：全日制学生，凭学生优惠卡（铁路学生优惠卡）购票。
优惠幅度：普速列车（K/T/Z）硬座、硬卧票价享受5折优惠；动车组和高铁（G/D/C）不享受学生折扣。
购票方式：需持学生优惠卡，12306网站或App购票时选择"学生"身份，系统会验证资格。
使用次数限制：每学年4次（寒暑假各2次），限始发地至学校之间的固定区间。
注意事项：学生票只适用于固定区间，跨区间不享受优惠；高铁全价，无学生折扣。
常见误区：很多学生误以为高铁也有学生票，实际上高铁（G/D/C）没有学生优惠折扣。""",
        tags=["学生票", "优惠", "折扣", "学生优惠卡", "政策"],
        metadata={"policy_version": "2024"}
    ),

    KnowledgeDoc(
        doc_id="policy_003",
        category="policy",
        title="儿童票和婴儿票规则",
        content="""儿童票规则：
1.2米以下儿童：免票，但无座位，须由成人陪同；若需要座位，须购儿童票。
1.2-1.5米儿童：购儿童票，票价为成人全价票的50%；高铁儿童票按成人二等座全价的50%计算。
1.5米以上儿童：须购成人全价票。
同一成人最多携带2名免票儿童，超过须为多余儿童购儿童票。
婴儿（不足1.2米）不需要票，但不占座；1.2米至1.5米需购儿童票；1.5米以上需购成人票。
注意：高铁儿童票无座位等级之分，统一按二等座全价的5折购票，不区分座位类型。""",
        tags=["儿童票", "婴儿", "免票", "优惠", "政策"],
        metadata={"policy_version": "2024"}
    ),

    KnowledgeDoc(
        doc_id="policy_004",
        category="policy",
        title="候补购票与抢票策略",
        content="""候补购票机制：
触发条件：目标车次、座位类型已无余票时，可申请候补。
候补成功时机：其他旅客退票时，系统自动优先分配给候补旅客；开售提前期（一般出发前30天）放票时也会触发候补分配。
候补成功率影响因素：热门线路节假日候补成功率低（约20-40%）；普通工作日候补成功率较高（约50-70%）；出发时间越近，候补成功率越低。
抢票策略建议：
1. 同时候补多个相邻时间段的车次，提高总体成功率。
2. 在退票高峰时间（出发前24小时、48小时、15天节点）刷新候补。
3. 若目标日期难以购到，考虑提前一天出发，次日到达目的地。
4. 中转方案：目标直达票无法购到时，可购买中转票作为保底方案。
注意：候补成功后会自动扣款，需确保支付账户余额充足。""",
        tags=["候补", "抢票", "购票策略", "余票", "放票"],
        metadata={}
    ),

    KnowledgeDoc(
        doc_id="policy_005",
        category="policy",
        title="高铁实名制与身份证规则",
        content="""高铁实名制规定：
购票：须凭有效身份证件（居民身份证、护照、港澳台通行证等）购票，一人一票实名。
检票：进站检票须本人携带购票所用证件，人证合一方可进站。
儿童：未满16周岁儿童出行，须携带户口本或身份证（或儿童专用证件）。
外籍人士：凭护照购票，进站须出示护照。
常见问题：
- 同一身份证同一时间只能购买一张同一车次的票，不可重复购买。
- 若同行多人，每人须用自己的身份证分别购票。
- 遗失车票：凭购票记录和本人身份证在车站窗口可补办，但需支付工本费。
- 身份证遗失：可凭户籍所在地派出所开具的临时身份证明，或在12306申请电子临时身份证明。""",
        tags=["实名制", "身份证", "进站", "检票", "政策"],
        metadata={"policy_version": "2024"}
    ),

    # ── 出行经验与小贴士 ──────────────────────────────────────────────────────
    KnowledgeDoc(
        doc_id="tip_001",
        category="tip",
        title="缺票情况下的中转策略",
        content="""当目标直达票无票时，中转方案是重要的备选策略：
选择中转枢纽的原则：
1. 优先选择在目标线路上的中间站，避免绕路。
2. 换乘枢纽应是高铁大站，保证第二程车次频次充足（每小时至少2-3班）。
3. 换乘时间不宜过短（建议≥45分钟），也不宜过长（>3小时则不如等直达）。
推荐中转枢纽：
- 北京←→广州方向：可选武汉、长沙南、郑州东中转。
- 北京←→上海方向：可选南京南中转（南京南往返两侧都车次密集）。
- 上海←→成都方向：可选武汉、重庆北、郑州中转。
- 华东←→西北方向：郑州东是最佳中转枢纽（京广+徐兰交汇）。
注意事项：中转购票须分开购买两张票，两段均需余票，且应注意换乘时间预留足够。""",
        tags=["中转策略", "缺票", "换乘枢纽", "方案规划"],
        metadata={}
    ),

    KnowledgeDoc(
        doc_id="tip_002",
        category="tip",
        title="春运和节假日购票经验",
        content="""春运购票时间节点（出发前30天开售）：
- 12306从出发前30天的早上8点开始预售（部分热门线路提前）。
- 春节前7天和春节后7天是最难购票的时间段。
- 建议在开售第一天即刻抢购，错过则考虑候补或中转。
节假日购票建议：
1. 黄金周（十一、五一）：提前20天以上购票，热门线路30天预售即刻秒光。
2. 清明、端午：提前10-15天购票。
3. 平日出行：提前5-7天即可，少部分线路可当天购到。
错峰出行建议：
- 节假日前2-3天和后2-3天票源最紧张，节假日当天反而可能有余票（很多人选择错峰）。
- 工作日出行比周末容易购票，尤其是周二、周三。
- 早班车（6-8点）和晚班车（20点以后）通常比中午时段容易购到。""",
        tags=["春运", "节假日", "购票策略", "预售", "错峰"],
        metadata={}
    ),

    KnowledgeDoc(
        doc_id="tip_003",
        category="tip",
        title="老人、残障人士铁路出行",
        content="""老年人乘火车注意事项：
- 70岁以上老人可在车站窗口优先购票、优先检票进站。
- 部分老人使用老年机，可至车站窗口人工购票，无需使用12306 App。
- 建议老人乘坐时选择靠近厕所的座位，方便如厕；高铁二等座9号、10号车厢靠近餐车和盥洗室。
- 硬卧建议选择中铺（比下铺私密，比上铺方便），行动不便老人建议选下铺。
残障人士优惠：
- 持残疾证可享受火车票半价优惠（需在车站窗口或12306认证后购票）。
- 免费政策：一级肢体残疾或盲人可免费，需携带残疾证。
- 无障碍设施：高铁列车设有无障碍座位，可联系12306客服预约。
注意：老年人和残障人士购票，窗口人工购票可核实证件，12306 App也支持证件认证。""",
        tags=["老人", "残障", "优惠", "无障碍", "特殊旅客"],
        metadata={}
    ),

    KnowledgeDoc(
        doc_id="tip_004",
        category="tip",
        title="高铁晚点与应急处理",
        content="""高铁晚点情况处理：
高铁准点率约92-95%，大部分情况下不会晚点，但以下情况可能导致延误：
- 恶劣天气（暴雪、大雾、强风）
- 前序车次晚点连带
- 线路维护、突发故障
晚点后的权益：
- 延误2小时以内：不予补偿。
- 延误2-4小时：补偿票价的50%。
- 延误4小时以上：补偿票价的100%。
- 可申请全额退票（无手续费）。
中转旅客注意：若因第一程晚点导致错过第二程，需向乘务员声明"中转超时"，可申请免费改签第二程；但若两张票是独立购买，铁路方面不承担第二程改签费用，需自行承担。
建议：中转旅客两程之间留足换乘时间，尤其是春运和恶劣天气季节。""",
        tags=["晚点", "延误", "补偿", "中转风险", "应急"],
        metadata={}
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# 3. BM25 检索器（关键词）
# ══════════════════════════════════════════════════════════════════════════════

class BM25Retriever:
    def __init__(self, docs: List[KnowledgeDoc]):
        from rank_bm25 import BM25Okapi
        self.docs = docs
        # 对每个文档的 title + content + tags 做 jieba 分词
        tokenized = []
        for d in docs:
            text = d.title + " " + d.content + " " + " ".join(d.tags)
            tokens = list(jieba.cut(text))
            tokenized.append(tokens)
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"[BM25] 索引构建完成，文档数: {len(docs)}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeDoc, float]]:
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        # 归一化分数
        max_score = max(scores) if max(scores) > 0 else 1.0
        norm_scores = [s / max_score for s in scores]
        ranked = sorted(enumerate(norm_scores), key=lambda x: x[1], reverse=True)
        results = [(self.docs[i], score) for i, score in ranked[:top_k] if score > 0]
        logger.info(f"[BM25] 查询: '{query}' → 返回 {len(results)} 条结果")
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. 向量检索器（FAISS + ZhipuAI Embedding）
# ══════════════════════════════════════════════════════════════════════════════

class VectorRetriever:
    def __init__(self, docs: List[KnowledgeDoc]):
        self.docs = docs
        self.index = None
        self.embeddings = None
        self._build_index()

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """调用 ZhipuAI embedding-3 获取向量"""
        api_key = os.environ.get("ZHIPUAI_API_KEY", "")
        if not api_key:
            return None
        try:
            from zhipuai import ZhipuAI
            client = ZhipuAI(api_key=api_key)
            resp = client.embeddings.create(
                model="embedding-3",
                input=text,
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.warning(f"[VectorRetriever] Embedding 调用失败: {e}")
            return None

    def _build_index(self):
        """构建 FAISS 索引"""
        import numpy as np

        logger.info("[VectorRetriever] 开始构建向量索引...")
        vectors = []

        for i, doc in enumerate(self.docs):
            text = doc.title + "\n" + doc.content[:500]  # 截断避免过长
            vec = self._get_embedding(text)
            if vec is None:
                # Embedding 不可用：用随机向量占位（保证结构完整，但无语义）
                logger.warning(f"[VectorRetriever] 文档{i}({doc.doc_id}) 使用随机向量占位")
                vec = [0.0] * 1024  # embedding-3 维度为 2048，占位用 1024
            vectors.append(vec)

        if not vectors:
            logger.error("[VectorRetriever] 向量为空，索引构建失败")
            return

        dim = len(vectors[0])
        self.embeddings = np.array(vectors, dtype=np.float32)

        try:
            import faiss
            self.index = faiss.IndexFlatIP(dim)  # 内积（余弦相似度需先归一化）
            # L2 归一化
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            logger.info(f"[VectorRetriever] FAISS 索引构建完成，维度={dim}，文档数={self.index.ntotal}")
        except Exception as e:
            logger.error(f"[VectorRetriever] FAISS 构建失败: {e}")
            self.index = None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeDoc, float]]:
        if self.index is None:
            logger.warning("[VectorRetriever] 索引不可用，跳过向量检索")
            return []

        import numpy as np
        import faiss

        vec = self._get_embedding(query)
        if vec is None:
            logger.warning("[VectorRetriever] Query embedding 获取失败")
            return []

        q = np.array([vec], dtype=np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.docs):
                results.append((self.docs[idx], float(score)))

        logger.info(f"[VectorRetriever] 查询: '{query}' → 返回 {len(results)} 条结果")
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. 混合检索器（BM25 + Vector，RRF 融合）
# ══════════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    混合检索：BM25（关键词）+ FAISS（语义）
    使用 Reciprocal Rank Fusion (RRF) 融合分数
    """

    def __init__(
        self,
        docs: List[KnowledgeDoc],
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        rrf_k: int = 60,
    ):
        self.docs = docs
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k

        logger.info("[HybridRetriever] 初始化 BM25 检索器...")
        self.bm25 = BM25Retriever(docs)

        logger.info("[HybridRetriever] 初始化向量检索器（需要 Embedding API）...")
        self.vector = VectorRetriever(docs)

        logger.info("[HybridRetriever] 混合检索器初始化完成")

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> List[Tuple[KnowledgeDoc, float]]:
        """
        混合检索
        
        Args:
            query:           检索查询
            top_k:           返回前k条
            category_filter: 可选，按分类过滤（route/hub/policy/tip）
        """
        logger.info(f"[HybridRetriever] 检索: '{query}' (filter={category_filter})")

        # BM25 检索
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        # 向量检索
        vector_results = self.vector.search(query, top_k=top_k * 2)

        # RRF 融合
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, KnowledgeDoc] = {}

        # BM25 贡献
        for rank, (doc, score) in enumerate(bm25_results):
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)
            doc_scores[doc.doc_id] = doc_scores.get(doc.doc_id, 0) + rrf_score
            doc_map[doc.doc_id] = doc

        # 向量检索贡献
        for rank, (doc, score) in enumerate(vector_results):
            rrf_score = self.vector_weight / (self.rrf_k + rank + 1)
            doc_scores[doc.doc_id] = doc_scores.get(doc.doc_id, 0) + rrf_score
            doc_map[doc.doc_id] = doc

        # 排序
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked:
            doc = doc_map[doc_id]
            # 分类过滤
            if category_filter and doc.category != category_filter:
                continue
            results.append((doc, score))
            if len(results) >= top_k:
                break

        logger.info(f"[HybridRetriever] 融合结果: {len(results)} 条")
        for doc, score in results:
            logger.info(f"  [{doc.category}] {doc.title} (score={score:.4f})")

        return results

    def format_context(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
    ) -> str:
        """
        将检索结果格式化为 LLM 可用的上下文字符串
        """
        results = self.search(query, top_k=top_k, category_filter=category_filter)
        if not results:
            return "（未找到相关知识库内容）"

        parts = ["【相关知识库参考】\n"]
        for i, (doc, score) in enumerate(results, 1):
            parts.append(f"--- 参考{i}：{doc.title} ---")
            parts.append(doc.content)
            parts.append("")

        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 测试入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🚄 模块三：RAG 知识库层 - 测试\n")

    print("正在构建知识库索引（首次需调用 Embedding API）...\n")
    retriever = HybridRetriever(KNOWLEDGE_DOCS)

    test_queries = [
        ("北京去广州没有直达票，怎么中转？", None),
        ("郑州换乘需要多少时间？", "hub"),
        ("学生票高铁有折扣吗？", "policy"),
        ("春运怎么抢票？", None),
        ("中转晚点了怎么办？", "tip"),
        ("上海到成都的路线怎么走？", "route"),
    ]

    for query, category in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        if category:
            print(f"类别过滤: {category}")
        context = retriever.format_context(query, top_k=2, category_filter=category)
        print(context[:600] + ("..." if len(context) > 600 else ""))

    print("\n✅ 模块三测试完成")