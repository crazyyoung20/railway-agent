"""
铁路出行智能 Agent v3：多轮对话 + 短期/长期记忆
============================================================
核心升级：
  短期记忆（会话内）：LangGraph Checkpointer（MemorySaver / SqliteSaver）
                      → 同一 thread_id 的所有轮次消息自动持久化
                      → Agent 能看到完整对话历史，支持追问/修改

  长期记忆（跨会话）：LangGraph BaseStore（InMemoryStore / AsyncRedisStore）
                      → 命名空间 (user_id, "preferences") 存储用户偏好
                      → 命名空间 (user_id, "trips") 存储历史出行记录
                      → 每次对话结束后写入，下次对话开始时注入 System Prompt

架构图：
  用户请求 (thread_id, user_id)
       │
       ▼
  [inject_memory] ── 读长期记忆 → 注入 SystemMessage
       │
       ▼
  [agent] ── ReAct LLM 推理（能看到全部历史消息）
       │  ↕ tool_calls / 无工具调用
       ▼
  [tools] ── Skills 执行
       │
       ▼  (规划完成后)
  [save_memory] ── 提取本轮偏好/行程 → 写入长期记忆 Store
       │
       ▼
  [finalize]
       │
       ▼
      END

  ┌─────────────────────────────────────────────────────┐
  │  短期记忆 (Checkpointer)                             │
  │  key = thread_id                                     │
  │  存储：完整 AgentState（messages + 所有字段）         │
  │  同一 thread 下每次 invoke 自动续接历史               │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  长期记忆 (Store)                                    │
  │  namespace = (user_id, "preferences")               │
  │    key="profile" → 偏好座位/最快最稳/常用城市        │
  │  namespace = (user_id, "trips")                     │
  │    key=trip_id  → 历史查询记录（最近N条）            │
  └─────────────────────────────────────────────────────┘
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime
from typing import Annotated, TypedDict, Sequence, Optional

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage
)
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# 通过 SkillLoader 按 Agent Skills 规范加载工具
# 扫描 skills/ 目录 → 解析 SKILL.md frontmatter → 动态 import scripts/tool.py
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AGENT_DIR)
from skill_loader import SkillLoader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("railway_agent_v3")


def _load_tools() -> list:
    """从 skills/ 目录按 Agent Skills 规范加载所有工具"""
    skills_dir = os.path.join(_AGENT_DIR, "skills")
    loader = SkillLoader(skills_dir)
    tools = loader.load_all()
    logger.info(
        f"[SkillLoader] 加载了 {len(loader.list_skills())} 个 skill，"
        f"{len(tools)} 个工具：{[t.name for t in tools]}"
    )
    return tools


# ══════════════════════════════════════════════════════════════════════════════
# 1. Agent State
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    # ── 核心消息链（Checkpointer 自动跨轮次持久化）──────────────────────────
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ── 当前轮次元信息 ────────────────────────────────────────────────────────
    user_id:   str    # 用户唯一标识（跨会话长期记忆的 key）
    thread_id: str    # 会话唯一标识（短期记忆的 key）
    iteration: int

    # ── 本轮提取的信息（写入长期记忆用）──────────────────────────────────────
    detected_preference: str   # 本轮检测到的偏好（最快/最稳/最便宜）
    detected_origin:     str
    detected_destination: str
    detected_date:       str
    final_answer: str


# ══════════════════════════════════════════════════════════════════════════════
# 2. 工具 & LLM
# ══════════════════════════════════════════════════════════════════════════════

# 通过 SkillLoader 加载，不再硬编码 import
TOOLS = _load_tools()

def _load_dotenv(path=".env"):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

def build_llm(mock=False):
    if mock:
        return None

    # 优先使用config.py统一配置
    try:
        from config import settings
        llm_config = settings.llm

        if llm_config.api_key:
            return ChatOpenAI(
                model=llm_config.model,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                timeout=llm_config.timeout,
            ).bind_tools(TOOLS)
    except ImportError:
        pass

    # 降级到原有环境变量方式（兼容旧代码）
    zhipu_key = os.environ.get("ZHIPUAI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    volcengine_key = os.environ.get("VOLCENGINE_API_KEY", "")

    if volcengine_key:
        return ChatOpenAI(
            model=os.environ.get("VOLCENGINE_MODEL", "coding-plan"),
            api_key=volcengine_key,
            base_url=os.environ.get("VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3"),
            temperature=0.1,
        ).bind_tools(TOOLS)
    elif zhipu_key:
        return ChatOpenAI(
            model="glm-4-flash", api_key=zhipu_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            temperature=0.1,
        ).bind_tools(TOOLS)
    elif openai_key:
        return ChatOpenAI(model="gpt-4o-mini", api_key=openai_key,
                          temperature=0.1).bind_tools(TOOLS)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. System Prompt（含长期记忆注入占位符）
# ══════════════════════════════════════════════════════════════════════════════

BASE_SYSTEM_PROMPT = """你是一个专业的铁路出行规划助手，运行于 ReAct 架构。

## 用户长期偏好（来自历史记忆）
{long_term_memory}

## 工作流程
1. 标准化站名（normalize_station）
2. 检索相关知识（search_railway_knowledge）
3. 查询直达票（query_tickets）
4. 如需中转：获取枢纽（get_transfer_hubs）→ 查询两段票 → 评估风险（assess_transfer_risk）
5. 综合推荐，结合用户历史偏好个性化输出

## 多轮对话能力
- 你能记住本次会话内所有对话内容，用户可以追问、修改条件
- 示例："换个最便宜的方案" / "改成后天出发" / "武汉换乘那个方案详细说说"

## 偏好关键词
- 最快 / 最稳 / 最便宜 / 最舒适

今天日期：{today}
"""

def build_system_prompt(long_term_memory: dict) -> str:
    """将长期记忆注入 System Prompt"""
    mem_lines = []

    profile = long_term_memory.get("profile", {})
    if profile:
        if profile.get("preferred_seat"):
            mem_lines.append(f"- 偏好座位：{profile['preferred_seat']}")
        if profile.get("preferred_priority"):
            mem_lines.append(f"- 出行优先级：{profile['preferred_priority']}")
        if profile.get("frequent_cities"):
            mem_lines.append(f"- 常用城市：{', '.join(profile['frequent_cities'])}")

    trips = long_term_memory.get("recent_trips", [])
    if trips:
        mem_lines.append(f"- 最近查询过的路线：")
        for t in trips[-3:]:  # 最近3条
            mem_lines.append(f"    · {t.get('origin','?')}→{t.get('destination','?')} "
                             f"({t.get('date','?')}) 偏好:{t.get('preference','无')}")

    mem_text = "\n".join(mem_lines) if mem_lines else "（暂无历史记录，这是首次对话）"

    return BASE_SYSTEM_PROMPT.format(
        long_term_memory=mem_text,
        today=datetime.now().strftime("%Y-%m-%d"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. 长期记忆读写工具函数
# ══════════════════════════════════════════════════════════════════════════════

def read_long_term_memory(store: InMemoryStore, user_id: str) -> dict:
    """从 Store 读取用户长期记忆"""
    result = {}
    try:
        item = store.get((user_id, "preferences"), "profile")
        result["profile"] = item.value if item else {}
    except Exception:
        result["profile"] = {}
    try:
        trips = store.search((user_id, "trips"))
        result["recent_trips"] = [t.value for t in trips]
    except Exception:
        result["recent_trips"] = []
    logger.info(f"[Memory:Read] user={user_id} profile={bool(result['profile'])} trips={len(result['recent_trips'])}")
    return result


def write_long_term_memory(store: InMemoryStore, user_id: str, state: AgentState):
    """将本轮对话提取的信息写入长期记忆"""
    preference  = state.get("detected_preference", "")
    origin      = state.get("detected_origin", "")
    destination = state.get("detected_destination", "")

    if preference or origin:
        try:
            item = store.get((user_id, "preferences"), "profile")
            profile = item.value if item else {}
        except Exception:
            profile = {}
        if preference:
            profile["preferred_priority"] = preference
        cities = set(profile.get("frequent_cities", []))
        if origin:      cities.add(origin)
        if destination: cities.add(destination)
        profile["frequent_cities"] = list(cities)[:10]
        profile["last_updated"] = datetime.now().isoformat()
        store.put((user_id, "preferences"), "profile", profile)
        logger.info(f"[Memory:Write] 偏好: {profile}")

    if origin and destination:
        trip_id = f"trip_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        store.put((user_id, "trips"), trip_id, {
            "origin": origin, "destination": destination,
            "date": state.get("detected_date", ""),
            "preference": preference,
            "recorded_at": datetime.now().isoformat(),
        })
        logger.info(f"[Memory:Write] 行程: {origin}→{destination}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 图节点
# ══════════════════════════════════════════════════════════════════════════════

MAX_ITERATIONS = 12

def make_inject_memory_node(store: InMemoryStore):
    """注入长期记忆节点（工厂函数，绑定 store）"""
    def inject_memory(state: AgentState) -> dict:
        """
        读取用户长期记忆，将其注入为第一条 SystemMessage。
        注意：Checkpointer 会保留历史 messages，
        所以我们只在没有 SystemMessage 时才注入（避免重复）。
        """
        messages = list(state.get("messages", []))
        user_id = state.get("user_id", "anonymous")

        # 检查是否已有 SystemMessage（防止每轮都重复注入）
        has_system = any(isinstance(m, SystemMessage) for m in messages)

        if not has_system:
            long_term = read_long_term_memory(store, user_id)
            system_msg = SystemMessage(content=build_system_prompt(long_term))
            messages = [system_msg] + messages
            logger.info(f"[Node:InjectMemory] 首次注入长期记忆 user={user_id}")
        else:
            logger.info(f"[Node:InjectMemory] SystemMessage 已存在，跳过注入")

        return {"messages": messages}

    return inject_memory


def make_agent_node(llm):
    """ReAct 推理节点"""
    def agent_node(state: AgentState) -> dict:
        iteration = state.get("iteration", 0)
        logger.info(f"[Node:Agent] iteration={iteration}")

        response = llm.invoke(list(state["messages"]))
        tool_count = len(response.tool_calls) if hasattr(response, "tool_calls") else 0
        logger.info(f"[Node:Agent] 返回 tool_calls={tool_count}")

        # 简单提取偏好关键词（实际生产中可用 NER）
        content = response.content or ""
        detected_pref = ""
        for kw in ["最快", "最稳", "最便宜", "最舒适"]:
            # 检查整个对话历史
            all_text = " ".join(
                m.content for m in state["messages"] if hasattr(m, "content") and m.content
            )
            if kw in all_text:
                detected_pref = kw
                break

        return {
            "messages": [response],
            "iteration": iteration + 1,
            "detected_preference": detected_pref,
        }

    return agent_node


def make_save_memory_node(store: InMemoryStore):
    """保存长期记忆节点"""
    def save_memory(state: AgentState) -> dict:
        user_id = state.get("user_id", "anonymous")
        logger.info(f"[Node:SaveMemory] 写入长期记忆 user={user_id}")
        write_long_term_memory(store, user_id, state)
        return {}

    return save_memory


def finalize_node(state: AgentState) -> dict:
    """提取最终答案"""
    messages = state["messages"]
    final = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final = msg.content
            break
    return {"final_answer": final}


# ══════════════════════════════════════════════════════════════════════════════
# 6. 路由条件
# ══════════════════════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last = messages[-1] if messages else None
    iteration = state.get("iteration", 0)

    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    if iteration >= MAX_ITERATIONS:
        return "save_memory"

    return "save_memory"


# ══════════════════════════════════════════════════════════════════════════════
# 7. Agent 主类
# ══════════════════════════════════════════════════════════════════════════════

class RailwayAgentV3:
    """
    多轮对话 + 短期/长期记忆 Railway Agent

    短期记忆：LangGraph MemorySaver（Checkpointer）
              → 同 thread_id 自动续接历史消息，支持追问

    长期记忆：LangGraph InMemoryStore
              → 跨会话存储用户偏好和历史行程
              → 生产环境可替换为 AsyncRedisStore / PostgresStore
    """

    def __init__(self, mock=False):
        _load_dotenv()

        # 短期记忆：Checkpointer（按 thread_id 存储完整 State）
        self.checkpointer = MemorySaver()

        # 长期记忆：Store（按 user_id 存储用户画像和历史）
        self.store = InMemoryStore()

        self.mock = mock
        self.llm = build_llm(mock=mock)

        if not mock and self.llm is None:
            raise RuntimeError("未找到 API Key，请配置 ZHIPUAI_API_KEY 或 OPENAI_API_KEY")

        self.graph = self._build_graph()
        logger.info("[AgentV3] 初始化完成（短期记忆=MemorySaver，长期记忆=InMemoryStore）")

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        # 注册节点（工厂函数绑定 store/llm）
        graph.add_node("inject_memory", make_inject_memory_node(self.store))
        graph.add_node("tools",         ToolNode(TOOLS))
        graph.add_node("save_memory",   make_save_memory_node(self.store))
        graph.add_node("finalize",      finalize_node)

        if self.mock:
            graph.add_node("agent", self._mock_agent_node)
        else:
            graph.add_node("agent", make_agent_node(self.llm))

        # 流程边
        graph.set_entry_point("inject_memory")
        graph.add_edge("inject_memory", "agent")

        graph.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "save_memory": "save_memory"},
        )
        graph.add_edge("tools",       "agent")        # ReAct 循环
        graph.add_edge("save_memory", "finalize")
        graph.add_edge("finalize",    END)

        # 编译时传入 Checkpointer（实现短期记忆）
        return graph.compile(checkpointer=self.checkpointer)

    def _mock_agent_node(self, state: AgentState) -> dict:
        """无 LLM 时的 Mock 节点，用于演示图结构"""
        iteration = state.get("iteration", 0)
        msgs = list(state["messages"])
        last_human = next(
            (m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), ""
        )

        if iteration == 0:
            response = AIMessage(content=f"[Mock] 已理解您的需求：『{last_human}』，正在规划...")
        else:
            response = AIMessage(content="[Mock] 规划完成。（实际输出需配置 LLM API Key）")

        return {"messages": [response], "iteration": iteration + 1}

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def chat(
        self,
        user_input: str,
        thread_id: str,
        user_id: str = "default",
    ) -> dict:
        """
        多轮对话接口。

        Args:
            user_input: 用户本轮输入
            thread_id:  会话ID（同一 thread_id = 同一对话，自动续接历史）
            user_id:    用户ID（长期记忆的 namespace key）

        Returns:
            dict: final_answer, iterations, thread_id, user_id
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[Chat] thread={thread_id} user={user_id}")
        logger.info(f"[Chat] 输入: '{user_input}'")

        # LangGraph Config：thread_id 控制短期记忆（Checkpointer）
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # 初始状态（Checkpointer 会自动合并历史 messages）
        input_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "user_id": user_id,
            "thread_id": thread_id,
            "iteration": 0,
            "detected_preference": "",
            "detected_origin": "",
            "detected_destination": "",
            "detected_date": "",
            "final_answer": "",
        }

        final_state = self.graph.invoke(input_state, config=config)

        result = {
            "thread_id":    thread_id,
            "user_id":      user_id,
            "user_input":   user_input,
            "final_answer": final_state.get("final_answer", ""),
            "iterations":   final_state.get("iteration", 0),
            "message_count": len(final_state.get("messages", [])),
        }
        logger.info(f"[Chat] 完成，消息总数={result['message_count']}")
        return result

    def get_history(self, thread_id: str) -> list:
        """
        获取指定 thread 的完整对话历史。
        LangGraph Checkpointer 自动维护，直接读取即可。
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = self.graph.get_state(config)
            messages = snapshot.values.get("messages", [])
            return [
                {
                    "role": "user" if isinstance(m, HumanMessage)
                             else ("assistant" if isinstance(m, AIMessage) else "system"),
                    "content": m.content,
                }
                for m in messages
                if not isinstance(m, SystemMessage)  # 过滤掉 System Prompt
            ]
        except Exception as e:
            logger.warning(f"[History] 读取失败: {e}")
            return []

    def get_user_memory(self, user_id: str) -> dict:
        """获取用户长期记忆（调试/展示用）"""
        return read_long_term_memory(self.store, user_id)

    def update_user_preference(self, user_id: str, preference: dict):
        """手动更新用户偏好（如前端设置页面调用）"""
        try:
            items = self.store.search(namespace=(user_id, "preferences"))
            profile = items[0].value if items else {}
        except Exception:
            profile = {}
        profile.update(preference)
        profile["last_updated"] = datetime.now().isoformat()
        self.store.put(namespace=(user_id, "preferences"), key="profile", value=profile)
        logger.info(f"[Memory:Update] user={user_id} → {profile}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. 演示：多轮对话 + 记忆效果
# ══════════════════════════════════════════════════════════════════════════════

def demo_multi_turn(agent: RailwayAgentV3):
    """演示多轮对话能力"""
    print("\n" + "="*65)
    print("  演示一：多轮对话（短期记忆）")
    print("="*65)

    user_id   = "user_zhang_san"
    thread_id = f"session_{uuid.uuid4().hex[:8]}"

    print(f"  会话 ID: {thread_id}")
    print(f"  用户 ID: {user_id}")

    conversations = [
        "北京去广州明天出发，帮我规划方案",
        "换个最便宜的方案",          # 追问：修改偏好，Agent 记得目的地
        "把出发日期改成后天",          # 追问：只改日期
        "武汉换乘那个方案，详细说说换乘注意事项",  # 追问：展开某个方案
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n  [第{i}轮] 用户: {user_input}")
        result = agent.chat(user_input, thread_id=thread_id, user_id=user_id)
        answer = result["final_answer"]
        # 截断显示
        print(f"  [第{i}轮] Agent: {answer[:200]}{'...' if len(answer)>200 else ''}")
        print(f"           消息总数: {result['message_count']}  迭代: {result['iterations']}")

    # 展示短期记忆：打印完整对话历史
    print(f"\n  ── 本次会话完整历史（共{len(agent.get_history(thread_id))}条）──")
    for msg in agent.get_history(thread_id):
        role = "👤" if msg["role"] == "user" else "🤖"
        print(f"  {role} [{msg['role']}]: {msg['content'][:80]}{'...' if len(msg['content'])>80 else ''}")


def demo_long_term_memory(agent: RailwayAgentV3):
    """演示长期记忆：新会话自动继承历史偏好"""
    print("\n" + "="*65)
    print("  演示二：长期记忆（跨会话）")
    print("="*65)

    user_id = "user_li_si"

    # 预设用户偏好（模拟之前的历史积累）
    agent.update_user_preference(user_id, {
        "preferred_priority": "最稳",
        "preferred_seat": "二等座",
        "frequent_cities": ["上海", "北京", "成都"],
    })
    # 模拟历史行程
    agent.store.put(
        namespace=(user_id, "trips"),
        key="trip_20260310",
        value={"origin": "上海", "destination": "北京",
               "date": "2026-03-10", "preference": "最稳"},
    )
    agent.store.put(
        namespace=(user_id, "trips"),
        key="trip_20260315",
        value={"origin": "北京", "destination": "成都",
               "date": "2026-03-15", "preference": "最便宜"},
    )

    # 全新会话（不同 thread_id）
    new_thread = f"session_{uuid.uuid4().hex[:8]}"
    print(f"\n  新会话 ID: {new_thread}（全新会话，但有历史记忆）")
    print(f"  用户 ID:   {user_id}")
    print(f"  预设偏好:  最稳 / 二等座 / 常用城市: 上海、北京、成都")

    result = agent.chat(
        "帮我规划上海到成都的出行方案",
        thread_id=new_thread,
        user_id=user_id,
    )
    answer = result["final_answer"]
    print(f"\n  Agent: {answer[:300]}{'...' if len(answer)>300 else ''}")
    print(f"  （Agent 应自动应用历史偏好：最稳 + 二等座）")

    # 展示长期记忆内容
    memory = agent.get_user_memory(user_id)
    print(f"\n  ── 用户长期记忆内容 ──")
    print(f"  偏好: {memory.get('profile', {})}")
    print(f"  历史行程数: {len(memory.get('recent_trips', []))}")


if __name__ == "__main__":
    _load_dotenv()

    has_key = bool(
        os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )

    print("\n🚄 铁路出行智能 Agent v3")
    print("   多轮对话 + 短期/长期记忆（LangGraph Checkpointer + Store）\n")

    use_mock = not has_key
    if use_mock:
        print("⚠️  未检测到 API Key，使用 Mock 模式（演示记忆机制）\n")

    agent = RailwayAgentV3(mock=use_mock)

    # ── 演示记忆机制 ────────────────────────────────────────────────────────

    print("\n" + "="*65)
    print("  核心演示：短期记忆（Checkpointer）")
    print("="*65)

    user_id   = "demo_user"
    thread_id = f"session_{uuid.uuid4().hex[:8]}"

    print(f"\n  同一 thread_id={thread_id} 下的多轮对话：\n")

    # 第1轮
    r1 = agent.chat("北京去广州明天出发", thread_id=thread_id, user_id=user_id)
    print(f"  第1轮 → 消息总数: {r1['message_count']}")
    print(f"    Agent: {r1['final_answer'][:120]}...")

    # 第2轮（追问，Agent 应记得目的地是广州）
    r2 = agent.chat("换个最便宜的", thread_id=thread_id, user_id=user_id)
    print(f"\n  第2轮（追问'换个最便宜的'）→ 消息总数: {r2['message_count']}")
    print(f"    Agent: {r2['final_answer'][:120]}...")
    print(f"    ✅ 消息数从{r1['message_count']}增长到{r2['message_count']}，说明历史被保留")

    # 验证对话历史
    history = agent.get_history(thread_id)
    print(f"\n  对话历史（{len(history)}条）:")
    for h in history:
        role = "👤" if h["role"] == "user" else "🤖"
        print(f"    {role} {h['content'][:60]}{'...' if len(h['content'])>60 else ''}")

    print("\n" + "="*65)
    print("  核心演示：长期记忆（InMemoryStore）")
    print("="*65)

    # 预设偏好
    agent.update_user_preference(user_id, {
        "preferred_priority": "最稳",
        "preferred_seat": "一等座",
        "frequent_cities": ["北京", "广州"],
    })

    # 新会话（新 thread_id = 清空短期记忆，但长期记忆保留）
    new_thread = f"session_{uuid.uuid4().hex[:8]}"
    r3 = agent.chat(
        "帮我规划一下从北京到上海",
        thread_id=new_thread,
        user_id=user_id,
    )
    print(f"\n  新会话（thread={new_thread}）→ 消息总数: {r3['message_count']}")
    print(f"  Agent: {r3['final_answer'][:200]}...")

    memory = agent.get_user_memory(user_id)
    print(f"\n  用户长期记忆: {memory.get('profile', {})}")
    print(f"  ✅ 新会话能看到旧会话写入的偏好（偏好最稳 + 一等座）")

    print(f"\n{'='*65}")
    print("  运行完成！")
    print(f"{'='*65}")