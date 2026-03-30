"""
简单测试脚本：验证火山引擎Coding Plan API和分层架构
"""
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

print("=" * 70)
print("  铁路出行智能 Agent v4 - 测试")
print("=" * 70)
print(f"\n  LLM Provider: {os.environ.get('LLM_PROVIDER', 'unknown')}")
print(f"  Model: {os.environ.get('LLM_MODEL', 'unknown')}")
print(f"  Base URL: {os.environ.get('LLM_BASE_URL', 'unknown')}")
print()

# 测试配置加载
try:
    from config import settings
    print("✅ 配置加载成功")
    print(f"   Provider: {settings.llm.provider}")
    print(f"   Model: {settings.llm.model}")
except Exception as e:
    print(f"❌ 配置加载失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 测试Skill加载
try:
    from skill_loader import SkillLoader
    from pathlib import Path
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(skills_dir)
    tools = loader.load_all()
    print(f"✅ Skills加载成功，共 {len(tools)} 个工具")
    for t in tools:
        print(f"   - {t.name}")
except Exception as e:
    print(f"❌ Skills加载失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 测试核心模块
try:
    from core import (
        default_cache,
        default_router,
    )
    print("✅ 核心模块加载成功")

    # 测试路由
    test_queries = [
        "北京到上海明天",
        "石家庄到三亚，想最快但预算只有500",
    ]
    for q in test_queries:
        decision = default_router.route(q)
        print(f"   路由测试: '{q}' → {decision.route.value} (confidence: {decision.confidence:.2f})")

except Exception as e:
    print(f"❌ 核心模块加载失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("  基础模块测试完成！")
print("=" * 70)
print()
print("  如需测试完整Agent，请运行:")
print("    python -c 'from hybrid_agent import create_hybrid_agent; agent = create_hybrid_agent(); result = agent.chat(\"北京到上海明天\"); print(result.final_answer)'")
print()
