"""
测试Agent运行
"""
import os
import sys

# 加载.env文件
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

print("=" * 70)
print("  测试 RailwayAgentV3")
print("=" * 70)
print()

# 测试原有的Agent
try:
    from agent import RailwayAgentV3

    print("✅ 正在初始化 Agent...")
    agent = RailwayAgentV3()
    print("✅ Agent 初始化成功！")
    print()

    # 先测试Skill单独运行
    print("--- 测试 Skills ---")
    from skill_loader import SkillLoader
    from pathlib import Path
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(skills_dir)
    tools = loader.load_all()

    tools_dict = {t.name: t for t in tools}

    if "normalize_station" in tools_dict:
        print("\n测试 normalize_station:")
        result = tools_dict["normalize_station"].invoke({"station_name": "北京"})
        print(f"  输入: 北京 → 输出: {result}")

    print()
    print("=" * 70)
    print("  基础测试通过！")
    print("=" * 70)
    print()
    print("  如需测试完整对话，需要确认火山引擎API端点是否正确。")
    print("  你当前的配置:")
    print(f"    Provider: {os.environ.get('LLM_PROVIDER')}")
    print(f"    Model: {os.environ.get('LLM_MODEL')}")
    print(f"    Base URL: {os.environ.get('LLM_BASE_URL')}")
    print()

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
