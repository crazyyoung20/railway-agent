"""
Skill Loader — Agent Skills 规范实现
解析 SKILL.md YAML frontmatter，加载并注册 LangChain 工具

规范参考：https://agentskills.io/specification
数据结构对齐：
  class SkillMetadata(TypedDict):
      path: str
      name: str
      description: str
      license: str | None
      compatibility: str | None
      metadata: dict[str, str]
      allowed_tools: list[str]
"""

import os
import sys
import importlib.util
import logging
from pathlib import Path
from typing import TypedDict, Optional

logger = logging.getLogger("skill_loader")


# ══════════════════════════════════════════════════════════════════════════════
# SkillMetadata — 对齐 Agent Skills 规范 TypedDict
# ══════════════════════════════════════════════════════════════════════════════

class SkillMetadata(TypedDict):
    """Metadata for a skill per Agent Skills specification."""
    path:          str
    name:          str
    description:   str
    license:       Optional[str]
    compatibility: Optional[str]
    metadata:      dict[str, str]
    allowed_tools: list[str]


# ══════════════════════════════════════════════════════════════════════════════
# SKILL.md 解析器（手写，不依赖 yaml 库）
# ══════════════════════════════════════════════════════════════════════════════

def _parse_frontmatter(skill_md_path: Path) -> SkillMetadata:
    """
    解析 SKILL.md 的 YAML frontmatter（--- ... --- 之间的内容）。
    返回 SkillMetadata TypedDict。
    """
    text = skill_md_path.read_text(encoding="utf-8")

    # 提取 frontmatter
    if not text.startswith("---"):
        raise ValueError(f"{skill_md_path}: 缺少 YAML frontmatter (必须以 --- 开头)")

    end = text.index("---", 3)
    fm_text = text[3:end].strip()

    # 简单行解析（支持 key: value 和 key: >\n  multiline）
    raw: dict = {}
    lines = fm_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()

            # 多行值（>）
            if val == ">":
                parts = []
                i += 1
                while i < len(lines) and lines[i].startswith(" "):
                    parts.append(lines[i].strip())
                    i += 1
                raw[key] = " ".join(parts)
                continue

            # 嵌套 dict（metadata: / allowed_tools: 下一行缩进）
            if val == "" and i + 1 < len(lines) and lines[i+1].startswith(" "):
                sub: dict | list = {} if ":" in lines[i+1] else []
                i += 1
                while i < len(lines) and lines[i].startswith(" "):
                    sub_line = lines[i].strip()
                    if sub_line.startswith("-"):
                        if isinstance(sub, list):
                            sub.append(sub_line.lstrip("- ").strip())
                    elif ":" in sub_line:
                        sk, _, sv = sub_line.partition(":")
                        if isinstance(sub, dict):
                            sub[sk.strip()] = sv.strip().strip('"')
                    i += 1
                raw[key] = sub
                continue

            raw[key] = val.strip('"')
        i += 1

    # 构造 SkillMetadata
    return SkillMetadata(
        path=str(skill_md_path.parent),
        name=raw.get("name", skill_md_path.parent.name),
        description=raw.get("description", ""),
        license=raw.get("license"),
        compatibility=raw.get("compatibility"),
        metadata=raw.get("metadata", {}),
        allowed_tools=raw.get("allowed_tools", []),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Skill 加载器
# ══════════════════════════════════════════════════════════════════════════════

class SkillLoader:
    """
    扫描 skills/ 目录，解析每个 skill 的 SKILL.md，
    动态加载 scripts/tool.py 中的 LangChain 工具。
    """

    # 每个 skill 的工具函数名（tool.py 中对外暴露的变量名）
    TOOL_EXPORTS = {
        "ticket-query":       ["query_tickets"],
        "station-normalizer": ["normalize_station"],
        "transfer-hub":       ["get_transfer_hubs", "assess_transfer_risk_tool"],
        "rag-knowledge":      ["search_railway_knowledge"],
        "knowledge-graph":    ["search_train_details", "search_station_details", "compare_trains", "search_trains_by_feature"],
    }

    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        self._metadata:  dict[str, SkillMetadata] = {}
        self._tools:     dict[str, list]           = {}

    def load_all(self) -> list:
        """
        加载 skills_dir 下所有合法 skill（含 SKILL.md 的子目录）。
        返回所有 LangChain tool 的扁平列表。
        """
        all_tools = []
        skill_dirs = sorted(
            d for d in self.skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        )

        for skill_dir in skill_dirs:
            try:
                meta = self._load_skill(skill_dir)
                tools = self._tools.get(meta["name"], [])
                logger.info(
                    f"[SkillLoader] ✅ {meta['name']} "
                    f"v{meta['metadata'].get('version','?')} "
                    f"— {len(tools)} tool(s)"
                )
                all_tools.extend(tools)
            except Exception as e:
                logger.warning(f"[SkillLoader] ⚠️  {skill_dir.name} 加载失败: {e}")

        logger.info(f"[SkillLoader] 共加载 {len(skill_dirs)} 个 skill，{len(all_tools)} 个工具")
        return all_tools

    def get_metadata(self, skill_name: str) -> Optional[SkillMetadata]:
        return self._metadata.get(skill_name)

    def list_skills(self) -> list[SkillMetadata]:
        return list(self._metadata.values())

    # ── 内部 ──────────────────────────────────────────────────────────────────

    def _load_skill(self, skill_dir: Path) -> SkillMetadata:
        """加载单个 skill：解析 SKILL.md + 动态导入 scripts/tool.py"""
        skill_md = skill_dir / "SKILL.md"
        meta = _parse_frontmatter(skill_md)
        self._metadata[meta["name"]] = meta

        # 加载工具
        tool_py = skill_dir / "scripts" / "tool.py"
        if tool_py.exists():
            tools = self._import_tools(meta["name"], tool_py)
            self._tools[meta["name"]] = tools
        else:
            logger.warning(f"[SkillLoader] {meta['name']}: scripts/tool.py 不存在，跳过工具加载")
            self._tools[meta["name"]] = []

        return meta

    def _import_tools(self, skill_name: str, tool_py: Path) -> list:
        """动态导入 tool.py，提取工具对象"""
        module_name = f"skill_{skill_name.replace('-', '_')}_tool"

        spec = importlib.util.spec_from_file_location(module_name, tool_py)
        module = importlib.util.module_from_spec(spec)

        # 确保 skills 目录在 sys.path 中（tool.py 可能有相对导入）
        skill_parent = str(tool_py.parent.parent.parent)
        if skill_parent not in sys.path:
            sys.path.insert(0, skill_parent)

        spec.loader.exec_module(module)

        # 按 TOOL_EXPORTS 提取，或自动发现带 .name 属性的对象（LangChain tool 标志）
        tool_names = self.TOOL_EXPORTS.get(skill_name, [])
        tools = []

        if tool_names:
            for name in tool_names:
                obj = getattr(module, name, None)
                if obj is not None:
                    tools.append(obj)
                else:
                    logger.warning(f"[SkillLoader] {skill_name}: 未找到导出 '{name}'")
        else:
            # 自动发现：LangChain tool 有 .name 和 .invoke 属性
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (hasattr(obj, "name") and hasattr(obj, "invoke")
                        and not attr_name.startswith("_")):
                    tools.append(obj)

        return tools


# ══════════════════════════════════════════════════════════════════════════════
# 便捷函数：一行加载所有工具
# ══════════════════════════════════════════════════════════════════════════════

def load_skills(skills_dir: str | Path = None) -> list:
    """
    加载指定目录下所有 skill，返回工具列表。

    用法：
        tools = load_skills("skills/")
        agent = RailwayAgentGraph(tools=tools)
    """
    if skills_dir is None:
        skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(skills_dir)
    return loader.load_all()


# ══════════════════════════════════════════════════════════════════════════════
# 测试
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(skills_dir)
    tools = loader.load_all()

    print("\n── Skill 元数据 ──────────────────────────────────────")
    for meta in loader.list_skills():
        print(f"\n  name:          {meta['name']}")
        print(f"  description:   {meta['description'][:60]}...")
        print(f"  license:       {meta['license']}")
        print(f"  compatibility: {meta['compatibility']}")
        print(f"  metadata:      {meta['metadata']}")
        print(f"  allowed_tools: {meta['allowed_tools']}")
        print(f"  path:          {meta['path']}")

    print(f"\n── 加载的工具 ({'共' + str(len(tools)) + '个'}) ───────────────────────────")
    for t in tools:
        print(f"  • {t.name:30s} — {t.description[:55]}...")

    print("\n── 快速调用测试 ──────────────────────────────────────")
    import json
    r = tools[0].invoke({"station_name": "北京"})
    d = json.loads(r)
    print(f"  normalize_station('北京') → primary='{d['primary']}'  confidence={d['confidence']}")
