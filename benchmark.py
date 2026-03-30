"""
Railway Agent v4 - 基准测试脚本
测试覆盖率和延迟分布
"""
import os
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Any
from collections import defaultdict

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@dataclass
class TestCase:
    """测试用例"""
    query: str
    expected_layer: str  # cache, pipeline, agent
    category: str  # 直达查询, 中转查询, 车次查询等


@dataclass
class TestResult:
    """测试结果"""
    query: str
    expected_layer: str
    actual_layer: str
    latency_ms: int
    success: bool
    category: str
    confidence: float = 0.0
    reason: str = ""


@dataclass
class BenchmarkReport:
    """基准测试报告"""
    total_cases: int = 0
    results: List[TestResult] = field(default_factory=list)
    layer_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latency_by_layer: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    category_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    routing_accuracy: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: TestResult):
        self.results.append(result)
        self.layer_distribution[result.actual_layer] += 1
        self.latency_by_layer[result.actual_layer].append(result.latency_ms)
        self.category_distribution[result.category] += 1

    def generate_report(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("  Railway Agent v4 - 基准测试报告")
        lines.append("=" * 70)
        lines.append("")

        # 1. 测试概览
        lines.append("📊 测试概览")
        lines.append("-" * 40)
        lines.append(f"  总测试用例数: {len(self.results)}")
        lines.append("")

        # 2. 分层分布（覆盖率）
        lines.append("🏗️ 分层分布（实际覆盖率）")
        lines.append("-" * 40)
        total = len(self.results)
        for layer, count in sorted(self.layer_distribution.items(), key=lambda x: x[1], reverse=True):
            pct = count / total * 100
            lines.append(f"  {layer:12s}: {count:3d} ({pct:5.1f}%)")

        # Pipeline 覆盖率
        pipeline_count = self.layer_distribution.get("pipeline", 0)
        pipeline_pct = pipeline_count / total * 100 if total > 0 else 0
        lines.append("")
        lines.append(f"  📈 Pipeline层覆盖率: {pipeline_pct:.1f}%")
        lines.append(f"     (即简单查询不走LLM的比例)")
        lines.append("")

        # 3. 延迟统计
        lines.append("⏱️ 延迟统计（按层）")
        lines.append("-" * 40)
        for layer, latencies in sorted(self.latency_by_layer.items()):
            if latencies:
                avg = sum(latencies) / len(latencies)
                min_lat = min(latencies)
                max_lat = max(latencies)
                # 计算P95
                sorted_lat = sorted(latencies)
                p95_idx = int(len(sorted_lat) * 0.95)
                p95 = sorted_lat[min(p95_idx, len(sorted_lat)-1)]
                lines.append(f"  {layer}:")
                lines.append(f"    平均: {avg:6.1f}ms | P95: {p95:6.1f}ms | 范围: [{min_lat}ms - {max_lat}ms]")
        lines.append("")

        # 4. 分类统计
        lines.append("📂 查询分类统计")
        lines.append("-" * 40)
        for category, count in sorted(self.category_distribution.items(), key=lambda x: x[1], reverse=True):
            pct = count / total * 100
            lines.append(f"  {category:12s}: {count:3d} ({pct:5.1f}%)")
        lines.append("")

        # 5. 详细结果
        lines.append("📋 详细测试结果")
        lines.append("-" * 70)
        lines.append(f"  {'查询':<30s} {'期望':<10s} {'实际':<10s} {'延迟':>8s} {'置信度':>8s}")
        lines.append("  " + "-" * 66)
        for r in self.results:
            status = "✅" if r.success else "❌"
            lines.append(f"  {status} {r.query:<28s} {r.expected_layer:<10s} {r.actual_layer:<10s} {r.latency_ms:>6d}ms {r.confidence:>7.2f}")
        lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


def create_test_cases() -> List[TestCase]:
    """创建测试用例集"""
    cases = []

    # 1. 简单直达查询（应该走Pipeline）
    simple_direct_cases = [
        ("北京到上海明天", "直达车次查询"),
        ("上海到杭州", "直达车次查询"),
        ("广州到深圳", "直达车次查询"),
        ("成都到重庆后天", "直达车次查询"),
        ("西安到郑州", "直达车次查询"),
        ("武汉到长沙", "直达车次查询"),
        ("南京到苏州", "直达车次查询"),
        ("天津到北京", "直达车次查询"),
        ("杭州到宁波", "直达车次查询"),
        ("青岛到济南", "直达车次查询"),
        ("厦门到福州", "直达车次查询"),
        ("大连到沈阳", "直达车次查询"),
        ("合肥到南京", "直达车次查询"),
        ("南昌到长沙", "直达车次查询"),
        ("贵阳到昆明", "直达车次查询"),
    ]
    for query, cat in simple_direct_cases:
        cases.append(TestCase(query, "pipeline", cat))

    # 2. 带日期的直达查询（应该走Pipeline）
    date_cases = [
        ("2025-04-01 北京到上海", "直达车次查询"),
        ("明天从广州到成都", "直达车次查询"),
        ("后天深圳到武汉", "直达车次查询"),
    ]
    for query, cat in date_cases:
        cases.append(TestCase(query, "pipeline", cat))

    # 3. 车次信息查询（应该走Pipeline）
    train_info_cases = [
        ("G1次列车", "车次信息查询"),
        ("G1有充电口吗", "车次信息查询"),
        ("D123次列车设施", "车次信息查询"),
        ("查询G101", "车次信息查询"),
    ]
    for query, cat in train_info_cases:
        cases.append(TestCase(query, "pipeline", cat))

    # 4. 车站信息查询（应该走Pipeline）
    station_info_cases = [
        ("北京南站", "车站信息查询"),
        ("上海虹桥站在哪", "车站信息查询"),
        ("广州南站有地铁吗", "车站信息查询"),
    ]
    for query, cat in station_info_cases:
        cases.append(TestCase(query, "pipeline", cat))

    # 5. 复杂查询（应该走Agent）
    complex_cases = [
        ("石家庄到三亚，想最快但预算只有500", "预算限制查询"),
        ("北京到广州，推荐最稳的中转方案", "中转推荐查询"),
        ("帮我比较G1和G3", "车次对比查询"),
        ("北京到上海和到杭州哪个更快", "多目的地比较"),
        ("我想从北京去成都，中途想在西安玩一天", "自定义中转查询"),
        ("哪个车次最快", "多轮对话-上下文"),
        ("换个便宜的", "多轮对话-修改条件"),
        ("帮我分析一下各方案的优缺点", "分析推荐查询"),
        ("中转方案有什么风险", "风险评估查询"),
    ]
    for query, cat in complex_cases:
        cases.append(TestCase(query, "agent", cat))

    # 6. 边缘/模糊查询（可能走Agent或Pipeline，取决于实现）
    edge_cases = [
        ("我想去北京", "不完整查询"),
        ("火车票怎么买", "通用问题"),
        ("高铁和动车有什么区别", "知识问答"),
    ]
    for query, cat in edge_cases:
        cases.append(TestCase(query, "agent", cat))  # 期望走Agent，但实际可能不同

    return cases


def run_benchmark():
    """运行基准测试"""
    print("=" * 70)
    print("  Railway Agent v4 - 基准测试")
    print("=" * 70)
    print()

    # 1. 初始化
    print("🔧 初始化组件...")
    try:
        from hybrid_agent import create_hybrid_agent
        from core import default_router

        agent = create_hybrid_agent()
        print("✅ Agent初始化成功")
    except Exception as e:
        print(f"❌ Agent初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # 2. 创建测试用例
    test_cases = create_test_cases()
    print(f"📝 加载测试用例: {len(test_cases)} 个")
    print()

    # 3. 运行测试
    report = BenchmarkReport(total_cases=len(test_cases))

    print("🚀 开始测试...")
    print("-" * 70)

    for i, case in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] 测试: {case.query[:30]}...", end=" ")

        try:
            start = time.time()

            # 先测试路由决策
            route_decision = default_router.route(case.query)

            # 执行完整请求
            result = agent.chat(case.query, user_id="benchmark_user")

            elapsed = int((time.time() - start) * 1000)

            actual_layer = result.route_info.get("layer", "unknown")
            success = actual_layer == case.expected_layer

            test_result = TestResult(
                query=case.query,
                expected_layer=case.expected_layer,
                actual_layer=actual_layer,
                latency_ms=elapsed,
                success=success,
                category=case.category,
                confidence=getattr(route_decision, 'confidence', 0.0),
                reason=getattr(route_decision, 'reason', '')
            )
            report.add_result(test_result)

            status = "✅" if success else "⚠️"
            print(f"→ {actual_layer} ({elapsed}ms) {status}")

            # 避免请求过快
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ 错误: {e}")
            # 记录失败
            test_result = TestResult(
                query=case.query,
                expected_layer=case.expected_layer,
                actual_layer="error",
                latency_ms=0,
                success=False,
                category=case.category
            )
            report.add_result(test_result)

    print()
    print("-" * 70)
    print()

    # 4. 输出报告
    print(report.generate_report())

    # 5. 保存报告到文件
    report_file = Path("benchmark_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report.generate_report())
    print(f"📄 报告已保存到: {report_file.absolute()}")

    return report


if __name__ == "__main__":
    run_benchmark()
