#!/usr/bin/env python3
"""
任务性能分析脚本

分析 SE_Perf 运行的任务性能，包括：
1. LLM 达到最大重试次数 (attempt=10/10) 的统计
2. 每个任务的评估耗时统计

用法:
    python utils/analyze_task_performance.py <trajectory_dir> [--compare <other_dir>]
    
示例:
    python utils/analyze_task_performance.py trajectories_perf/deepseek-v3/Plan-Random-Local-Global-45its_20251218_160428
    
    # 对比两个目录
    python utils/analyze_task_performance.py trajectories_perf/deepseek-v3/Plan-Random-Local-Global-45its_20251218_160428 \
        --compare trajectories_perf/deepseek-v3/Plan-Weighted-45its_20251218_153854
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple


class TaskStats(NamedTuple):
    """任务统计结果"""

    task_name: str
    # LLM 重试相关
    max_retry_count: int  # attempt=10/10 的次数
    total_limiting_count: int  # 限流次数
    total_llm_calls: int  # 总 LLM 调用次数
    # 评估耗时相关
    eval_count: int  # 评估次数
    total_eval_time: float  # 总评估时间 (秒)
    avg_eval_time: float  # 平均评估时间 (秒)
    max_eval_time: float  # 最大评估时间 (秒)
    min_eval_time: float  # 最小评估时间 (秒)


def analyze_se_framework_log(log_path: Path) -> dict:
    """分析 se_framework.log 文件"""
    stats = {
        "max_retry_count": 0,
        "total_limiting_count": 0,
        "total_llm_calls": 0,
    }

    if not log_path.exists():
        return stats

    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # 统计 attempt=10/10 (达到最大重试)
        stats["max_retry_count"] = len(re.findall(r"attempt=10/10", content))

        # 统计限流次数
        stats["total_limiting_count"] = len(re.findall(r"5513-chatGPt\.limiting", content))

        # 统计 LLM 调用次数
        stats["total_llm_calls"] = len(re.findall(r"调用LLM:", content))

    except Exception as e:
        print(f"Warning: 无法分析 {log_path}: {e}", file=sys.stderr)

    return stats


def analyze_perfagent_logs(task_dir: Path) -> dict:
    """分析 perfagent.log 文件获取评估耗时"""
    stats = {
        "eval_count": 0,
        "total_eval_time": 0.0,
        "max_eval_time": 0.0,
        "min_eval_time": float("inf"),
        "eval_times": [],
    }

    task_name = task_dir.name

    # 查找所有 iteration_*/task_name/perfagent.log
    for iteration_dir in task_dir.glob("iteration_*"):
        inner_perfagent = iteration_dir / task_name / "perfagent.log"
        if inner_perfagent.exists():
            try:
                with open(inner_perfagent, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # 提取 "耗时 XXXs" 格式的时间
                times = re.findall(r"耗时 ([\d.]+)s", content)
                for t in times:
                    try:
                        time_val = float(t)
                        stats["eval_times"].append(time_val)
                        stats["eval_count"] += 1
                        stats["total_eval_time"] += time_val
                        stats["max_eval_time"] = max(stats["max_eval_time"], time_val)
                        stats["min_eval_time"] = min(stats["min_eval_time"], time_val)
                    except ValueError:
                        pass
            except Exception:
                pass

    if stats["min_eval_time"] == float("inf"):
        stats["min_eval_time"] = 0.0

    return stats


def analyze_directory(traj_dir: Path) -> list[TaskStats]:
    """分析整个轨迹目录"""
    results = []

    if not traj_dir.exists():
        print(f"Error: 目录不存在: {traj_dir}", file=sys.stderr)
        return results

    # 遍历所有任务目录
    task_dirs = [d for d in traj_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    for task_dir in sorted(task_dirs):
        task_name = task_dir.name

        # 分析 se_framework.log
        se_log = task_dir / "se_framework.log"
        se_stats = analyze_se_framework_log(se_log)

        # 分析 perfagent.log
        eval_stats = analyze_perfagent_logs(task_dir)

        # 计算平均值
        avg_eval_time = 0.0
        if eval_stats["eval_count"] > 0:
            avg_eval_time = eval_stats["total_eval_time"] / eval_stats["eval_count"]

        results.append(
            TaskStats(
                task_name=task_name,
                max_retry_count=se_stats["max_retry_count"],
                total_limiting_count=se_stats["total_limiting_count"],
                total_llm_calls=se_stats["total_llm_calls"],
                eval_count=eval_stats["eval_count"],
                total_eval_time=eval_stats["total_eval_time"],
                avg_eval_time=avg_eval_time,
                max_eval_time=eval_stats["max_eval_time"],
                min_eval_time=eval_stats["min_eval_time"],
            )
        )

    return results


def print_stats(results: list[TaskStats], title: str):
    """打印统计结果"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # 总体统计
    total_max_retry = sum(r.max_retry_count for r in results)
    total_limiting = sum(r.total_limiting_count for r in results)
    total_llm_calls = sum(r.total_llm_calls for r in results)
    tasks_with_retry = sum(1 for r in results if r.max_retry_count > 0)

    print("\n📊 总体统计:")
    print(f"  - 任务总数: {len(results)}")
    print(f"  - 有最大重试的任务数: {tasks_with_retry}")
    print(f"  - 总达到最大重试次数 (attempt=10/10): {total_max_retry}")
    print(f"  - 总限流次数: {total_limiting}")
    print(f"  - 总 LLM 调用次数: {total_llm_calls}")

    # LLM 最大重试 TOP 20
    print("\n🔴 LLM 达到最大重试次数 TOP 20 (attempt=10/10):")
    sorted_by_retry = sorted(results, key=lambda x: x.max_retry_count, reverse=True)[:20]
    for r in sorted_by_retry:
        if r.max_retry_count > 0:
            print(f"  {r.task_name}: {r.max_retry_count} 次")

    # 评估耗时 TOP 20 (按平均耗时)
    print("\n⏱️  评估耗时 TOP 20 (按平均耗时排序):")
    sorted_by_avg = sorted(results, key=lambda x: x.avg_eval_time, reverse=True)[:20]
    for r in sorted_by_avg:
        if r.eval_count > 0:
            print(f"  {r.task_name}: 次数={r.eval_count}, 平均={r.avg_eval_time:.1f}s, 最大={r.max_eval_time:.1f}s")

    # 异常情况 (最大评估时间 > 300s)
    print("\n⚠️  异常评估耗时 (单次 > 300s):")
    sorted_by_max = sorted(results, key=lambda x: x.max_eval_time, reverse=True)
    for r in sorted_by_max:
        if r.max_eval_time > 300:
            print(f"  {r.task_name}: 最大={r.max_eval_time:.1f}s ({r.max_eval_time / 60:.1f}分钟)")


def compare_stats(results1: list[TaskStats], results2: list[TaskStats], title1: str, title2: str):
    """对比两个目录的统计结果"""
    print(f"\n{'=' * 80}")
    print(f"  对比分析: {title1} vs {title2}")
    print(f"{'=' * 80}")

    # 创建查找字典
    dict1 = {r.task_name: r for r in results1}
    dict2 = {r.task_name: r for r in results2}

    # 总体对比
    total1_retry = sum(r.max_retry_count for r in results1)
    total2_retry = sum(r.max_retry_count for r in results2)
    total1_limiting = sum(r.total_limiting_count for r in results1)
    total2_limiting = sum(r.total_limiting_count for r in results2)
    total1_llm = sum(r.total_llm_calls for r in results1)
    total2_llm = sum(r.total_llm_calls for r in results2)

    print("\n📊 总体对比:")
    print(f"  {'指标':<30} {title1:>15} {title2:>15} {'差异':>10}")
    print(f"  {'-' * 70}")
    print(f"  {'任务数':<30} {len(results1):>15} {len(results2):>15}")
    print(
        f"  {'达到最大重试次数':<30} {total1_retry:>15} {total2_retry:>15} {total1_retry / max(total2_retry, 1):.1f}x"
    )
    print(
        f"  {'总限流次数':<30} {total1_limiting:>15} {total2_limiting:>15} {total1_limiting / max(total2_limiting, 1):.1f}x"
    )
    print(f"  {'总LLM调用次数':<30} {total1_llm:>15} {total2_llm:>15} {total1_llm / max(total2_llm, 1):.1f}x")

    # 相同任务对比
    common_tasks = set(dict1.keys()) & set(dict2.keys())
    print(f"\n⏱️  相同任务评估耗时对比 (共 {len(common_tasks)} 个):")
    print(f"  {'任务名':<50} {title1:>12} {title2:>12} {'差异':>8}")
    print(f"  {'-' * 85}")

    comparisons = []
    for task in common_tasks:
        r1, r2 = dict1[task], dict2[task]
        if r1.avg_eval_time > 0 and r2.avg_eval_time > 0:
            diff = (r1.avg_eval_time - r2.avg_eval_time) / r2.avg_eval_time * 100
            comparisons.append((task, r1.avg_eval_time, r2.avg_eval_time, diff))

    # 按差异排序
    comparisons.sort(key=lambda x: x[3], reverse=True)
    for task, avg1, avg2, diff in comparisons[:15]:
        sign = "+" if diff > 0 else ""
        print(f"  {task:<50} {avg1:>10.1f}s {avg2:>10.1f}s {sign}{diff:>6.1f}%")


def export_json(results: list[TaskStats], output_path: Path):
    """导出结果为 JSON"""
    data = []
    for r in results:
        data.append(
            {
                "task_name": r.task_name,
                "max_retry_count": r.max_retry_count,
                "total_limiting_count": r.total_limiting_count,
                "total_llm_calls": r.total_llm_calls,
                "eval_count": r.eval_count,
                "total_eval_time": r.total_eval_time,
                "avg_eval_time": r.avg_eval_time,
                "max_eval_time": r.max_eval_time,
                "min_eval_time": r.min_eval_time,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n📁 结果已导出到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="分析 SE_Perf 任务性能", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    parser.add_argument("trajectory_dir", type=str, help="轨迹目录路径")
    parser.add_argument("--compare", type=str, help="对比目录路径")
    parser.add_argument("--output", "-o", type=str, help="输出 JSON 文件路径")

    args = parser.parse_args()

    traj_dir = Path(args.trajectory_dir)

    print(f"正在分析: {traj_dir}")
    results = analyze_directory(traj_dir)

    if not results:
        print("未找到任何任务数据")
        return 1

    title = traj_dir.name
    print_stats(results, title)

    if args.compare:
        compare_dir = Path(args.compare)
        print(f"\n正在分析对比目录: {compare_dir}")
        compare_results = analyze_directory(compare_dir)
        if compare_results:
            compare_title = compare_dir.name
            print_stats(compare_results, compare_title)
            compare_stats(results, compare_results, title, compare_title)

    if args.output:
        export_json(results, Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
