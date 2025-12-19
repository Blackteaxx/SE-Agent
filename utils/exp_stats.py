#!/usr/bin/env python3
"""
实验统计分析脚本 (exp_stats.py)

分析 SE_Perf 实验运行统计，包括：
1. 每个任务的总运行时间
2. LLM 调用次数和重试统计
3. 评估耗时统计

用法:
    python utils/exp_stats.py <trajectory_dir> [--compare <other_dir>]
    
示例:
    python utils/exp_stats.py trajectories_perf/deepseek-v3/Plan-Random-Local-Global-45its_20251218_160428
    
    # 对比两个目录
    python utils/exp_stats.py trajectories_perf/deepseek-v3/Plan-Random-Local-Global-45its_20251218_160428 \
        --compare trajectories_perf/deepseek-v3/Plan-Weighted-45its_20251218_153854
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

# 预编译正则表达式 (全局，避免重复编译)
RE_MAX_RETRY = re.compile(rb"attempt=10/10")
RE_LIMITING = re.compile(rb"5513-chatGPt\.limiting")
RE_LLM_CALL = re.compile(r"调用LLM:".encode())
# 只匹配 step_2 的评估时间（step_1 是初始化，耗时 0.00s）
RE_EVAL_TIME = re.compile(rb"step_2.*\xe8\x80\x97\xe6\x97\xb6 ([\d.]+)s")  # "step_2...耗时 XXXs"
# 提取日志时间戳: "2025-12-18 16:04:38,703"
RE_LOG_TIMESTAMP = re.compile(rb"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


class TaskStats(NamedTuple):
    """任务统计结果"""

    task_name: str
    # 任务运行时间
    total_run_time: float  # 总运行时间 (秒)
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


def parse_log_timestamp(ts_bytes: bytes) -> datetime | None:
    """解析日志时间戳"""
    try:
        ts_str = ts_bytes.decode("utf-8")
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
    except (ValueError, UnicodeDecodeError):
        return None


def analyze_se_framework_log(log_path: Path) -> dict:
    """分析 se_framework.log 文件（使用二进制模式 + 预编译正则，更快）"""
    stats = {
        "max_retry_count": 0,
        "total_limiting_count": 0,
        "total_llm_calls": 0,
        "total_run_time": 0.0,
    }

    if not log_path.exists():
        return stats

    try:
        # 使用二进制模式读取，避免编码转换开销
        with open(log_path, "rb") as f:
            content = f.read()

        # 使用预编译的正则表达式
        stats["max_retry_count"] = len(RE_MAX_RETRY.findall(content))
        stats["total_limiting_count"] = len(RE_LIMITING.findall(content))
        stats["total_llm_calls"] = len(RE_LLM_CALL.findall(content))

        # 提取开始和结束时间，计算总运行时间
        lines = content.split(b"\n")
        start_time = None
        end_time = None
        end_time_marker = None

        # 找第一个有效时间戳
        for line in lines:
            match = RE_LOG_TIMESTAMP.match(line)
            if match:
                start_time = parse_log_timestamp(match.group(1))
                break

        # 找到首次出现“生成最终结果 final.json”所在行的时间戳作为结束时间
        for line in lines:
            if b"final.json" in line:
                try:
                    text = line.decode("utf-8", errors="ignore")
                    if "生成最终结果 final.json" in text:
                        m = RE_LOG_TIMESTAMP.match(line)
                        if m:
                            end_time_marker = parse_log_timestamp(m.group(1))
                            break
                except Exception:
                    continue

        # 找最后一个有效时间戳
        for line in reversed(lines):
            match = RE_LOG_TIMESTAMP.match(line)
            if match:
                end_time = parse_log_timestamp(match.group(1))
                break

        if start_time:
            chosen_end = end_time_marker or end_time
            if chosen_end:
                stats["total_run_time"] = (chosen_end - start_time).total_seconds()

    except Exception as e:
        print(f"Warning: 无法分析 {log_path}: {e}", file=sys.stderr)

    return stats


def analyze_perfagent_logs(task_dir: Path) -> dict:
    """分析 perfagent.log 文件获取评估耗时（使用二进制模式 + 预编译正则，更快）"""
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
                # 使用二进制模式读取
                with open(inner_perfagent, "rb") as f:
                    content = f.read()

                # 使用预编译的正则表达式提取 step_2 的评估时间（跳过 step_1 初始化）
                times = RE_EVAL_TIME.findall(content)
                for t in times:
                    try:
                        time_val = float(t.decode("utf-8"))
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


def analyze_single_task(task_dir: Path) -> TaskStats:
    """分析单个任务目录（供多进程调用）"""
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

    return TaskStats(
        task_name=task_name,
        total_run_time=se_stats["total_run_time"],
        max_retry_count=se_stats["max_retry_count"],
        total_limiting_count=se_stats["total_limiting_count"],
        total_llm_calls=se_stats["total_llm_calls"],
        eval_count=eval_stats["eval_count"],
        total_eval_time=eval_stats["total_eval_time"],
        avg_eval_time=avg_eval_time,
        max_eval_time=eval_stats["max_eval_time"],
        min_eval_time=eval_stats["min_eval_time"],
    )


def analyze_directory(traj_dir: Path, max_workers: int | None = None) -> list[TaskStats]:
    """分析整个轨迹目录（使用多进程并行加速）"""
    results = []

    if not traj_dir.exists():
        print(f"Error: 目录不存在: {traj_dir}", file=sys.stderr)
        return results

    # 遍历所有任务目录
    task_dirs = [d for d in traj_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not task_dirs:
        return results

    # 默认使用 CPU 核心数
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(task_dirs))

    print(f"  使用 {max_workers} 个进程并行分析 {len(task_dirs)} 个任务...")

    # 使用多进程并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_task, task_dir): task_dir for task_dir in task_dirs}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task_dir = futures[future]
                print(f"Warning: 分析任务 {task_dir.name} 失败: {e}", file=sys.stderr)

    # 按任务名排序
    results.sort(key=lambda x: x.task_name)

    return results


def format_duration(seconds: float) -> str:
    """格式化时间，显示为 小时:分钟:秒"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.0f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.0f}s"
    else:
        return f"{secs:.1f}s"


def print_stats(results: list[TaskStats], title: str):
    """打印统计结果"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # 运行时间统计
    run_times = [r.total_run_time for r in results if r.total_run_time > 0]
    total_run_time = sum(run_times)
    avg_run_time = total_run_time / len(run_times) if run_times else 0
    max_run_time = max(run_times) if run_times else 0
    min_run_time = min(run_times) if run_times else 0

    # 总体统计
    total_max_retry = sum(r.max_retry_count for r in results)
    total_limiting = sum(r.total_limiting_count for r in results)
    total_llm_calls = sum(r.total_llm_calls for r in results)
    tasks_with_retry = sum(1 for r in results if r.max_retry_count > 0)

    print("\n📊 总体统计:")
    print(f"  - 任务总数: {len(results)}")
    print(f"  - 总运行时间: {format_duration(total_run_time)} (平均: {format_duration(avg_run_time)})")
    print(f"  - 运行时间范围: {format_duration(min_run_time)} ~ {format_duration(max_run_time)}")
    print(f"  - 有最大重试的任务数: {tasks_with_retry}")
    print(f"  - 总达到最大重试次数 (attempt=10/10): {total_max_retry}")
    print(f"  - 总限流次数: {total_limiting}")
    print(f"  - 总 LLM 调用次数: {total_llm_calls}")

    # 运行时间 TOP 20
    print("\n🕐 运行时间 TOP 20:")
    sorted_by_runtime = sorted(results, key=lambda x: x.total_run_time, reverse=True)[:20]
    for r in sorted_by_runtime:
        if r.total_run_time > 0:
            print(f"  {r.task_name}: {format_duration(r.total_run_time)}")

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
                "total_run_time": r.total_run_time,
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
        description="分析 SE_Perf 实验统计（运行时间、LLM调用、评估耗时）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("trajectory_dir", type=str, help="轨迹目录路径")
    parser.add_argument("--compare", type=str, help="对比目录路径")
    parser.add_argument("--output", "-o", type=str, help="输出 JSON 文件路径")
    parser.add_argument("--workers", "-w", type=int, default=None, help="并行进程数 (默认: CPU核心数)")

    args = parser.parse_args()

    traj_dir = Path(args.trajectory_dir)

    print(f"正在分析: {traj_dir}")
    results = analyze_directory(traj_dir, max_workers=args.workers)

    if not results:
        print("未找到任何任务数据")
        return 1

    title = traj_dir.name
    print_stats(results, title)

    if args.compare:
        compare_dir = Path(args.compare)
        print(f"\n正在分析对比目录: {compare_dir}")
        compare_results = analyze_directory(compare_dir, max_workers=args.workers)
        if compare_results:
            compare_title = compare_dir.name
            print_stats(compare_results, compare_title)
            compare_stats(results, compare_results, title, compare_title)

    if args.output:
        export_json(results, Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
