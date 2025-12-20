#!/usr/bin/env python3
"""
实验统计分析脚本 (exp_stats.py)

分析 SE_Perf 实验运行统计，包括：
1. 每个任务的总运行时间
2. LLM 调用次数和耗时统计
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

# ============================================================================
# 新格式 perfagent.log 的正则表达式
# ============================================================================

# 提取迭代号: "[迭代 1 开始]" 或 "[迭代 1 完成]"
RE_ITER_START = re.compile(r"\[迭代 (\d+) 开始\]")
RE_ITER_END = re.compile(r"\[迭代 (\d+) 完成\]")

# 提取性能评估耗时: "性能评估耗时: 14.10s"
RE_EVAL_TIME = re.compile(r"性能评估耗时[：:]\s*([\d.]+)s")

# 提取 LLM 调用耗时: "[LLM调用完成] 耗时: 31.80s" 或 "LLM调用耗时: 31.80s"
RE_LLM_TIME = re.compile(r"(?:\[LLM调用完成\]\s*耗时[：:]|LLM调用耗时[：:])\s*([\d.]+)s")

# 提取迭代总耗时: "迭代总耗时: 45.90s"
RE_ITER_TOTAL_TIME = re.compile(r"迭代总耗时[：:]\s*([\d.]+)s")

# 提取总运行时间: "总耗时: 45.90s" 或 "# 总耗时: 45.90s"
RE_TOTAL_TIME = re.compile(r"总耗时[：:]\s*([\d.]+)s")

# 提取执行迭代数: "执行迭代数: 1"
RE_ITER_COUNT = re.compile(r"执行迭代数[：:]\s*(\d+)")

# 提取成功改进迭代数: "成功改进迭代数: 1"
RE_SUCCESS_ITER_COUNT = re.compile(r"成功改进迭代数[：:]\s*(\d+)")

# 提取优化成功标记: "优化成功: ✅ 是" 或 "优化成功: ❌ 否"
RE_OPT_SUCCESS = re.compile(r"优化成功[：:]\s*(✅|❌)")

# 提取 pass_rate: "pass_rate: 8.00%"
RE_PASS_RATE = re.compile(r"pass_rate[：:]\s*([\d.]+)%")

# 提取最终 integral: "最终 integral: inf MB*s" 或 "integral: infMB*s"
RE_FINAL_INTEGRAL = re.compile(r"(?:最终\s*)?integral[：:]\s*([\d.]+|inf)(?:\s*MB\*s|MB\*s)?")

# 提取改进幅度: "改进幅度: 0.00%"
RE_IMPROVEMENT = re.compile(r"改进幅度[：:]\s*([-\d.]+)%")

# ============================================================================
# 旧格式兼容 (se_framework.log)
# ============================================================================
RE_MAX_RETRY = re.compile(rb"attempt=10/10")
RE_LIMITING = re.compile(rb"5513-chatGPt\.limiting")
RE_LLM_CALL = re.compile(r"调用LLM:".encode())
RE_LOG_TIMESTAMP = re.compile(rb"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


class EvalDetail(NamedTuple):
    """单次评估的详细信息"""

    iteration: int  # 迭代号
    eval_time: float  # 评估时间（秒）
    llm_time: float  # LLM调用时间（秒）
    iter_total_time: float  # 迭代总时间（秒）


class BestIterInfo(NamedTuple):
    """最优迭代信息"""

    best_iteration: int  # 达到最优性能的迭代次数
    best_performance: float  # 最优性能值
    total_iterations: int  # 总迭代次数
    first_valid_iteration: int  # 第一次有效（非 Infinity）的迭代
    performance_history: tuple[tuple[int, float], ...]  # (iteration, performance) 历史


class TaskStats(NamedTuple):
    """任务统计结果"""

    task_name: str
    # 任务运行时间
    total_run_time: float  # 总运行时间 (秒)
    # LLM 重试相关
    max_retry_count: int  # attempt=10/10 的次数
    total_limiting_count: int  # 限流次数
    total_llm_calls: int  # 总 LLM 调用次数
    # LLM 耗时相关
    total_llm_time: float  # 总 LLM 调用时间 (秒)
    avg_llm_time: float  # 平均 LLM 调用时间 (秒)
    # 评估耗时相关
    eval_count: int  # 评估次数
    total_eval_time: float  # 总评估时间 (秒)
    avg_eval_time: float  # 平均评估时间 (秒)
    max_eval_time: float  # 最大评估时间 (秒)
    min_eval_time: float  # 最小评估时间 (秒)
    # 迭代相关
    iter_count: int  # 执行迭代数
    success_iter_count: int  # 成功改进迭代数
    # 结果相关
    opt_success: bool  # 优化是否成功
    final_pass_rate: float  # 最终 pass_rate
    improvement_pct: float  # 改进幅度
    # 异常评估详情（包含迭代号）
    max_eval_detail: EvalDetail | None  # 最大评估时间的详情
    eval_details: tuple[EvalDetail, ...]  # 所有评估详情
    # 最优迭代信息
    best_iter_info: BestIterInfo | None  # 达到最优性能的迭代信息


def parse_log_timestamp(ts_bytes: bytes) -> datetime | None:
    """解析日志时间戳"""
    try:
        ts_str = ts_bytes.decode("utf-8")
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
    except (ValueError, UnicodeDecodeError):
        return None


def analyze_traj_pool(task_dir: Path) -> BestIterInfo | None:
    """
    分析 traj.pool 文件，找到达到最优性能的迭代次数。

    Returns:
        BestIterInfo | None: 最优迭代信息，如果无法解析则返回 None
    """
    traj_pool_path = task_dir / "traj.pool"
    if not traj_pool_path.exists():
        return None

    try:
        with open(traj_pool_path, encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return None

        # traj.pool 的结构是 {"task_name": {"problem": ..., "iter1_sol1": {...}, "iter2_sol1": {...}, ...}}
        task_name = task_dir.name
        task_data = data.get(task_name)
        if not isinstance(task_data, dict):
            # 尝试获取第一个键
            for key in data:
                if isinstance(data[key], dict) and "problem" in data[key]:
                    task_data = data[key]
                    break

        if not task_data:
            return None

        # 提取所有迭代的性能数据
        performance_history: list[tuple[int, float]] = []

        for key, value in task_data.items():
            if key == "problem" or not isinstance(value, dict):
                continue

            iteration = value.get("iteration")
            performance = value.get("performance")

            if iteration is None or performance is None:
                continue

            try:
                iter_num = int(iteration)
                # 处理 "Infinity" 字符串和数字
                if isinstance(performance, str):
                    if performance.lower() in ("infinity", "inf"):
                        perf_val = float("inf")
                    else:
                        perf_val = float(performance)
                else:
                    perf_val = float(performance)

                performance_history.append((iter_num, perf_val))
            except (ValueError, TypeError):
                continue

        if not performance_history:
            return None

        # 按迭代号排序
        performance_history.sort(key=lambda x: x[0])

        # 找到最优性能（最小值，排除 inf）
        finite_perfs = [(it, p) for it, p in performance_history if p != float("inf")]

        if not finite_perfs:
            # 所有都是 inf
            return BestIterInfo(
                best_iteration=0,
                best_performance=float("inf"),
                total_iterations=len(performance_history),
                first_valid_iteration=0,
                performance_history=tuple(performance_history),
            )

        # 找到第一次有效的迭代
        first_valid_iteration = finite_perfs[0][0]

        # 找到最优性能的迭代（累积最优）
        best_perf = float("inf")
        best_iter = 0

        for it, p in performance_history:
            if p < best_perf:
                best_perf = p
                best_iter = it

        return BestIterInfo(
            best_iteration=best_iter,
            best_performance=best_perf,
            total_iterations=len(performance_history),
            first_valid_iteration=first_valid_iteration,
            performance_history=tuple(performance_history),
        )

    except Exception as e:
        print(f"Warning: 分析 {traj_pool_path} 失败: {e}", file=sys.stderr)
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

        # 找到首次出现"生成最终结果 final.json"所在行的时间戳作为结束时间
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


def analyze_perfagent_log_new_format(log_content: str, iteration_num: int) -> dict | None:
    """分析新格式的 perfagent.log 内容，返回单次迭代的统计信息"""
    result = {
        "iteration": iteration_num,
        "eval_time": 0.0,
        "llm_time": 0.0,
        "iter_total_time": 0.0,
        "total_run_time": 0.0,
        "iter_count": 0,
        "success_iter_count": 0,
        "opt_success": False,
        "final_pass_rate": 0.0,
        "improvement_pct": 0.0,
    }

    # 提取性能评估耗时（可能有多个，取最后一个或求和）
    eval_times = RE_EVAL_TIME.findall(log_content)
    if eval_times:
        # 取所有评估时间的总和（一个迭代可能有多次评估）
        result["eval_time"] = sum(float(t) for t in eval_times)

    # 提取 LLM 调用耗时（可能有多个）
    llm_times = RE_LLM_TIME.findall(log_content)
    if llm_times:
        result["llm_time"] = sum(float(t) for t in llm_times)

    # 提取迭代总耗时
    iter_total_match = RE_ITER_TOTAL_TIME.search(log_content)
    if iter_total_match:
        result["iter_total_time"] = float(iter_total_match.group(1))

    # 提取总运行时间（从最后一个匹配）
    total_time_matches = RE_TOTAL_TIME.findall(log_content)
    if total_time_matches:
        result["total_run_time"] = float(total_time_matches[-1])

    # 提取执行迭代数
    iter_count_match = RE_ITER_COUNT.search(log_content)
    if iter_count_match:
        result["iter_count"] = int(iter_count_match.group(1))

    # 提取成功改进迭代数
    success_iter_match = RE_SUCCESS_ITER_COUNT.search(log_content)
    if success_iter_match:
        result["success_iter_count"] = int(success_iter_match.group(1))

    # 提取优化成功标记
    opt_success_match = RE_OPT_SUCCESS.search(log_content)
    if opt_success_match:
        result["opt_success"] = opt_success_match.group(1) == "✅"

    # 提取最终 pass_rate
    pass_rate_matches = RE_PASS_RATE.findall(log_content)
    if pass_rate_matches:
        result["final_pass_rate"] = float(pass_rate_matches[-1])

    # 提取改进幅度
    improvement_match = RE_IMPROVEMENT.search(log_content)
    if improvement_match:
        result["improvement_pct"] = float(improvement_match.group(1))

    return result


# 旧格式 batch 日志的时间戳正则
RE_BATCH_LOG_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def analyze_perfagent_log_batch_format(log_content: str, iteration_num: int) -> dict | None:
    """
    分析旧格式的 perfagent.log (batch 日志)，从时间戳推算运行时间。

    旧格式日志特点：
    - 是 perfagent.run_batch 的输出
    - 不包含 [LLM调用完成] 耗时: 信息
    - 可以从时间戳差推算大致运行时间
    """
    result = {
        "iteration": iteration_num,
        "eval_time": 0.0,
        "llm_time": 0.0,
        "iter_total_time": 0.0,
        "total_run_time": 0.0,
        "iter_count": 1,  # batch 格式通常是单次迭代
        "success_iter_count": 0,
        "opt_success": False,
        "final_pass_rate": 0.0,
        "improvement_pct": 0.0,
    }

    lines = log_content.strip().split("\n")
    if not lines:
        return None

    # 检查是否是 batch 格式日志（包含 "perfagent.run_batch" 或 "PerfAgent 批量运行"）
    is_batch_format = any("run_batch" in line or "批量运行" in line for line in lines[:10])
    if not is_batch_format:
        return None

    # 提取时间戳
    timestamps: list[datetime] = []
    for line in lines:
        match = RE_BATCH_LOG_TIMESTAMP.match(line)
        if match:
            try:
                ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
                timestamps.append(ts)
            except ValueError:
                pass

    if len(timestamps) >= 2:
        start_time = timestamps[0]
        end_time = timestamps[-1]
        total_time = (end_time - start_time).total_seconds()

        # 这个时间主要是 LLM 调用时间（评估时间通常较短）
        result["iter_total_time"] = total_time
        result["llm_time"] = total_time  # 近似为 LLM 时间
        result["total_run_time"] = total_time

    # 检查是否优化成功
    if "优化成功" in log_content:
        result["opt_success"] = True
        result["success_iter_count"] = 1

    return result


def analyze_perfagent_logs(task_dir: Path) -> dict:
    """分析 perfagent.log 文件获取评估耗时（新格式）"""
    stats = {
        "eval_count": 0,
        "total_eval_time": 0.0,
        "max_eval_time": 0.0,
        "min_eval_time": float("inf"),
        "eval_times": [],
        "eval_details": [],  # 存储 EvalDetail
        "max_eval_detail": None,
        # LLM 耗时
        "total_llm_time": 0.0,
        "llm_times": [],
        # 总运行时间（从 perfagent.log 获取，更准确）
        "total_run_time": 0.0,
        # 迭代统计
        "iter_count": 0,
        "success_iter_count": 0,
        # 结果统计
        "opt_success": False,
        "final_pass_rate": 0.0,
        "improvement_pct": 0.0,
    }

    task_name = task_dir.name
    last_log_stats = None  # 保存最后一个 iteration 的完整统计

    # 查找所有 iteration_*/task_name/perfagent.log 或 iteration_*/perfagent.log
    iteration_dirs = sorted(
        task_dir.glob("iteration_*"), key=lambda x: int(x.name.split("_")[1]) if "_" in x.name else 0
    )

    for iteration_dir in iteration_dirs:
        # 尝试两种路径格式：
        # 1. 新格式: iteration_X/task_name/perfagent.log (SE_Perf with Local-Global memory)
        # 2. 旧格式: iteration_X/perfagent.log (older SE_Perf runs)
        inner_perfagent = iteration_dir / task_name / "perfagent.log"
        if not inner_perfagent.exists():
            inner_perfagent = iteration_dir / "perfagent.log"
        if inner_perfagent.exists():
            try:
                # 提取迭代号
                iter_name = iteration_dir.name
                iteration_num = int(iter_name.split("_")[1]) if "_" in iter_name else 0

                # 读取日志内容
                with open(inner_perfagent, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # 优先尝试新格式解析
                log_stats = analyze_perfagent_log_new_format(content, iteration_num)

                # 如果新格式解析没有 LLM 时间信息，尝试旧格式（batch 日志）
                if log_stats and log_stats["llm_time"] == 0 and log_stats["eval_time"] == 0:
                    batch_stats = analyze_perfagent_log_batch_format(content, iteration_num)
                    if batch_stats and batch_stats["llm_time"] > 0:
                        log_stats = batch_stats

                if log_stats:
                    last_log_stats = log_stats

                    eval_time = log_stats["eval_time"]
                    llm_time = log_stats["llm_time"]
                    iter_total_time = log_stats["iter_total_time"]

                    if eval_time > 0:
                        detail = EvalDetail(
                            iteration=iteration_num,
                            eval_time=eval_time,
                            llm_time=llm_time,
                            iter_total_time=iter_total_time,
                        )
                        stats["eval_times"].append(eval_time)
                        stats["eval_details"].append(detail)
                        stats["eval_count"] += 1
                        stats["total_eval_time"] += eval_time

                        if eval_time > stats["max_eval_time"]:
                            stats["max_eval_time"] = eval_time
                            stats["max_eval_detail"] = detail

                        stats["min_eval_time"] = min(stats["min_eval_time"], eval_time)

                    if llm_time > 0:
                        stats["llm_times"].append(llm_time)
                        stats["total_llm_time"] += llm_time

            except Exception as e:
                print(f"Warning: 分析 {inner_perfagent} 失败: {e}", file=sys.stderr)

    # 从最后一个 iteration 的日志获取汇总信息
    if last_log_stats:
        stats["total_run_time"] = last_log_stats.get("total_run_time", 0.0)
        stats["iter_count"] = last_log_stats.get("iter_count", 0)
        stats["success_iter_count"] = last_log_stats.get("success_iter_count", 0)
        stats["opt_success"] = last_log_stats.get("opt_success", False)
        stats["final_pass_rate"] = last_log_stats.get("final_pass_rate", 0.0)
        stats["improvement_pct"] = last_log_stats.get("improvement_pct", 0.0)

    if stats["min_eval_time"] == float("inf"):
        stats["min_eval_time"] = 0.0

    return stats


def analyze_single_task(task_dir: Path) -> TaskStats:
    """分析单个任务目录（供多进程调用）"""
    task_name = task_dir.name

    # 分析 se_framework.log
    se_log = task_dir / "se_framework.log"
    se_stats = analyze_se_framework_log(se_log)

    # 分析 perfagent.log（新格式）
    eval_stats = analyze_perfagent_logs(task_dir)

    # 分析 traj.pool（最优迭代信息）
    best_iter_info = analyze_traj_pool(task_dir)

    # 计算平均值
    avg_eval_time = 0.0
    if eval_stats["eval_count"] > 0:
        avg_eval_time = eval_stats["total_eval_time"] / eval_stats["eval_count"]

    avg_llm_time = 0.0
    llm_call_count = len(eval_stats["llm_times"])
    if llm_call_count > 0:
        avg_llm_time = eval_stats["total_llm_time"] / llm_call_count

    # 优先使用 perfagent.log 中的总运行时间（更准确）
    total_run_time = eval_stats["total_run_time"]
    if total_run_time <= 0:
        total_run_time = se_stats["total_run_time"]

    return TaskStats(
        task_name=task_name,
        total_run_time=total_run_time,
        max_retry_count=se_stats["max_retry_count"],
        total_limiting_count=se_stats["total_limiting_count"],
        total_llm_calls=se_stats["total_llm_calls"] or llm_call_count,
        total_llm_time=eval_stats["total_llm_time"],
        avg_llm_time=avg_llm_time,
        eval_count=eval_stats["eval_count"],
        total_eval_time=eval_stats["total_eval_time"],
        avg_eval_time=avg_eval_time,
        max_eval_time=eval_stats["max_eval_time"],
        min_eval_time=eval_stats["min_eval_time"],
        iter_count=eval_stats["iter_count"],
        success_iter_count=eval_stats["success_iter_count"],
        opt_success=eval_stats["opt_success"],
        final_pass_rate=eval_stats["final_pass_rate"],
        improvement_pct=eval_stats["improvement_pct"],
        max_eval_detail=eval_stats["max_eval_detail"],
        eval_details=tuple(eval_stats["eval_details"]),
        best_iter_info=best_iter_info,
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


def print_best_iteration_stats(results: list[TaskStats]):
    """打印最优迭代次数统计（用于迭代预算选取）"""
    # 收集有效的最优迭代数据
    best_iters: list[int] = []
    first_valid_iters: list[int] = []
    tasks_with_best_info = 0
    tasks_never_valid = 0  # 从未达到有效性能的任务

    for r in results:
        if r.best_iter_info is None:
            continue
        tasks_with_best_info += 1

        if r.best_iter_info.best_performance == float("inf"):
            tasks_never_valid += 1
        else:
            best_iters.append(r.best_iter_info.best_iteration)
            if r.best_iter_info.first_valid_iteration > 0:
                first_valid_iters.append(r.best_iter_info.first_valid_iteration)

    if not best_iters:
        print("\n🎯 最优迭代次数统计:")
        print("  (无有效数据)")
        return

    # 计算统计量
    best_iters.sort()
    n = len(best_iters)

    avg_best = sum(best_iters) / n
    median_best = best_iters[n // 2] if n % 2 == 1 else (best_iters[n // 2 - 1] + best_iters[n // 2]) / 2
    min_best = min(best_iters)
    max_best = max(best_iters)

    # 计算分位数
    def percentile(data: list[int], p: float) -> float:
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])

    p25 = percentile(best_iters, 25)
    p50 = percentile(best_iters, 50)
    p75 = percentile(best_iters, 75)
    p90 = percentile(best_iters, 90)
    p95 = percentile(best_iters, 95)

    print("\n🎯 最优迭代次数统计（用于迭代预算选取）:")
    print(f"  - 有效任务数: {n}/{tasks_with_best_info} (从未达到有效性能: {tasks_never_valid})")
    print(f"  - 平均达到最优的迭代次数: {avg_best:.1f}")
    print(f"  - 中位数: {median_best:.1f}")
    print(f"  - 范围: {min_best} ~ {max_best}")
    print("  - 分位数:")
    print(f"      25%: {p25:.0f} 次迭代")
    print(f"      50%: {p50:.0f} 次迭代")
    print(f"      75%: {p75:.0f} 次迭代")
    print(f"      90%: {p90:.0f} 次迭代")
    print(f"      95%: {p95:.0f} 次迭代")

    # 迭代次数分布直方图
    print("\n📊 达到最优性能的迭代次数分布:")
    # 定义区间
    bins = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 35), (36, 40), (41, 45), (46, 50)]
    bin_counts: dict[str, int] = {}
    for low, high in bins:
        count = sum(1 for x in best_iters if low <= x <= high)
        if count > 0 or low <= 20:  # 只显示有数据的区间或前几个区间
            bin_counts[f"{low}-{high}"] = count

    # 超过 50 的
    over_50 = sum(1 for x in best_iters if x > 50)
    if over_50 > 0:
        bin_counts[">50"] = over_50

    max_count = max(bin_counts.values()) if bin_counts else 1
    for bin_name, count in bin_counts.items():
        bar_len = int(count / max_count * 30)
        bar = "█" * bar_len
        pct = count / n * 100
        print(f"  {bin_name:>6} 次: {bar:<30} {count:>3} 个 ({pct:.1f}%)")

    # 累积分布
    print("\n📈 累积分布（迭代预算建议）:")
    cumulative_targets = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    for target in cumulative_targets:
        count = sum(1 for x in best_iters if x <= target)
        pct = count / n * 100
        print(f"  ≤{target:>2} 次迭代: {count:>3}/{n} 任务达到最优 ({pct:.1f}%)")

    # 第一次有效迭代统计
    if first_valid_iters:
        first_valid_iters.sort()
        avg_first = sum(first_valid_iters) / len(first_valid_iters)
        median_first = first_valid_iters[len(first_valid_iters) // 2]
        print("\n📍 第一次达到有效性能（非 Infinity）的迭代:")
        print(f"  - 平均: {avg_first:.1f} 次")
        print(f"  - 中位数: {median_first} 次")
        print(f"  - 范围: {min(first_valid_iters)} ~ {max(first_valid_iters)}")


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

    # LLM 耗时统计
    total_llm_time = sum(r.total_llm_time for r in results)
    total_eval_time = sum(r.total_eval_time for r in results)

    # 总体统计
    total_max_retry = sum(r.max_retry_count for r in results)
    total_limiting = sum(r.total_limiting_count for r in results)
    total_llm_calls = sum(r.total_llm_calls for r in results)
    tasks_with_retry = sum(1 for r in results if r.max_retry_count > 0)

    # 优化结果统计
    success_count = sum(1 for r in results if r.opt_success)
    total_iter_count = sum(r.iter_count for r in results)
    total_success_iter = sum(r.success_iter_count for r in results)

    print("\n📊 总体统计:")
    print(f"  - 任务总数: {len(results)}")
    print(f"  - 总运行时间: {format_duration(total_run_time)} (平均: {format_duration(avg_run_time)})")
    print(f"  - 运行时间范围: {format_duration(min_run_time)} ~ {format_duration(max_run_time)}")
    print(
        f"  - 总 LLM 调用时间: {format_duration(total_llm_time)} ({total_llm_time / max(total_run_time, 1) * 100:.1f}%)"
    )
    print(f"  - 总评估时间: {format_duration(total_eval_time)} ({total_eval_time / max(total_run_time, 1) * 100:.1f}%)")
    print(f"  - 有最大重试的任务数: {tasks_with_retry}")
    print(f"  - 总达到最大重试次数 (attempt=10/10): {total_max_retry}")
    print(f"  - 总限流次数: {total_limiting}")
    print(f"  - 总 LLM 调用次数: {total_llm_calls}")

    print("\n📈 优化结果统计:")
    print(f"  - 优化成功任务数: {success_count}/{len(results)} ({success_count / max(len(results), 1) * 100:.1f}%)")
    print(f"  - 总执行迭代数: {total_iter_count}")
    print(f"  - 总成功改进迭代数: {total_success_iter}")

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

    # LLM 耗时 TOP 20
    print("\n🤖 LLM 调用耗时 TOP 20 (按总耗时排序):")
    sorted_by_llm = sorted(results, key=lambda x: x.total_llm_time, reverse=True)[:20]
    for r in sorted_by_llm:
        if r.total_llm_time > 0:
            print(f"  {r.task_name}: 总计={format_duration(r.total_llm_time)}, 平均={r.avg_llm_time:.1f}s")

    # 异常情况 (最大评估时间 > 300s)
    print("\n⚠️  异常评估耗时 (单次 > 300s):")
    sorted_by_max = sorted(results, key=lambda x: x.max_eval_time, reverse=True)
    found_anomaly = False
    for r in sorted_by_max:
        if r.max_eval_time > 300:
            found_anomaly = True
            iter_info = f"iter_{r.max_eval_detail.iteration}" if r.max_eval_detail else "?"
            print(f"  {r.task_name} [{iter_info}]: 最大={r.max_eval_time:.1f}s ({r.max_eval_time / 60:.1f}分钟)")
    if not found_anomaly:
        print("  (无异常)")

    # Pass Rate 统计
    print("\n📋 Pass Rate 分布:")
    pass_rate_bins = {"0%": 0, "1-50%": 0, "51-99%": 0, "100%": 0}
    for r in results:
        if r.final_pass_rate == 0:
            pass_rate_bins["0%"] += 1
        elif r.final_pass_rate == 100:
            pass_rate_bins["100%"] += 1
        elif r.final_pass_rate <= 50:
            pass_rate_bins["1-50%"] += 1
        else:
            pass_rate_bins["51-99%"] += 1
    for bin_name, count in pass_rate_bins.items():
        pct = count / max(len(results), 1) * 100
        print(f"  {bin_name}: {count} 个任务 ({pct:.1f}%)")

    # 最优迭代次数统计
    print_best_iteration_stats(results)


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
    total1_llm_time = sum(r.total_llm_time for r in results1)
    total2_llm_time = sum(r.total_llm_time for r in results2)

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
    print(
        f"  {'总LLM调用时间(s)':<30} {total1_llm_time:>15.0f} {total2_llm_time:>15.0f} {total1_llm_time / max(total2_llm_time, 1):.1f}x"
    )

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
                "total_llm_time": r.total_llm_time,
                "avg_llm_time": r.avg_llm_time,
                "eval_count": r.eval_count,
                "total_eval_time": r.total_eval_time,
                "avg_eval_time": r.avg_eval_time,
                "max_eval_time": r.max_eval_time,
                "min_eval_time": r.min_eval_time,
                "iter_count": r.iter_count,
                "success_iter_count": r.success_iter_count,
                "opt_success": r.opt_success,
                "final_pass_rate": r.final_pass_rate,
                "improvement_pct": r.improvement_pct,
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
