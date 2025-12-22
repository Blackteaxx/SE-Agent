#!/usr/bin/env python3
"""
从实验目录中提取前 K 次迭代的最佳解决方案。

功能：
1. 遍历实验目录下的所有任务
2. 判断任务是否完成（有 traj.pool 文件）
3. 从 traj.pool 中提取前 K 次迭代中性能最优的解决方案
4. 输出为 final.json 格式

用法:
    python utils/extract_best_solutions.py <experiment_dir> -k <K> -o <output_path>

示例:
    # 提取前 10 次迭代的最佳解决方案
    python utils/extract_best_solutions.py trajectories_perf/deepseek-v3/Plan-Weighted-Local-Global-45its_20251220_074720 -k 10 -o output/final_k10.json

    # 提取前 20 次迭代的最佳解决方案
    python utils/extract_best_solutions.py trajectories_perf/deepseek-v3/Plan-Weighted-Local-Global-45its_20251220_074720 -k 20 -o output/final_k20.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple


class SolutionInfo(NamedTuple):
    """解决方案信息"""

    code: str
    performance: float
    iteration: int
    label: str


def parse_performance(perf_value) -> float:
    """解析性能值，返回浮点数（Infinity 返回 float('inf')）"""
    if perf_value is None:
        return float("inf")
    if isinstance(perf_value, str):
        if perf_value.lower() in ("infinity", "inf"):
            return float("inf")
        try:
            return float(perf_value)
        except ValueError:
            return float("inf")
    try:
        return float(perf_value)
    except (ValueError, TypeError):
        return float("inf")


def is_task_completed(task_dir: Path) -> bool:
    """
    判断任务是否完成。

    完成的标志：
    1. 存在 traj.pool 文件且不为空
    2. 或者存在 final.json 文件
    """
    final_json = task_dir / "final.json"

    if final_json.exists():
        try:
            with open(final_json, encoding="utf-8") as f:
                data = json.load(f)
                if data:
                    return True
        except Exception:
            pass

    return False


def extract_best_solution_from_traj_pool(
    traj_pool_path: Path, max_iterations: int, task_name: str
) -> SolutionInfo | None:
    """
    从 traj.pool 中提取前 max_iterations 次迭代中性能最优的解决方案。

    Args:
        traj_pool_path: traj.pool 文件路径
        max_iterations: 最大迭代次数
        task_name: 任务名称

    Returns:
        最优解决方案信息，如果没有有效解决方案则返回 None
    """
    if not traj_pool_path.exists():
        return None

    try:
        with open(traj_pool_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    # 获取任务数据（可能是直接的任务名，也可能需要搜索）
    task_data = data.get(task_name)
    if not isinstance(task_data, dict):
        # 尝试获取第一个非空的任务
        for key, value in data.items():
            if isinstance(value, dict) and "problem" in value:
                task_data = value
                break

    if not task_data:
        return None

    # 收集所有在 max_iterations 范围内的解决方案
    candidates: list[SolutionInfo] = []

    for key, value in task_data.items():
        if key == "problem" or not isinstance(value, dict):
            continue

        iteration = value.get("iteration")
        performance = value.get("performance")
        code = value.get("code")

        if iteration is None or code is None:
            continue

        try:
            iter_num = int(iteration)
        except (ValueError, TypeError):
            continue

        # 只考虑前 max_iterations 次迭代
        if iter_num > max_iterations:
            continue

        perf_val = parse_performance(performance)

        # 跳过没有代码的解决方案
        if not code or not code.strip():
            continue

        candidates.append(
            SolutionInfo(
                code=code,
                performance=perf_val,
                iteration=iter_num,
                label=key,
            )
        )

    if not candidates:
        return None

    # 过滤掉 Infinity 的解决方案
    valid_candidates = [c for c in candidates if c.performance != float("inf")]

    if not valid_candidates:
        # 如果所有解决方案都是 Infinity，跳过该任务
        return None

    # 选择性能最优的（最小值）
    best = min(valid_candidates, key=lambda x: (x.performance, -x.iteration))
    return best


def extract_best_solutions(
    experiment_dir: Path,
    max_iterations: int,
    output_path: Path | None = None,
    verbose: bool = True,
) -> dict[str, str]:
    """
    从实验目录中提取所有已完成任务的最佳解决方案。

    Args:
        experiment_dir: 实验目录路径
        max_iterations: 最大迭代次数
        output_path: 输出文件路径（可选）
        verbose: 是否打印详细信息

    Returns:
        任务名到最佳代码的映射
    """
    if not experiment_dir.exists():
        print(f"Error: 实验目录不存在: {experiment_dir}", file=sys.stderr)
        return {}

    # 收集所有任务目录
    task_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    task_dirs.sort(key=lambda x: x.name)

    if verbose:
        print(f"发现 {len(task_dirs)} 个任务目录")
        print(f"提取前 {max_iterations} 次迭代的最佳解决方案...")

    results: dict[str, str] = {}
    stats = {
        "total": len(task_dirs),
        "completed": 0,
        "extracted": 0,
        "no_valid_solution": 0,
        "skipped": 0,
    }

    for task_dir in task_dirs:
        task_name = task_dir.name

        # 检查任务是否完成
        if not is_task_completed(task_dir):
            stats["skipped"] += 1
            if verbose:
                print(f"  [跳过] {task_name}: 任务未完成")
            continue

        stats["completed"] += 1

        # 提取最佳解决方案
        traj_pool_path = task_dir / "traj.pool"
        best_solution = extract_best_solution_from_traj_pool(traj_pool_path, max_iterations, task_name)

        if best_solution is None:
            stats["no_valid_solution"] += 1
            if verbose:
                print(f"  [跳过] {task_name}: 没有找到有效解决方案 (全为 Infinity 或无解)")
            continue

        stats["extracted"] += 1
        results[task_name] = best_solution.code

        if verbose:
            perf_str = f"{best_solution.performance:.6f}" if best_solution.performance != float("inf") else "Infinity"
            print(f"  [提取] {task_name}: 迭代 {best_solution.iteration}, 性能 {perf_str}")

    if verbose:
        print("\n统计:")
        print(f"  - 任务总数: {stats['total']}")
        print(f"  - 已完成: {stats['completed']}")
        print(f"  - 成功提取: {stats['extracted']}")
        print(f"  - 无有效解: {stats['no_valid_solution']}")
        print(f"  - 跳过 (未完成): {stats['skipped']}")

    # 输出到文件
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n已保存到: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="从实验目录中提取前 K 次迭代的最佳解决方案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("experiment_dir", type=str, help="实验目录路径")
    parser.add_argument(
        "-k",
        "--max-iterations",
        type=int,
        required=True,
        help="最大迭代次数（只考虑前 K 次迭代）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="输出文件路径",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="静默模式，不打印详细信息",
    )

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_path = Path(args.output)

    results = extract_best_solutions(
        experiment_dir=experiment_dir,
        max_iterations=args.max_iterations,
        output_path=output_path,
        verbose=not args.quiet,
    )

    if not results:
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
