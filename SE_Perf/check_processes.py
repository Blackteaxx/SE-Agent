#!/usr/bin/env python3
"""
检查僵尸进程和孤儿进程的 Python 脚本
可以更详细地分析进程状态和关系
"""

import subprocess
from datetime import datetime


def run_cmd(cmd: list[str]) -> str:
    """执行命令并返回输出"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def parse_ps_line(line: str) -> dict | None:
    """解析 ps aux 输出的一行"""
    parts = line.split()
    if len(parts) < 11:
        return None
    try:
        return {
            "user": parts[0],
            "pid": int(parts[1]),
            "cpu": float(parts[2]),
            "mem": float(parts[3]),
            "vsz": parts[4],
            "rss": parts[5],
            "tty": parts[6],
            "stat": parts[7],
            "start": parts[8],
            "time": parts[9],
            "cmd": " ".join(parts[10:]),
        }
    except (ValueError, IndexError):
        return None


def check_zombies():
    """检查僵尸进程"""
    print("=== 检查僵尸进程 ===")
    output = run_cmd(["ps", "aux"])
    zombies = []
    for line in output.split("\n"):
        if len(line) > 0 and line.split()[7] == "Z":
            proc = parse_ps_line(line)
            if proc:
                zombies.append(proc)

    if not zombies:
        print("✓ 未发现僵尸进程")
    else:
        print(f"⚠️  发现 {len(zombies)} 个僵尸进程：")
        for z in zombies:
            print(f"  PID: {z['pid']:8d} | CMD: {z['cmd'][:80]}")
    return zombies


def check_perf_processes():
    """检查 perf_run.py 相关进程"""
    print("\n=== 检查 perf_run.py 相关进程 ===")
    output = run_cmd(["ps", "aux"])
    perf_procs = []
    for line in output.split("\n"):
        if "perf_run.py" in line and "grep" not in line:
            proc = parse_ps_line(line)
            if proc:
                perf_procs.append(proc)

    if not perf_procs:
        print("✓ 未发现 perf_run.py 进程")
    else:
        print(f"⚠️  发现 {len(perf_procs)} 个 perf_run.py 进程：")
        for p in perf_procs:
            stat_icon = "⚠️" if p["stat"] in ["Z", "T"] else "  "
            print(
                f"{stat_icon} PID: {p['pid']:8d} | STAT: {p['stat']:4s} | CPU: {p['cpu']:6.2f}% | "
                f"MEM: {p['mem']:6.2f}% | CMD: {p['cmd'][:100]}"
            )
    return perf_procs


def check_instance_runner():
    """检查 instance_runner.py 进程"""
    print("\n=== 检查 instance_runner.py 进程 ===")
    output = run_cmd(["ps", "aux"])
    runner_procs = []
    for line in output.split("\n"):
        if "instance_runner.py" in line and "grep" not in line:
            proc = parse_ps_line(line)
            if proc:
                runner_procs.append(proc)

    if not runner_procs:
        print("✓ 未发现 instance_runner.py 进程（可能已结束）")
    else:
        print(f"发现 {len(runner_procs)} 个 instance_runner.py 进程：")
        for p in runner_procs:
            print(
                f"  PID: {p['pid']:8d} | STAT: {p['stat']:4s} | CPU: {p['cpu']:6.2f}% | "
                f"MEM: {p['mem']:6.2f}% | CMD: {p['cmd'][:100]}"
            )
    return runner_procs


def check_process_tree(pid: int):
    """检查进程树"""
    print(f"\n=== 进程树 (PID: {pid}) ===")
    try:
        # 尝试使用 pstree
        tree = run_cmd(["pstree", "-p", str(pid)])
        if tree and "Error" not in tree:
            print(tree)
        else:
            # 回退到 ps
            children = run_cmd(["ps", "--ppid", str(pid), "-o", "pid,cmd"])
            if children:
                print(f"子进程：\n{children}")
            else:
                print("无子进程")
    except Exception as e:
        print(f"无法获取进程树: {e}")


def check_long_running():
    """检查长时间运行的进程"""
    print("\n=== 检查长时间运行的 Python 进程 ===")
    output = run_cmd(["ps", "-eo", "pid,etime,cmd"])
    long_running = []
    for line in output.split("\n"):
        if ("instance_runner" in line or "perf_run" in line) and "grep" not in line:
            parts = line.split()
            if len(parts) >= 3:
                etime = parts[1]  # elapsed time
                # 简单检查：如果包含 "days" 或时间格式很长，可能是长时间运行
                if "days" in etime or "-" in etime:
                    long_running.append(line)

    if not long_running:
        print("✓ 未发现异常长时间运行的进程")
    else:
        print("⚠️  发现长时间运行的进程：")
        for proc in long_running[:10]:  # 只显示前10个
            print(f"  {proc}")
    return long_running


def check_resource_usage():
    """检查资源使用情况"""
    print("\n=== 检查资源使用情况 ===")
    output = run_cmd(["ps", "aux"])
    high_cpu = []
    high_mem = []
    for line in output.split("\n"):
        if "python" in line.lower() and "grep" not in line:
            proc = parse_ps_line(line)
            if proc:
                if proc["cpu"] > 50.0:
                    high_cpu.append(proc)
                if proc["mem"] > 10.0:
                    high_mem.append(proc)

    if high_cpu:
        print(f"⚠️  发现 {len(high_cpu)} 个高 CPU 使用率的进程（>50%）：")
        for p in sorted(high_cpu, key=lambda x: x["cpu"], reverse=True)[:10]:
            print(f"  PID: {p['pid']:8d} | CPU: {p['cpu']:6.2f}% | CMD: {p['cmd'][:80]}")
    else:
        print("✓ 未发现异常高 CPU 使用的进程")

    if high_mem:
        print(f"\n⚠️  发现 {len(high_mem)} 个高内存使用率的进程（>10%）：")
        for p in sorted(high_mem, key=lambda x: x["mem"], reverse=True)[:10]:
            print(f"  PID: {p['pid']:8d} | MEM: {p['mem']:6.2f}% | CMD: {p['cmd'][:80]}")
    else:
        print("✓ 未发现异常高内存使用的进程")


def main():
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    zombies = check_zombies()
    runner_procs = check_instance_runner()
    perf_procs = check_perf_processes()
    long_running = check_long_running()
    check_resource_usage()

    # 如果有 instance_runner 进程，显示其进程树
    if runner_procs:
        for proc in runner_procs:
            check_process_tree(proc["pid"])

    # 总结
    print("\n=== 总结 ===")
    print(f"僵尸进程数: {len(zombies)}")
    print(f"instance_runner.py 进程数: {len(runner_procs)}")
    print(f"perf_run.py 进程数: {len(perf_procs)}")
    print(f"长时间运行进程数: {len(long_running)}")

    if zombies:
        print("\n⚠️  建议：如果发现僵尸进程，可以尝试：")
        print("  1. 等待父进程清理（通常会自动清理）")
        print("  2. 如果父进程已死，僵尸进程会在 init 进程接管后自动清理")
        print("  3. 重启系统可以清理所有僵尸进程（最后手段）")

    if len(perf_procs) > 35:
        print(f"\n⚠️  警告：发现 {len(perf_procs)} 个 perf_run.py 进程，超过预期的并行数（35）")
        print("  可能存在进程泄漏，建议检查是否有进程卡住或未正确退出")


if __name__ == "__main__":
    main()
