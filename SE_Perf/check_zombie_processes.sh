#!/bin/bash
# 检查僵尸进程和孤儿进程的脚本

echo "=== 检查僵尸进程 ==="
# 查找所有僵尸进程（状态为 Z）
zombies=$(ps aux | awk '$8 ~ /^Z/ { print $2, $8, $11, $12, $13 }')
if [ -z "$zombies" ]; then
    echo "✓ 未发现僵尸进程"
else
    echo "⚠️  发现僵尸进程："
    echo "$zombies"
fi

echo ""
echo "=== 检查 instance_runner.py 相关进程 ==="
# 查找所有 instance_runner.py 进程
runner_procs=$(ps aux | grep -E "instance_runner\.py" | grep -v grep)
if [ -z "$runner_procs" ]; then
    echo "✓ 未发现 instance_runner.py 进程"
else
    echo "发现 instance_runner.py 进程："
    echo "$runner_procs"
fi

echo ""
echo "=== 检查 perf_run.py 相关进程 ==="
# 查找所有 perf_run.py 进程
perf_procs=$(ps aux | grep -E "perf_run\.py" | grep -v grep)
if [ -z "$perf_procs" ]; then
    echo "✓ 未发现 perf_run.py 进程"
else
    echo "⚠️  发现 perf_run.py 进程："
    echo "$perf_procs"
    echo ""
    echo "进程详情（PID, 状态, CPU%, 内存%, 启动时间, 命令）："
    ps aux | grep -E "perf_run\.py" | grep -v grep | awk '{printf "PID: %-8s STAT: %-4s CPU: %-6s MEM: %-6s START: %-10s CMD: %s\n", $2, $8, $3, $4, $9, $11" "$12" "$13" "$14" "$15}'
fi

echo ""
echo "=== 检查长时间运行的 Python 进程（超过 24 小时）==="
# 查找运行时间超过 24 小时的 Python 进程
long_running=$(ps -eo pid,etime,cmd | grep -E "python.*(instance_runner|perf_run)" | grep -v grep | awk '{print $0}')
if [ -z "$long_running" ]; then
    echo "✓ 未发现长时间运行的进程"
else
    echo "⚠️  发现长时间运行的进程："
    echo "$long_running"
fi

echo ""
echo "=== 检查进程树（instance_runner 及其子进程）==="
# 查找 instance_runner 的 PID
runner_pid=$(ps aux | grep -E "instance_runner\.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$runner_pid" ]; then
    echo "instance_runner.py (PID: $runner_pid) 的进程树："
    pstree -p $runner_pid 2>/dev/null || echo "无法显示进程树（可能需要安装 pstree）"
else
    echo "未找到 instance_runner.py 进程"
fi

echo ""
echo "=== 统计信息 ==="
# 统计各种状态的进程数
total_procs=$(ps aux | wc -l)
zombie_count=$(ps aux | awk '$8 ~ /^Z/ { count++ } END { print count+0 }')
python_procs=$(ps aux | grep python | grep -v grep | wc -l)
perf_procs_count=$(ps aux | grep -E "perf_run\.py" | grep -v grep | wc -l)

echo "总进程数: $total_procs"
echo "僵尸进程数: $zombie_count"
echo "Python 进程数: $python_procs"
echo "perf_run.py 进程数: $perf_procs_count"

echo ""
echo "=== 检查是否有进程占用过多资源 ==="
# 查找 CPU 或内存使用率高的进程
high_cpu=$(ps aux | awk '$3 > 50.0 && $11 ~ /python/ { print $2, $3, $4, $11, $12, $13 }' | head -10)
if [ -n "$high_cpu" ]; then
    echo "⚠️  发现高 CPU 使用率的 Python 进程（>50%）："
    echo "$high_cpu"
else
    echo "✓ 未发现异常高 CPU 使用的进程"
fi

high_mem=$(ps aux | awk '$4 > 10.0 && $11 ~ /python/ { print $2, $3, $4, $11, $12, $13 }' | head -10)
if [ -n "$high_mem" ]; then
    echo "⚠️  发现高内存使用率的 Python 进程（>10%）："
    echo "$high_mem"
fi

