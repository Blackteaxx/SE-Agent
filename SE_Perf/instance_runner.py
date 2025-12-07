#!/usr/bin/env python3
"""
按实例运行的入口脚本（控制全局并行度），并调用 SE_Perf/perf_run.py。

目标：
- 将原本“一个迭代中所有实例一起跑”的方式，改为“以实例为单位的独立运行”。
- 通过本脚本统一控制全局并行程度（同时启动多少个实例的 perf_run）。
- 对实例目录进行临时重组：外层为实例名文件夹，内层才是 JSON（每次仅包含一个 JSON），以供 perf_run 仅处理该实例。
- 对输出目录进行改写：使用 `output_dir/instance_name` 作为运行根，方便下游按实例聚合。

用法示例：
  python SE_Perf/instance_runner.py --config configs/se_perf_configs/deepseek-v3.1-Plan-AutoSelect.yaml \
         --max-parallel 2 --mode execute

可选：
- 使用 `--dry-run` 仅预览本脚本将执行的命令，不实际调用 perf_run。
- 使用 `--limit` 仅选择前 N 个实例进行快速试跑。
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import signal
import threading
from datetime import datetime
from pathlib import Path

import yaml
from core.utils.collect_outputs import collect_outputs

# ensure 'core' package is importable when running as script
sys.path.insert(0, str(Path(__file__).parent))


def _load_yaml(path: Path) -> dict:
    """读取 YAML 文件为字典。"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_abs(path_like: str | Path, base: Path) -> Path:
    """将路径转换为绝对路径（相对路径以 base 为根）。"""
    p = Path(path_like)
    return p if p.is_absolute() else (base / p)


def _discover_instances(instances_dir: Path) -> list[Path]:
    """枚举实例 JSON 文件。"""
    return sorted(instances_dir.glob("*.json"))


def _prepare_temp_space(inst_files: list[Path], tmp_root: Path) -> dict[str, Path]:
    """
    生成临时空间结构：
    - tmp_root/instance_name/instance_name.json

    返回：{instance_name: tmp_instance_dir}
    """
    tmp_root.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Path] = {}
    for fp in inst_files:
        name = fp.stem
        inst_dir = tmp_root / name
        inst_dir.mkdir(parents=True, exist_ok=True)
        target = inst_dir / f"{name}.json"
        shutil.copy(fp, target)
        mapping[name] = inst_dir
    return mapping


def _write_per_instance_config(
    base_cfg: dict,
    tmp_instance_dir: Path,
    instance_name: str,
    config_out_path: Path,
    timestamp: str,
    override_base_out: str | None = None,
) -> Path:
    """
    基于原始 SE 配置生成“单实例”配置：
    - instances.instances_dir 指向 tmp_instance_dir（仅含该实例 JSON）
    - output_dir 在原有基础上追加 "/instance_name"（perf_run 内部会再拼接迭代目录）
    """
    cfg = dict(base_cfg)  # 浅拷贝足够（字段均为原生类型或嵌套字典）

    instances = cfg.get("instances", {}) or {}
    instances["instances_dir"] = str(tmp_instance_dir)
    cfg["instances"] = instances

    orig_out = str(cfg.get("output_dir", "trajectories_perf/run_{timestamp}")).rstrip("/")
    base_out = str(override_base_out) if override_base_out else orig_out
    final_base_out = base_out.replace("{timestamp}", timestamp)
    cfg["output_dir"] = f"{final_base_out}/{instance_name}"

    # 如果配置了 Global Memory 且 backend 为 chroma，且未显式指定 persist_path
    # 则自动将 persist_path 设置为 {cfg["output_dir"]}/global_memory
    # 这样每个实例的 ChromaDB 就会存储在各自的输出目录下，避免并发写入冲突
    global_memory_bank = cfg.get("global_memory_bank")
    if isinstance(global_memory_bank, dict) and global_memory_bank.get("enabled"):
        memory = global_memory_bank.get("memory", {})
        if memory.get("backend") == "chroma":
            chroma_cfg = memory.get("chroma", {})
            # 强制覆盖 persist_path 为实例输出目录下的 global_memory
            chroma_cfg["persist_path"] = f"{final_base_out}/global_memory"
            memory["chroma"] = chroma_cfg
            global_memory_bank["memory"] = memory
            cfg["global_memory_bank"] = global_memory_bank

    config_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return config_out_path


def _build_perf_cmd(config_path: Path, mode: str, project_root: Path) -> list[str]:
    """
    构建调用 perf_run.py 的命令（不执行，仅返回命令列表）。
    """
    return [
        sys.executable,
        str(project_root / "SE_Perf" / "perf_run.py"),
        "--config",
        str(config_path),
        "--mode",
        mode,
    ]


import json


def _log_token_usage(output_dir: Path, logger=None):
    """
    统计并记录 Token 使用情况
    """
    token_log_file = output_dir / "token_usage.jsonl"
    if not token_log_file.exists():
        return

    total_prompt = 0
    total_completion = 0
    total = 0
    by_context: dict[str, dict[str, int]] = {}

    try:
        with open(token_log_file, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    pt = int(rec.get("prompt_tokens") or 0)
                    ct = int(rec.get("completion_tokens") or 0)
                    tt = int(rec.get("total_tokens") or (pt + ct))
                    ctx = str(rec.get("context") or "unknown")

                    total_prompt += pt
                    total_completion += ct
                    total += tt

                    agg = by_context.setdefault(ctx, {"prompt": 0, "completion": 0, "total": 0})
                    agg["prompt"] += pt
                    agg["completion"] += ct
                    agg["total"] += tt
                except Exception:
                    continue

        print("\n📈 Token 使用统计:")
        print(f"  Total: {total} (Prompt: {total_prompt}, Completion: {total_completion})")
        if by_context:
            print("  按上下文分类:")
            for ctx, vals in by_context.items():
                print(f"    - {ctx}: prompt={vals['prompt']}, completion={vals['completion']}, total={vals['total']}")

        # 如果提供了 logger，则记录详细 JSON
        if logger:
            logger.info(
                json.dumps(
                    {
                        "token_usage_total": {"prompt": total_prompt, "completion": total_completion, "total": total},
                        "by_context": by_context,
                        "token_log_file": str(token_log_file),
                    },
                    ensure_ascii=False,
                )
            )
    except Exception:
        pass


def _aggregate_token_stats(instance_output_dirs: list[Path], base_root_dir: str):
    """
    聚合所有实例的 Token 消耗
    """
    total_prompt = 0
    total_completion = 0
    total = 0
    by_context: dict[str, dict[str, int]] = {}

    print("\n=== 全局 Token 消耗统计 ===")

    for inst_dir in instance_output_dirs:
        token_file = inst_dir / "token_usage.jsonl"
        if not token_file.exists():
            continue

        try:
            with open(token_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        pt = int(rec.get("prompt_tokens") or 0)
                        ct = int(rec.get("completion_tokens") or 0)
                        tt = int(rec.get("total_tokens") or (pt + ct))
                        ctx = str(rec.get("context") or "unknown")

                        total_prompt += pt
                        total_completion += ct
                        total += tt

                        agg = by_context.setdefault(ctx, {"prompt": 0, "completion": 0, "total": 0})
                        agg["prompt"] += pt
                        agg["completion"] += ct
                        agg["total"] += tt
                    except Exception:
                        continue
        except Exception:
            pass

    print(f"📈 Total All Instances: {total} (Prompt: {total_prompt}, Completion: {total_completion})")
    if by_context:
        print("  按上下文分类 (All Instances):")
        for ctx, vals in by_context.items():
            print(f"    - {ctx}: prompt={vals['prompt']}, completion={vals['completion']}, total={vals['total']}")

    # 尝试写入总的 token_usage.json 到根目录
    try:
        root_token_file = Path(base_root_dir) / "total_token_usage.json"
        with open(root_token_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "token_usage_total": {"prompt": total_prompt, "completion": total_completion, "total": total},
                    "by_context": by_context,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"已保存全局 Token 统计至: {root_token_file}")
    except Exception as e:
        print(f"保存全局 Token 统计失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="按实例运行的入口脚本（控制并行度），封装调用 perf_run.py")
    parser.add_argument("--config", required=True, help="SE 配置文件路径（作为基础模板）")
    parser.add_argument("--instances-dir", help="覆盖配置中的 instances_dir（可选）")
    parser.add_argument("--max-parallel", type=int, default=1, help="同时并发的实例数（全局并行度）")
    parser.add_argument("--mode", choices=["demo", "execute"], default="execute", help="perf_run.py 的运行模式")
    parser.add_argument("--output-dir", help="覆盖配置中的 output_dir（可选，用于续跑或指定输出根目录）")
    parser.add_argument("--limit", type=int, help="仅处理前 N 个实例（快速试跑）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览命令，不实际执行")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_cfg_path = _ensure_abs(args.config, project_root)
    base_cfg = _load_yaml(base_cfg_path)

    # 解析实例目录（优先 CLI 覆盖）
    if args.instances_dir:
        inst_dir = _ensure_abs(args.instances_dir, project_root)
    else:
        inst_dir = _ensure_abs((base_cfg.get("instances", {}) or {}).get("instances_dir", "instances"), project_root)

    inst_files = _discover_instances(inst_dir)
    if args.limit is not None:
        inst_files = inst_files[: max(0, int(args.limit))]

    if not inst_files:
        print(f"未在 {inst_dir} 找到实例 JSON 文件")
        sys.exit(1)

    import uuid

    # 临时空间：tmp/instance_runner/<timestamp-like>_<random_suffix>
    # 添加随机后缀防止多次运行时目录冲突
    random_suffix = str(uuid.uuid4())[:8]
    tmp_root = project_root / "tmp" / "instance_runner" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_suffix}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    mapping = _prepare_temp_space(inst_files, tmp_root)

    # 生成 timestamp 并计算输出根目录（支持 CLI 覆盖）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        # 如果传入的覆盖目录包含占位符，则替换；否则按原样使用
        override_root = str(_ensure_abs(args.output_dir, project_root))
        base_root_dir = override_root.replace("{timestamp}", timestamp)
    else:
        base_root_dir = base_cfg.get("output_dir", "trajectories_perf/run_{timestamp}").replace(
            "{timestamp}", timestamp
        )

    # 构建 per-instance 配置文件并运行
    per_instance_cfg_paths: dict[str, Path] = {}
    for name, tmp_inst_dir in mapping.items():
        cfg_out = tmp_inst_dir / "se_config.yaml"
        per_instance_cfg_paths[name] = _write_per_instance_config(
            base_cfg,
            tmp_inst_dir,
            name,
            cfg_out,
            timestamp,
            override_base_out=base_root_dir,
        )

    # 并行执行（简单的批次调度：窗口大小 = max_parallel）
    names = list(per_instance_cfg_paths.keys())
    total = len(names)
    max_parallel = max(1, int(args.max_parallel))
    i = 0
    successes = 0
    failures = 0
    print(f"准备运行 {total} 个实例；并行度 = {max_parallel}；模式 = {args.mode}；dry_run = {args.dry_run}")

    # 准备一个全局列表来跟踪正在运行的子进程
    active_processes: list[subprocess.Popen] = []
    active_processes_lock = threading.Lock()

    def _cleanup_processes():
        """终止所有活跃的子进程。"""
        with active_processes_lock:
            # 无论是否有记录的 active_processes，都尝试清理整个进程组
            # 这可以捕获那些还没来得及加入 active_processes 列表的孙子进程
            try:
                # 给整个进程组发送 SIGTERM
                os.killpg(0, signal.SIGTERM)
            except Exception:
                pass

            if not active_processes:
                return

            print(f"\n正在终止 {len(active_processes)} 个活跃子进程...")
            for p in active_processes:
                try:
                    if p.poll() is None:  # 仅终止还在运行的进程
                        p.terminate()
                except Exception:
                    pass
            # 等待一小段时间让进程优雅退出
            try:
                for p in active_processes:
                    try:
                        p.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        p.kill()  # 强制杀死
                    except Exception:
                        pass
            except Exception:
                pass

    def _signal_handler(sig, frame):
        """处理终止信号。"""
        print(f"\n接收到信号 {sig}，准备退出...")
        _cleanup_processes()
        sys.exit(1)

    # 注册信号处理
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 创建一个进程组，以便能够一次性杀死所有子进程
    # 注意：在某些环境中，os.setsid() 可能会改变进程组 ID，这在 shell 脚本或 CI 环境中可能会有副作用。
    # 但对于交互式运行，这是确保彻底清理子进程的好方法。
    # 为了避免影响主进程的信号处理，我们只在创建新进程组时这样做。
    # 但这里我们是在主进程中运行，所以我们更倾向于在 _cleanup_processes 中处理。
    # 不过，如果 instance_runner.py 被杀死，我们希望它的子进程也被杀死。
    # 一个常见的做法是把当前进程设为进程组组长。
    try:
        os.setpgrp()
    except Exception:
        pass

    # 包装 subprocess.run 以便跟踪 Popen 对象
    def _run_process(cmd, cwd):
        try:
            # 使用 Popen 替代 subprocess.run 以便获取句柄
            with subprocess.Popen(cmd, cwd=cwd, text=True) as proc:
                with active_processes_lock:
                    active_processes.append(proc)

                try:
                    proc.wait()
                    return subprocess.CompletedProcess(proc.args, proc.returncode)
                finally:
                    with active_processes_lock:
                        if proc in active_processes:
                            active_processes.remove(proc)
        except Exception as e:
            raise e

    if args.dry_run:
        for name in names:
            cfg_path = per_instance_cfg_paths[name]
            cmd = _build_perf_cmd(cfg_path, args.mode, project_root)
            print(f"[DRY-RUN] {name}: {' '.join(cmd)}")
            successes += 1
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_map: dict[concurrent.futures.Future, str] = {}
            for name in names:
                cfg_path = per_instance_cfg_paths[name]
                cmd = _build_perf_cmd(cfg_path, args.mode, project_root)
                # 改为调用 _run_process
                fut = executor.submit(_run_process, cmd, cwd=str(project_root))
                future_map[fut] = name
            for fut in concurrent.futures.as_completed(future_map):
                name = future_map[fut]
                try:
                    res = fut.result()
                    rc = getattr(res, "returncode", None)
                    if rc == 0:
                        print(f"✅ 完成实例: {name}")
                        successes += 1
                    else:
                        print(f"❌ 失败实例: {name}（返回码 {rc}）")
                        failures += 1
                except Exception as e:
                    print(f"❌ 失败实例: {name}（异常 {e}）")
                    failures += 1

    print(f"运行结束：成功 {successes}/{total}；失败 {failures}")

    if not args.dry_run:
        try:
            res = collect_outputs(base_root_dir, names)
            print(f"聚合 traj.pool: {res.get('traj_pool')}")
            print(f"聚合 all_hist.json: {res.get('all_hist')}")
            print(f"聚合 final.json: {res.get('final')}")
        except Exception:
            print("批次聚合失败")

        # 统计所有实例的 Token 消耗
        try:
            # per_instance_cfg_paths 里的 key 是 instance_name，但我们需要的是实际的输出目录
            # 在 main 函数前面我们计算了: final_base_out = orig_out.replace("{timestamp}", timestamp)
            # 并且 per instance 的 output_dir 是 f"{final_base_out}/{instance_name}"
            # 所以我们可以重建这些路径

            instance_output_dirs = []
            for name in names:
                inst_out_dir = Path(base_root_dir) / name
                instance_output_dirs.append(inst_out_dir)

            _aggregate_token_stats(instance_output_dirs, base_root_dir)
        except Exception as e:
            print(f"Token 聚合统计失败: {e}")


if __name__ == "__main__":
    main()
