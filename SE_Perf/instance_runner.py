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
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


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
    base_cfg: dict, tmp_instance_dir: Path, instance_name: str, config_out_path: Path, timestamp: str
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
    # 使用调用者传入的已生成 timestamp 替换占位符
    final_base_out = orig_out.replace("{timestamp}", timestamp)
    cfg["output_dir"] = f"{final_base_out}/{instance_name}"

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


def main():
    parser = argparse.ArgumentParser(description="按实例运行的入口脚本（控制并行度），封装调用 perf_run.py")
    parser.add_argument("--config", required=True, help="SE 配置文件路径（作为基础模板）")
    parser.add_argument("--instances-dir", help="覆盖配置中的 instances_dir（可选）")
    parser.add_argument("--max-parallel", type=int, default=1, help="同时并发的实例数（全局并行度）")
    parser.add_argument("--mode", choices=["demo", "execute"], default="execute", help="perf_run.py 的运行模式")
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

    # 临时空间：tmp/instance_runner/<timestamp-like>
    tmp_root = project_root / "tmp" / "instance_runner"
    tmp_root.mkdir(parents=True, exist_ok=True)

    mapping = _prepare_temp_space(inst_files, tmp_root)

    # 生成 timestamp 并替换输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root_dir = base_cfg["output_dir"].replace("{timestamp}", timestamp)

    # 构建 per-instance 配置文件并运行
    per_instance_cfg_paths: dict[str, Path] = {}
    for name, tmp_inst_dir in mapping.items():
        cfg_out = tmp_inst_dir / "se_config.yaml"
        per_instance_cfg_paths[name] = _write_per_instance_config(base_cfg, tmp_inst_dir, name, cfg_out, timestamp)

    # 并行执行（简单的批次调度：窗口大小 = max_parallel）
    names = list(per_instance_cfg_paths.keys())
    total = len(names)
    max_parallel = max(1, int(args.max_parallel))
    i = 0
    successes = 0
    failures = 0
    print(f"准备运行 {total} 个实例；并行度 = {max_parallel}；模式 = {args.mode}；dry_run = {args.dry_run}")

    while i < total:
        batch = names[i : i + max_parallel]
        procs: list[tuple[str, subprocess.Popen]] = []
        # 启动当前批次
        for name in batch:
            cfg_path = per_instance_cfg_paths[name]
            cmd = _build_perf_cmd(cfg_path, args.mode, project_root)
            if args.dry_run:
                print(f"[DRY-RUN] {name}: {' '.join(cmd)}")
                successes += 1
            else:
                try:
                    p = subprocess.Popen(
                        cmd,
                        cwd=str(project_root),
                        text=True,
                    )
                except Exception as e:
                    print(f"❌ 启动失败: {name}（{e}）")
                    failures += 1
                    continue
                procs.append((name, p))
        # 等待批次完成
        if not args.dry_run:
            for name, p in procs:
                rc = p.wait()
                if rc == 0:
                    print(f"✅ 完成实例: {name}")
                    successes += 1
                else:
                    print(f"❌ 失败实例: {name}（返回码 {rc}）")
                    failures += 1
        i += max_parallel

    print(f"运行结束：成功 {successes}/{total}；失败 {failures}")

    if not args.dry_run:
        try:
            agg_pool: dict[str, dict] = {}
            agg_hist: dict[str, dict] = {}
            agg_final: dict[str, str] = {}

            for name in names:
                inst_root = base_root_dir / name
                pool_path = inst_root / "traj.pool"
                preds_path = inst_root / "preds.json"
                final_path = inst_root / "final.json"

                pool_data: dict[str, dict] = {}
                try:
                    if pool_path.exists():
                        pool_data = json.loads(pool_path.read_text(encoding="utf-8")) or {}
                        for k, v in pool_data.items():
                            agg_pool[k] = v
                except Exception:
                    pass

                try:
                    if preds_path.exists():
                        preds = json.loads(preds_path.read_text(encoding="utf-8")) or {}
                        for inst_id, entries in preds.items():
                            iteration_map: dict[str, dict] = {}
                            for e in entries or []:
                                it = e.get("iteration")
                                code = e.get("code") or ""
                                runtime = e.get("runtime")
                                if it is None:
                                    continue
                                iteration_map[str(it)] = {"code": code, "runtime": runtime}
                            problem = None
                            try:
                                pinfo = pool_data.get(inst_id) or {}
                                problem = pinfo.get("problem")
                            except Exception:
                                problem = None
                            agg_hist[inst_id] = {"problem": problem, "iteration": iteration_map}
                except Exception:
                    pass

                try:
                    if final_path.exists():
                        fdata = json.loads(final_path.read_text(encoding="utf-8")) or {}
                        for inst_id, code in fdata.items():
                            agg_final[inst_id] = code
                except Exception:
                    pass

            base_root_dir.mkdir(parents=True, exist_ok=True)
            out_pool = base_root_dir / "traj.pool"
            out_hist = base_root_dir / "all_hist.json"
            out_final = base_root_dir / "final.json"

            try:
                out_pool.write_text(json.dumps(agg_pool, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"聚合 traj.pool: {out_pool}")
            except Exception:
                print("聚合 traj.pool 失败")

            try:
                out_hist.write_text(json.dumps(agg_hist, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"聚合 all_hist.json: {out_hist}")
            except Exception:
                print("聚合 all_hist.json 失败")

            try:
                out_final.write_text(json.dumps(agg_final, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"聚合 final.json: {out_final}")
            except Exception:
                print("聚合 final.json 失败")
        except Exception:
            print("批次聚合失败")


if __name__ == "__main__":
    main()
