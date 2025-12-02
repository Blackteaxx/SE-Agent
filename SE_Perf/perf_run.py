#!/usr/bin/env python3
"""
PerfAgent 集成执行脚本
模仿 SE/basic_run.py 的结构，在 SE 框架中驱动 perfagent 的单/多实例性能优化。
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

# 添加SE目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入SE日志系统
from core.utils.se_logger import get_se_logger, setup_se_logging
from core.utils.traj_extractor import TrajExtractor
from core.utils.traj_pool_manager import TrajPoolManager
from core.utils.trajectory_processor import TrajectoryProcessor

# 导入operator系统
from operators import create_operator


def _prepare_initial_code_dir(
    initial_code_dir: Path, input_labels: list[str], traj_pool_manager: TrajPoolManager, logger
) -> Path | None:
    """
    准备初始代码目录：从轨迹池中提取指定标签的代码，写入到 initial_code_dir。
    用于支持基于已有轨迹继续优化的场景。
    """
    try:
        initial_code_dir.mkdir(parents=True, exist_ok=True)
        written_instances: set[str] = set()

        pool_data = traj_pool_manager.get_all_trajectories() or {}
        for label in input_labels:
            found_inst = None
            found_entry = None
            for inst_name, entry in pool_data.items():
                if not isinstance(entry, dict):
                    continue
                if label in entry and isinstance(entry[label], dict):
                    found_inst = str(inst_name)
                    found_entry = entry[label]
                    break
            if not found_inst or not isinstance(found_entry, dict):
                logger.warning(f"初始代码准备：未找到轨迹 {label}")
                continue
            code = (
                found_entry.get("content")
                or ((found_entry.get("summary") or {}).get("final_solution") or {}).get("code")
                or ""
            )
            if not code:
                logger.warning(f"初始代码准备：轨迹 {label} 缺少代码内容")
                continue
            if found_inst in written_instances:
                logger.info(f"初始代码准备：实例 {found_inst} 已存在，跳过重复标签 {label}")
                continue
            file_path = initial_code_dir / f"{found_inst}.py"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            written_instances.add(found_inst)
        if not written_instances:
            logger.warning("初始代码准备：没有任何代码文件被写入")
            return None
        return initial_code_dir
    except Exception as e:
        logger.error(f"初始代码准备失败: {e}")
        return None


def _execute_operator_step(
    step: dict, se_config: dict, traj_pool_manager: TrajPoolManager, workspace_dir: str, logger
) -> dict:
    """
    执行单个算子步骤。
    根据 operator_name 创建算子实例并调用其 run 方法。
    """
    operator_name = step.get("operator")
    if not operator_name:
        logger.error("算子步骤缺少 operator 字段")
        return {}

    # 将选择模式注入算子配置（算子内部优先使用 step['selection_mode']，其次使用 config['operator_selection_mode']）
    op_cfg = dict(se_config) if isinstance(se_config, dict) else {}
    try:
        if isinstance(step, dict) and step.get("selection_mode"):
            op_cfg["operator_selection_mode"] = step.get("selection_mode")
    except Exception:
        pass
    operator = create_operator(operator_name, op_cfg)
    if not operator:
        logger.error(f"无法创建算子实例: {operator_name}")
        return {}

    result = {}
    try:
        result = operator.run(step, traj_pool_manager, workspace_dir)
    except Exception as e:
        logger.error(f"算子执行失败: {operator_name}, {e}")
        return {}

    if isinstance(result.get("initial_code_dir"), str):
        p = Path(result["initial_code_dir"]) if result.get("initial_code_dir") else None
        if p and p.exists():
            logger.info(f"算子返回初始代码目录: {p}")
        gen_cnt = result.get("generated_count")
        try:
            if gen_cnt is not None:
                logger.info(f"算子生成初始代码数量: {int(gen_cnt)}")
        except Exception:
            pass

    # 历史兼容：不再支持 add_to_pool，算子必须返回 initial_code_dir
    return result


def _summarize_iteration_to_pool(
    iteration_dir: Path,
    iteration_index: int,
    traj_pool_manager: TrajPoolManager,
    se_config: dict,
    logger,
    label_prefix: str | None = None,
    source_labels: list[str] | None = None,
    source_labels_map: dict[str, list[str]] | None = None,
    operator_name: str | None = None,
) -> None:
    """
    将一次迭代生成的轨迹数据（.tra 文件等）提取并汇总到轨迹池中。
    包含提取实例数据、格式化轨迹条目、并调用 traj_pool_manager 进行持久化。
    """
    try:
        extractor = TrajExtractor()
        # 包含性能指标
        extracted = extractor.extract_instance_data(iteration_dir, include_metrics=True)
        if not extracted:
            logger.warning("本迭代没有有效的实例数据用于轨迹池总结")
            return
        trajectories_to_process = []
        for item in extracted:
            try:
                instance_name, problem_description, tra_content, patch_content, perf_metrics = item
            except Exception:
                # 兼容不含 metrics 的旧格式
                instance_name, problem_description, tra_content, patch_content = item
                perf_metrics = None
            label = str(label_prefix) if label_prefix else f"iter{iteration_index}"
            per_inst_src = None
            try:
                if source_labels_map and isinstance(source_labels_map, dict):
                    per_inst_src = source_labels_map.get(str(instance_name))
            except Exception:
                per_inst_src = None
            trajectories_to_process.append(
                {
                    "label": label,
                    "instance_name": instance_name,
                    "problem_description": problem_description,
                    "trajectory_content": tra_content,
                    "patch_content": patch_content,
                    "iteration": iteration_index,
                    "performance": (perf_metrics or {}).get("final_performance"),
                    "source_dir": str(iteration_dir / instance_name),
                    "source_entry_labels": list(per_inst_src or []),
                    "operator_name": str(operator_name) if operator_name is not None else None,
                }
            )
        traj_pool_manager.summarize_and_add_trajectories(
            trajectories_to_process, num_workers=se_config.get("num_workers")
        )
        pool_stats = traj_pool_manager.get_pool_stats()
        logger.info(f"轨迹池更新: {pool_stats['total_trajectories']}条轨迹")
    except Exception as pool_error:
        logger.error(f"迭代轨迹池更新失败: {pool_error}")


def write_iteration_preds(base_dir: Path, logger) -> Path | None:
    """
    聚合当前迭代各实例的结果，生成 preds.json。

    - passed：直接依据 final_performance 是否为 inf（或字符串表示的无穷）判断。
    - runtime：输出最终性能值（若存在），否则回退到初始评估的修剪均值。
    - code：优先使用 optimized_code，回退到 initial_code。

    返回生成的 preds.json 路径，失败返回 None。
    """
    preds = {}
    try:
        for inst_dir in base_dir.iterdir():
            if not inst_dir.is_dir():
                continue
            res_file = inst_dir / "result.json"
            if not res_file.exists():
                continue
            try:
                with open(res_file, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            instance_id = data.get("instance_id", inst_dir.name)
            code = data.get("optimized_code", "")

            # 读取最终性能值并用于 passed 判断
            final_perf = data.get("final_performance")

            # runtime：优先 final_performance，否则 initial trimmed_mean
            runtime = final_perf

            # passed：final_performance 为 inf 则 False，否则 True
            passed = not math.isinf(float(final_perf)) if final_perf is not None else False
            preds[str(instance_id)] = {
                "code": code,
                "passed": passed,
                "runtime": runtime,
            }

        preds_path = base_dir / "preds.json"
        with open(preds_path, "w", encoding="utf-8") as pf:
            json.dump(preds, pf, indent=2, ensure_ascii=False)
        print(f"📝 已生成 preds.json: {preds_path}")
        logger.info(f"已生成 preds.json: {preds_path}")
        return preds_path
    except Exception as e:
        logger.warning(f"生成 preds.json 失败: {e}")
        return None


def aggregate_all_iterations_preds(root_output_dir: Path, logger) -> Path | None:
    """
    汇总所有 iteration_* 目录下的 preds.json，添加迭代号并写入运行根目录的 preds.json。

    变更：不再过滤未通过项。对于未通过的实例，其 code 字段明确设为空字符串""，以避免后续输出缺失。

    输出结构示例：
    {
      "inst1": [
        {"iteration": 1, "code": "...", "runtime": 1.23},
        {"iteration": 2, "code": "...", "runtime": 1.11}
      ],
      "inst2": [
        {"iteration": 2, "code": "", "runtime": "Infinity"}
      ]
    }
    """
    aggregated: dict[str, list[dict]] = {}
    try:
        for iter_dir in sorted(root_output_dir.glob("iteration_*")):
            if not iter_dir.is_dir():
                continue
            # 解析迭代号
            try:
                iter_num = int(iter_dir.name.split("_")[-1])
            except Exception:
                iter_num = None

            preds_file = iter_dir / "preds.json"
            if not preds_file.exists():
                continue
            try:
                with open(preds_file, encoding="utf-8") as pf:
                    preds = json.load(pf)
            except Exception:
                continue

            for instance_id, info in preds.items():
                try:
                    passed = bool(info.get("passed", False))
                    # 未通过的实例，code 明确置为空字符串
                    code = info.get("code", "") if passed else ""
                    runtime = info.get("runtime")
                    entry = {"iteration": iter_num, "code": code, "runtime": runtime}
                    aggregated.setdefault(str(instance_id), []).append(entry)
                except Exception:
                    continue

        agg_path = root_output_dir / "preds.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        print(f"📝 汇总 preds.json: {agg_path}")
        logger.info(f"汇总 preds.json: {agg_path}")
        return agg_path
    except Exception as e:
        logger.warning(f"汇总 preds.json 失败: {e}")
        return None


def write_final_json_from_preds(aggregated_preds_path: Path, root_output_dir: Path, logger) -> Path | None:
    """
    从运行根目录的 preds.json（汇总）选择每个实例 runtime 最小的解，写入 final.json。

    文件结构：
    {
      "instance_name": "code"
    }
    """
    try:
        with open(aggregated_preds_path, encoding="utf-8") as f:
            aggregated = json.load(f)
    except Exception as e:
        logger.warning(f"读取汇总 preds.json 失败: {e}")
        return None

    def to_float(rt):
        try:
            if rt is None:
                return float("inf")
            if isinstance(rt, (int, float)):
                return float(rt)
            if isinstance(rt, str):
                lowered = rt.strip().lower()
                if lowered in ("inf", "infinity", "nan"):
                    return float("inf")
                return float(rt)
            return float("inf")
        except Exception:
            return float("inf")

    final_map: dict[str, str] = {}
    try:
        # 先根据最小 runtime 选择最佳解；若均未通过（runtime 为 inf），code 会是空字符串
        for instance_id, entries in aggregated.items():
            if not isinstance(entries, list) or not entries:
                continue
            try:
                best = min(entries, key=lambda e: to_float(e.get("runtime")))
            except Exception:
                continue
            final_map[str(instance_id)] = best.get("code", "") or ""

        # 注：不再进行“补齐空字符串”，final.json 仅依据汇总 preds.json 的最小 runtime 选择结果。

        final_path = root_output_dir / "final.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_map, f, indent=2, ensure_ascii=False)
        print(f"🏁 生成 final.json: {final_path}")
        logger.info(f"生成 final.json: {final_path}")
        return final_path
    except Exception as e:
        logger.warning(f"生成 final.json 失败: {e}")
        return None


def create_temp_perf_config(
    base_config_path: str | None,
    se_model_cfg: dict,
    logger,
    extra_overrides: dict | None = None,
) -> Path | None:
    """基于基础配置生成一个临时 PerfAgent 配置文件，并按需覆盖字段。

    - 覆盖模型相关字段（来自 SE 主模型设置）
    - 覆盖顶层控制字段（目前支持 max_iterations），用于与 SE 配置对齐

    返回临时配置文件路径；若失败则返回 None。
    """
    try:
        perf_cfg = {}
        if base_config_path:
            with open(base_config_path, encoding="utf-8") as f:
                perf_cfg = yaml.safe_load(f) or {}

        # 仅覆盖 PerfAgent 支持的模型字段
        allowed_keys = [
            "name",
            "api_base",
            "api_key",
            "max_input_tokens",
            "max_output_tokens",
            "temperature",
        ]
        override_model = {
            k: v
            for k, v in (se_model_cfg or {}).items()
            if k in allowed_keys and v is not None and (str(v).strip() != "")
        }

        perf_cfg.setdefault("model", {})
        perf_cfg["model"].update(override_model)

        # 顶层覆盖：支持从 SE 配置传入的 max_iterations
        if extra_overrides:
            if "max_iterations" in extra_overrides:
                mi = extra_overrides.get("max_iterations")
                if mi is not None and str(mi).strip() != "":
                    try:
                        perf_cfg["max_iterations"] = int(mi)
                    except Exception:
                        # 若无法转换为整数，仍按原值写入，避免中断
                        perf_cfg["max_iterations"] = mi

        # 生成临时 YAML 文件
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(perf_cfg, tmp, sort_keys=False, allow_unicode=True)
        tmp_path = Path(tmp.name)
        tmp.close()

        print(f"🧩 已生成临时PerfAgent配置: {tmp_path}")
        logger.info(f"临时PerfAgent配置(模型覆盖): {json.dumps(override_model, ensure_ascii=False)}")
        if extra_overrides and "max_iterations" in extra_overrides:
            logger.info(f"临时PerfAgent配置(迭代覆盖): max_iterations={perf_cfg.get('max_iterations')}")
        return tmp_path
    except Exception as e_cfg:
        logger.warning(f"生成临时PerfAgent配置失败: {e_cfg}")
        return None


def call_perfagent(iteration_params, logger, dry_run=False):
    """
    直接调用 perfagent.run_batch 的批量执行接口，运行本次迭代的实例优化。

    Args:
        iteration_params: 包含配置路径、实例目录、输出目录等参数的字典
        logger: 日志记录器
        dry_run: 若为 True，仅打印命令预览而不实际执行（用于演示模式）

    Returns:
        dict: 执行结果，包含 status, returncode 等
    """
    base_config_path = iteration_params.get("perf_base_config")

    try:
        # 基础配置 + SE 主模型配置覆盖 => 生成临时 PerfAgent 配置
        logger.debug(f"使用PerfAgent基础配置: {base_config_path}")

        if dry_run:
            # 在演示模式下也显示将要执行的命令（包含关键参数），便于核对
            base_dir = Path(iteration_params["output_dir"]).resolve()
            base_dir.mkdir(parents=True, exist_ok=True)
            se_model_cfg = iteration_params.get("model") or {}
            temp_config_path = (
                create_temp_perf_config(
                    base_config_path,
                    se_model_cfg,
                    logger,
                    extra_overrides={
                        "max_iterations": iteration_params.get("max_iterations"),
                    },
                )
                or base_config_path
            )
            cmd_preview = [sys.executable, "-m", "perfagent.run_batch"]
            if temp_config_path:
                cmd_preview.extend(["--config", str(temp_config_path)])
            cmd_preview.extend(
                [
                    "--instances-dir",
                    str(iteration_params.get("instances_dir")),
                    "--base-dir",
                    str(base_dir),
                    "--max-workers",
                    str(iteration_params.get("num_workers", 1)),
                ]
            )
            operator_params = iteration_params.get("operator_params", {}) or {}
            icd = operator_params.get("initial_code_dir")
            itd = operator_params.get("instance_templates_dir")
            if icd:
                cmd_preview.extend(["--initial-code-dir", str(icd)])
            if itd:
                cmd_preview.extend(["--instance-templates-dir", str(itd)])
            print(f"🚀 [DEMO] PerfAgent命令预览: {' '.join(cmd_preview)}")
            logger.warning("演示模式：跳过PerfAgent实际执行")
            return {"status": "skipped", "reason": "dry_run", "preview_cmd": " ".join(cmd_preview)}

        # 目标路径和命令（批量执行脚本）
        se_root = Path(__file__).parent
        project_root = se_root.parent

        # 选择实例目录
        instances_dir = iteration_params.get("instances_dir")

        # 基础目录：使用当前迭代的输出目录，让每个实例在其子目录下生成日志与轨迹
        base_dir = Path(iteration_params["output_dir"]).resolve()
        base_dir.mkdir(parents=True, exist_ok=True)

        # 封装：生成临时配置（包含模型覆盖）；失败则回退到基础配置
        se_model_cfg = iteration_params.get("model") or {}
        if base_config_path:
            print(f"📋 使用基础配置文件: {base_config_path}")
        temp_config_path = (
            create_temp_perf_config(
                base_config_path,
                se_model_cfg,
                logger,
                extra_overrides={
                    "max_iterations": iteration_params.get("max_iterations"),
                },
            )
            or base_config_path
        )

        # 优先使用基础配置；operator 仅返回 initial_code_dir（不再使用 instance_templates_dir）
        # 组装命令：先放 --config（若有），再依次加入其他参数，避免解析冲突
        cmd = [sys.executable, "-m", "perfagent.run_batch"]
        if temp_config_path:
            cmd.extend(["--config", str(temp_config_path)])
        cmd.extend(
            [
                "--instances-dir",
                str(instances_dir),
                "--base-dir",
                str(base_dir),
                "--max-workers",
                str(iteration_params.get("num_workers", 1)),
            ]
        )

        # 传递算子输出参数
        operator_params = iteration_params.get("operator_params", {}) or {}
        icd = operator_params.get("initial_code_dir")
        if icd:
            cmd.extend(["--initial-code-dir", str(icd)])
        itd = operator_params.get("instance_templates_dir")
        if itd:
            cmd.extend(["--instance-templates-dir", str(itd)])

        print(f"🚀 执行PerfAgent批量命令: {' '.join(cmd)}")
        print(f"📁 工作目录: {project_root}")
        print("=" * 60)

        result = subprocess.run(cmd, cwd=str(project_root), text=True)

        print("=" * 60)
        if result.returncode == 0:
            logger.info("PerfAgent批量迭代执行成功")
            print("✅ PerfAgent批量迭代执行成功")

            preds_path = write_iteration_preds(base_dir, logger)
            return {
                "status": "success",
                "summary": "success",
                "base_dir": str(base_dir),
                "preds_file": str(preds_path) if preds_path else None,
            }
        else:
            logger.error(f"PerfAgent批量迭代执行失败，返回码: {result.returncode}")
            print(f"❌ PerfAgent批量迭代执行失败，返回码: {result.returncode}")
            return {"status": "failed", "returncode": result.returncode}

    except Exception as e:
        logger.error(f"调用PerfAgent时出错: {e}", exc_info=True)
        return {"status": "error", "exception": str(e)}
    finally:
        # 无临时配置需要清理
        pass


def main():
    """
    主函数：策略驱动的 PerfAgent 多迭代执行。
    解析命令行参数和 SE 配置文件，按步骤执行配置中的算子和 PerfAgent 迭代。
    """

    parser = argparse.ArgumentParser(description="SE 框架 PerfAgent 多迭代执行脚本")
    parser.add_argument("--config", default="SE/configs/se_configs/dpsk.yaml", help="SE 配置文件路径")
    parser.add_argument(
        "--mode", choices=["demo", "execute"], default="execute", help="运行模式: demo=演示模式, execute=直接执行"
    )
    args = parser.parse_args()

    print("=== SE PerfAgent 多迭代执行 ===")
    print(f"配置文件: {args.config}")
    print(f"运行模式: {args.mode}")

    try:
        # 读取 SE 配置文件
        with open(args.config, encoding="utf-8") as f:
            se_config = yaml.safe_load(f)

        # 生成 timestamp 并替换输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = se_config["output_dir"].replace("{timestamp}", timestamp)

        # 设置日志系统
        log_file = setup_se_logging(output_dir)
        print(f"日志文件: {log_file}")

        logger = get_se_logger("perf_run", emoji="⚡")
        logger.info("SE PerfAgent 多迭代执行启动")
        logger.debug(f"使用配置文件: {args.config}")
        logger.info(f"生成timestamp: {timestamp}")
        logger.info(f"实际输出目录: {output_dir}")

        # 设置全局 token 统计日志文件路径（按运行输出目录隔离）
        os.environ["SE_TOKEN_LOG_PATH"] = str(Path(output_dir) / "token_usage.jsonl")

        # 初始化轨迹池管理器
        traj_pool_path = os.path.join(output_dir, "traj.pool")

        # 创建LLM客户端用于轨迹总结
        llm_client = None
        try:
            from core.utils.llm_client import LLMClient

            # 使用operator_models配置，如果没有则使用主模型配置
            llm_client = LLMClient.from_se_config(se_config, use_operator_model=True)
            logger.info(f"LLM客户端初始化成功: {llm_client.config['name']}")
        except Exception as e:
            logger.warning(f"LLM客户端初始化失败，将使用备用总结: {e}")

        # 将 se_config 中的并发控制传入 TrajPoolManager
        traj_pool_manager = TrajPoolManager(traj_pool_path, llm_client, num_workers=se_config.get("num_workers"))
        traj_pool_manager.initialize_pool()
        logger.info(f"轨迹池初始化: {traj_pool_path}")
        print(f"🏊 轨迹池: {traj_pool_path}")

        print("\n📊 配置概览:")
        print(f"  基础配置: {se_config['base_config']}")
        print(f"  模型: {se_config['model']['name']}")
        print(f"  实例目录: {se_config['instances']['instances_dir']}")
        print(f"  输出目录: {output_dir}")

        # ============ 开始 PerfAgent 多迭代执行 ============

        iterations = se_config.get("strategy", {}).get("iterations", [])
        print(f"  迭代次数: {len(iterations)}")

        try:
            existing = [int(p.name.split("_")[-1]) for p in Path(output_dir).glob("iteration_*") if p.is_dir()]
            next_iteration_index = (max(existing) if existing else 0) + 1
        except Exception:
            next_iteration_index = 1

        for step_idx, iteration in enumerate(iterations, 1):
            operator_name = iteration.get("operator")
            is_filter_operator = str(operator_name) in ("filter", "filter_trajectories")

            def build_common_params(out_dir: str) -> dict:
                return {
                    "perf_base_config": iteration.get("perf_base_config"),
                    "operator": operator_name,
                    "model": se_config.get("model", {}),
                    "instances_dir": se_config.get("instances", {}).get("instances_dir", ""),
                    "output_dir": out_dir,
                    "max_iterations": se_config.get("max_iterations", 10),
                    "num_workers": se_config.get("num_workers", 1),
                }

            # ============ 执行算子 ============
            if operator_name == "plan":
                print("🔧 执行算子: plan (展开为多次迭代)")
                step = {
                    "operator": "plan",
                    "num": iteration.get("num"),
                    "trajectory_labels": iteration.get("trajectory_labels"),
                }
                op_result = _execute_operator_step(step, se_config, traj_pool_manager, output_dir, logger)
                plans = op_result.get("plans") or []
                for plan in plans:
                    label = str(plan.get("label")) if plan.get("label") else None
                    per_inst = plan.get("per_instance_requirements") or {}

                    iteration_output_dir = f"{output_dir}/iteration_{next_iteration_index}"
                    system_prompt_dir = Path(iteration_output_dir) / "system_prompt"
                    try:
                        system_prompt_dir.mkdir(parents=True, exist_ok=True)
                        for inst_name, req in per_inst.items():
                            try:
                                file_path = system_prompt_dir / f"{inst_name}.yaml"
                                data = {"prompts": {"additional_requirements": str(req)}}
                                with open(file_path, "w", encoding="utf-8") as f:
                                    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
                            except Exception:
                                pass
                        instance_templates_dir_for_run = str(system_prompt_dir)
                    except Exception:
                        instance_templates_dir_for_run = None

                    iteration_params = build_common_params(iteration_output_dir)
                    if instance_templates_dir_for_run:
                        iteration_params["operator_params"] = {"instance_templates_dir": instance_templates_dir_for_run}

                    print(f"\n=== 第{next_iteration_index}次PerfAgent迭代调用 ===")
                    print(f"使用配置: {iteration.get('perf_base_config', 'None')}")
                    print(f"算子: plan -> {label}")
                    print(f"输出目录: {iteration_output_dir}")

                    if args.mode == "execute":
                        logger.info(f"直接执行模式：迭代 {next_iteration_index}")
                        result = call_perfagent(iteration_params, logger, dry_run=False)
                        print(f"执行结果: {result['status']}")
                        if result.get("status") == "success":
                            try:
                                processor = TrajectoryProcessor()
                                iteration_dir = Path(iteration_output_dir)
                                tra_stats = processor.process_iteration_directory(iteration_dir)

                                if tra_stats and tra_stats.get("total_tra_files", 0) > 0:
                                    _summarize_iteration_to_pool(
                                        iteration_dir,
                                        next_iteration_index,
                                        traj_pool_manager,
                                        se_config,
                                        logger,
                                        label_prefix=label,
                                        source_labels=[],
                                        source_labels_map=None,
                                        operator_name=operator_name,
                                    )
                                else:
                                    logger.warning(f"迭代 {next_iteration_index} 未生成.tra文件")
                                    print("⚠️ 未生成.tra文件（可能没有有效轨迹）")
                            except Exception as tra_error:
                                logger.error(f"迭代 {next_iteration_index} 生成.tra文件失败: {tra_error}")
                                print(f"⚠️ .tra文件生成失败: {tra_error}")
                    else:
                        logger.info(f"演示模式：迭代 {next_iteration_index}")
                        result = call_perfagent(iteration_params, logger, dry_run=True)
                        print(f"演示结果: {result['status']}")
                        print("📝 演示模式：跳过.tra文件生成与轨迹池更新")

                    next_iteration_index += 1
                continue

            # 非 plan 算子路径
            # 常规算子执行逻辑：
            # 1. 准备算子输入（源轨迹标签、目标标签等）
            # 2. 执行算子（_execute_operator_step）
            # 3. 获取算子输出（initial_code_dir 或 instance_templates_dir）
            # 4. 构建 PerfAgent 参数并执行
            initial_code_dir_for_run: str | None = None
            instance_templates_dir_for_run: str | None = None

            iteration_output_dir = f"{output_dir}/iteration_{next_iteration_index}"
            iteration_params = build_common_params(iteration_output_dir)

            if operator_name:
                print(f"🔧 执行算子: {operator_name}")
                src_labels: list[str] = []
                if isinstance(iteration.get("source_trajectories"), list):
                    src_labels = [str(x) for x in iteration.get("source_trajectories")]
                elif iteration.get("source_trajectory"):
                    src_labels = [str(iteration.get("source_trajectory"))]

                outputs = []
                if iteration.get("trajectory_label"):
                    outputs = [{"label": str(iteration.get("trajectory_label"))}]

                strat_cfg = iteration.get("filter_strategy") or iteration.get("strategy") or {}

                step = {
                    "operator": operator_name,
                    "inputs": [{"label": l} for l in src_labels],
                    "outputs": outputs,
                    "strategy": strat_cfg,
                }
                op_result = _execute_operator_step(step, se_config, traj_pool_manager, iteration_output_dir, logger)
                if isinstance(op_result.get("initial_code_dir"), str):
                    initial_code_dir_for_run = op_result["initial_code_dir"]
                if isinstance(op_result.get("instance_templates_dir"), str):
                    instance_templates_dir_for_run = op_result["instance_templates_dir"]
                source_labels_map = op_result.get("source_entry_labels_per_instance") or {}
                if is_filter_operator:
                    logger.info("过滤算子步骤，执行后跳过PerfAgent")
                    try:
                        ff = op_result.get("filtered_out_file")
                        pi = op_result.get("per_instance") or {}
                        kept_total = sum(len(v.get("kept_labels", [])) for v in pi.values())
                        deleted_total = sum(len(v.get("deleted_labels", [])) for v in pi.values())
                        if ff:
                            logger.info(f"过滤输出文件: {ff}")
                        logger.info(f"过滤摘要: 保留 {kept_total} 条, 删除 {deleted_total} 条, 实例 {len(pi)} 个")
                    except Exception:
                        pass
            else:
                print("🔄 无算子处理")
                logger.debug("当前步骤无算子处理")

            if initial_code_dir_for_run or instance_templates_dir_for_run:
                op_params = {}
                if initial_code_dir_for_run:
                    op_params["initial_code_dir"] = initial_code_dir_for_run
                if instance_templates_dir_for_run:
                    op_params["instance_templates_dir"] = instance_templates_dir_for_run
                iteration_params["operator_params"] = op_params

            logger.debug(f"迭代参数: {json.dumps(iteration_params, ensure_ascii=False)}")
            print(f"使用配置: {iteration.get('perf_base_config', 'None')}")
            print(f"算子: {iteration.get('operator', 'None')}")
            print(f"输出目录: {iteration_output_dir}")

            # ============ 执行 PerfAgent  ============

            if args.mode == "execute" and not is_filter_operator:
                logger.info(f"直接执行模式：迭代 {next_iteration_index}")
                result = call_perfagent(iteration_params, logger, dry_run=False)
                try:
                    logger.info(f"PerfAgent返回状态: {result.get('status')}")
                except Exception:
                    pass
                print(f"执行结果: {result['status']}")

                # ============ 处理 PerfAgent 执行结果 ============

                # 成功则生成.tra并更新 traj.pool
                # .tra 直接使用 history 生成
                # traj.pool 轨迹总结通过 LLM Summary 生成
                if result.get("status") == "success":
                    try:
                        processor = TrajectoryProcessor()
                        iteration_dir = Path(iteration_output_dir)

                        # 处理当前迭代目录下的所有实例，生成 .tra 文件
                        tra_stats = processor.process_iteration_directory(iteration_dir)
                        if tra_stats and tra_stats.get("total_tra_files", 0) > 0:
                            prefix = iteration.get("trajectory_label")
                            # 将生成的轨迹汇总到全局轨迹池
                            _summarize_iteration_to_pool(
                                iteration_dir,
                                next_iteration_index,
                                traj_pool_manager,
                                se_config,
                                logger,
                                label_prefix=prefix,
                                source_labels=src_labels,
                                source_labels_map=source_labels_map if isinstance(source_labels_map, dict) else None,
                                operator_name=operator_name,
                            )
                        else:
                            logger.warning(f"迭代 {next_iteration_index} 未生成.tra文件")
                            print("⚠️ 未生成.tra文件（可能没有有效轨迹）")
                    except Exception as tra_error:
                        logger.error(f"迭代 {next_iteration_index} 生成.tra文件失败: {tra_error}")
                        print(f"⚠️ .tra文件生成失败: {tra_error}")
                next_iteration_index += 1

            elif args.mode == "execute" and is_filter_operator:
                logger.info("跳过PerfAgent执行（过滤算子）")
                print("⏭️ 跳过PerfAgent执行（filter算子）")
            else:
                logger.info("演示模式：本步骤")
                if not is_filter_operator:
                    result = call_perfagent(iteration_params, logger, dry_run=True)
                    print(f"演示结果: {result['status']}")
                    print("📝 演示模式：跳过.tra文件生成与轨迹池更新")
                    next_iteration_index += 1
                else:
                    logger.info("演示模式下跳过PerfAgent（过滤算子）")
                    print("⏭️ 演示模式下跳过PerfAgent（filter算子）")

        logger.info("所有PerfAgent迭代准备完成")

        print("\n🎯 执行总结:")
        try:
            parsed_iterations = len(se_config.get("strategy", {}).get("iterations", []))
        except Exception:
            parsed_iterations = 0
        print(f"  ✅ 解析{parsed_iterations}个迭代配置")
        print(f"  ✅ 时间戳: {timestamp}")
        print(f"  ✅ 日志文件: {log_file}")
        print(f"  📁 输出目录: {output_dir}")
        try:
            final_pool_stats = traj_pool_manager.get_pool_stats()
            print(f"  🏊 轨迹池: {final_pool_stats.get('total_trajectories', 0)}条轨迹")
            print(f"  🏊 轨迹池文件: {traj_pool_path}")
        except Exception:
            pass

        logger.info("SE PerfAgent 多迭代执行完成")

        # ===== 统计 token 使用 =====
        print("\n📊 统计 token 使用:")

        # 读取并汇总本次运行的 token 使用情况
        token_log_file = Path(output_dir) / "token_usage.jsonl"
        total_prompt = 0
        total_completion = 0
        total = 0
        by_context: dict[str, dict[str, int]] = {}
        try:
            if token_log_file.exists():
                with open(token_log_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
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
            pass

        print("\n📈 Token 使用统计:")
        print(f"  输入tokens: {total_prompt}")
        print(f"  输出tokens: {total_completion}")
        print(f"  总计tokens: {total}")
        if by_context:
            print("  按上下文分类:")
            for ctx, vals in by_context.items():
                print(f"    - {ctx}: prompt={vals['prompt']}, completion={vals['completion']}, total={vals['total']}")
        logger.info(
            json.dumps(
                {
                    "token_usage_total": {
                        "prompt": total_prompt,
                        "completion": total_completion,
                        "total": total,
                    },
                    "by_context": by_context,
                    "token_log_file": str(token_log_file),
                },
                ensure_ascii=False,
            )
        )

        # ================================== 依据输出选择最佳 Solution ==================================

        logger.info("开始选择每个任务的最优解")
        try:
            # 1) 汇总所有迭代的 preds.json 到运行根目录 preds.json（含 iteration 字段）
            root_output_dir = Path(output_dir)
            agg_preds_path = aggregate_all_iterations_preds(root_output_dir, logger)

            # 2) 从汇总 preds.json 里为每个实例选取 runtime 最小的解，生成 final.json
            if agg_preds_path and Path(agg_preds_path).exists():
                write_final_json_from_preds(Path(agg_preds_path), root_output_dir, logger)
            else:
                logger.warning("未找到汇总 preds.json，跳过 final.json 生成")
        except Exception as sel_err:
            logger.warning(f"选择最优解失败: {sel_err}")

    except Exception as e:
        if "logger" in locals():
            logger.error(f"运行出错: {e}", exc_info=True)
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
