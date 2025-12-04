#!/usr/bin/env python3
"""
PerfAgent 集成执行脚本

功能：
    在 SE 框架中驱动 PerfAgent 进行单次或多次迭代的性能优化。
    模仿 SE/basic_run.py 的结构，支持策略驱动的执行流程。
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
from typing import Any

import yaml

# 添加 SE 根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入 SE 核心模块
from core.utils.local_memory_manager import LocalMemoryManager
from core.utils.se_logger import get_se_logger, setup_se_logging
from core.utils.traj_extractor import TrajExtractor
from core.utils.traj_pool_manager import TrajPoolManager
from core.utils.trajectory_processor import TrajectoryProcessor
from operators import create_operator

# --- 辅助函数 ---


def _execute_operator_step(
    step_config: dict[str, Any],
    se_config: dict[str, Any],
    traj_pool_manager: TrajPoolManager,
    workspace_dir: str,
    logger,
) -> dict[str, Any]:
    """
    执行单个算子步骤。

    Args:
        step_config: 算子步骤配置
        se_config: SE 全局配置
        traj_pool_manager: 轨迹池管理器
        workspace_dir: 工作目录
        logger: 日志记录器

    Returns:
        算子执行结果字典
    """
    operator_name = step_config.get("operator")
    if not operator_name:
        logger.error("算子执行错误：步骤配置缺少 'operator' 字段")
        return {}

    # 合并配置：优先使用 step_config 中的设置，其次是 se_config
    operator_config = dict(se_config) if isinstance(se_config, dict) else {}

    if step_config.get("selection_mode"):
        operator_config["operator_selection_mode"] = step_config.get("selection_mode")
    if step_config.get("prompt_config"):
        operator_config["prompt_config"] = step_config.get("prompt_config")

    operator_instance = create_operator(operator_name, operator_config)
    if not operator_instance:
        logger.error(f"无法创建算子实例: {operator_name}")
        return {}

    result = {}
    try:
        result = operator_instance.run(step_config, traj_pool_manager, workspace_dir)
    except Exception as e:
        logger.error(f"算子 '{operator_name}' 执行失败: {e}")
        return {}

    # 记录结果日志
    initial_code_path = result.get("initial_code_dir")
    if isinstance(initial_code_path, str) and initial_code_path:
        path_obj = Path(initial_code_path)
        if path_obj.exists():
            logger.info(f"算子返回初始代码目录: {path_obj}")

    generated_count = result.get("generated_count")
    if generated_count is not None:
        try:
            logger.info(f"算子生成初始代码数量: {int(generated_count)}")
        except (ValueError, TypeError):
            pass

    return result


def _summarize_iteration_to_pool(
    iteration_dir: Path,
    iteration_index: int,
    traj_pool_manager: TrajPoolManager,
    se_config: dict[str, Any],
    logger,
    label_prefix: str | None = None,
    source_labels_map: dict[str, list[str]] | None = None,
    operator_name: str | None = None,
) -> None:
    """
    提取迭代结果并更新到轨迹池。
    """
    try:
        # 更新 prompt_config
        prompt_config = se_config.get("prompt_config")
        if isinstance(prompt_config, dict):
            traj_pool_manager.prompt_config = prompt_config

        extractor = TrajExtractor()
        # 提取实例数据，包含性能指标
        extracted_data = extractor.extract_instance_data(iteration_dir, include_metrics=True)

        if not extracted_data:
            logger.warning(f"迭代 {iteration_index}：没有有效的实例数据用于轨迹池总结")
            return

        trajectories_to_process = []
        for item in extracted_data:
            # 兼容旧格式解包
            if len(item) == 5:
                instance_name, problem_desc, tra_content, patch_content, perf_metrics = item
            else:
                instance_name, problem_desc, tra_content, patch_content = item
                perf_metrics = None

            label = str(label_prefix) if label_prefix else f"iter{iteration_index}"

            # 获取源标签
            instance_source_labels = None
            if source_labels_map and isinstance(source_labels_map, dict):
                instance_source_labels = source_labels_map.get(str(instance_name))

            trajectories_to_process.append(
                {
                    "label": label,
                    "instance_name": instance_name,
                    "problem_description": problem_desc,
                    "trajectory_content": tra_content,
                    "patch_content": patch_content,
                    "iteration": iteration_index,
                    "performance": (perf_metrics or {}).get("performance"),
                    "source_dir": str(iteration_dir / instance_name),
                    "source_entry_labels": list(instance_source_labels or []),
                    "operator_name": str(operator_name) if operator_name else None,
                    "perf_metrics": perf_metrics,
                }
            )

        traj_pool_manager.summarize_and_add_trajectories(
            trajectories_to_process, num_workers=se_config.get("num_workers")
        )

        pool_stats = traj_pool_manager.get_pool_stats()
        logger.info(f"轨迹池更新完毕: 当前共 {pool_stats.get('total_trajectories', 'unknown')} 条轨迹")

    except Exception as e:
        logger.error(f"迭代轨迹池更新失败: {e}")


def _extract_optimization_info(perf_config_path: str | None) -> tuple[str | None, str | None]:
    """
    从 PerfAgent 配置文件中提取优化目标和语言配置。
    """
    if not perf_config_path:
        return None, None

    try:
        with open(perf_config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        opt_target = config.get("optimization", {}).get("target")
        language = config.get("language_cfg", {}).get("language")

        target_str = str(opt_target) if isinstance(opt_target, str) and opt_target.strip() else None
        language_str = str(language) if isinstance(language, str) and language.strip() else None

        return target_str, language_str
    except Exception:
        return None, None


def write_iteration_preds(base_dir: Path, logger) -> Path | None:
    """
    聚合当前迭代各实例的结果，生成 preds.json。
    """
    predictions = {}
    try:
        for instance_dir in base_dir.iterdir():
            if not instance_dir.is_dir():
                continue

            result_file = instance_dir / "result.json"
            if not result_file.exists():
                continue

            try:
                with open(result_file, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            instance_id = data.get("instance_id", instance_dir.name)
            code = data.get("optimized_code", "")
            final_perf = data.get("final_performance")

            # 只要 final_perf 不是无穷大，就认为通过
            is_passed = False
            if final_perf is not None:
                try:
                    is_passed = not math.isinf(float(final_perf))
                except (ValueError, TypeError):
                    is_passed = False

            predictions[str(instance_id)] = {
                "code": code,
                "passed": is_passed,
                "runtime": final_perf,
            }

        preds_path = base_dir / "preds.json"
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        logger.info(f"已生成迭代预测汇总: {preds_path}")
        return preds_path

    except Exception as e:
        logger.warning(f"生成 preds.json 失败: {e}")
        return None


def aggregate_all_iterations_preds(root_output_dir: Path, logger) -> Path | None:
    """
    汇总所有 iteration_* 目录下的 preds.json 到根目录。
    """
    aggregated_data: dict[str, list[dict]] = {}

    try:
        # 遍历所有迭代目录，按数字顺序排序
        iteration_dirs = sorted(root_output_dir.glob("iteration_*"), key=lambda p: p.name)

        for iter_dir in iteration_dirs:
            if not iter_dir.is_dir():
                continue

            # 解析迭代号
            try:
                iter_num = int(iter_dir.name.split("_")[-1])
            except ValueError:
                continue

            preds_file = iter_dir / "preds.json"
            if not preds_file.exists():
                continue

            try:
                with open(preds_file, encoding="utf-8") as f:
                    preds = json.load(f)
            except Exception:
                continue

            for instance_id, info in preds.items():
                try:
                    passed = bool(info.get("passed", False))
                    # 未通过的实例，code 置为空字符串
                    code = info.get("code", "") if passed else ""
                    runtime = info.get("runtime")

                    entry = {"iteration": iter_num, "code": code, "runtime": runtime}
                    aggregated_data.setdefault(str(instance_id), []).append(entry)
                except Exception:
                    continue

        agg_path = root_output_dir / "preds.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(aggregated_data, f, indent=2, ensure_ascii=False)

        logger.info(f"汇总所有迭代预测结果: {agg_path}")
        return agg_path

    except Exception as e:
        logger.warning(f"汇总 preds.json 失败: {e}")
        return None


def write_final_json_from_preds(aggregated_preds_path: Path, root_output_dir: Path, logger) -> Path | None:
    """
    从汇总的 preds.json 中选择最佳结果（runtime 最小）写入 final.json。
    """
    try:
        with open(aggregated_preds_path, encoding="utf-8") as f:
            aggregated_data = json.load(f)
    except Exception as e:
        logger.warning(f"读取汇总 preds.json 失败: {e}")
        return None

    def _parse_runtime(rt_val):
        """解析 runtime 值，异常情况返回无穷大"""
        try:
            if rt_val is None:
                return float("inf")
            if isinstance(rt_val, (int, float)):
                return float(rt_val)
            if isinstance(rt_val, str):
                lowered = rt_val.strip().lower()
                if lowered in ("inf", "infinity", "nan"):
                    return float("inf")
                return float(rt_val)
            return float("inf")
        except Exception:
            return float("inf")

    final_result_map: dict[str, str] = {}

    try:
        for instance_id, entries in aggregated_data.items():
            if not isinstance(entries, list) or not entries:
                continue

            # 找到 runtime 最小的条目
            try:
                best_entry = min(entries, key=lambda e: _parse_runtime(e.get("runtime")))
            except ValueError:
                continue

            final_result_map[str(instance_id)] = best_entry.get("code", "") or ""

        final_path = root_output_dir / "final.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_result_map, f, indent=2, ensure_ascii=False)

        logger.info(f"生成最终结果 final.json: {final_path}")
        return final_path

    except Exception as e:
        logger.warning(f"生成 final.json 失败: {e}")
        return None


def create_temp_perf_config(
    base_config_path: str | None,
    se_model_config: dict[str, Any],
    logger,
    extra_overrides: dict[str, Any] | None = None,
) -> Path | None:
    """
    生成临时的 PerfAgent 配置文件。
    """
    try:
        perf_config = {}
        if base_config_path:
            try:
                with open(base_config_path, encoding="utf-8") as f:
                    perf_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"无法读取基础配置文件 {base_config_path}: {e}")

        # 模型参数覆盖白名单
        allowed_model_keys = {"name", "api_base", "api_key", "max_input_tokens", "max_output_tokens", "temperature"}

        model_overrides = {
            k: v
            for k, v in (se_model_config or {}).items()
            if k in allowed_model_keys and v is not None and str(v).strip() != ""
        }

        perf_config.setdefault("model", {}).update(model_overrides)

        # 其他顶层参数覆盖（如 max_iterations）
        if extra_overrides:
            for key, val in extra_overrides.items():
                if val is not None and str(val).strip() != "":
                    # 尝试转为 int，如果失败则保持原值
                    if key == "max_iterations":
                        try:
                            perf_config[key] = int(val)
                        except (ValueError, TypeError):
                            perf_config[key] = val
                    else:
                        perf_config[key] = val

        # 写入临时文件
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp_file:
            yaml.safe_dump(perf_config, tmp_file, sort_keys=False, allow_unicode=True)
            temp_path = Path(tmp_file.name)

        logger.info(f"生成临时 PerfAgent 配置: {temp_path}")
        logger.debug(f"模型覆盖参数: {json.dumps(model_overrides, ensure_ascii=False)}")

        return temp_path

    except Exception as e:
        logger.warning(f"生成临时配置失败: {e}")
        return None


def call_perfagent(iteration_params: dict[str, Any], logger, dry_run: bool = False) -> dict[str, Any]:
    """
    调用 PerfAgent 执行批量优化。
    """
    base_config_path = iteration_params.get("perf_base_config")
    output_dir = Path(iteration_params["output_dir"]).resolve()
    instances_dir = iteration_params.get("instances_dir")
    num_workers = iteration_params.get("num_workers", 1)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 生成临时配置
        se_model_config = iteration_params.get("model") or {}
        temp_config_path = create_temp_perf_config(
            base_config_path,
            se_model_config,
            logger,
            extra_overrides={
                "max_iterations": iteration_params.get("max_iterations"),
            },
        )

        # 构建命令
        cmd = [sys.executable, "-m", "perfagent.run_batch"]

        # 配置文件（优先使用临时生成的）
        config_path_to_use = temp_config_path if temp_config_path else base_config_path
        if config_path_to_use:
            cmd.extend(["--config", str(config_path_to_use)])

        cmd.extend(
            [
                "--instances-dir",
                str(instances_dir),
                "--base-dir",
                str(output_dir),
                "--max-workers",
                str(num_workers),
            ]
        )

        # 算子传递的参数
        operator_params = iteration_params.get("operator_params") or {}
        initial_code_dir = operator_params.get("initial_code_dir")
        instance_templates_dir = operator_params.get("instance_templates_dir")

        if initial_code_dir:
            cmd.extend(["--initial-code-dir", str(initial_code_dir)])
        if instance_templates_dir:
            cmd.extend(["--instance-templates-dir", str(instance_templates_dir)])

        cmd_str = " ".join(cmd)

        if dry_run:
            logger.info("演示模式：跳过实际执行")
            print(f"🚀 [DEMO] PerfAgent 命令预览: {cmd_str}")
            return {"status": "skipped", "reason": "dry_run", "preview_cmd": cmd_str}

        logger.info(f"执行 PerfAgent 命令: {cmd_str}")
        print(f"🚀 执行 PerfAgent: {cmd_str}")

        # 执行命令
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent), text=True)

        if result.returncode == 0:
            logger.info("PerfAgent 执行成功")
            print("✅ PerfAgent 执行成功")
            # 生成当前迭代的预测结果
            preds_path = write_iteration_preds(output_dir, logger)
            return {
                "status": "success",
                "summary": "success",
                "base_dir": str(output_dir),
                "preds_file": str(preds_path) if preds_path else None,
            }
        else:
            logger.error(f"PerfAgent 执行失败，返回码: {result.returncode}")
            print(f"❌ PerfAgent 执行失败，返回码: {result.returncode}")
            return {"status": "failed", "returncode": result.returncode}

    except Exception as e:
        logger.error(f"调用 PerfAgent 异常: {e}", exc_info=True)
        return {"status": "error", "exception": str(e)}
    finally:
        # 这里可以添加删除临时配置文件的逻辑，如果需要的话
        pass


# --- 辅助函数 ---


def _process_and_summarize(
    iter_dir: Path,
    iter_idx: int,
    step_config: dict,
    se_config: dict,
    pool_manager: TrajPoolManager,
    logger,
    label_prefix=None,
    source_labels=None,
    source_labels_map=None,
    operator_name=None,
):
    """
    后处理：生成 .tra 文件并更新轨迹池
    """
    try:
        processor = TrajectoryProcessor()
        tra_stats = processor.process_iteration_directory(iter_dir)

        if tra_stats and tra_stats.get("total_tra_files", 0) > 0:
            # 提取优化目标配置
            perf_cfg_path = step_config.get("perf_base_config")
            opt_target, lang_val = _extract_optimization_info(perf_cfg_path)

            # 更新 summarizer 配置
            if opt_target or lang_val:
                pc = se_config.setdefault("prompt_config", {})
                if isinstance(pc, dict):
                    scfg = pc.setdefault("summarizer", {})
                    if opt_target:
                        scfg["optimization_target"] = opt_target
                    if lang_val:
                        scfg["language"] = lang_val

            _summarize_iteration_to_pool(
                iter_dir,
                iter_idx,
                pool_manager,
                se_config,
                logger,
                label_prefix=label_prefix,
                source_labels_map=source_labels_map,
                operator_name=operator_name,
            )
            try:
                mm = getattr(pool_manager, "memory_manager", None)
                if mm is not None:
                    mem = mm.load()
                    ckpt_path = Path(iter_dir) / f"memory_iter_{iter_idx}.json"
                    with open(ckpt_path, "w", encoding="utf-8") as f:
                        json.dump(mem, f, ensure_ascii=False, indent=2)
                    logger.info(f"已保存迭代 {iter_idx} 的记忆快照: {ckpt_path}")
            except Exception as e:
                logger.warning(f"保存迭代 {iter_idx} 记忆快照失败: {e}")
        else:
            logger.warning(f"迭代 {iter_idx} 未生成 .tra 文件")
    except Exception as e:
        logger.error(f"迭代 {iter_idx} 后处理失败: {e}")


def _print_final_summary(se_config, timestamp, log_file, output_dir, traj_pool_manager, logger):
    """
    打印和记录最终执行摘要
    """
    logger.info("所有任务执行完成")
    print("\n🎯 执行完成")
    print(f"  日志: {log_file}")
    print(f"  输出: {output_dir}")

    # 汇总 preds.json 并生成 final.json
    try:
        root_dir = Path(output_dir)
        agg_path = aggregate_all_iterations_preds(root_dir, logger)
        if agg_path:
            write_final_json_from_preds(agg_path, root_dir, logger)
    except Exception as e:
        logger.warning(f"生成最终结果文件失败: {e}")

    # 统计 Token
    _log_token_usage(output_dir, logger)


def _log_token_usage(output_dir, logger):
    """
    统计并记录 Token 使用情况
    """
    token_log_file = Path(output_dir) / "token_usage.jsonl"
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


# --- 主流程 ---


def main():
    """
    主函数：策略驱动的 PerfAgent 多迭代执行入口。
    """
    parser = argparse.ArgumentParser(description="SE 框架 PerfAgent 多迭代执行脚本")
    parser.add_argument("--config", default="SE/configs/se_configs/dpsk.yaml", help="SE 配置文件路径")
    parser.add_argument("--mode", choices=["demo", "execute"], default="execute", help="运行模式")
    args = parser.parse_args()

    print("=== SE PerfAgent 多迭代执行 ===")

    try:
        # 1. 加载配置
        with open(args.config, encoding="utf-8") as f:
            se_config = yaml.safe_load(f)

        # 2. 准备输出环境
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = se_config["output_dir"].replace("{timestamp}", timestamp)

        # 初始化日志
        log_file = setup_se_logging(output_dir)
        logger = get_se_logger("perf_run", emoji="⚡")

        logger.info(f"启动执行: {args.config}, 模式: {args.mode}")
        logger.info(f"输出目录: {output_dir}")

        # Token统计与LLM I/O日志文件路径
        os.environ["SE_TOKEN_LOG_PATH"] = str(Path(output_dir) / "token_usage.jsonl")
        os.environ["SE_LLM_IO_LOG_PATH"] = str(Path(output_dir) / "llm_io.jsonl")

        # 3. 初始化核心组件
        traj_pool_path = os.path.join(output_dir, "traj.pool")

        # LLM Client
        llm_client = None
        try:
            from core.utils.llm_client import LLMClient

            llm_client = LLMClient.from_se_config(se_config, use_operator_model=True)
        except Exception as e:
            logger.warning(f"LLM客户端初始化失败: {e}")

        # Local Memory Manager
        local_memory = None
        memory_config = se_config.get("local_memory")
        if isinstance(memory_config, dict) and memory_config.get("enabled", True):
            try:
                memory_path = Path(output_dir) / "memory.json"
                local_memory = LocalMemoryManager(memory_path, llm_client=llm_client)
                local_memory.initialize()
                logger.info("LocalMemoryManager 已启用")
            except Exception as e:
                logger.warning(f"LocalMemoryManager 初始化失败: {e}")

        local_memory_text = None
        try:
            if local_memory is not None:
                mem = local_memory.load()
                local_memory_text = local_memory.render_as_markdown(mem)
        except Exception:
            local_memory_text = None

        # Trajectory Pool Manager
        traj_pool_manager = TrajPoolManager(
            traj_pool_path,
            llm_client,
            num_workers=se_config.get("num_workers"),
            memory_manager=local_memory,
            prompt_config=se_config.get("prompt_config"),
        )
        traj_pool_manager.initialize_pool()

        # 4. 执行迭代策略
        iterations = se_config.get("strategy", {}).get("iterations", [])
        logger.info(f"计划执行 {len(iterations)} 个迭代步骤")

        # 确定起始迭代编号
        existing_iters = [int(p.name.split("_")[-1]) for p in Path(output_dir).glob("iteration_*") if p.is_dir()]
        next_iteration_idx = (max(existing_iters) if existing_iters else 0) + 1

        for step_config in iterations:
            operator_name = step_config.get("operator")
            is_filter_operator = str(operator_name) in ("filter", "filter_trajectories")

            # 构建当前迭代的基础参数
            current_iter_dir = f"{output_dir}/iteration_{next_iteration_idx}"
            iter_params = {
                "perf_base_config": step_config.get("perf_base_config"),
                "operator": operator_name,
                "model": se_config.get("model", {}),
                "instances_dir": se_config.get("instances", {}).get("instances_dir", ""),
                "output_dir": current_iter_dir,
                "max_iterations": se_config.get("max_iterations", 10),
                "num_workers": se_config.get("num_workers", 1),
            }

            try:
                if local_memory is not None:
                    _mem_latest = local_memory.load()
                    local_memory_text = local_memory.render_as_markdown(_mem_latest)
            except Exception:
                pass

            # --- 分支 A: Plan 算子 (特殊处理: 展开为多实例配置) ---
            if operator_name == "plan":
                logger.info("执行算子: Plan")
                # 构建 Plan 算子参数
                plan_step = {
                    "operator": "plan",
                    "num": step_config.get("num"),
                    "trajectory_labels": step_config.get("trajectory_labels"),
                }
                op_result = _execute_operator_step(plan_step, se_config, traj_pool_manager, output_dir, logger)

                plans = op_result.get("plans") or []
                for plan in plans:
                    # 为每个 plan 创建单独的迭代目录
                    plan_iter_dir = f"{output_dir}/iteration_{next_iteration_idx}"
                    plan_label = plan.get("label")
                    per_inst_reqs = plan.get("per_instance_requirements") or {}

                    # 准备 system_prompt 目录
                    sys_prompt_dir = Path(plan_iter_dir) / "system_prompt"
                    sys_prompt_dir.mkdir(parents=True, exist_ok=True)

                    rendered_template = None
                    try:
                        base_cfg_path = step_config.get("perf_base_config")
                        if base_cfg_path:
                            with open(base_cfg_path, encoding="utf-8") as f:
                                base_cfg_data = yaml.safe_load(f) or {}
                            tmpl = (base_cfg_data.get("prompts", {}) or {}).get("system_template")
                            if isinstance(tmpl, str) and tmpl:
                                if local_memory_text:
                                    rendered_template = tmpl.replace("{local_memory}", str(local_memory_text))
                                else:
                                    rendered_template = tmpl
                    except Exception:
                        rendered_template = None

                    for inst_name, req in per_inst_reqs.items():
                        try:
                            data = {"prompts": {"additional_requirements": str(req)}}
                            if isinstance(rendered_template, str) and rendered_template:
                                data["prompts"]["system_template"] = rendered_template
                            with open(sys_prompt_dir / f"{inst_name}.yaml", "w", encoding="utf-8") as f:
                                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
                        except Exception:
                            pass

                    # 更新迭代参数
                    iter_params["output_dir"] = plan_iter_dir
                    iter_params["operator_params"] = {"instance_templates_dir": str(sys_prompt_dir)}

                    print(f"\n=== 迭代 {next_iteration_idx} (Plan: {plan_label}) ===")

                    # 执行 PerfAgent
                    run_result = call_perfagent(iter_params, logger, dry_run=(args.mode == "demo"))

                    # 后处理：生成轨迹
                    if run_result.get("status") == "success" and args.mode == "execute":
                        _process_and_summarize(
                            Path(plan_iter_dir),
                            next_iteration_idx,
                            step_config,
                            se_config,
                            traj_pool_manager,
                            logger,
                            label_prefix=plan_label,
                            operator_name=operator_name,
                        )

                    next_iteration_idx += 1
                continue

            # --- 分支 B: 普通算子或无算子 ---
            initial_code_dir = None
            instance_templates_dir = None
            source_labels_map = None

            if operator_name:
                logger.info(f"执行算子: {operator_name}")

                # 准备算子输入
                src_labels = []
                if isinstance(step_config.get("source_trajectories"), list):
                    src_labels = [str(x) for x in step_config.get("source_trajectories")]
                elif step_config.get("source_trajectory"):
                    src_labels = [str(step_config.get("source_trajectory"))]

                op_step_config = {
                    "operator": operator_name,
                    "inputs": [{"label": l} for l in src_labels],
                    "outputs": [{"label": str(step_config.get("trajectory_label"))}]
                    if step_config.get("trajectory_label")
                    else [],
                    "strategy": step_config.get("filter_strategy") or step_config.get("strategy") or {},
                }

                op_result = _execute_operator_step(
                    op_step_config, se_config, traj_pool_manager, current_iter_dir, logger
                )

                initial_code_dir = op_result.get("initial_code_dir")
                instance_templates_dir = op_result.get("instance_templates_dir")
                source_labels_map = op_result.get("source_entry_labels_per_instance")

                # Filter 算子特殊逻辑：跳过 PerfAgent 执行
                if is_filter_operator:
                    logger.info("Filter 算子执行完毕，跳过后续 PerfAgent 运行")
                    continue

                rendered_template = None
                try:
                    base_cfg_path = step_config.get("perf_base_config")
                    if base_cfg_path:
                        with open(base_cfg_path, encoding="utf-8") as f:
                            base_cfg_data = yaml.safe_load(f) or {}
                        tmpl = (base_cfg_data.get("prompts", {}) or {}).get("system_template")
                        if isinstance(tmpl, str) and tmpl:
                            if local_memory_text:
                                rendered_template = tmpl.replace("{local_memory}", str(local_memory_text))
                            else:
                                rendered_template = tmpl
                except Exception:
                    rendered_template = None

                if isinstance(rendered_template, str) and rendered_template:
                    try:
                        if instance_templates_dir:
                            p = Path(instance_templates_dir)
                            if p.exists():
                                for fp in p.glob("*.yaml"):
                                    try:
                                        with open(fp, encoding="utf-8") as f:
                                            d = yaml.safe_load(f) or {}
                                        pm = d.get("prompts") or {}
                                        pm["system_template"] = rendered_template
                                        d["prompts"] = pm
                                        with open(fp, "w", encoding="utf-8") as f:
                                            yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)
                                    except Exception:
                                        pass
                        else:
                            sys_prompt_dir = Path(current_iter_dir) / "system_prompt"
                            sys_prompt_dir.mkdir(parents=True, exist_ok=True)
                            inst_dir = Path(iter_params.get("instances_dir") or "")
                            for fp in inst_dir.glob("*.json"):
                                try:
                                    with open(sys_prompt_dir / f"{fp.stem}.yaml", "w", encoding="utf-8") as f:
                                        yaml.safe_dump(
                                            {"prompts": {"system_template": rendered_template}},
                                            f,
                                            allow_unicode=True,
                                            sort_keys=False,
                                        )
                                except Exception:
                                    pass
                            instance_templates_dir = str(sys_prompt_dir)
                    except Exception:
                        pass

            # 设置算子输出参数
            iter_params["operator_params"] = {}
            if initial_code_dir:
                iter_params["operator_params"]["initial_code_dir"] = initial_code_dir
            if instance_templates_dir:
                iter_params["operator_params"]["instance_templates_dir"] = instance_templates_dir

            print(f"\n=== 迭代 {next_iteration_idx} ===")

            # 执行 PerfAgent
            run_result = call_perfagent(iter_params, logger, dry_run=(args.mode == "demo"))

            # 后处理
            if run_result.get("status") == "success" and args.mode == "execute":
                # 确定源标签用于记录
                src_labels_for_summary = []
                if isinstance(step_config.get("source_trajectories"), list):
                    src_labels_for_summary = [str(x) for x in step_config.get("source_trajectories")]

                _process_and_summarize(
                    Path(current_iter_dir),
                    next_iteration_idx,
                    step_config,
                    se_config,
                    traj_pool_manager,
                    logger,
                    label_prefix=step_config.get("trajectory_label"),
                    source_labels=src_labels_for_summary,
                    source_labels_map=source_labels_map,
                    operator_name=operator_name,
                )

            next_iteration_idx += 1

        # 5. 最终汇总
        _print_final_summary(se_config, timestamp, log_file, output_dir, traj_pool_manager, logger)

    except Exception as e:
        if "logger" in locals():
            logger.error(f"程序运行异常: {e}", exc_info=True)
        print(f"程序运行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
