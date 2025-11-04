#!/usr/bin/env python3
"""
PerfAgent 集成执行脚本
模仿 SE/basic_run.py 的结构，在 SE 框架中驱动 perfagent 的单/多实例性能优化。
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# 添加SE目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入SE日志系统
from core.utils.se_logger import get_se_logger, setup_se_logging
from core.utils.traj_extractor import TrajExtractor
from core.utils.traj_pool_manager import TrajPoolManager
from core.utils.trajectory_processor import TrajectoryProcessor

# 导入operator系统
from operators import create_operator, list_operators


def call_operator(operator_name, workspace_dir, current_iteration, se_config, logger):
    """
    调用指定的operator处理

    Args:
        operator_name: operator名称
        workspace_dir: 工作空间根目录 (不带迭代号)
        current_iteration: 当前迭代号
        se_config: SE配置字典
        logger: 日志记录器

    Returns:
        operator返回的参数字典 (如 {'instance_templates_dir': 'path'}) 或 None表示失败
    """
    try:
        logger.info(f"开始调用operator: {operator_name}")

        # 动态创建operator实例
        operator = create_operator(operator_name, se_config)
        if not operator:
            logger.error(f"无法创建operator实例: {operator_name}")
            return None

        logger.info(f"成功创建operator实例: {operator.__class__.__name__}")

        # 调用operator.process()方法
        result = operator.process(
            workspace_dir=workspace_dir,
            current_iteration=current_iteration,
            num_workers=se_config.get("num_workers", 1),
        )

        if result:
            logger.info(f"Operator {operator_name} 执行成功，返回: {list(result.keys())}")
            return result
        else:
            logger.warning(f"Operator {operator_name} 执行成功但返回空结果")
            return None

    except Exception as e:
        logger.error(f"Operator {operator_name} 执行失败: {e}", exc_info=True)
        return None

def write_iteration_preds(base_dir: Path, logger) -> Optional[Path]:
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
                with open(res_file, "r", encoding="utf-8") as f:
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


def aggregate_all_iterations_preds(root_output_dir: Path, logger) -> Optional[Path]:
    """
    汇总所有 iteration_* 目录下的 preds.json，过滤未通过项，添加迭代号并写入运行根目录的 preds.json。

    输出结构示例：
    {
      "inst1": [
        {"iteration": 1, "code": "...", "runtime": 1.23},
        {"iteration": 2, "code": "...", "runtime": 1.11}
      ],
      "inst2": [
        {"iteration": 2, "code": "...", "runtime": 0.98}
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
                with open(preds_file, "r", encoding="utf-8") as pf:
                    preds = json.load(pf)
            except Exception:
                continue

            for instance_id, info in preds.items():
                try:
                    if not bool(info.get("passed", False)):
                        continue
                    code = info.get("code", "")
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


def write_final_json_from_preds(aggregated_preds_path: Path, root_output_dir: Path, logger) -> Optional[Path]:
    """
    从运行根目录的 preds.json（汇总）选择每个实例 runtime 最小的解，写入 final.json。

    文件结构：
    {
      "instance_name": "code"
    }
    """
    try:
        with open(aggregated_preds_path, "r", encoding="utf-8") as f:
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
        for instance_id, entries in aggregated.items():
            if not isinstance(entries, list) or not entries:
                continue
            try:
                best = min(entries, key=lambda e: to_float(e.get("runtime")))
            except Exception:
                continue
            final_map[str(instance_id)] = best.get("code", "")

        final_path = root_output_dir / "final.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_map, f, indent=2, ensure_ascii=False)
        print(f"🏁 生成 final.json: {final_path}")
        logger.info(f"生成 final.json: {final_path}")
        return final_path
    except Exception as e:
        logger.warning(f"生成 final.json 失败: {e}")
        return None

def call_perfagent(iteration_params, logger, dry_run=False):
    """
    直接调用 perfagent.run_batch 的批量执行接口，运行本次迭代的实例优化
    """
    base_config_path = iteration_params.get("perf_base_config")

    try:
        # 使用基础配置文件，不创建临时配置
        logger.debug(f"使用PerfAgent基础配置: {base_config_path}")
        if base_config_path:
            print(f"📋 使用基础配置文件: {base_config_path}")

        if dry_run:
            logger.warning("演示模式：跳过PerfAgent实际执行")
            return {"status": "skipped", "reason": "dry_run"}

        # 目标路径和命令（批量执行脚本）
        se_root = Path(__file__).parent
        project_root = se_root.parent

        # 选择实例目录
        instances_dir = iteration_params.get("instances_dir")

        # 基础目录：使用当前迭代的输出目录，让每个实例在其子目录下生成日志与轨迹
        base_dir = Path(iteration_params["output_dir"])
        
        # 优先使用基础配置；如果提供 instance_templates_dir，则交由 run_batch 做每任务合并
        # 组装命令
        cmd = [
            sys.executable,
            "-m",
            "perfagent.run_batch",
            "--config",
            str(base_config_path),
            "--instances-dir",
            str(instances_dir),
            "--base-dir",
            str(base_dir),
            "--max-workers",
            str(iteration_params.get("num_workers", 1)),
        ]

        # 若 operator 返回 instance_templates_dir，则传给 run_batch 做每任务合并
        operator_params = iteration_params.get("operator_params", {}) or {}
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


def _get_nested(data: dict, path: str):
    cur = data
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def _normalize_text_or_list(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        # 以项目符号的形式组合列表内容
        items = []
        for v in val:
            if isinstance(v, str):
                t = v.strip()
            else:
                t = str(v).strip()
            if t:
                items.append(f"- {t}")
        return "\n".join(items)
    # 其他类型回退为字符串
    try:
        return str(val).strip()
    except Exception:
        return ""


def build_additional_requirements_from_dir(templates_dir: Path, logger) -> str:
    """从 YAML 模板目录构建 additional_requirements 文本。

    支持的键（按优先级聚合）：
    - additional_requirements
    - templates.additional_requirements
    - agent.templates.additional_requirements
    - system_template_append
    - templates.system_template_append
    - agent.templates.system_template_append
    - system_template
    - templates.system_template
    - agent.templates.system_template
    """
    if not templates_dir or not Path(templates_dir).exists():
        return ""

    pieces = []
    try:
        yaml_files = list(Path(templates_dir).glob("*.y*ml"))
        for yf in yaml_files:
            try:
                with open(yf, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"读取模板 {yf} 失败: {e}")
                continue

            key_paths = [
                "additional_requirements",
                "templates.additional_requirements",
                "agent.templates.additional_requirements",
                "system_template_append",
                "templates.system_template_append",
                "agent.templates.system_template_append",
                "system_template",
                "templates.system_template",
                "agent.templates.system_template",
            ]

            for kp in key_paths:
                val = _get_nested(data, kp)
                text = _normalize_text_or_list(val)
                if text:
                    pieces.append(text)
    except Exception as e:
        logger.warning(f"扫描模板目录失败: {e}")

    # 合并为一个文本块
    merged = "\n\n".join(pieces).strip()
    return merged


def create_temp_perf_config(iteration_params, base_config_path, logger) -> Optional[Path]:
    """创建临时 PerfAgent 配置，将算子生成的 instance_templates_dir 合并为 prompts.additional_requirements。

    - 读取基础 PerfAgent 配置 YAML
    - 从 operator_params.instance_templates_dir 构建 additional_requirements 文本
    - 若基础配置已有 prompts.additional_requirements，则进行合并拼接
    - 移除 prompts.instance_templates_dir 避免歧义
    - 写出临时 YAML 文件并返回路径；若无可注入内容则返回 None
    """
    if not base_config_path:
        return None
    try:
        with open(base_config_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"读取基础PerfAgent配置失败: {e}")
        return None

    operator_params = iteration_params.get("operator_params", {}) or {}

    # 1) 优先从算子输出目录构建
    add_texts = []
    itd = operator_params.get("instance_templates_dir")
    if itd:
        try:
            txt = build_additional_requirements_from_dir(Path(itd), logger)
            if txt:
                add_texts.append(txt)
        except Exception as e:
            logger.warning(f"解析算子模板目录失败: {e}")

    # 2) 直接从算子返回 additional_requirements（若有）
    op_additional = operator_params.get("additional_requirements")
    if op_additional:
        txt = _normalize_text_or_list(op_additional)
        if txt:
            add_texts.append(txt)

    # 3) 基础配置中已有的 additional_requirements（若有），也并入
    existing_base = None
    try:
        existing_base = base_cfg.get("prompts", {}).get("additional_requirements")
    except Exception:
        existing_base = None
    if existing_base:
        txt = _normalize_text_or_list(existing_base)
        if txt:
            add_texts.append(txt)

    # 若没有任何附加内容，不生成临时配置
    merged_text = "\n\n".join([t for t in add_texts if t]).strip()
    if not merged_text:
        return None

    # 注入到 prompts.additional_requirements，并移除旧字段
    if "prompts" not in base_cfg or base_cfg.get("prompts") is None:
        base_cfg["prompts"] = {}
    base_cfg["prompts"]["additional_requirements"] = merged_text
    if "instance_templates_dir" in base_cfg["prompts"]:
        base_cfg["prompts"].pop("instance_templates_dir", None)

    # 写出临时配置
    fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="perfagent_iteration_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            yaml.safe_dump(base_cfg, tmp, allow_unicode=True, sort_keys=False)
    except Exception as e:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        logger.warning(f"写出临时PerfAgent配置失败: {e}")
        return None

    return Path(temp_path)

def generate_per_task_configs(base_config_path: Path, instances_dir: Path, output_dir: Path, operator_result: dict, logger) -> Optional[Path]:
    """基于基础配置生成每任务配置，注入 additional_requirements。

    - 若 operator_result 提供 instance_templates_dir，则读取并构建附加要求文本。
    - 将该文本写入 prompts.additional_requirements 字段。
    - 为每个实例生成一个 <task_name>.yaml 配置文件。
    """
    if not base_config_path or not Path(base_config_path).exists():
        logger.warning("未提供有效的 PerfAgent 基础配置，跳过每任务配置生成")
        return None

    instances_path = Path(instances_dir)
    if not instances_path.exists():
        logger.warning("实例目录不存在，跳过每任务配置生成")
        return None

    # 解析 operator 结果
    add_req_text = ""
    try:
        if operator_result:
            itd = operator_result.get("instance_templates_dir")
            if itd:
                add_req_text = build_additional_requirements_from_dir(Path(itd), logger)
            else:
                # 允许算子直接返回 additional_requirements
                add_req_text = _normalize_text_or_list(operator_result.get("additional_requirements"))
    except Exception as e:
        logger.warning(f"处理 operator 结果失败: {e}")

    if not add_req_text:
        # 无附加要求则不生成每任务配置
        return None

    per_task_dir = Path(output_dir) / "per_task_configs"
    per_task_dir.mkdir(parents=True, exist_ok=True)

    # 读取基础配置一次
    try:
        with open(base_config_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"读取基础配置失败: {e}")
        return None

    # 注入 prompts.additional_requirements
    if "prompts" not in base_cfg or base_cfg.get("prompts") is None:
        base_cfg["prompts"] = {}
    base_cfg["prompts"]["additional_requirements"] = add_req_text
    # 移除旧字段以避免歧义（可选）
    if "instance_templates_dir" in base_cfg["prompts"]:
        base_cfg["prompts"].pop("instance_templates_dir", None)

    # 为每个实例写出专属配置
    for inst_file in instances_path.glob("*.json"):
        task_name = inst_file.stem
        cfg_path = per_task_dir / f"{task_name}.yaml"
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(base_cfg, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            logger.warning(f"写出每任务配置失败 {cfg_path}: {e}")

    return per_task_dir


def main():
    """主函数：策略驱动的 PerfAgent 多迭代执行"""

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
        with open(args.config, "r", encoding="utf-8") as f:
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

        traj_pool_manager = TrajPoolManager(traj_pool_path, llm_client)
        traj_pool_manager.initialize_pool()
        logger.info(f"轨迹池初始化: {traj_pool_path}")
        print(f"🏊 轨迹池: {traj_pool_path}")

        print(f"\n📊 配置概览:")
        print(f"  基础配置: {se_config['base_config']}")
        print(f"  模型: {se_config['model']['name']}")
        print(f"  实例目录: {se_config['instances']['instances_dir']}")
        print(f"  输出目录: {output_dir}")
        print(f"  迭代次数: {len(se_config['strategy']['iterations'])}")

        # 执行策略中的每个迭代
        iterations = se_config["strategy"]["iterations"]
        for i, iteration in enumerate(iterations, 1):
            logger.info(f"开始第{i}次PerfAgent迭代")
            print(f"\n=== 第{i}次PerfAgent迭代调用 ===")

            iteration_output_dir = f"{output_dir}/iteration_{i}"

            # 构建 PerfAgent 迭代参数（保持与 SE 的结构兼容）
            iteration_params = {
                "perf_base_config": iteration.get("perf_base_config"),  # 可选：指定 PerfAgent 的基础配置
                "operator": iteration.get("operator"),  # 可选：指定算子
                "model": se_config.get("model", {}),
                "instances_dir": se_config.get("instances", {}).get("instances_dir", ""),
                "output_dir": iteration_output_dir,
                "max_iterations": se_config.get("max_iterations", 10),
                "num_workers": se_config.get("num_workers", 1),
            }

            # 处理operator返回的额外参数
            operator_name = iteration.get("operator")
            if operator_name:
                print(f"🔧 调用算子: {operator_name}")
                logger.info(f"执行算子: {operator_name}")

                # 调用operator处理（传递workspace_dir而不是iteration_output_dir）
                operator_result = call_operator(operator_name, output_dir, i, se_config, logger)
                if operator_result:
                    iteration_params["operator_params"] = operator_result
                    print(f"✅ Operator {operator_name} 执行成功")
                    print(f"📋 生成参数: {list(operator_result.keys())}")
                    # 临时配置在 call_perfagent 中生成与清理，无需这里处理
                else:
                    print(f"⚠️  Operator {operator_name} 执行失败，继续执行但不使用增强")
                    logger.warning(f"Operator {operator_name} 执行失败，继续执行但不使用增强")
            else:
                print(f"🔄 无算子处理")
                logger.debug(f"第{i}次迭代无算子处理")

            logger.debug(f"第{i}次PerfAgent迭代参数: {json.dumps(iteration_params, ensure_ascii=False)}")
            print(f"使用配置: {iteration.get('perf_base_config', 'None')}")
            print(f"算子: {iteration.get('operator', 'None')}")
            print(f"输出目录: {iteration_output_dir}")

            # 执行 PerfAgent
            if args.mode == "execute":
                logger.info(f"直接执行模式：第{i}次PerfAgent迭代")
                result = call_perfagent(iteration_params, logger, dry_run=False)
                print(f"执行结果: {result['status']}")

                # 成功则生成.tra并更新轨迹池
                if result.get("status") == "success":
                    logger.info(f"开始为第{i}次迭代生成.tra文件")
                    # 生成 .tra 文件
                    try:
                        processor = TrajectoryProcessor()
                        iteration_dir = Path(iteration_output_dir)

                        # 处理当前迭代目录下的所有实例
                        tra_stats = processor.process_iteration_directory(iteration_dir)
                        if tra_stats and tra_stats.get("total_tra_files", 0) > 0:
                            logger.info(
                                f"第{i}次PerfAgent迭代.tra文件生成完成: "
                                f"{tra_stats['total_tra_files']}个文件, ~{tra_stats['total_tokens']}tokens"
                            )
                            print(f"📝 生成了 {tra_stats['total_tra_files']} 个.tra文件")

                            # 更新轨迹池
                            try:
                                extractor = TrajExtractor()
                                instance_data_list = extractor.extract_instance_data(iteration_dir)
                                if instance_data_list:
                                    for (
                                        instance_name,
                                        problem_description,
                                        trajectory_content,
                                        patch_content,
                                    ) in instance_data_list:
                                        traj_pool_manager.add_iteration_summary(
                                            instance_name=instance_name,
                                            iteration=i,
                                            trajectory_content=trajectory_content,
                                            patch_content=patch_content,
                                            problem_description=problem_description or None,
                                        )
                                    logger.info(f"成功提取并处理了 {len(instance_data_list)} 个实例")
                                else:
                                    logger.warning(f"第{i}次迭代没有找到有效的实例数据")
                                    print("⚠️ 没有找到有效的实例数据")

                                pool_stats = traj_pool_manager.get_pool_stats()
                                logger.info(
                                    f"轨迹池更新完成: {pool_stats['total_instances']}实例, {pool_stats['total_iterations']}总迭代"
                                )
                                print(
                                    f"🏊 轨迹池更新: {pool_stats['total_instances']}实例, {pool_stats['total_iterations']}总迭代"
                                )
                            except Exception as pool_error:
                                logger.error(f"第{i}次迭代轨迹池更新失败: {pool_error}")
                                print(f"⚠️ 轨迹池更新失败: {pool_error}")
                        else:
                            logger.warning(f"第{i}次迭代未生成.tra文件")
                            print("⚠️ 未生成.tra文件（可能没有有效轨迹）")
                    except Exception as tra_error:
                        logger.error(f"第{i}次迭代生成.tra文件失败: {tra_error}")
                        print(f"⚠️ .tra文件生成失败: {tra_error}")
            else:
                logger.info(f"演示模式：第{i}次PerfAgent迭代")
                result = call_perfagent(iteration_params, logger, dry_run=True)
                print(f"演示结果: {result['status']}")
                print("📝 演示模式：跳过.tra文件生成与轨迹池更新")

            if result.get("status") == "failed":
                logger.error(f"第{i}次PerfAgent迭代执行失败，停止后续迭代")
                break

        logger.info("所有PerfAgent迭代准备完成")

        print(f"\n🎯 执行总结:")
        print(f"  ✅ 解析{len(iterations)}个迭代配置")
        print(f"  ✅ 时间戳: {timestamp}")
        print(f"  ✅ 日志文件: {log_file}")
        print(f"  📁 输出目录: {output_dir}")
        try:
            final_pool_stats = traj_pool_manager.get_pool_stats()
            print(
                f"  🏊 轨迹池: {final_pool_stats['total_instances']}实例, {final_pool_stats['total_iterations']}总迭代"
            )
            print(f"  🏊 轨迹池文件: {traj_pool_path}")
        except Exception:
            pass

        logger.info("SE PerfAgent 多迭代执行完成")

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

