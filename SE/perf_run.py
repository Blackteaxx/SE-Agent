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
    直接调用 perfagent.run_batch 的批量执行接口，运行本次迭代的实例优化
    """
    base_config_path = iteration_params.get("perf_base_config")

    try:
        # 基础配置 + SE 主模型配置覆盖 => 生成临时 PerfAgent 配置
        logger.debug(f"使用PerfAgent基础配置: {base_config_path}")

        if dry_run:
            logger.warning("演示模式：跳过PerfAgent实际执行")
            return {"status": "skipped", "reason": "dry_run"}

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

        # 优先使用基础配置；如果提供 instance_templates_dir，则交由 run_batch 做每任务合并
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
                print("🔄 无算子处理")
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

                # 成功则生成.tra并更新 traj.pool
                # .tra 直接使用 history 生成
                # traj.pool 轨迹总结通过 LLM Summary 生成
                if result.get("status") == "success":
                    logger.info(f"开始为第{i}次迭代生成.tra文件")
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
                                    # 交由 TrajPoolManager 并行生成并批量写入，避免并发写文件
                                    processed_count = traj_pool_manager.summarize_and_add_iteration_batch(
                                        instance_data_list, iteration=i, num_workers=se_config.get("num_workers")
                                    )
                                    logger.info(f"成功提取并处理了 {processed_count} 个实例")
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

        print("\n🎯 执行总结:")
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
