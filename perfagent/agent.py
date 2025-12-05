"""
PerfAgent 核心类

实现代码性能优化的主要逻辑，包括迭代优化、diff 应用、性能评估等功能。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PerfAgentConfig
from .diff_applier import DiffApplier
from .effibench.benchmark import run_performance_benchmark
from .llm_client import LLMClient
from .trajectory import TrajectoryLogger
from .utils.log import get_se_logger


@dataclass
class EffiBenchXInstance:
    id: str
    title: str
    title_slug: str
    description: str
    description_md: str
    source: str
    url: str
    type: str
    starter_code: str | None = None
    solutions: dict[str, dict[str, str]] = field(default_factory=dict)
    language: str | None = None
    generated_tests: list[dict[str, Any]] = field(default_factory=list)
    evaluator: str | None = None
    # 任务名（来源于实例文件名，不含扩展名）
    task_name: str | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EffiBenchXInstance":
        # Robustly parse generated_tests when it can be a list or a JSON string
        gt_raw = data.get("generated_tests", [])
        if isinstance(gt_raw, str):
            try:
                gt_parsed = json.loads(gt_raw)
            except Exception:
                gt_parsed = []
        elif isinstance(gt_raw, list):
            gt_parsed = gt_raw
        else:
            gt_parsed = []

        return EffiBenchXInstance(
            id=str(data.get("id", "unknown")),
            title=data.get("title", ""),
            title_slug=data.get("title_slug", ""),
            description=data.get("description", ""),
            description_md=data.get("description_md", ""),
            source=data.get("source", ""),
            url=data.get("url", ""),
            type=data.get("type", ""),
            starter_code=data.get("starter_code"),
            solutions=data.get("solutions", {}),
            language=data.get("language"),
            generated_tests=gt_parsed,
            evaluator=data.get("evaluator"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "title_slug": self.title_slug,
            "description": self.description,
            "description_md": self.description_md,
            "source": self.source,
            "url": self.url,
            "type": self.type,
            "starter_code": self.starter_code,
            "solutions": self.solutions,
            "language": self.language,
            "generated_tests": self.generated_tests,
            "evaluator": self.evaluator,
            "task_name": self.task_name,
        }


class PerfAgent:
    """性能优化 Agent"""

    def __init__(self, config: PerfAgentConfig):
        self.config = config

        # 简化逻辑：凭据存在即初始化 LLMClient，无需 use_llm 标志
        self.llm_client = None
        if self.config.model.api_base and self.config.model.api_key:
            client_cfg = {
                "name": self.config.model.name,
                "api_base": self.config.model.api_base,
                "api_key": self.config.model.api_key,
                "max_output_tokens": self.config.model.max_output_tokens,
                "request_timeout": self.config.model.request_timeout,
                "max_retries": self.config.model.max_retries,
                "retry_delay": self.config.model.retry_delay,
                "log_inputs_outputs": self.config.model.log_inputs_outputs,
                "log_sanitize": self.config.model.log_sanitize,
            }
            # 将 LLM I/O 独立写入 log_dir/llm_io.log
            io_log_file = Path(self.config.logging.log_dir) / "llm_io.log"
            self.llm_client = LLMClient(
                client_cfg,
                io_log_path=io_log_file,
                log_inputs_outputs=self.config.model.log_inputs_outputs,
                log_sanitize=self.config.model.log_sanitize,
                request_timeout=self.config.model.request_timeout,
            )

        self.diff_applier = DiffApplier()

        # 设置日志：统一绑定到单一文件
        # 使用包含日志目录名的唯一 logger 名称，避免并发实例复用同名导致串写
        agent_logger_name = f"perfagent.agent.{Path(self.config.logging.log_dir).name}"
        get_se_logger(
            agent_logger_name,
            Path(self.config.logging.log_dir) / "perfagent.log",
            emoji="🔧",
            level=getattr(logging, self.config.logging.log_level.upper()),
            also_stream=False,
        )
        self.logger = logging.getLogger(agent_logger_name)

        # 优化历史
        self.optimization_history: list[dict[str, Any]] = []

        # 初始代码来源："default" | "text" | "dir"
        self._initial_code_source: str = "default"

    def _normalize_language(self, lang: str | None) -> str:
        # 标准化语言名称
        if not lang:
            return "python3"
        l = lang.lower()
        if l in ("python", "py", "python3"):
            return "python3"
        if l in ("cpp", "c++", "cxx"):
            return "cpp"
        if l in ("javascript", "js"):
            return "javascript"
        if l in ("java",):
            return "java"
        return l

    def _get_default_placeholder(self, language: str | None = None) -> str:
        """获取默认占位符代码（根据语言）"""
        lang = self._normalize_language(language or self.config.language_cfg.language)
        placeholder_map = {
            "python3": "# Start your code here\n",
            "cpp": "// Start your code here\n",
            "java": "// Start your code here\n",
            "javascript": "// Start your code here\n",
            "golang": "// Start your code here\n",
        }
        return placeholder_map.get(lang, "# Start your code here\n")

    def _extract_initial_code(
        self, instance: EffiBenchXInstance, language: str | None = None, optimization_target: str | None = None
    ) -> str:
        """从配置/文件系统注入或生成初始代码。

        优先级：
        1) 配置 overrides.initial_code_text（直接文本）
        2) 配置 overrides.initial_code_dir（按实例名匹配文件）
        3) 默认占位符代码（根据语言）
        """
        try:
            # 默认来源
            self._initial_code_source = "default"
            # 1) 直接文本覆盖
            override_text = getattr(getattr(self.config, "overrides", None), "initial_code_text", None)
            if isinstance(override_text, str) and override_text.strip():
                self._initial_code_source = "text"
                return override_text if override_text.endswith("\n") else override_text + "\n"

            # 2) 目录覆盖（按实例名匹配文件）
            code_dir = getattr(getattr(self.config, "overrides", None), "initial_code_dir", None)
            task_name = getattr(instance, "task_name", None) or getattr(instance, "id", None)
            if code_dir and task_name:
                lang = self._normalize_language(language or self.config.language_cfg.language)
                # 语言扩展映射
                ext_map = {
                    "python3": [".py"],
                    "cpp": [".cpp", ".cc", ".cxx"],
                    "java": [".java"],
                    "javascript": [".js", ".mjs"],
                    "golang": [".go"],
                }
                candidates: list[Path] = []
                for ext in ext_map.get(lang, []):
                    candidates.append(Path(code_dir) / f"{task_name}{ext}")
                # 退化：任意匹配同名文件（不区分扩展名）
                try:
                    for fp in Path(code_dir).iterdir():
                        if fp.is_file() and fp.stem == task_name and fp not in candidates:
                            candidates.append(fp)
                except Exception:
                    pass

                for fp in candidates:
                    try:
                        if fp.exists():
                            code = fp.read_text(encoding="utf-8")
                            if isinstance(code, str) and code.strip():
                                self.logger.info(f"使用覆盖初始代码: {fp}")
                                self._initial_code_source = "dir"
                                return code if code.endswith("\n") else code + "\n"
                    except Exception as e:
                        self.logger.warning(f"读取初始代码文件失败 {fp}: {e}")
        except Exception as e:
            # 覆盖流程失败则回退到占位符
            self.logger.warning(f"初始代码覆盖失败，使用默认占位符: {e}")

        # 3) 默认占位符（保持现有测试兼容）
        return self._get_default_placeholder(language)

    def _prepare_test_cases(self, instance: EffiBenchXInstance) -> list[dict[str, Any]]:
        """准备测试用例（实例仅为 dataclass）"""
        return instance.generated_tests or []

    def _detect_language(self, instance: EffiBenchXInstance) -> str:
        """检测编程语言（仅保留以兼容调用路径，但不使用）"""
        return self._normalize_language(self.config.language_cfg.language)

    def _evaluate_performance(
        self, language: str, code: str, test_cases: list[dict[str, Any]], instance: Any
    ) -> dict[str, Any]:
        """评估代码性能，保持参数兼容"""

        # 如果代码与占位符代码相同，返回默认失败结构
        if code == self._get_default_placeholder(language):
            perf = {
                "original_n": 0,
                "n": 0,
                "runtime": float("inf"),
                "memory": float("inf"),
                "integral": float("inf"),
                "analysis": {
                    "runtime": {
                        "original_n": 0,
                        "n": 0,
                        "mean": float("inf"),
                        "std": float("inf"),
                        "min": float("inf"),
                        "max": float("inf"),
                        "max_diff": float("inf"),
                        "95%_CI": (float("inf"), float("inf")),
                        "trimmed_mean": float("inf"),
                    },
                    "memory": {
                        "original_n": 0,
                        "n": 0,
                        "mean": float("inf"),
                        "std": float("inf"),
                        "min": float("inf"),
                        "max": float("inf"),
                        "max_diff": float("inf"),
                        "95%_CI": (float("inf"), float("inf")),
                        "trimmed_mean": float("inf"),
                    },
                    "integral": {
                        "original_n": 0,
                        "n": 0,
                        "mean": float("inf"),
                        "std": float("inf"),
                        "min": float("inf"),
                        "max": float("inf"),
                        "max_diff": float("inf"),
                        "95%_CI": (float("inf"), float("inf")),
                        "trimmed_mean": float("inf"),
                    },
                },
            }
            return {
                "performance_analysis": perf,
                "first_run_details": [],
                "failed_submission_exit_codes": [],
                "pass_rates": [],
                "pass_rate_consistent": True,
            }

        # 若 evaluator 或测试用例缺失/格式不合法，直接返回默认结构以避免长时间的后端调用
        evaluator = getattr(instance, "evaluator", None)
        tc_valid = bool(test_cases) and isinstance(test_cases, list) and isinstance(test_cases[0], dict)
        if not evaluator or not tc_valid:
            perf = {
                "original_n": 0,
                "n": 0,
                "runtime": float("inf"),
                "memory": float("inf"),
                "integral": float("inf"),
                "analysis": {
                    "runtime": {
                        "original_n": 0,
                        "n": 0,
                        "mean": float("inf"),
                        "std": float("inf"),
                        "min": float("inf"),
                        "max": float("inf"),
                        "max_diff": float("inf"),
                        "95%_CI": (float("inf"), float("inf")),
                        "trimmed_mean": float("inf"),
                    },
                    "memory": {
                        "original_n": 0,
                        "n": 0,
                        "mean": float("inf"),
                        "std": float("inf"),
                        "min": float("inf"),
                        "max": float("inf"),
                        "max_diff": float("inf"),
                        "95%_CI": (float("inf"), float("inf")),
                        "trimmed_mean": float("inf"),
                    },
                    "integral": {
                        "original_n": 0,
                        "n": 0,
                        "mean": float("inf"),
                        "std": float("inf"),
                        "min": float("inf"),
                        "max": float("inf"),
                        "max_diff": float("inf"),
                        "95%_CI": (float("inf"), float("inf")),
                        "trimmed_mean": float("inf"),
                    },
                },
            }
            return {
                "performance_analysis": perf,
                "first_run_details": [],
                "failed_test_details": [],
                "failed_submission_exit_codes": [],
                "pass_rates": [],
                "pass_rate_consistent": True,
            }

        # 级联评估：先用 benchmark 进行一次运行（num_runs=1），若未全部通过则直接返回
        try:
            single_run_summary = run_performance_benchmark(
                lang=language,
                solution=code,
                test_cases=test_cases,
                evaluator=evaluator,
                num_runs=1,
                time_limit=self.config.runtime.time_limit,
                memory_limit=self.config.runtime.memory_limit,
                trim_ratio=self.config.runtime.trim_ratio,
                max_workers=self.config.runtime.max_workers,
            )
        except Exception as e:
            # 单次运行失败则回退到默认失败结构，保持与现有测试兼容
            self.logger.warning(f"单次运行评估失败，返回默认性能结构: {e}")
            perf = {
                "original_n": 0,
                "n": 0,
                "runtime": float("inf"),
                "memory": float("inf"),
                "integral": float("inf"),
                "analysis": {
                    "runtime": {
                        "original_n": 0,
                        "n": 0,
                        "mean": float("inf"),
                        "std": float("inf"),
                        "min": float("inf"),
                        "max": float("inf"),
                        "max_diff": float("inf"),
                        "95%_CI": (float("inf"), float("inf")),
                        "trimmed_mean": float("inf"),
                    },
                    "memory": {
                        "original_n": 0,
                        "n": 0,
                        "mean": float("inf"),
                        "std": float("inf"),
                        "min": float("inf"),
                        "max": float("inf"),
                        "max_diff": float("inf"),
                        "95%_CI": (float("inf"), float("inf")),
                        "trimmed_mean": float("inf"),
                    },
                    "integral": {
                        "original_n": 0,
                        "n": 0,
                        "mean": float("inf"),
                        "std": float("inf"),
                        "min": float("inf"),
                        "max": float("inf"),
                        "max_diff": float("inf"),
                        "95%_CI": (float("inf"), float("inf")),
                        "trimmed_mean": float("inf"),
                    },
                },
            }
            return {
                "performance_analysis": perf,
                "first_run_details": [],
                "failed_test_details": [],
                "pass_rates": [],
                "pass_rate_consistent": True,
            }

        # 计算单次运行通过率（优先使用返回的 pass_rates）
        pr_list = single_run_summary.get("pass_rates", [])
        if pr_list:
            single_pass_rate = float(pr_list[0])
        else:
            try:
                first_run_details = single_run_summary.get("first_run_details", [])
                total_cases = len(first_run_details) if first_run_details else 0
                num_passed = sum(1 for tc in (first_run_details or []) if tc.get("passed", False))
                single_pass_rate = num_passed / total_cases if total_cases > 0 else 0.0
            except Exception:
                single_pass_rate = 0.0

        # 若未全部通过，直接返回单次运行的结果（不进行多次性能评估）
        if single_pass_rate < 1.0:
            return single_run_summary

        # 所有测试用例通过，进行正式的多次性能评估
        try:
            result = run_performance_benchmark(
                lang=language,
                solution=code,
                test_cases=test_cases,
                evaluator=evaluator,
                num_runs=self.config.runtime.num_runs,
                time_limit=self.config.runtime.time_limit,
                memory_limit=self.config.runtime.memory_limit,
                trim_ratio=self.config.runtime.trim_ratio,
                max_workers=self.config.runtime.max_workers,
            )
            return result
        except Exception as e:
            self.logger.error(f"性能评估失败: {e}")
            return {
                "performance_analysis": {"trimmed_mean": float("inf")},
                "first_run_details": [],
                "failed_test_details": [],
                "pass_rates": [],
                "pass_rate_consistent": False,
            }

    def run(self, instance: EffiBenchXInstance) -> dict[str, Any]:
        """运行性能优化流程（仅使用配置语言，实例为 dataclass）"""
        inst = instance
        # 优先使用文件名（task_name）作为实例 ID，若不存在则回退到 JSON 中的 id
        instance_id = getattr(inst, "task_name", None) or getattr(inst, "id", "unknown")

        # 初始化轨迹记录器（统一日志目录到 config.logging.log_dir）
        trajectory = TrajectoryLogger(
            instance_id,
            self.config.logging.trajectory_dir,
            log_dir=self.config.logging.log_dir,
        )

        try:
            self.logger.info(f"开始优化实例: {instance_id}")

            # 使用配置中的语言
            language = self._normalize_language(self.config.language_cfg.language)

            # 设置轨迹语言与优化方向
            trajectory.metadata.language = language
            trajectory.metadata.optimization_target = self.config.optimization.target

            # 将系统提示在对话历史最开头记录一次
            system_prompt_header = self._build_system_prompt(
                language=language,
                optimization_target=self.config.optimization.target,
                task_description=inst.description_md,
            )
            trajectory.add_history(role="system", content=system_prompt_header, message_type="system_prompt")

            # 提取初始代码与测试用例
            initial_code = self._extract_initial_code(
                inst, language=language, optimization_target=self.config.optimization.target
            )
            test_cases = self._prepare_test_cases(inst)

            if not initial_code:
                raise ValueError("无法提取初始代码")

            # 若接受了外部初始代码（文本或目录），则初始评估计为第1次迭代
            iter_offset = 1 if self._initial_code_source in ("text", "dir") else 0

            # 初始化当前代码与最佳性能
            current_code = initial_code
            best_performance = float("inf")
            best_code = initial_code
            latest_optimized_code = current_code

            # 评估初始性能
            step_id = trajectory.start_step(
                "initial_evaluation", query="Evaluate the initial code performance.", code_snapshot=current_code
            )
            initial_performance = self._evaluate_performance(language, current_code, test_cases, inst)
            initial_evaluation_summary = {
                "performance_analysis": initial_performance.get("performance_analysis", {}),
                "failed_test_details": initial_performance.get("failed_test_details", [])[:3],
                "pass_rates": initial_performance.get("pass_rates", []),
                "pass_rate_consistent": initial_performance.get("pass_rate_consistent", False),
            }
            initial_summary_text = self._build_summary_text(
                iteration=1 if iter_offset else 0,
                code_changed=False,
                diff_text=None,
                benchmark_results=initial_performance,
                current_program=current_code,
            )
            trajectory.end_step(
                step_id,
                response=initial_summary_text,
                thought="收集初始性能基线以指导后续优化",
                code_changed=False,
                performance_metrics=initial_evaluation_summary,
                code_snapshot=current_code,
            )

            def _extract_pass_rate(results: dict[str, Any]) -> float:
                pr_list = results.get("pass_rates") or []
                try:
                    if isinstance(pr_list, list) and pr_list:
                        return float(min(float(p) for p in pr_list))
                except Exception:
                    pass
                try:
                    fr = results.get("first_run_details") or []
                    total = len(fr)
                    passed = sum(1 for tc in fr if tc.get("passed", False))
                    return (passed / total) if total > 0 else 0.0
                except Exception:
                    return 0.0

            best_pass_rate = _extract_pass_rate(initial_performance)
            target = self.config.optimization.target
            init_metric = initial_evaluation_summary["performance_analysis"].get(target, float("inf"))
            if init_metric <= best_performance:
                best_performance = init_metric
                best_code = current_code

            # 记录当前代码对应的最新评估结果（用于提示构造）
            current_benchmark_results = initial_performance

            # 迭代优化
            no_improve_count = 0  # 连续未改进计数（跨迭代累积）

            # 主迭代循环
            # 若存在外部初始代码（文本或目录），初始评估记为第1次迭代，优化循环次数相应减一
            remaining_iterations = max(0, self.config.max_iterations - iter_offset)
            for iteration in range(remaining_iterations):
                self.logger.info(f"开始第 {iteration + 1 + iter_offset} 次迭代")

                # 生成优化建议
                opt_prompt = self._build_optimization_prompt(
                    current_program=current_code,
                    language=language,
                    benchmark_results=current_benchmark_results,
                )
                step_id = trajectory.start_step(
                    "generate_optimization",
                    query=opt_prompt,
                    code_snapshot=current_code,
                )

                # multi-turn chat: 构造消息序列（保留最近会话上下文）
                system_prompt = self._build_system_prompt(
                    language=language,
                    optimization_target=self.config.optimization.target,
                    task_description=inst.description_md,
                )
                messages = self._build_messages(system_prompt, trajectory.history, opt_prompt)

                if self.llm_client:
                    optimization_response = self.llm_client.call_llm(
                        messages,
                        temperature=self.config.model.temperature,
                        max_tokens=self.config.model.max_output_tokens,
                        usage_context="perfagent.optimize",
                    )
                else:
                    # 保守回退：LLM 未配置时返回空建议，避免引入无效 diff
                    optimization_response = "LLM 未配置或不可用，跳过本次优化建议。请检查 API 配置。"

                # 提取代码变更
                diff_text = None
                optimized_code = None

                if self.config.optimization.code_generation_mode == "direct":
                    optimized_code = self._extract_full_code_from_response(optimization_response)
                    if not optimized_code:
                        summary_text = self._build_summary_text(
                            iteration=iteration + 1 + iter_offset,
                            code_changed=False,
                            diff_text=None,
                            benchmark_results=None,
                            current_program=current_code,
                            error_message="无法从响应中提取有效的完整代码",
                        )
                        trajectory.end_step(
                            step_id,
                            response=optimization_response,
                            thought="未能提取有效的完整代码区块",
                            code_changed=False,
                            diff=None,
                            error="无法从响应中提取有效的完整代码",
                            code_snapshot=current_code,
                            summary=summary_text,
                        )
                        continue
                else:
                    # 提取 diff
                    diff_text = self._extract_diff_from_response(optimization_response)

                    if not diff_text:
                        summary_text = self._build_summary_text(
                            iteration=iteration + 1 + iter_offset,
                            code_changed=False,
                            diff_text=None,
                            benchmark_results=None,
                            current_program=current_code,
                            error_message="无法从响应中提取有效的 diff",
                        )
                        trajectory.end_step(
                            step_id,
                            response=optimization_response,
                            thought="未能提取有效的 SEARCH/REPLACE 区块",
                            code_changed=False,
                            diff=None,
                            error="无法从响应中提取有效的 diff",
                            code_snapshot=current_code,
                            summary=summary_text,
                        )
                        continue

                # 应用变更
                try:
                    if self.config.optimization.code_generation_mode == "diff":
                        optimized_code = self.diff_applier.apply_diff(current_code, diff_text)

                    # 如果代码未发生变化，仅结束该步骤并跳过迭代
                    if optimized_code == current_code:
                        summary_text = self._build_summary_text(
                            iteration=iteration + 1 + iter_offset,
                            code_changed=False,
                            diff_text=diff_text,
                            benchmark_results=current_benchmark_results,
                            current_program=current_code,
                        )
                        trajectory.end_step(
                            step_id,
                            response=optimization_response,
                            thought="diff 应用后代码未变化，跳过",
                            code_changed=False,
                            diff=diff_text,
                            code_snapshot=current_code,
                            summary=summary_text,
                        )
                        self.logger.warning("代码未发生变化，跳过此次迭代")
                        # 记录未改进一次并检查早停
                        no_improve_count += 1
                        if self.config.early_stop_no_improve and no_improve_count >= self.config.early_stop_no_improve:
                            self.logger.info(f"连续未改进达到阈值 {self.config.early_stop_no_improve}，提前停止。")
                            break
                        continue

                    # 评估优化后的性能，并将结果作为 performance_metrics 附加到 generate_optimization
                    try:
                        latest_optimized_code = optimized_code
                        self.logger.info("开始评估优化后的代码性能")
                        performance_result = self._evaluate_performance(language, optimized_code, test_cases, inst)

                        target = self.config.optimization.target
                        current_performance = performance_result.get("performance_analysis", {}).get(
                            target, float("inf")
                        )
                        current_pass_rate = _extract_pass_rate(performance_result)

                        # 仅保留核心评估结果
                        evaluation_summary = {
                            "performance_analysis": performance_result.get("performance_analysis", {}),
                            "failed_test_details": performance_result.get("failed_test_details", [])[:3],
                            "pass_rates": performance_result.get("pass_rates", []),
                            "pass_rate_consistent": performance_result.get("pass_rate_consistent", False),
                        }

                        # 记录优化历史
                        self.optimization_history.append(
                            {
                                "iteration": iteration + 1 + iter_offset,
                                "diff": diff_text,
                                "performance_before": best_performance,
                                "performance_after": current_performance,
                                "improvement": best_performance - current_performance,
                                # 强制转换为 Python bool，避免 numpy.bool_ 导致 JSON 序列化错误
                                "success": bool(
                                    (current_pass_rate > best_pass_rate)
                                    or (
                                        current_pass_rate == best_pass_rate
                                        and current_pass_rate == 1.0
                                        and current_performance < best_performance
                                    )
                                ),
                            }
                        )

                        improved = False
                        if current_pass_rate > best_pass_rate:
                            improved = True
                        elif (
                            current_pass_rate == best_pass_rate
                            and current_pass_rate == 1.0
                            and current_performance < best_performance
                        ):
                            improved = True

                        if improved:
                            best_pass_rate = current_pass_rate
                            best_performance = current_performance
                            best_code = optimized_code
                            self.logger.info(
                                f"采用更优代码: pass_rate {best_pass_rate:.2f}, {target} {best_performance:.4f}"
                            )
                            no_improve_count = 0
                        else:
                            self.logger.info(
                                f"未改进: pass_rate {current_pass_rate:.2f} vs {best_pass_rate:.2f}; {target} {current_performance:.4f} vs {best_performance:.4f}"
                            )
                            no_improve_count += 1

                        # 根据配置决定是否采用优化后的代码作为下一轮基础
                        if self.config.optimization.adopt_only_if_improved:
                            if improved:
                                current_code = optimized_code
                            else:
                                current_code = best_code
                        else:
                            current_code = optimized_code
                        # 更新最新评估结果，供下一轮提示生成使用
                        current_benchmark_results = performance_result

                        adopted = True
                        if self.config.optimization.adopt_only_if_improved:
                            adopted = improved
                        summary_text = self._build_summary_text(
                            iteration=iteration + 1 + iter_offset,
                            code_changed=adopted,
                            diff_text=diff_text,
                            benchmark_results=performance_result,
                            current_program=current_code,
                        )
                        trajectory.end_step(
                            step_id,
                            response=optimization_response,
                            thought=("应用 diff 并完成性能评估" if adopted else "评估未改进，未采用优化"),
                            code_changed=adopted,
                            diff=diff_text,
                            performance_metrics=evaluation_summary,
                            code_snapshot=current_code,
                            summary=summary_text,
                        )

                        # 早停检查（评估后）
                        if self.config.early_stop_no_improve and no_improve_count >= self.config.early_stop_no_improve:
                            self.logger.info(f"连续未改进达到阈值 {self.config.early_stop_no_improve}，提前停止。")
                            break

                    except Exception as e:
                        summary_text = self._build_summary_text(
                            iteration=iteration + 1,
                            code_changed=True,
                            diff_text=diff_text,
                            benchmark_results=None,
                            current_program=current_code,
                            error_message=f"性能评估失败: {e}",
                        )
                        trajectory.end_step(
                            step_id,
                            response=optimization_response,
                            thought="性能评估阶段发生异常",
                            code_changed=True,
                            diff=diff_text,
                            performance_metrics=None,
                            error=f"性能评估失败: {e}",
                            code_snapshot=current_code,
                            summary=summary_text,
                        )
                        continue

                except Exception as e:
                    summary_text = self._build_summary_text(
                        iteration=iteration + 1 + iter_offset,
                        code_changed=False,
                        diff_text=diff_text,
                        benchmark_results=None,
                        current_program=current_code,
                        error_message=f"应用 diff 失败: {e}",
                    )
                    trajectory.end_step(
                        step_id,
                        response=optimization_response,
                        thought="应用 diff 阶段发生异常",
                        code_changed=None,
                        diff=diff_text,
                        performance_metrics=None,
                        error=f"应用 diff 失败: {e}",
                        code_snapshot=current_code,
                        summary=summary_text,
                    )
                    continue

            # 完成优化
            # 计算 success 时确保参与比较的值为原生 Python 类型
            target = self.config.optimization.target
            initial_trimmed = initial_performance.get("performance_analysis", {}).get(target, float("inf"))
            try:
                item_fn = getattr(initial_trimmed, "item", None)
                if callable(item_fn):
                    initial_trimmed = item_fn()
            except Exception:
                pass
            if isinstance(initial_trimmed, str):
                s = initial_trimmed.strip().lower()
                if s in ("inf", "+inf", "infinity", "+infinity"):
                    initial_trimmed = float("inf")
                elif s in ("-inf", "-infinity"):
                    initial_trimmed = float("-inf")
                elif s == "nan":
                    initial_trimmed = float("nan")
                else:
                    try:
                        initial_trimmed = float(initial_trimmed)
                    except Exception:
                        initial_trimmed = float("inf")

            try:
                bp_item = getattr(best_performance, "item", None)
                if callable(bp_item):
                    best_performance = bp_item()
            except Exception:
                pass

            final_result = {
                "instance_id": instance_id,
                "initial_code": initial_code,
                "optimized_code": latest_optimized_code,
                "initial_performance": initial_trimmed,
                "final_performance": best_performance,
                # 总迭代数 = 初始评估(若存在) + 实际优化循环次数
                "total_iterations": (1 if self._initial_code_source in ("text", "dir") else 0) + remaining_iterations,
                "optimization_history": self.optimization_history,
                # 显式转换为 Python bool，避免 numpy.bool_
                "success": bool(best_performance < initial_trimmed),
            }

            unit = "s" if target == "runtime" else ("MB" if target == "memory" else "MB*s")
            final_result["language"] = language
            final_result["optimization_target"] = target
            final_result["performance_unit"] = unit

            try:
                md_metrics, md_artifacts = self._build_metrics_and_artifacts(current_benchmark_results)
                final_result["final_artifacts"] = self._format_artifacts_md(md_artifacts)
            except Exception:
                final_result["final_artifacts"] = None

            # 记录最终轨迹
            trajectory_file = trajectory.finalize(
                success=final_result["success"],
                final_performance={
                    "target": self.config.optimization.target,
                    "trimmed_mean": best_performance,
                    "unit": unit,
                },
                final_submission_code=latest_optimized_code,
            )

            final_result["trajectory_file"] = trajectory_file

            return final_result

        except Exception as e:
            self.logger.error(f"优化过程失败: {e}")
            try:
                trajectory.finalize(success=False, error_message=str(e), final_submission_code=best_code)
            except Exception:
                trajectory.finalize(success=False, error_message=str(e), final_submission_code=None)
            raise

    def _build_optimization_prompt(
        self,
        current_program: str,
        language: str,
        benchmark_results: dict[str, Any],
    ) -> str:
        """构建优化提示词，填充当前程序、评估指标与构件(section)。"""
        if self.config.optimization.code_generation_mode == "direct":
            return self.config.prompts.optimization_template
        # 构造 metrics 与 artifacts
        metrics_dict, artifacts_dict = self._build_metrics_and_artifacts(benchmark_results)
        # 以 Markdown 格式化，便于模型阅读
        current_metrics_str = self._format_metrics_md(metrics_dict)
        current_artifacts_str = self._format_artifacts_md(artifacts_dict)
        current_program_md = f"```\n{current_program}\n```"

        try:
            return self.config.prompts.optimization_template.format(
                current_program=current_program_md,
                current_metrics=current_metrics_str,
                current_artifacts_section=current_artifacts_str,
                language=language,
            )
        except Exception:
            # 若模板占位符不匹配，回退为一个通用提示
            return (
                "# Task\n"
                "请分析以下程序信息，并根据系统提示生成 `## Thinking` 与 `## Diffs`：\n\n"
                "## Current Program\n" + current_program_md + "\n\n"
                "## Current Metrics\n" + current_metrics_str + "\n\n"
                "## Current Artifacts\n" + current_artifacts_str
            )

    def _build_system_prompt(self, language: str, optimization_target: str, task_description: str) -> str:
        """格式化系统提示词，填充语言/优化目标/任务描述/附加要求。"""
        tmpl = self.config.prompts.system_template
        additional = self.config.prompts.additional_requirements or ""
        local_memory = getattr(self.config.prompts, "local_memory", None) or ""
        if tmpl:
            try:
                return tmpl.format(
                    language=language,
                    optimization_target=optimization_target,
                    task_description=task_description,
                    additional_requirements=additional,
                    local_memory=local_memory,
                )
            except Exception:
                return tmpl
        # 默认提示
        return (
            f"你是一个专业的代码性能优化专家。目标是提升 {optimization_target}。\n"
            f"当前语言：{language}。任务描述：{task_description}\n\n"
            f"附加要求：{additional}\n\n"
            f"本地记忆：{local_memory}"
        )

    def _build_metrics_and_artifacts(self, benchmark_results: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """根据基准评估结果构造 current_metrics 与 current_artifacts_section。"""
        performance_metrics = benchmark_results.get("performance_analysis", {})
        failed_test_details = benchmark_results.get("failed_test_details", []) or []

        # 失败情况：汇总失败信息并返回错误指标
        target = self.config.optimization.target
        target_value = performance_metrics.get(target, float("inf"))
        if failed_test_details or target_value == float("inf"):
            num_failed = len(failed_test_details)
            num_total = len(benchmark_results.get("first_run_details", []))
            pass_rate = (num_total - num_failed) / num_total if num_total > 0 else 0

            representative_failures: dict[str, Any] = {}
            for failure in failed_test_details:
                status = failure.get("status", "unknown")
                if status not in representative_failures:
                    representative_failures[status] = failure

            failure_details_summary: list[str] = []
            for status, failure in representative_failures.items():
                text = failure.get("text", "No additional error text.")
                if isinstance(text, str) and len(text) > 300:
                    text = text[-300:] + "..."
                failure_details_summary.append(f"- Status: {status}, Details (last 300 chars of Output): {text}")

            failures_text = "\n".join(failure_details_summary)
            all_statuses = ", ".join(representative_failures.keys())

            error_artifacts = {
                "error_type": f"SolutionFailedTests (statuses: {all_statuses})",
                "error_message": (f"Solution passed {pass_rate:.2%} of test cases. Failure details:\n{failures_text}"),
                "suggestion": (
                    "Review the solution to ensure it correctly handles all test cases, including edge cases."
                ),
            }

            metrics = {
                "pass_rate": pass_rate,
                f"trimmed_mean_{target}": "Infinity",
                "target": target,
                "error": (
                    f"Solution failed {len(failed_test_details)} test case(s) with statuses: {all_statuses}. See artifacts for details."
                ),
            }
            return metrics, error_artifacts

        # 成功情况：计算时间分数与综合分数
        pass_rate = 1.0
        trimmed_mean_runtime = performance_metrics.get(target, float("inf"))

        metrics = {
            "pass_rate": pass_rate,
            f"trimmed_mean_{target}": trimmed_mean_runtime,
            "target": target,
        }
        artifacts = {"details": "All test cases passed."}
        return metrics, artifacts

    def _format_metrics_md(self, metrics: dict[str, Any]) -> str:
        """将性能指标格式化为 Markdown 文本。"""
        lines: list[str] = []
        # pass_rate -> 百分比
        pr = metrics.get("pass_rate")
        if pr is not None:
            try:
                pr_pct = f"{float(pr) * 100:.2f}%"
            except Exception:
                pr_pct = str(pr)
            lines.append(f"- Pass rate: {pr_pct}")

        # trimmed_mean_target
        tmr_key = next((k for k in metrics.keys() if k.startswith("trimmed_mean_")), None)
        tmr = metrics.get(tmr_key) if tmr_key else None
        if tmr is not None and tmr_key:
            tgt = tmr_key.split("_", 2)[-1]
            unit = "s" if tgt == "runtime" else ("MB" if tgt == "memory" else "MB*s")
            if isinstance(tmr, (int, float)):
                if tmr == float("inf"):
                    lines.append(f"- Trimmed mean {tgt}: Infinity")
                else:
                    lines.append(f"- Trimmed mean {tgt}: {float(tmr):.6f} {unit}")
            else:
                val = str(tmr)
                low = val.strip().lower()
                if low in ("inf", "+inf", "infinity", "+infinity"):
                    lines.append(f"- Trimmed mean {tgt}: Infinity")
                else:
                    lines.append(f"- Trimmed mean {tgt}: {val} {unit}")

        # 错误信息（仅在失败时存在）
        err = metrics.get("error")
        if err:
            lines.append(f"- Error: {err}")

        return "\n".join(lines) if lines else "- No metrics available."

    def _format_artifacts_md(self, artifacts: dict[str, Any]) -> str:
        """将构件信息格式化为 Markdown 文本。"""
        if not artifacts:
            return "- No artifacts available."
        lines: list[str] = []
        for k, v in artifacts.items():
            if isinstance(v, str) and "\n" in v:
                indented = "\n  ".join(v.split("\n"))
                lines.append(f"- {k}: {indented}")
            else:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def _build_summary_text(
        self,
        iteration: int,
        code_changed: bool,
        diff_text: str | None,
        benchmark_results: dict[str, Any] | None,
        current_program: str | None = None,
        error_message: str | None = None,
    ) -> str:
        """构建一步迭代的 Markdown 摘要文本，包含程序更新、当前程序、指标与构件。

        - metrics/artifacts 由 `_build_metrics_and_artifacts` 生成并通过 `_format_*_md` 格式化。
        - 无评估或失败时，输出错误信息和占位构件。
        """
        # 构造指标与构件
        if benchmark_results:
            metrics_dict, artifacts_dict = self._build_metrics_and_artifacts(benchmark_results)
        else:
            metrics_dict = {}
            artifacts_dict = {}
            if error_message:
                metrics_dict["error"] = error_message
                if not artifacts_dict:
                    artifacts_dict["details"] = "No evaluation due to error."

        metrics_md = self._format_metrics_md(metrics_dict)
        artifacts_md = self._format_artifacts_md(artifacts_dict)
        diff_size = len(diff_text) if diff_text else 0

        prog_text = current_program or ""

        return (
            "## Program Update\n"
            f"- Iteration: {iteration}\n"
            f"- Code changed: {'yes' if code_changed else 'no'}\n"
            f"- Diff size: {diff_size} chars\n\n"
            "## Current Program\n" + prog_text + "\n\n"
            "## Current Metrics\n" + metrics_md + "\n\n"
            "## Current Artifacts\n" + artifacts_md
        )

    def _extract_full_code_from_response(self, response: str) -> str:
        """从模型响应中提取完整代码（Markdown 代码块）。"""
        if not response:
            return ""
        # 匹配 ```language ... ```
        # 尝试匹配 python, cpp, java, etc. 或者不指定
        pattern = r"```(?:\w+)?\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # 返回最后一个匹配的代码块，通常是最终代码
            return matches[-1].strip()
        return ""

    def _extract_diff_from_response(self, response: str) -> str:
        """从模型响应中提取 diff
        仅支持 SEARCH/REPLACE 区块格式。
        """
        if not response:
            return ""
        if "<<<<<<< SEARCH" in response and ">>>>>>> REPLACE" in response:
            try:
                start_idx = response.find("<<<<<<< SEARCH")
                end_idx = response.rfind(">>>>>>> REPLACE")
                if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                    return response[start_idx : end_idx + len(">>>>>>> REPLACE")].strip()
            except Exception:
                return ""
        return ""

    def _build_messages(
        self, system_prompt: str, history: list[dict[str, Any]], user_prompt: str, limit: int = 200
    ) -> list[dict[str, str]]:
        use_all = bool(getattr(self.config.prompts, "include_all_history", False))
        if use_all:
            msgs: list[dict[str, str]] = []
            tail = history[-limit:] if len(history) > limit else history
            for h in tail:
                role = h.get("role")
                content = h.get("content", "")
                if role in ("system", "user", "assistant") and content:
                    msgs.append({"role": role, "content": content})
            return msgs
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
