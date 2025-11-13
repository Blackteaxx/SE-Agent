#!/usr/bin/env python3

"""
Trajectory Pool Summary Operator

分析轨迹池中的全部历史尝试，综合其优点与失败模式，
生成新的 Python 初始代码（仅代码），用于下一次迭代评估。
"""

import json
import textwrap
from pathlib import Path
from typing import Any

from operators import TemplateOperator


class TrajPoolSummaryOperator(TemplateOperator):
    """轨迹池总结算子：综合历史轨迹，生成新的 Python 初始代码"""

    def get_name(self) -> str:
        return "traj_pool_summary"

    def get_strategy_prefix(self) -> str:
        return "RISK-AWARE PROBLEM SOLVING GUIDANCE"

    def _discover_instances(self, workspace_dir: Path, current_iteration: int) -> list[dict[str, Any]]:
        """
        重写实例发现逻辑，直接查找工作目录中的traj.pool文件

        Args:
            workspace_dir: 工作目录路径
            current_iteration: 当前迭代号

        Returns:
            实例信息列表
        """
        instances = []

        # 通过父类方法加载 traj.pool 数据映射
        pool_data = self._load_traj_pool(workspace_dir)
        if not pool_data:
            return instances

        # 为每个实例创建实例信息
        for instance_name, instance_data in pool_data.items():
            if isinstance(instance_data, dict) and len(instance_data) > 0:
                # 检查是否有数字键（迭代数据）
                has_iteration_data = any(key.isdigit() for key in instance_data.keys())
                if has_iteration_data:
                    instances.append(
                        {
                            "instance_name": instance_name,
                            "instance_dir": workspace_dir,  # 使用工作目录作为实例目录
                            "trajectory_file": workspace_dir / "traj.pool",  # 指向 traj.pool 文件
                            "previous_iteration": current_iteration - 1,
                            "pool_data": instance_data,  # 附加池数据用于后续处理
                            "problem_description": instance_data.get("problem", {}),
                        }
                    )

        self.logger.info(f"发现 {len(instances)} 个可处理的实例")
        return instances

    def _extract_problem_statement(self, trajectory_data: dict[str, Any]) -> str:
        """
        重写问题陈述提取，返回占位符
        因为我们在_generate_content中直接使用pool_data中的问题陈述
        """
        return "placeholder"

    # 移除本地加载方法，统一由父类提供 _load_traj_pool

    def _format_approaches_data(self, approaches_data: dict[str, Any]) -> str:
        """格式化历史尝试数据为通用的嵌套文本结构。

        - 保留字典键原始顺序
        - 两空格缩进层级
        - 列表项以 "- " 前缀展示
        - 多行字符串块式缩进
        - 跳过空值/空列表/空字典
        """

        def _fmt(value: Any, indent: int) -> str:
            prefix = "  " * indent
            if isinstance(value, dict):
                lines: list[str] = []
                for k, v in value.items():
                    if v is None or v == "" or (isinstance(v, (list, dict)) and len(v) == 0):
                        continue
                    if isinstance(v, (int, float)):
                        lines.append(f"{prefix}{k}: {v}")
                    elif isinstance(v, bool):
                        lines.append(f"{prefix}{k}: {'true' if v else 'false'}")
                    elif isinstance(v, str) and "\n" not in v:
                        lines.append(f"{prefix}{k}: {v}")
                    else:
                        lines.append(f"{prefix}{k}:")
                        child = _fmt(v, indent + 1)
                        if child:
                            lines.append(child)
                return "\n".join(lines)
            if isinstance(value, list):
                lines: list[str] = []
                for item in value:
                    if item is None or item == "":
                        continue
                    if isinstance(item, (int, float)):
                        lines.append(f"{prefix}- {item}")
                    elif isinstance(item, bool):
                        lines.append(f"{prefix}- {'true' if item else 'false'}")
                    elif isinstance(item, str) and "\n" not in item:
                        lines.append(f"{prefix}- {item}")
                    else:
                        child = _fmt(item, indent + 1)
                        if child:
                            child_lines = child.splitlines()
                            if child_lines:
                                child_prefix = "  " * (indent + 1)
                                first = child_lines[0]
                                if first.startswith(child_prefix):
                                    first = first[len(child_prefix) :]
                                lines.append(f"{prefix}- {first}")
                                for cl in child_lines[1:]:
                                    if cl.startswith(child_prefix):
                                        cl = cl[len(child_prefix) :]
                                    lines.append(f"{prefix}  {cl}")
                return "\n".join(lines)
            if isinstance(value, str):
                if "\n" in value:
                    return textwrap.indent(value, "  " * indent)
                return f"{prefix}{value}"
            if isinstance(value, bool):
                return f"{prefix}{'true' if value else 'false'}"
            if isinstance(value, (int, float)):
                return f"{prefix}{value}"
            return f"{prefix}{str(value)}"

        parts: list[str] = []
        for key, data in approaches_data.items():
            if key == "problem":
                continue
            if isinstance(data, dict):
                parts.append(f"ATTEMPT {key}:")
                body = _fmt(data, 0)
                if body:
                    parts.append(body)
        return "\n".join(parts)

    def _generate_synthesized_code(self, problem_statement: str, approaches_data: dict[str, Any]) -> str:
        """
        综合历史轨迹，生成新的 Python 初始代码。强制```py```代码块，并在未满足时重试与提取。
        """
        formatted_attempts = self._format_approaches_data(approaches_data)

        def build_prompt(attempt: int) -> tuple[str, str]:
            # Persona: 强调“综合”和“高性能实现”
            # Task: 明确要求生成“完整的、合成的”Python程序，基于所有历史经验
            # Context: 提供了所有轨迹的详细技术组件摘要
            # Format: 严格要求代码块格式，且代码必须自包含
            system_prompt = """You are an expert algorithmic strategist and performance engineer. Your task is to synthesize a SUPERIOR, COMPLETE Python 3 program by integrating the BEST PROVEN components across all prior trajectories while strictly avoiding their failure modes.

CRITICAL OUTPUT RULES:
- The output MUST be a single, runnable Python 3 program.
- Wrap the code in a fenced block: ```python ... ```.
- The code must read from STDIN and print to STDOUT.

SYNTHESIS REQUIREMENTS:
- **Component-Level Integration:** actively select the best ALGORITHM, the best DATA STRUCTURE, and the best I/O TECHNIQUE from the entire history.
    - E.g., "Use the O(N) stack algorithm from Attempt A, but implement it with the pre-allocated array from Attempt B, and use the fast byte-I/O from Attempt C."
- **Failure Avoidance:** If any attempt failed due to a specific technique (e.g., "recursion caused stack overflow"), your synthesized code MUST NOT use that technique.
- **Self-Contained & Explicit:** The final code must be a definitive, standalone implementation. Do NOT use comments like `# originally from Trajectory 1`.

Your goal is to produce the definitive, optimal solution that surpasses all individual previous attempts.
"""

            prompt = (
                f"Generate a superior synthesized Python 3 solution based on this history:\n\n"
                f"PROBLEM STATEMENT:\n{textwrap.indent(str(problem_statement), '  ')}\n\n"
                f"ALL PREVIOUS TRAJECTORIES (COMPONENT ANALYSIS):\n{textwrap.indent(formatted_attempts, '  ')}\n\n"
                "Create a synthesized program that integrates the strongest components and avoids prior failure modes.\n\n"
                "REQUIREMENTS FOR THE SYNTHESIZED CODE:\n"
                "1.  **Explicit Integration:** actively combine complementary strengths (e.g., best algorithm + best I/O).\n"
                "2.  **Avoid Specific Failures:** Do not repeat any technique explicitly flagged as causing a failure (e.g., correctness regression, TLE).\n"
                "3.  **No Placeholders:** The code must stand alone as a definitive implementation.\n"
                "4.  **Performance Focus:** Prefer faster techniques (e.g., `sys.stdin.buffer.read`) unless they are known to break correctness for this specific problem.\n\n"
                "Output the complete synthesized Python 3 code block and necessary analysis."
            )
            return prompt, system_prompt

        code_only = self._require_py_block_with_retry(build_prompt, max_retries=2)
        # 返回纯代码文本
        return code_only or ""

    def _generate_content(
        self, instance_info: dict[str, Any], problem_statement: str, trajectory_data: dict[str, Any]
    ) -> str:
        """生成轨迹池总结内容"""
        instance_name = instance_info["instance_name"]

        # 直接使用附加的池数据
        approaches_data = instance_info.get("pool_data", {})
        if not approaches_data:
            self.logger.warning(f"跳过 {instance_name}: 无轨迹池数据")
            return ""

        # 使用实例信息中的问题陈述，若为字典则转为简字符串
        pool_problem_raw = instance_info.get("problem_description", "N/A")
        try:
            pool_problem = (
                json.dumps(pool_problem_raw, ensure_ascii=False, indent=2)
                if isinstance(pool_problem_raw, (dict, list))
                else str(pool_problem_raw)
            )
        except Exception:
            pool_problem = str(pool_problem_raw)

        # 获取所有迭代数据（数字键）
        iteration_data = {k: v for k, v in approaches_data.items() if k.isdigit() and isinstance(v, dict)}

        if not iteration_data:
            self.logger.warning(f"跳过 {instance_name}: 无有效迭代数据")
            return ""

        self.logger.info(f"分析 {instance_name}: {len(iteration_data)} 个历史尝试")

        # 生成综合初始代码
        code = self._generate_synthesized_code(pool_problem, iteration_data)
        return code


# 注册算子
from operators import register_operator

register_operator("traj_pool_summary", TrajPoolSummaryOperator)
