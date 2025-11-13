#!/usr/bin/env python3

"""
Crossover Operator

当轨迹池中有效条数大于等于2时，结合两条轨迹的特性生成新的策略。
当有效条数不足时，记录错误并跳过处理。
"""

import textwrap
from typing import Any

from operators import TemplateOperator


class CrossoverOperator(TemplateOperator):
    """交叉算子：综合两条轨迹的优点，生成新的初始代码"""

    def get_name(self) -> str:
        return "crossover"

    def get_strategy_prefix(self) -> str:
        return "CROSSOVER STRATEGY"

    # 轨迹池加载统一由父类实现 _load_traj_pool

    def _get_valid_iterations(self, approaches_data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """获取有效的迭代数据"""
        valid_iterations = []

        for key, data in approaches_data.items():
            if key == "problem":
                continue

            if isinstance(data, dict) and key.isdigit():
                # 检查是否有基本的策略信息
                # if data.get('strategy') or data.get('modified_files') or data.get('key_changes'):
                # 先全部加入，后续再筛选
                valid_iterations.append((key, data))

        # 按迭代号排序
        valid_iterations.sort(key=lambda x: int(x[0]))

        return valid_iterations

    def _format_trajectory_data(self, iteration_key: str, data: dict[str, Any]) -> str:
        """格式化单条轨迹数据为通用的嵌套文本结构。

        - 保留字典的原始键顺序
        - 使用两个空格缩进层级
        - 列表项以 "- " 前缀展示
        - 多行字符串使用块式缩进
        - 空值/空列表/空字典跳过
        """

        def _fmt(value: Any, indent: int) -> str:
            prefix = "  " * indent
            # 字典：按原始顺序输出 key: value（或 key: 换行 + 子块）
            if isinstance(value, dict):
                lines: list[str] = []
                for k, v in value.items():
                    if v is None or v == "" or (isinstance(v, (list, dict)) and len(v) == 0):
                        continue
                    # 基本类型单行输出；多行字符串或嵌套结构块式输出
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
            # 列表：每个元素一行，以 - 前缀；复杂元素分行缩进
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
                                # 去掉子块自身缩进，避免在 "- " 之后产生重复空格
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
            # 字符串：多行使用块式缩进
            if isinstance(value, str):
                if "\n" in value:
                    return textwrap.indent(value, "  " * indent)
                return f"{prefix}{value}"
            # 其他基本类型
            if isinstance(value, bool):
                return f"{prefix}{'true' if value else 'false'}"
            if isinstance(value, (int, float)):
                return f"{prefix}{value}"
            return f"{prefix}{str(value)}"

        parts: list[str] = [f"ITERATION {iteration_key}:"]
        body = _fmt(data, 0)
        if body:
            parts.append(body)
        return "\n".join(parts)

    def _generate_crossover_code(self, problem_statement: str, trajectory1: str, trajectory2: str) -> str:
        """
        综合两条轨迹的优势，生成新的 Python 初始代码。

        输出必须置于 ```py 代码块中；若不满足，将进行重试并仅提取代码块内容。
        """

        # 使用统一重试助手，强制输出置于```py```代码块并提取
        def build_prompt(attempt: int) -> tuple[str, str]:
            # Persona: 强调“综合”和“高性能实现”
            # Task: 明确要求生成“完整的、合成的”Python程序
            # Context: 提供了两条轨迹的总结，作为“合成素材”
            # Format: 严格要求代码块格式，且代码必须自包含（无 T1/T2 代号）
            system_prompt = """You are an expert algorithmic strategist and competitive programmer. Your task is to synthesize a SUPERIOR, COMPLETE Python 3 program by intelligently combining the best elements of two prior optimization trajectories.

CRITICAL OUTPUT RULES:
- The output MUST be a single, runnable Python 3 program.
- Wrap the code in a fenced block: ```python ... ```.
- The code must read from STDIN and print to STDOUT.

CROSSOVER SYNTHESIS REQUIREMENTS:
- **Synthesize, Don't Just Concatenate:** You must intellectually combine the strengths.
    - E.g., If T1 had a better CORE ALGORITHM but slow I/O, and T2 had fast I/O but a worse algorithm, your code MUST implement T1's algorithm using T2's I/O technique.
    - E.g., If T1 used a logically correct STACK but slow `list`, and T2 used a fast pre-allocated ARRAY but broke correctness, your code MUST implement T1's correct logic using T2's fast array structure.
- **Self-Contained & Explicit:** The final code must NOT contain comments like `# implementation from T1`. It must just BE the implementation.

Your goal is to produce a hybrid solution that is demonstrably better (faster, more correct, or more robust) than either individual parent trajectory.
"""

            prompt = f"""Generate a superior hybrid Python 3 solution by synthesizing these two trajectories:

PROBLEM STATEMENT:
{textwrap.indent(problem_statement, "  ")}

TRAJECTORY 1 SUMMARY:
{textwrap.indent(trajectory1, "  ")}

TRAJECTORY 2 SUMMARY:
{textwrap.indent(trajectory2, "  ")}

REQUIREMENTS FOR THE NEW HYBRID CODE:
1.  **Explicit Synthesis:** actively combine complementary strengths (e.g., Algorithm from T1 + I/O from T2).
2.  **Avoid Shared Weaknesses:** If both failed at X, your code must explicitly use a different technique for X.
3.  **No Placeholders:** Do not use "T1", "T2" in comments or variable names. The code must stand alone as a definitive implementation.
4.  **Correctness & Performance:** Aim for the highest possible performance without sacrificing correctness.

Output the complete synthesized Python 3 code block and necessary analysis."""

            return prompt, system_prompt

        code_only = self._require_py_block_with_retry(build_prompt, max_retries=2)
        # 返回纯代码文本
        return code_only or ""

    def _generate_content(
        self, instance_info: dict[str, Any], problem_statement: str, trajectory_data: dict[str, Any]
    ) -> str:
        """生成交叉策略内容"""
        instance_name = instance_info["instance_name"]

        # 加载轨迹池数据（从workspace_dir，通过instance_dir计算）
        workspace_dir = instance_info["instance_dir"].parent.parent
        approaches_data = self._load_traj_pool(workspace_dir, instance_name)
        if not approaches_data:
            self.logger.warning(f"跳过 {instance_name}: 无轨迹池数据")
            return ""

        # 获取有效的迭代数据
        valid_iterations = self._get_valid_iterations(approaches_data)

        if len(valid_iterations) < 2:
            self.logger.error(f"跳过 {instance_name}: 轨迹池有效条数不足 (需要>=2, 实际={len(valid_iterations)})")
            return ""

        # 选择最近的两条轨迹进行交叉
        # 可以选择最后两条，或者选择效果最好的两条，这里选择最后两条
        iteration1_key, iteration1_data = valid_iterations[-2]
        iteration2_key, iteration2_data = valid_iterations[-1]

        # 格式化轨迹数据
        trajectory1_formatted = self._format_trajectory_data(iteration1_key, iteration1_data)
        trajectory2_formatted = self._format_trajectory_data(iteration2_key, iteration2_data)

        self.logger.info(f"分析 {instance_name}: 交叉迭代 {iteration1_key} 和 {iteration2_key}")

        # 生成交叉初始代码
        code = self._generate_crossover_code(
            problem_statement,
            trajectory1_formatted,
            trajectory2_formatted,
        )

        return code


# 注册算子
from operators import register_operator

register_operator("crossover", CrossoverOperator)
