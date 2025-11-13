#!/usr/bin/env python3

"""
Alternative Strategy Operator

基于最近一次失败尝试生成截然不同的替代解决策略，
避免重复相同的错误方法。
"""

import textwrap
from typing import Any

from operators import TemplateOperator


class AlternativeStrategyOperator(TemplateOperator):
    """替代策略算子：基于最近轨迹生成“截然不同”的初始代码"""

    def get_name(self) -> str:
        return "alternative_strategy"

    def get_strategy_prefix(self) -> str:
        return "ALTERNATIVE SOLUTION STRATEGY"

    # 统一由父类提供 _load_traj_pool，实现复用

    def _get_latest_approach(self, approaches_data: dict[str, Any]) -> str:
        """将最近一次尝试的嵌套 dict/list 数据通用格式化为可读文本。

        不依赖固定字段，保持原始键顺序，递归缩进显示；
        对多行字符串做块状缩进；列表使用 "- " 项标识。
        """
        if not approaches_data:
            return ""

        # 找到最大的迭代号
        iteration_nums = []
        for key in approaches_data.keys():
            if key != "problem" and key.isdigit():
                iteration_nums.append(int(key))

        if not iteration_nums:
            return ""

        latest_iteration = max(iteration_nums)
        latest_data = approaches_data.get(str(latest_iteration), {})

        def indent_str(level: int) -> str:
            return "  " * level

        def fmt_value(val: Any, level: int) -> str:
            # 基本类型直接转文字
            if val is None:
                return "null"
            if isinstance(val, (int, float)):
                return str(val)
            if isinstance(val, bool):
                return "true" if val else "false"
            if isinstance(val, str):
                if "\n" in val:
                    # 多行字符串块
                    lines = val.splitlines()
                    pad = indent_str(level + 1)
                    return "|\n" + "\n".join(f"{pad}{line}" for line in lines)
                return val
            if isinstance(val, dict):
                # 保持原始顺序
                lines = []
                for k, v in val.items():
                    key_line = f"{indent_str(level)}{k}:"
                    if isinstance(v, (dict, list)) or (isinstance(v, str) and "\n" in v):
                        lines.append(key_line)
                        lines.append(fmt_value(v, level + 1))
                    else:
                        lines.append(f"{key_line} {fmt_value(v, 0)}")
                return "\n".join(lines)
            if isinstance(val, list):
                lines = []
                for item in val:
                    if isinstance(item, (dict, list)) or (isinstance(item, str) and "\n" in item):
                        lines.append(f"{indent_str(level)}-")
                        lines.append(fmt_value(item, level + 1))
                    else:
                        lines.append(f"{indent_str(level)}- {fmt_value(item, 0)}")
                return "\n".join(lines)
            # 其它类型兜底为字符串
            return str(val)

        header = f"Latest Iteration: {latest_iteration}"
        body = fmt_value(latest_data, 0)
        return f"{header}\n{body}"

    #     def _generate_alternative_strategy(self, problem_statement: str, previous_approach: str) -> str:
    #         """生成截然不同的替代策略"""

    #         system_prompt = """You are an expert software engineering strategist specializing in breakthrough problem-solving. Your task is to generate a fundamentally different approach to a software engineering problem, based on analyzing a previous failed attempt.

    # You will be given a problem and a previous approach that FAILED (possibly due to cost limits, early termination, or strategic inadequacy). Create a completely orthogonal strategy that:
    # 1. Uses different investigation paradigms (e.g., runtime analysis vs static analysis)
    # 2. Approaches from unconventional angles (e.g., user impact vs code structure)
    # 3. Employs alternative tools and techniques
    # 4. Follows different logical progression

    # CRITICAL: Your strategy must be architecturally dissimilar to avoid the same limitations and blind spots.

    # SPECIAL FOCUS: If the previous approach failed due to early termination or cost limits, prioritize:
    # - More focused, direct approaches
    # - Faster problem identification techniques
    # - Incremental validation methods
    # - Minimal viable change strategies

    # IMPORTANT:
    # - Respond with plain text, no formatting
    # - Keep response under 200 words for system prompt efficiency
    # - Focus on cognitive framework rather than code specifics
    # - Provide actionable strategic guidance"""

    #         prompt = f"""Generate a radically different solution strategy:

    # PROBLEM:
    # {problem_statement}...

    # PREVIOUS FAILED APPROACH:
    # {previous_approach}...

    # Requirements for alternative strategy:
    # 1. Adopt different investigation paradigm (e.g., empirical vs theoretical)
    # 2. Start from alternative entry point (e.g., dependencies vs core logic)
    # 3. Use non-linear logical sequence (e.g., symptom-to-cause vs cause-to-symptom)
    # 4. Integrate unconventional techniques (e.g., profiling, fuzzing, visualization)
    # 5. Prioritize overlooked aspects (e.g., performance, edge cases, integration)

    # Provide a concise strategic framework that enables an AI agent to approach this problem through an entirely different methodology. Focus on WHY this approach differs and HOW it circumvents previous limitations.

    # Keep response under 200 words."""

    def _generate_alternative_code(self, problem_statement: str, previous_approach: str) -> str:
        """
        基于问题与最近轨迹，生成“截然不同”的 Python 初始代码。

        输出必须置于 ```py 代码块中；若不满足，将进行重试并仅提取代码块内容。
        """

        def build_prompt(attempt: int):
            system_prompt = """You are an expert algorithmic strategist and performance engineer. Your job is to produce a COMPLETE Python program for competitive programming tasks.

CRITICAL OUTPUT RULES:
- Wrap the entire program in a fenced block starting with ```py and ending with ```.
- Output ONLY Python code inside the fence (no text, no comments, no Markdown outside the block).
- The program must be self-contained and runnable: read from STDIN and write to STDOUT.

ALIGNMENT WITH ALTERNATIVE STRATEGY REQUIREMENTS (from multi-agent setting):
- Deliver a fundamentally different approach than the previous trajectory.
- Prioritize a different bottleneck or angle (e.g., shift between algorithm vs I/O parsing).
- Use a different algorithmic paradigm or data structure (e.g., Greedy vs DP, deque/heap vs list).
- Do NOT provide incremental refinements, micro-optimizations, or simple bugfixes of the previous approach.
- Special focus: if the previous attempt focused on core algorithmics, consider an I/O-centric technique; if it focused on I/O, propose a different core algorithm/data structure; explore better Big-O when feasible (e.g., aim for O(N) over O(N log N)).

IMPLEMENTATION GUIDELINES:
- Prefer fast I/O when appropriate (e.g., sys.stdin.buffer.read()).
- Keep code clear, robust, and suitable for typical judge environments.
"""

            prompt = (
                f"PROBLEM:\n{textwrap.indent(problem_statement, '  ')}\n\n"
                f"PREVIOUS (EXISTING) TRAJECTORY SUMMARY:\n{textwrap.indent(previous_approach, '  ')}\n\n"
                "Requirements for the new alternative approach code:\n"
                "1. Must NOT be an incremental refinement of the previous approach.\n"
                "2. Must be qualitatively different: switch paradigm, data structures, or bottleneck focus.\n"
                "3. Consider alternative bottlenecks (algorithm vs I/O parsing).\n"
                "4. Aim for different Big-O when feasible.\n\n"
                "Task:\nProduce a self-contained Python program that reads from stdin and prints to stdout, implementing the different approach."
            )
            return prompt, system_prompt

        code_only = self._require_py_block_with_retry(build_prompt, max_retries=2)
        # 返回纯代码文本，不再包裹 ```py fence
        return code_only or ""

    def _generate_content(
        self, instance_info: dict[str, Any], problem_statement: str, trajectory_data: dict[str, Any]
    ) -> str:
        """生成替代策略内容"""
        instance_dir = instance_info["instance_dir"]
        instance_name = instance_info["instance_name"]

        # 加载轨迹池数据（从workspace_dir，通过instance_dir计算）
        workspace_dir = instance_dir.parent.parent
        approaches_data = self._load_traj_pool(workspace_dir, instance_name)
        if not approaches_data:
            self.logger.warning(f"跳过 {instance_name}: 无轨迹池数据")
            return ""

        # 获取最近一次尝试
        latest_approach = self._get_latest_approach(approaches_data)
        if not latest_approach:
            self.logger.warning(f"跳过 {instance_name}: 无最近尝试数据")
            return ""

        self.logger.info(f"分析 {instance_name}: 基于最近尝试生成替代初始代码")

        # 使用统一重试助手，强制输出置于```py```代码块并提取
        def build_prompt(attempt: int) -> tuple[str, str]:
            # Persona: 强调“算法专家”和“高性能实现者”
            # Task: 明确要求生成“完整的、不同的”Python程序
            # Context: 提供了先前策略的总结，作为“需要避开/改变”的基线
            # Format: 严格要求代码块格式
            system_prompt = """You are an expert algorithmic strategist and competitive programmer. Your task is to generate a COMPLETE, SELF-CONTAINED Python 3 program that solves a given problem using a FUNDAMENTALLY DIFFERENT approach than a previously tried strategy.

CRITICAL OUTPUT RULES:
- The output MUST be a single, runnable Python 3 program.
- Wrap the code in a fenced block: ```python ... ```.
- The code must read from STDIN and print to STDOUT.

STRATEGIC DIVERGENCE REQUIREMENTS:
- You MUST adopt a different optimization paradigm.
    - If previous was ALGO-focused (e.g., stack, DP), shift focus to I/O or DATA STRUCTURES.
    - If previous was I/O-focused, shift back to a better CORE ALGORITHM.
- Explore different trade-offs (e.g., use more memory for less time, or simpler code with slightly worse theoretical Big-O if practical performance might be better).

Your goal is to explore the solution space, NOT just to micro-optimize the previous attempt.
"""

            prompt = f"""Generate a radically different Python 3 solution for this problem:

PROBLEM STATEMENT:
{textwrap.indent(problem_statement, "  ")}

PREVIOUS (EXISTING) TRAJECTORY SUMMARY:
{textwrap.indent(latest_approach, "  ")}

REQUIREMENTS FOR THE NEW ALTERNATIVE CODE:
1.  **Diverge Strategically:** Do NOT just fix bugs or tweak the previous code. Use a different algorithm (e.g., Greedy vs. DP), data structure (e.g., array vs. hash map), or I/O technique.
2.  **Self-Contained:** The code must import all necessary standard libraries and run standalone.
3.  **Correctness Goal:** Aim for a correct solution, even if it explores a different performance profile.

Output the complete Python 3 code block necessary analysis."""

            return prompt, system_prompt

        code_only = self._require_py_block_with_retry(build_prompt, max_retries=2)
        # 返回纯代码文本，不再包裹 ```py fence
        return code_only or ""


# 注册算子
from operators import register_operator

register_operator("alternative_strategy", AlternativeStrategyOperator)
