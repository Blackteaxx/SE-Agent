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
    """替代策略算子，针对最近失败尝试生成正交的解决方案"""

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

    def _generate_alternative_strategy(self, problem_statement: str, previous_approach: str) -> str:
        """
        针对 PerfAgent 任务，生成一个截然不同的替代优化策略，用于探索多样性。
        (V3 - 专注于多样性，而非失败)

        Args:
            problem_statement: 问题的描述。
            previous_approach_summary: 之前轨迹的总结 (e.g., "Iter 1: Correct O(N) stack, 1.07s...")
                                    这个解不一定是错的。

        Returns:
            一个简短的、截然不同的策略字符串。
        """

        # Persona: 强调创造力和“跳出思维定式”
        # Task: 任务是基于“现有”的解来生成“不同”的策略
        # Context: 明确指出“现有解”可能是好的，我们的目标是多样性
        # Format: 保持简短、纯文本、面向战略
        system_prompt = """You are an expert algorithmic strategist and performance engineer, specializing in creative, out-of-the-box problem-solving and code optimization. Your task is to generate a fundamentally different *optimization strategy* for a code efficiency problem, based on analyzing a previous, existing solution.

    You will be given a problem and a summary of a previous trajectory. This previous solution might be correct, incorrect, fast, or slow. Your task is **not** to fix it, but to provide a **completely different path** to a solution.

    Your goal is to create a completely *orthogonal* strategy to explore a different part of the solution space.
    1.  Focus on a different bottleneck (e.g., I/O and parsing vs. core computation).
    2.  Use a different algorithmic paradigm (e.g., Greedy vs. Dynamic Programming).
    3.  Employ alternative data structures (e.g., `list` stack vs. `deque` vs. `heapq`).

    CRITICAL: Your strategy must be *algorithmically* dissimilar. A simple micro-optimization or bugfix of the previous approach is **not** an acceptable answer.

    SPECIAL FOCUS:
    -   If the previous approach focused on **algorithmic optimization** (e.g., stack logic), propose a strategy that focuses on **I/O optimization** (e.g., custom byte parsing).
    -   If the previous approach focused on **I/O**, propose a different **core algorithm** or **data structure**.
    -   If the previous approach was `O(N log N)`, explore if an `O(N)` solution is possible.
    -   The goal is **strategic diversity**, not incremental improvement.

    IMPORTANT:
    -   Respond with plain text, no formatting.
    -   Keep response under 200 words for system prompt efficiency.
    -   Focus on the strategic shift (the 'what' and 'why'), not specific code.
    -   Provide actionable strategic guidance for an AI agent.
    """

        # User Prompt: 明确要求“非增量”和“定性不同”
        prompt = f"""Generate a radically different solution strategy for exploration:

    PROBLEM:
    {textwrap.indent(problem_statement, "  ")}

    PREVIOUS (EXISTING) STRATEGY SUMMARY:
    {textwrap.indent(previous_approach, "  ")}

    Requirements for new alternative strategy:
    1.  Must **not** be an incremental refinement of the previous strategy (e.g., no simple bugfixes or micro-optimizations).
    2.  Must be a *qualitatively different* approach (e.g., different algorithm, different data structure, or different optimization bottleneck).
    3.  Explore alternative bottlenecks (e.g., algorithm vs. I/O parsing).

    Provide a concise strategic framework that enables an AI agent to approach this problem through an entirely different optimization methodology. Focus on WHY this approach is *different* and what new trade-offs (e.g., simpler code vs. faster runtime, different Big-O) it explores.

    Keep response under 200 words.
    """

        return self._call_llm_api(prompt, system_prompt)

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

        self.logger.info(f"分析 {instance_name}: 基于最近尝试生成替代策略")

        # 生成替代策略
        strategy = self._generate_alternative_strategy(problem_statement, latest_approach)

        if not strategy:
            # 如果LLM调用失败，提供简单的默认替代策略
            strategy = "Try a more direct approach: focus on the specific error message, search for similar issues in the codebase, and make minimal targeted changes rather than broad modifications."

        return strategy


# 注册算子
from operators import register_operator

register_operator("alternative_strategy", AlternativeStrategyOperator)
