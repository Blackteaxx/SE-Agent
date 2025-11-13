#!/usr/bin/env python3

"""
Trajectory Analyzer Operator

直接分析 .tra 轨迹文件，提取详细的问题陈述和轨迹数据，
生成基于完整轨迹内容的解决策略。
"""

from typing import Any

from operators import TemplateOperator


class TrajectoryAnalyzerOperator(TemplateOperator):
    """轨迹分析算子：分析 .tra 轨迹，生成新的 Python 初始代码"""

    def get_name(self) -> str:
        return "trajectory_analyzer"

    def get_strategy_prefix(self) -> str:
        return "SOLUTION STRATEGY"

    def _extract_trajectory_analysis(self, trajectory_data: dict[str, Any]) -> str:
        """提取轨迹分析信息"""
        try:
            trajectory = trajectory_data.get("Trajectory", [])

            # 统计轨迹信息
            total_steps = len(trajectory)
            assistant_steps = len([item for item in trajectory if item.get("role") == "assistant"])
            user_steps = len([item for item in trajectory if item.get("role") == "user"])

            # 提取最后几个助手响应
            assistant_responses = []
            for item in reversed(trajectory):
                if item.get("role") == "assistant" and len(assistant_responses) < 3:
                    content = item.get("content", "")
                    if isinstance(content, list) and len(content) > 0:
                        text = content[0].get("text", "")
                    elif isinstance(content, str):
                        text = content
                    else:
                        continue

                    # 截取前200字符
                    assistant_responses.append(text[:200] + "..." if len(text) > 200 else text)

            # 检查是否有工具使用
            has_tools = any("function_call" in str(item) or "tool_calls" in str(item) for item in trajectory)

            analysis = f"""轨迹统计:
- 总步数: {total_steps}
- 助手响应: {assistant_steps}
- 用户输入: {user_steps}
- 工具使用: {"是" if has_tools else "否"}

最近的助手响应:
{chr(10).join(f"{i + 1}. {resp}" for i, resp in enumerate(assistant_responses))}"""

            return analysis

        except Exception as e:
            self.logger.error(f"提取轨迹分析失败: {e}")
            return ""

    def _generate_solution_code(self, problem_statement: str, trajectory_analysis: str, instance_name: str) -> str:
        """生成新的 Python 初始代码，强制```py```代码块并重试未满足情况"""

        def build_prompt(attempt: int):
            system_prompt = """You are an expert algorithmic strategist and performance engineer. Your task is to produce a COMPLETE Python program informed by trajectory analysis.

CRITICAL OUTPUT RULES:
- Wrap the entire program in a fenced block starting with ```py and ending with ```.
- Output ONLY Python code inside the fence (no text, no comments, no Markdown outside the block).
- The program must be self-contained and runnable: read from STDIN and write to STDOUT.

ALIGNMENT WITH TRAJECTORY ANALYZER REQUIREMENTS (from multi-agent setting):
- Generate a solution that is architecturally dissimilar to previous approaches implied by the analysis.
- Leverage novel paradigms or angles (e.g., runtime parsing focus vs static algorithmic focus).
- Incorporate alternative tools and techniques when appropriate (e.g., buffered I/O, different data structures).
- Establish a distinct logical progression and avoid prior failure modes.

IMPLEMENTATION GUIDELINES:
- Prefer fast I/O when appropriate (e.g., sys.stdin.buffer.read()).
- Keep code clear, robust, and suitable for typical judge environments.
"""

            prompt = (
                f"PROBLEM:\n{problem_statement}\n\n"
                f"TRAJECTORY ANALYSIS:\n{trajectory_analysis}\n\n"
                "Requirements for the divergent approach code:\n"
                "1. Adopt a profoundly different paradigm or entry point (e.g., empirical runtime-driven parsing).\n"
                "2. Use different algorithms/data structures than implied by prior attempts.\n"
                "3. Integrate unconventional techniques when helpful (profiling-ready I/O, deque/heap, etc.).\n"
                "4. Explicitly avoid known failure modes from the analysis.\n\n"
                "Task:\nProduce a self-contained Python program that reads from stdin and prints to stdout, implementing this divergent approach."
            )
            return prompt, system_prompt

        code_only = self._require_py_block_with_retry(build_prompt, max_retries=2)
        # 返回纯代码文本
        return code_only or ""

    def _generate_content(
        self, instance_info: dict[str, Any], problem_statement: str, trajectory_data: dict[str, Any]
    ) -> str:
        """生成轨迹分析策略内容"""
        instance_name = instance_info["instance_name"]

        # 提取详细的问题陈述
        detailed_problem = problem_statement

        # 提取轨迹分析
        trajectory_analysis = self._extract_trajectory_analysis(trajectory_data)

        self.logger.info(f"分析 {instance_name}: 基于完整轨迹数据生成策略")

        # 生成初始代码
        code = self._generate_solution_code(detailed_problem, trajectory_analysis, instance_name)

        return code


# 注册算子
from operators import register_operator

register_operator("trajectory_analyzer", TrajectoryAnalyzerOperator)
