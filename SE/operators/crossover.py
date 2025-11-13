#!/usr/bin/env python3

"""
Crossover Operator

当轨迹池中有效条数大于等于2时，结合两条轨迹的特性生成新的策略。
当有效条数不足时，记录错误并跳过处理。
"""

import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from operators import TemplateOperator


class CrossoverOperator(TemplateOperator):
    """交叉算子，结合两条轨迹的特性生成新策略"""
    
    def get_name(self) -> str:
        return "crossover"
    
    def get_strategy_prefix(self) -> str:
        return "CROSSOVER STRATEGY"
    
    # 轨迹池加载统一由父类实现 _load_traj_pool
    
    def _get_valid_iterations(self, approaches_data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
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
    
    def _format_trajectory_data(self, iteration_key: str, data: Dict[str, Any]) -> str:
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
                lines: List[str] = []
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
                lines: List[str] = []
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
                                    first = first[len(child_prefix):]
                                lines.append(f"{prefix}- {first}")
                                for cl in child_lines[1:]:
                                    if cl.startswith(child_prefix):
                                        cl = cl[len(child_prefix):]
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

        parts: List[str] = [f"ITERATION {iteration_key}:"]
        body = _fmt(data, 0)
        if body:
            parts.append(body)
        return "\n".join(parts)
    
    def _generate_crossover_strategy(self, problem_statement: str, trajectory1_summary: str, trajectory2_summary: str) -> str:
        """
        针对 PerfAgent 任务，生成一个“交叉”的混合优化策略。
        (V4 - 自包含描述，消除 T1/T2 代号)
        
        Args:
            problem_statement: 问题的描述。
            trajectory1_summary: 第一个轨迹的总结。
            trajectory2_summary: 第二个轨迹的总结。
        
        Returns:
            一个简短的、综合的混合策略字符串，不包含对原轨迹的直接引用。
        """

        # Persona: 强调“综合”和“清晰传达”
        # Task: 任务是“合成”并“重述”为一个全新的、独立的策略
        # Context: 明确指出下游 Agent 看不到原始轨迹，必须自包含
        # Format: 保持简短、纯文本、面向战略，严禁使用代号
        system_prompt = """You are an expert algorithmic strategist and performance engineer. Your task is to analyze two different optimization trajectories and synthesize a new *hybrid strategy* that combines their strengths.

    CRITICAL REQUIREMENT: The agent reading your output will NOT have access to the original trajectories (T1, T2). Therefore, your synthesized strategy must be **completely self-contained**.

    DO NOT use references like "T1", "Trajectory 1", "the first approach", etc.
    INSTEAD, explicitly describe the technique you are adopting.

    BAD OUTPUT: "Combine T1's stack logic with T2's I/O."
    GOOD OUTPUT: "Combine the O(N) monotonic stack algorithm with standard library `sys.stdin.buffer.read()` for fast I/O."

    Your goal is to create a unified, superior strategic directive that stands on its own.
    """

        # User Prompt: 再次强调“自包含”和“去代号化”
        prompt = f"""Analyze these two trajectories and create a superior, self-contained hybrid strategy:

    PROBLEM:
    {textwrap.indent(problem_statement, '  ')}

    TRAJECTORY 1 SUMMARY:
    {textwrap.indent(trajectory1_summary, '  ')}

    TRAJECTORY 2 SUMMARY:
    {textwrap.indent(trajectory2_summary, '  ')}

    Create a crossover strategy that synthesizes the best components of both.

    Requirements for the hybrid strategy text:
    1.  **NO PLACEHOLDERS:** Do not use "T1", "T2", "Approach 1", etc. Explicitly describe every technique you mention.
    2.  **Synthesize complementary strengths:** e.g., "Use the [specific algorithm from T1] combined with [specific data structure from T2]."
    3.  **Address shared weaknesses:** If both failed at X, explicitly state "Avoid [technique X] as it led to [failure mode] in previous attempts."
    4.  **Be directive:** Write it as a clear instruction for the next agent.

    Keep response under 250 words. Ensure it is perfectly readable without knowing the previous trajectories.
    """

        return self._call_llm_api(prompt, system_prompt)
    
    def _generate_content(self, instance_info: Dict[str, Any], problem_statement: str, trajectory_data: Dict[str, Any]) -> str:
        """生成交叉策略内容"""
        instance_name = instance_info['instance_name']
        
        # 加载轨迹池数据（从workspace_dir，通过instance_dir计算）
        workspace_dir = instance_info['instance_dir'].parent.parent
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
        
        # 生成交叉策略
        strategy = self._generate_crossover_strategy(
            problem_statement, 
            trajectory1_formatted, 
            trajectory2_formatted
        )
        
        if not strategy:
            # 如果LLM调用失败，提供默认交叉策略
            strategy = f"""Synthesize the most effective elements from both previous approaches. Start with the stronger analytical method from the first approach, then apply the more focused implementation technique from the second approach. Address the common limitations observed in both attempts by adding intermediate validation steps. This hybrid approach combines thorough analysis with targeted action, while incorporating safeguards against the pitfalls encountered in both previous attempts."""
        
        return strategy


# 注册算子
from operators import register_operator
register_operator("crossover", CrossoverOperator)