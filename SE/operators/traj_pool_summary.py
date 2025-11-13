#!/usr/bin/env python3

"""
Trajectory Pool Summary Operator

分析轨迹池中的历史失败尝试，识别常见盲区和风险点，
生成简洁的风险感知解决指导。
"""

from pathlib import Path
from typing import Any, Dict, List
import textwrap

from operators import TemplateOperator


class TrajPoolSummaryOperator(TemplateOperator):
    """轨迹池总结算子，生成风险感知的问题解决指导"""
    
    def get_name(self) -> str:
        return "traj_pool_summary"
    
    def get_strategy_prefix(self) -> str:
        return "RISK-AWARE PROBLEM SOLVING GUIDANCE"
    
    def _discover_instances(self, workspace_dir: Path, current_iteration: int) -> List[Dict[str, Any]]:
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
                    instances.append({
                        'instance_name': instance_name,
                        'instance_dir': workspace_dir,  # 使用工作目录作为实例目录
                        'trajectory_file': workspace_dir / 'traj.pool',  # 指向 traj.pool 文件
                        'previous_iteration': current_iteration - 1,
                        'pool_data': instance_data,  # 附加池数据用于后续处理
                        'problem_description': instance_data.get('problem', {})
                    })
        
        self.logger.info(f"发现 {len(instances)} 个可处理的实例")
        return instances
    
    def _extract_problem_statement(self, trajectory_data: Dict[str, Any]) -> str:
        """
        重写问题陈述提取，返回占位符
        因为我们在_generate_content中直接使用pool_data中的问题陈述
        """
        return "placeholder"
    
    # 移除本地加载方法，统一由父类提供 _load_traj_pool
    
    def _format_approaches_data(self, approaches_data: Dict[str, Any]) -> str:
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
                lines: List[str] = []
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
            if isinstance(value, str):
                if "\n" in value:
                    return textwrap.indent(value, "  " * indent)
                return f"{prefix}{value}"
            if isinstance(value, bool):
                return f"{prefix}{'true' if value else 'false'}"
            if isinstance(value, (int, float)):
                return f"{prefix}{value}"
            return f"{prefix}{str(value)}"

        parts: List[str] = []
        for key, data in approaches_data.items():
            if key == "problem":
                continue
            if isinstance(data, dict):
                parts.append(f"ATTEMPT {key}:")
                body = _fmt(data, 0)
                if body:
                    parts.append(body)
        return "\n".join(parts)
    
    def _generate_risk_aware_guidance(self, problem_statement: str, approaches_data: Dict[str, Any]) -> str:
        """
        (V5 - 自包含描述，消除代号引用)
        分析所有轨迹，生成用于下一次迭代的、综合了所有经验的、自包含的准确指导。
        """
        
        # Persona: 强调“综合分析”和“清晰指令”
        # Task: 任务是基于历史数据，发出一个独立的、无需上下文的下一步指令
        # Context: 明确指出下游 Agent 看不到原始轨迹，必须自包含
        # Format: 保持清晰的三段式结构，严禁使用代号
        system_prompt = """You are an expert algorithmic strategist and performance analyst. Your task is to analyze a set of all optimization trajectories and synthesize a single, actionable guidance for the next optimization iteration.

    CRITICAL REQUIREMENT: The agent reading your output will NOT have access to the original trajectory history (T1, T2, etc.). Therefore, your guidance must be **completely self-contained**.

    DO NOT use references like "Combine T1 with T3", "Avoid the mistake in Trajectory 2", etc.
    INSTEAD, explicitly describe the techniques: "Combine the O(N) monotonic stack algorithm (proven correct) with custom byte-level I/O parsing (proven fast)."

    Your goal is to provide a clear, confident, and optimal path forward that stands on its own.
    """
        
        formatted_attempts = self._format_approaches_data(approaches_data)
        
        # Prompt:
        # - 将 "FAILED ATTEMPTS" 改为 "PREVIOUS TRAJECTORIES"
        # - 保持你的三段式结构，它对于这个任务仍然非常有效
        prompt = f"""Analyze these solution trajectories and provide a single, self-contained actionable guidance for the *next* iteration:

    PROBLEM:
    {problem_statement}...

    PREVIOUS TRAJECTORIES:
    {formatted_attempts}...

    Provide concise guidance for the *next* attempt in this structure:

    KEY OBSERVATIONS:
    [List 2-3 key learnings from *all* attempts. e.g., "O(N) stack is correct," "I/O is a major bottleneck."]

    CRITICAL RISKS TO AVOID:
    [List 2-3 specific failure patterns to *not* repeat. e.g., "Simplifying the stack logic breaks correctness," "O(N^3) DP is too slow."]

    OPTIMAL NEXT STEP:
    [2-3 sentences with a single, clear directive. e.g., "Combine the O(N) stack algorithm from T1 with the fast byte-parsing I/O from T3. This is the most promising hybrid."]

    Keep total response under 200 words. Ensure it is perfectly readable without knowing the previous trajectories. Be specific and actionable."""
        
        return self._call_llm_api(prompt, system_prompt)
    
    def _generate_content(self, instance_info: Dict[str, Any], problem_statement: str, trajectory_data: Dict[str, Any]) -> str:
        """生成轨迹池总结内容"""
        instance_name = instance_info['instance_name']
        
        # 直接使用附加的池数据
        approaches_data = instance_info.get('pool_data', {})
        if not approaches_data:
            self.logger.warning(f"跳过 {instance_name}: 无轨迹池数据")
            return ""
        
        # 使用实例信息中的问题陈述
        pool_problem = instance_info.get('problem_description', 'N/A')
        
        # 获取所有迭代数据（数字键）
        iteration_data = {k: v for k, v in approaches_data.items() 
                         if k.isdigit() and isinstance(v, dict)}
        
        if not iteration_data:
            self.logger.warning(f"跳过 {instance_name}: 无有效迭代数据")
            return ""
        
        self.logger.info(f"分析 {instance_name}: {len(iteration_data)} 个历史尝试")
        
        # 生成风险感知指导
        guidance = self._generate_risk_aware_guidance(pool_problem, iteration_data)
        
        if not guidance:
            # 如果LLM调用失败，提供简化的默认指导
            guidance = "Be careful with changes that affect multiple files. Test each change incrementally. Focus on understanding the problem before implementing solutions."
        
        return guidance


# 注册算子
from operators import register_operator
register_operator("traj_pool_summary", TrajPoolSummaryOperator)