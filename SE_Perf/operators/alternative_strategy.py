#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alternative Strategy Operator

基于指定的输入轨迹，生成一个全新的、策略上截然不同的解决方案。
此算子旨在跳出局部最优，从不同维度（例如，算法、数据结构、I/O模式）探索解空间。
"""

import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict
import yaml

from core.utils.traj_pool_manager import TrajPoolManager

from operators.base import BaseOperator


class AlternativeStrategyOperator(BaseOperator):
    def get_name(self) -> str:
        return "alternative_strategy"

    """
    替代策略算子：
    根据 step_config 中指定的单个输入轨迹（input），
    生成一个策略迥异的新轨迹（output）。
    """

    def run(
        self,
        step_config: Dict[str, Any],
        traj_pool_manager: TrajPoolManager,
        workspace_dir: str,
    ) -> Dict[str, Any]:
        output_dir = Path(workspace_dir) / "system_prompt"
        output_dir.mkdir(parents=True, exist_ok=True)

        def _work(args):
            instance_name, entry = args
            try:
                if not isinstance(entry, dict):
                    return 0
                problem_statement = entry.get("problem")
                previous_approach_summary = None
                if isinstance(entry, dict):
                    previous_approach_summary = self._format_entry(entry)
                if not problem_statement or not previous_approach_summary:
                    return 0
                content = self._build_additional_requirements(previous_approach_summary)
                if not content:
                    return 0
                data = {"prompts": {"additional_requirements": content}}
                file_path = output_dir / f"{instance_name}.yaml"
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
                return 1
            except Exception:
                return 0

        num = self.config.get("num_workers", 1)
        try:
            max_workers = max(1, int(num))
        except Exception:
            max_workers = 1

        all_instances = traj_pool_manager.get_all_trajectories()
        written = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_work, (name, entry)) for name, entry in (all_instances or {}).items()]
            for fut in as_completed(futures):
                try:
                    written += int(fut.result() or 0)
                except Exception:
                    pass

        return {"instance_templates_dir": str(output_dir), "generated_count": written}

    def _build_additional_requirements(self, previous_approach: str) -> str:
        prev = textwrap.indent(previous_approach.strip(), "  ")
        req = f"""
### STRATEGY MODE: ALTERNATIVE SOLUTION STRATEGY
You are explicitly instructed to abandon the current optimization trajectory and implement a FUNDAMENTALLY DIFFERENT approach.

### PREVIOUS APPROACH SUMMARY
{prev}

### EXECUTION GUIDELINES
1. **Qualitative Shift**: You must NOT provide incremental refinements, micro-optimizations, or simple bugfixes to the code above.
2. **New Paradigm**: Switch the algorithmic paradigm or data structure entirely (e.g., if Greedy -> try DP; if List -> try Heap/Deque; if Iterative -> try Recursive).
3. **Shift Bottleneck Focus**: If the previous attempt focused heavily on Core Algorithmics, consider an I/O-centric technique (or vice versa).
4. **Target**: Aim for a better Big-O complexity (e.g., O(N) over O(N log N)) where feasible.
        """

        return req


# 注册算子
from .registry import register_operator

register_operator("alternative_strategy", AlternativeStrategyOperator)
