#!/usr/bin/env python3
"""
Crossover Operator

当轨迹池中有效条数大于等于2时，结合两条轨迹的特性生成新的策略。
当有效条数不足时，记录错误并跳过处理。
"""

import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from core.utils.traj_pool_manager import TrajPoolManager

from operators.base import BaseOperator


class CrossoverOperator(BaseOperator):
    """交叉算子：综合两条轨迹的优点，生成新的初始代码"""

    def get_name(self) -> str:
        return "crossover"

    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager: TrajPoolManager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        if not step_config.get("inputs") or len(step_config["inputs"]) != 2:
            return {}
        input_label1 = step_config["inputs"][0].get("label")
        input_label2 = step_config["inputs"][1].get("label")

        output_dir = Path(workspace_dir) / "system_prompt"
        output_dir.mkdir(parents=True, exist_ok=True)

        def _work(args):
            instance_name, entry = args
            try:
                if not isinstance(entry, dict):
                    return 0
                ref1 = entry.get(input_label1) if input_label1 else None
                ref2 = entry.get(input_label2) if input_label2 else None
                if not isinstance(ref1, dict) or not isinstance(ref2, dict):
                    return 0
                problem_statement = entry.get("problem")
                summary1 = self._format_entry({str(input_label1 or "iter1"): ref1}) if isinstance(ref1, dict) else ""
                summary2 = self._format_entry({str(input_label2 or "iter2"): ref2}) if isinstance(ref2, dict) else ""
                if not problem_statement or not summary1 or not summary2:
                    return 0
                content = self._build_additional_requirements(summary1, summary2)
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

    def _build_additional_requirements(self, trajectory1: str, trajectory2: str) -> str:
        t1 = textwrap.indent(trajectory1.strip(), "  ")
        t2 = textwrap.indent(trajectory2.strip(), "  ")
        req = f"""
### STRATEGY MODE: CROSSOVER STRATEGY
You are tasked with synthesizing a SUPERIOR hybrid solution by intelligently combining the best elements of two prior optimization trajectories described below.

### TRAJECTORY 1 SUMMARY
{t1}

### TRAJECTORY 2 SUMMARY
{t2}

### SYNTHESIS GUIDELINES
1. **Complementary Combination**: Actively combine specific strengths.
- Example: If T1 has a better Core Algorithm but slow I/O, and T2 has fast I/O but a naive algorithm, implement T1's algorithm using T2's I/O technique.
- Example: If T1 used a correct Stack logic but slow List, and T2 used a fast Array but had logic bugs, implement T1's logic using T2's structure.
2. **Avoid Shared Weaknesses**: If both trajectories failed at a specific sub-task, you must introduce a novel fix for that specific part.
3. **Seamless Integration**: Do not just concatenate code. The resulting logic must be a single, cohesive implementation.
        """
        return req


# 注册算子
from .registry import register_operator

register_operator("crossover", CrossoverOperator)
