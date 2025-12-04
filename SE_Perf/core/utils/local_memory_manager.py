#!/usr/bin/env python3

"""
Local Memory Manager

管理短期工作记忆（Local Memory），用于在迭代优化过程中：
- 维护全局状态（当前代数、最佳性能、最佳解ID、当前方法）
- 记录尝试过的高层方向及其成败（attempted_directions）
- 沉淀可迁移的成功/失败经验（reasoning_bank）

该模块参考 reasoningbank 的 Memory 设计思想，提供结构化的 JSON 存储与增量更新，
并在需要时调用 LLM 进行记忆提炼（Extraction）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .llm_client import LLMClient
from .se_logger import get_se_logger


class LocalMemoryManager:
    """
    本地记忆管理器（JSON 后端）

    存储结构（示例）：
    {
      "global_status": {
        "current_generation": 5,
        "current_solution_id": "Gen5_Sol_2",
        "best_solution_id": "Gen3_Sol_4"
      },
      "attempted_directions": [
        {"direction": "Use fast IO", "outcome": "Success", "source_ref": "iter_4", "evidence": "..."}
      ],
      "reasoning_bank": [
        {
          "type": "Success",
          "title": "Bitwise Operation Optimization",
          "description": "Replace modulo with bitwise AND for powers of 2.",
          "content": "Using x & (MOD-1) improved constant factor.",
          "evidence": [{
            "solution_id": "Gen5_Sol_2",
            "code_change": "Changed dp[i] % 1024 -> dp[i] & 1023",
            "metrics_delta": "Runtime: 150ms -> 120ms (-20%)",
            "context": "Effective when MOD=1024"
          }]
        }
      ]
    }
    """

    def __init__(
        self,
        memory_path: str | Path,
        llm_client: LLMClient | None = None,
        token_limit: int = 3000,
    ) -> None:
        """
        初始化本地记忆管理器。

        Args:
            memory_path: 记忆库 JSON 文件路径。
            llm_client: 可选的 LLM 客户端，用于进行记忆提炼。
            token_limit: 触发压缩的近似 token/字符阈值。
        """
        self.path = Path(memory_path)
        self.llm_client = llm_client
        self.token_limit = int(token_limit)
        self.logger = get_se_logger("local_memory", emoji="🧠")

    def initialize(self) -> None:
        """确保记忆库文件存在，若不存在则创建空结构。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            empty = {
                "global_status": {
                    "current_generation": 0,
                    "current_solution_id": None,
                    "best_solution_id": None,
                },
                "attempted_directions": [],
                "reasoning_bank": [],
            }
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(empty, f, ensure_ascii=False, indent=2)
            self.logger.info(f"初始化本地记忆库: {self.path}")

    def load(self) -> dict[str, Any]:
        """加载记忆库 JSON。"""
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"global_status": {}, "attempted_directions": [], "reasoning_bank": []}
        except Exception as e:
            self.logger.warning(f"加载本地记忆库失败: {e}")
            return {"global_status": {}, "attempted_directions": [], "reasoning_bank": []}

    def save(self, memory: dict[str, Any]) -> None:
        """保存记忆库 JSON。"""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存本地记忆库失败: {e}")
            raise

    def render_as_markdown(self, memory: dict[str, Any]) -> str:
        """
        将结构化记忆渲染为简洁的 Markdown 文本，便于注入 System Prompt。
        """
        gs = memory.get("global_status") or {}
        dirs = memory.get("attempted_directions") or []
        bank = memory.get("reasoning_bank") or []

        lines: list[str] = []
        lines.append("## Global Status")
        lines.append(f"- Generation: {gs.get('current_generation', 'N/A')}")
        lines.append(f"- Current Solution ID: {gs.get('current_solution_id', 'N/A')}")
        lines.append(f"- Best Solution ID: {gs.get('best_solution_id', 'N/A')}")
        lines.append("")
        lines.append("## Attempted Directions")
        for d in dirs:
            lines.append(f"- [{d.get('outcome', 'Unknown')}] {d.get('direction', '')} — {d.get('evidence', '')}")
        lines.append("")
        lines.append("## Reasoning Bank (Latest)")
        for item in bank:
            lines.append(f"- ({item.get('type', '')}) {item.get('title', '')} — {item.get('description', '')}")
        return "\n".join(lines)

    def _estimate_chars(self, memory: dict[str, Any]) -> int:
        """粗略估计记忆体量（按字符计）。"""
        try:
            return len(json.dumps(memory, ensure_ascii=False))
        except Exception:
            return 0

    def _format_metrics_delta(self, perf_old: float | None, perf_new: float | None) -> str:
        """将性能变化格式化为易读字符串。"""
        try:
            if perf_old is None or perf_new is None:
                return "N/A"
            if math.isinf(perf_old) and not math.isinf(perf_new):
                return f"Runtime: inf -> {perf_new}"
            if math.isinf(perf_new):
                return f"Runtime: {perf_old} -> inf"
            delta = perf_new - perf_old
            pct = (delta / perf_old * 100.0) if perf_old and not math.isinf(perf_old) else None
            if pct is None:
                return f"Runtime: {perf_old} -> {perf_new}"
            sign = "+" if pct >= 0 else ""
            return f"Runtime: {perf_old} -> {perf_new} ({sign}{pct:.1f}%)"
        except Exception:
            return "N/A"

    def _build_extraction_prompts(
        self,
        problem_description: str | None,
        perf_old: float | None,
        perf_new: float | None,
        source_entries: list[dict[str, Any]] | None,
        current_entry: dict[str, Any] | None,
        best_entry: dict[str, Any] | None,
        current_directions: list[dict[str, Any]],
        language: str = "",
        optimization_target: str = "",
        current_solution_id: str | None = None,
    ) -> tuple[str, str]:
        """
        构造记忆提炼的 System/User 提示词。
        根据性能变化分流进入 Success 或 Failure 分支。
        """
        # 1. Metric Analysis
        perf_diff = 0.0
        is_initial = False

        if perf_old is not None and perf_new is not None:
            # Handle inf
            if math.isinf(perf_old) and not math.isinf(perf_new):
                perf_diff = float("inf")  # Improvement
            elif not math.isinf(perf_old) and math.isinf(perf_new):
                perf_diff = float("-inf")  # Regression
            elif math.isinf(perf_old) and math.isinf(perf_new):
                perf_diff = 0.0
            else:
                perf_diff = perf_old - perf_new
        elif perf_new is not None:
            is_initial = True

        # 2. Extraction Branch
        if is_initial:
            return self._build_initial_prompt(
                problem_description,
                perf_new,
                current_directions,
                language,
                optimization_target,
                current_entry,
                best_entry,
            )
        elif perf_diff > 0:
            return self._build_success_prompt(
                problem_description,
                perf_old,
                perf_new,
                perf_diff,
                source_entries,
                current_entry,
                best_entry,
                current_directions,
                language,
                optimization_target,
                current_solution_id,
            )
        else:
            return self._build_failure_prompt(
                problem_description,
                perf_old,
                perf_new,
                perf_diff,
                source_entries,
                current_entry,
                best_entry,
                current_directions,
                language,
                optimization_target,
                current_solution_id,
            )

    def _build_initial_prompt(
        self,
        problem,
        perf_new,
        directions,
        language,
        target,
        current_entry: dict[str, Any] | None = None,
        best_entry: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        # 1. System Prompt for Baseline/Initial Solution
        system_prompt = """You are an expert Algorithm Optimization Specialist. You are analyzing the **initial solution** (Baseline) generated by an agent for a competitive programming problem.

## Goal
Since there is no previous version to compare against, your task is to **identify the algorithmic strategy** used in the Current Solution and initialize the agent's memory.

## Guidelines for Memory Extraction

1. **Identify Strategy**: Analyze the whole code. What is the core algorithmic paradigm? (e.g., Dynamic Programming, Greedy, BFS, Binary Search, or naive Brute Force).
2. **Establish Baseline**: The "Memory Item" should describe this fundamental approach.
3. **Initialize Directions**:
    - Extract the core approach and add it to "updated_directions".
    - Mark the outcome as "Baseline" or "Success" (since it is a valid starting point).
    - If the "Current Directions" list is empty, populate it with this detected strategy.

## Other Hints

- Memory Item Limit: You can add 0-3 new memory items to the reasoning bank.

## Input Data Provided
You will be given:
1. **Problem Description**: The algorithmic problem.
2. **Current Solution**: The generated code and its runtime/memory metrics.
3. **Best Solution**: The global best solution (for context).
4. **Optimization Target**: (e.g., runtime, memory).
5. **Language**: (e.g., C++, Python).
6. **Current Directions**: Likely empty or contains pre-set hints.

## Output Format
You must output a single JSON object strictly adhering to this schema:

```json
{
  "thought_process": "Briefly explain what algorithm the code uses (e.g., 'The code uses a hash map to store frequencies...').",
  "new_direction_item": {
      "direction": "Short strategy description (e.g., Approach: Dynamic Programming)",
      "outcome": "Baseline",
      "source_ref": "Current_Sol_ID",
      "evidence": "Initial implementation. Runtime: X ms."
    }
}
        """

        user_template = """
        
## Optimization Target
 
{optimization_target}

## Language

{language}
        
## Problem Description
        
{problem_description}
     
## Current Solution
        
{current_solution}

## Best Solution
        
{best_solution}

## Current Direction

{directions}
        """
        # Build formatted texts using TrajPoolManager.format_entry
        try:
            from .traj_pool_manager import TrajPoolManager
        except Exception:
            TrajPoolManager = None  # type: ignore

        def _fmt_entry_text(entry: dict | None) -> str:
            try:
                if TrajPoolManager and isinstance(entry, dict):
                    lbl = str(entry.get("label") or entry.get("solution_id") or "current")
                    return TrajPoolManager.format_entry({lbl: entry}, include_keys={"code", "perf_metrics"})
            except Exception:
                pass
            return "N/A"

        current_solution_text = _fmt_entry_text(current_entry)
        best_solution_text = _fmt_entry_text(best_entry)

        user_prompt = user_template.format(
            optimization_target=str(target or "Runtime"),
            language=str(language or "Unknown"),
            problem_description=str(problem or "N/A"),
            current_solution=current_solution_text,
            best_solution=best_solution_text,
            directions=json.dumps(directions or [], ensure_ascii=False),
        )

        return system_prompt, user_prompt

    def _build_success_prompt(
        self,
        problem,
        perf_old,
        perf_new,
        perf_diff,
        source_entries,
        current_entry,
        best_entry,
        directions,
        language,
        target,
        current_solution_id,
    ) -> tuple[str, str]:
        # 1. System Prompt
        system_prompt = """You are an expert Algorithm Optimization Specialist. You have just observed an evolutionary step where an agent **successfully optimized** a code solution.

## Goal
Your task is to analyze the code changes between the **Source Solution** and the **Current Solution**, explain *why* the performance improved, and update the agent's memory to guide future evolution.

## Guidelines for Memory Extraction

1. **Analyze the Diff**: Look strictly at the logical changes between the Source and Current solution. Ignore formatting changes.
2. **Verify Causality**: Connect the specific code change to the metric improvement. (e.g., "Replacing `cin` with `scanf` caused the 20ms speedup").
3. **Generalize**: The "Memory Item" you create should be a reusable tactic, not just a description of this specific problem.
4. **Update Directions**:
    - If the successful change corresponds to an existing item in "Current Directions", mark it as verified/successful.
    - If it's a new strategy, add it to the directions.
    - Remove directions that are now obsolete or clearly inferior to this new finding.

## Input Data Provided
You will be given:
1. **Problem Description**: The algorithmic problem being solved.
2. **Source Solution**: The parent code, summary and metrics before mutation.
3. **Current Solution**: The mutated code, summary and metrics that achieved better performance.
4. **Best Solution**: The global best solution code, summary and metrics found so far (for context).
5. **Optimization Target**: (e.g., runtime, memory, integral).
6. **Language**: The programming language used (e.g., C++, Python).
6. **Current Directions**: The active list of strategies the agent is currently exploring.

## Other Hints

- Noise/Neutral Classification: If the improvement is less than 3% or the absolute delta is within typical measurement jitter (e.g., < 0.05 seconds), classify the change as **Neutral** rather than Success.
- Neutral Handling: When outcome is Neutral, you may set `new_memory_item` to `null` (no new knowledge) or provide a concise item with `type: "Neutral"` if a general lesson exists (keep it minimal).
- Memory Item Limit: You can add 0-3 new memory items to the reasoning bank.

        ## Output Format
        You must output a single JSON object strictly adhering to this schema:

        ```json
        {
          "thought_process": "Briefly explain your reasoning here (max 2 sentences).",
          "new_direction_item": {
            "direction": "Short optimization strategy description (e.g., Use Fast I/O)",
            "outcome": "Success" | "Neutral",
            "source_ref": "Current_Sol_ID",
            "evidence": "Reduced runtime from X ms to Y ms."
          },
          "new_memory_item": {
            "type": "Success" | "Neutral",
            "title": "Concise Title (e.g., Bitwise Modulo Optimization)",
            "description": "One sentence summary of the technique.",
            "content": "Detailed insight on how to apply this optimization.",
            "evidence": [
              {
                "solution_id": "Current_Sol_ID",
                "code_change": "Brief snippet of what changed (e.g., i % 2 -> i & 1)",
                "metrics_delta": "Exact improvement (e.g., -20ms)",
                "context": "Conditions where this applies (e.g., when N is power of 2)"
              }
            ]
          }
        }
        ```

        Note: If the outcome is Neutral, set `new_direction_item.outcome` to "Neutral" and `new_memory_item` to `null` or an item with `type: "Neutral"`.
        """
        user_template = """
        
## Optimization Target
 
{optimization_target}

## Language

{language}
        
## Problem Description
        
{problem_description}

## Source Solutions
        
{source_solutions}
        
## Current Solution
        
{current_solution}

## Best Solution
        
{best_solution}

## Current Direction

{directions}
        """
        # Build formatted texts using TrajPoolManager.format_entry
        try:
            from .traj_pool_manager import TrajPoolManager
        except Exception:
            TrajPoolManager = None  # type: ignore

        def _fmt_entry_text(entry: dict | None) -> str:
            try:
                if TrajPoolManager and isinstance(entry, dict):
                    lbl = str(entry.get("label") or entry.get("solution_id") or "current")
                    return TrajPoolManager.format_entry({lbl: entry}, include_keys={"code", "perf_metrics"})
            except Exception:
                pass
            return "N/A"

        def _fmt_entries_text(entries: list[dict] | None) -> str:
            if not entries:
                return "N/A"
            texts: list[str] = []
            for e in entries:
                t = _fmt_entry_text(e)
                if t and t != "N/A":
                    texts.append(t)
            return "\n\n".join(texts) if texts else "N/A"

        source_solutions_text = _fmt_entries_text(source_entries)
        current_solution_text = _fmt_entry_text(current_entry)
        best_solution_text = _fmt_entry_text(best_entry)

        user_prompt = user_template.format(
            optimization_target=str(target or "Runtime"),
            language=str(language or "Unknown"),
            problem_description=str(problem or "N/A"),
            source_solutions=source_solutions_text,
            current_solution=current_solution_text,
            best_solution=best_solution_text,
            directions=json.dumps(directions or [], ensure_ascii=False),
        )

        return system_prompt, user_prompt

    def _build_failure_prompt(
        self,
        problem,
        perf_old,
        perf_new,
        perf_diff,
        source_entries,
        current_entry,
        best_entry,
        directions,
        language,
        target,
        current_solution_id,
    ) -> tuple[str, str]:
        # 1. System Prompt
        system_prompt = """You are an expert Algorithm Optimization Specialist. You have just observed an evolutionary step where an agent **failed** to optimize the task (either performance degraded, or the solution became incorrect).
## Goal
Your task is to analyze the changes between the **Source Solution** and the **Current Solution**, identify the root cause of the failure, and create a warning to prevent this mistake in the future.

## Guidelines for Memory Extraction
1. **Identify the Trap**: Did the agent trade too much memory for time? Did a logical simplification break edge cases? Did an algorithm switch (e.g., Recursion to Iteration) add too much overhead?
2. **Reflection**: Think *why* the agent thought this would work, and why it actually failed.
3. **Update Directions (Tabu List)**:
    - If this failed strategy is in "Current Directions", update its status to "Failed" with the specific reason.
    - If it's a new failure, add it to warn future agents.
    - **Crucial**: Do not delete successful directions from the list just because this specific attempt failed. Only penalize the specific strategy used here.
## Input Data Provided
You will be given:
1. **Problem Description**: The algorithmic problem being solved.
2. **Source Solution**: The parent code, summary and metrics before mutation.
3. **Current Solution**: The mutated code, summary and metrics that achieved better performance.
4. **Best Solution**: The global best solution code, summary and metrics found so far (for context).
5. **Optimization Target**: (e.g., runtime, memory, integral).
6. **Language**: The programming language used (e.g., C++, Python).
6. **Current Directions**: The active list of strategies the agent is currently exploring.

## Other Hints

- Noise/Neutral Classification: If the regression is less than 3% or the absolute delta is within typical measurement jitter (e.g., < 0.05 seconds), classify the change as **Neutral** rather than Failure.
- Neutral Handling: When outcome is Neutral, you may set `new_memory_item` to `null` (no new knowledge) or provide a concise item with `type: "Neutral"` if a general lesson exists (keep it minimal).
- Memory Item Limit: You can add 0-3 new memory items to the reasoning bank.

        ## Output Format
        You must output a single JSON object strictly adhering to this schema:

        ```json
        {
          "thought_process": "Briefly explain your reasoning here (max 2 sentences).",
          "new_direction_item": {
            "direction": "Short strategy description (e.g., Use Fast I/O)",
            "outcome": "Failed" | "Neutral",
            "source_ref": "Current_Sol_ID",
            "evidence": "Increased runtime from X ms to Y ms or caused incorrectness."
          },
          "new_memory_item": {
            "type": "Failure" | "Neutral",
            "title": "Concise Title (e.g., Recursion Stack Overhead)",
            "description": "One sentence summary of the failed technique.",
            "content": "Detailed insight on why this approach failed and under what conditions.",
            "evidence": [
              {
                "solution_id": "Current_Sol_ID",
                "code_change": "Brief snippet of what changed (e.g., for-loop to recursive call)",
                "metrics_delta": "Exact regression (e.g., +50ms)",
                "context": "Conditions where this is a bad idea (e.g., when recursion depth is high)"
              }
            ]
          }
        }
        ```

        Note: If the outcome is Neutral, set `new_direction_item.outcome` to "Neutral" and `new_memory_item` to `null` or an item with `type: "Neutral"`.
        """
        user_template = """
        
## Optimization Target
 
{optimization_target}

## Language

{language}
        
## Problem Description
        
{problem_description}

## Source Solutions
        
{source_solutions}
        
## Current Solution
        
{current_solution}

## Best Solution
        
{best_solution}

## Current Direction

{directions}
        """
        # Build formatted texts using TrajPoolManager.format_entry
        try:
            from .traj_pool_manager import TrajPoolManager
        except Exception:
            TrajPoolManager = None  # type: ignore

        def _fmt_entry_text(entry: dict | None) -> str:
            try:
                if TrajPoolManager and isinstance(entry, dict):
                    lbl = str(entry.get("label") or entry.get("solution_id") or "current")
                    return TrajPoolManager.format_entry({lbl: entry}, include_keys={"code", "perf_metrics"})
            except Exception:
                pass
            return "N/A"

        def _fmt_entries_text(entries: list[dict] | None) -> str:
            if not entries:
                return "N/A"
            texts: list[str] = []
            for e in entries:
                t = _fmt_entry_text(e)
                if t and t != "N/A":
                    texts.append(t)
            return "\n\n".join(texts) if texts else "N/A"

        source_solutions_text = _fmt_entries_text(source_entries)
        current_solution_text = _fmt_entry_text(current_entry)
        best_solution_text = _fmt_entry_text(best_entry)

        user_prompt = user_template.format(
            optimization_target=str(target or "Runtime"),
            language=str(language or "Unknown"),
            problem_description=str(problem or "N/A"),
            source_solutions=source_solutions_text,
            current_solution=current_solution_text,
            best_solution=best_solution_text,
            directions=json.dumps(directions or [], ensure_ascii=False),
        )

        return system_prompt, user_prompt

    def _parse_llm_json(self, text: str) -> dict[str, Any]:
        """提取并解析 LLM 返回的 JSON 内容。"""
        content = (text or "").strip()
        if not content:
            raise ValueError("空响应内容，无法解析为JSON")

        # 尝试直接解析完整JSON
        if content.startswith("{"):
            return json.loads(content)

        # 尝试提取JSON片段进行解析
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_content = content[start_idx:end_idx]
            return json.loads(json_content)

        # 未找到可解析的JSON片段
        raise ValueError("响应中未找到可解析的JSON内容")

    def _validate_memory_response(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("响应数据必须为JSON对象")
        # new_memory_item 可以在 Neutral 情况下为 null；也允许初始/噪声场景下为 null
        required_top = ["new_direction_item"]
        missing_top = [k for k in required_top if k not in data]
        if missing_top:
            raise ValueError(f"响应格式缺少键: {', '.join(missing_top)}")

        if data.get("new_direction_item") is None:
            return

        if not isinstance(data.get("new_direction_item"), dict):
            raise ValueError("new_direction_item必须为对象")
        item = data.get("new_memory_item")
        if item is None:
            # 允许为 null
            return
        if not isinstance(item, dict):
            raise ValueError("new_memory_item必须为对象")

        required_item = ["type", "title", "description", "content", "evidence"]
        missing_item = [k for k in required_item if k not in item]
        if missing_item:
            raise ValueError(f"new_memory_item缺少键: {', '.join(missing_item)}")

    def compress_if_needed(self, memory: dict[str, Any]) -> None:
        try:
            if self._estimate_chars(memory) <= self.token_limit:
                return
            if not self.llm_client:
                self.logger.warning("LLM不可用，跳过记忆压缩")
                return
            sys_prompt, user_prompt = self._build_compress_prompts(memory, self.token_limit)
            last_error: str | None = None
            for attempt in range(1, 4):
                try:
                    llm_response = self.llm_client.call_with_system_prompt(
                        system_prompt=sys_prompt,
                        user_prompt=user_prompt,
                        temperature=0.7,
                        max_tokens=10000,
                        usage_context="memory.compress",
                    )
                    self.logger.debug(f"LLM原始响应 (压缩，第{attempt}次):\n{llm_response}")
                    llm_response = self.llm_client.clean_think_tags(llm_response)
                    self.logger.debug(f"LLM清理后响应 (压缩，第{attempt}次):\n{llm_response}")
                    parsed = self._parse_llm_json(llm_response)
                    self._validate_compress_response(parsed)
                    ad = parsed.get("attempted_directions")
                    if isinstance(ad, list):
                        memory["attempted_directions"] = ad
                    rb = parsed.get("reasoning_bank")
                    if isinstance(rb, list):
                        memory["reasoning_bank"] = rb

                    self.logger.info("LLM记忆压缩成功")
                    break
                except ValueError as e:
                    last_error = "invalid_response_format"
                    self.logger.warning(f"LLM记忆压缩解析失败: 响应格式错误或无有效JSON片段 (第{attempt}次): {e}")
                except Exception as e:
                    last_error = "llm_call_failed"
                    self.logger.warning(f"LLM记忆压缩调用失败 (第{attempt}次): {e}")
            if last_error:
                self.logger.error(f"LLM记忆压缩最终失败: {last_error}")
        except Exception as e:
            self.logger.warning(f"压缩记忆失败: {e}")

    def _build_compress_prompts(self, memory: dict[str, Any], token_limit: int) -> tuple[str, str]:
        # 1. System Prompt: 同时管理 Reasoning Bank 和 Directions
        system_prompt = """You are the **Chief Knowledge Officer** for an evolutionary coding agent.
Your goal is to compress and consolidate the agent's entire memory (Reasoning Bank + Attempted Directions) to fit within token limits while preserving high-value insights.

## Task 1: Consolidate Reasoning Bank (Deep Knowledge)
1. **Merge**: Group insights covering the same core strategy (e.g., merge "Bitwise Mod" and "Bitwise And Opt").
2. **Aggregate Evidence**: For merged items, collect their structural evidence objects into a single list.
    - Limit to **Top-3** most distinct/impactful evidence items per strategy.
    - Keep the exact schema: `solution_id`, `code_change`, `metrics_delta`, `context`.
3. **Filter**: Discard low-impact items (<1% gain) unless unique.

## Task 2: Refine Attempted Directions (High-Level Guide)
1. **Deduplicate**: Merge identical or highly similar directions (e.g., "Use Fast I/O" and "Switch to scanf").
2. **Resolve Conflicts**: If a direction appears as both "Success" and "Failed":
    - If the latest attempt succeeded, mark as "Success" (and note the caveat in evidence).
    - If it consistently fails now, mark as "Failed".
    - If changes are within jitter (e.g., < 3% or < 0.05s absolute), mark as **Neutral**.
3. **Prune**: Remove directions that are vague, obsolete, or fully covered by a "Reasoning Bank" item (don't need to track it as a "direction" if it's already a proven "knowledge").
4. **Limit**: Keep the list concise (max 5-8 active directions).

## Output Format
Output a SINGLE JSON object with keys:
```json
{
    "thought_process": "Briefly explain your reasoning here (max 2 sentences).",
    "attempted_directions": [
        {
        "direction": "Concise strategy name",
        "outcome": "Success" | "Failed" | "Neutral",
        "source_ref": "Most relevant Solution ID (Single String)",
        "evidence": "Brief text summary of the outcome (2-3 sentences)"
        }
    ],
    "reasoning_bank": [
        {
        "type": "Success" | "Failure" | "Neutral",
        "title": "...",
        "description": "...",
        "content": "...",
        "related_operator": "...",
        "evidence": [ 
            {"solution_id": "...", "code_change": "...", "metrics_delta": "...", "context": "..."}
        ]
        }
    ],
}
        """

        # 2. User Prompt: 注入当前数据
        data_to_compress = {
            "reasoning_bank": memory.get("reasoning_bank", []),
            "attempted_directions": memory.get("attempted_directions", []),
        }

        current_memory_json = json.dumps(data_to_compress, indent=2)

        user_prompt = f"""
## Current Reasoning Bank (Overfilled)
{current_memory_json}

## Task
The current memory is too fragmented. 
Please **Compress** and **Consolidate** the list above.
**Merge** duplicate strategies and **Aggregate** their structural evidence.

Output ONLY the valid JSON object.
    """

        return system_prompt, user_prompt

    def _validate_compress_response(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("响应数据必须为JSON对象")
        rb = data.get("reasoning_bank")
        if not isinstance(rb, list):
            raise ValueError("reasoning_bank必须为列表")
        for item in rb:
            if not isinstance(item, dict):
                raise ValueError("reasoning_bank项必须为对象")
            for k in ("type", "title", "description", "content", "evidence"):
                if k not in item:
                    raise ValueError(f"reasoning_bank项缺少键: {k}")
            ev = item.get("evidence")
            if not isinstance(ev, list):
                raise ValueError("reasoning_bank.evidence必须为列表")
            for e in ev:
                if not isinstance(e, dict):
                    raise ValueError("evidence项必须为对象")
        ad = data.get("attempted_directions")
        if not isinstance(ad, list):
            raise ValueError("attempted_directions必须为列表")
        for item in ad:
            if not isinstance(item, dict):
                raise ValueError("attempted_directions项必须为对象")
            for k in ("direction", "outcome", "source_ref", "evidence"):
                if k not in item:
                    raise ValueError(f"attempted_directions项缺少键: {k}")

    def extract_and_update(
        self,
        instance_name: str,
        current_entry: dict[str, Any],
        source_entries: list[dict[str, Any]] | None = None,
        best_entry: dict[str, Any] | None = None,
        problem_description: str | None = None,
        language: str | None = None,
        optimization_target: str | None = None,
    ) -> None:
        """
        根据一次迭代的总结与性能数据，进行记忆提炼并更新本地记忆库。

        Args:
            instance_name: 实例名称。
            current_entry: 当前轨迹条目（包含 iteration, summary, code, perf_metrics 等）。
            source_entries: 来源轨迹条目列表（用于对比 diff 和性能变化）。
            best_entry: 当前最佳轨迹条目（用于参考）。
            problem_description: 问题描述。
            language: 编程语言。
            optimization_target: 优化目标（如 Runtime, Memory 等）。
        """
        memory = self.load()
        attempted = memory.get("attempted_directions") or []

        # Extract data from entries
        iteration = int(current_entry.get("iteration") or 0)
        perf_metrics = current_entry.get("perf_metrics")
        current_label = str(current_entry.get("label") or "")

        # 计算性能差异（old vs new）
        perf_old = None
        perf_new = None
        try:
            # New performance
            if perf_metrics:
                new_perf_val = perf_metrics.get("final_performance") or perf_metrics.get("performance")
                perf_new = float(new_perf_val) if new_perf_val is not None else None
            if perf_new is None:
                # Fallback to top-level performance field
                new_perf_val = current_entry.get("performance")
                perf_new = float(new_perf_val) if new_perf_val is not None else None

            # Old performance: Compare against ALL source entries (Best/Min)
            source_perfs = []
            if source_entries:
                for entry in source_entries:
                    val = None
                    # Try perf_metrics
                    entry_perf_metrics = entry.get("perf_metrics")
                    if entry_perf_metrics:
                        perf_val = entry_perf_metrics.get("final_performance") or entry_perf_metrics.get("performance")
                        val = float(perf_val) if perf_val is not None else None
                    # Try top-level
                    if val is None:
                        perf_val = entry.get("performance")
                        val = float(perf_val) if perf_val is not None else None

                    if val is not None:
                        source_perfs.append(val)

            if source_perfs:
                # Assuming that Lower is Better, so we take the minimum of source entries
                perf_old = min(source_perfs)
        except Exception:
            pass

        # LLM 提炼：生成 Direction Item + 生成 Reasoning Item
        new_direction_item: dict[str, Any] | None = None
        new_memory_item: dict[str, Any] | None = None
        if self.llm_client:
            try:
                sys_prompt, user_prompt = self._build_extraction_prompts(
                    problem_description,
                    perf_old,
                    perf_new,
                    source_entries,
                    current_entry,
                    best_entry,
                    attempted,
                    language=language,
                    optimization_target=optimization_target,
                    current_solution_id=current_label,
                )
                last_error: str | None = None
                for attempt in range(1, 4):
                    try:
                        llm_response = self.llm_client.call_with_system_prompt(
                            system_prompt=sys_prompt,
                            user_prompt=user_prompt,
                            temperature=0.7,
                            max_tokens=20000,
                            usage_context="local_memory.extract_and_update",
                        )
                        self.logger.debug(f"LLM原始响应 (第{attempt}次):\n{llm_response}")
                        llm_response = self.llm_client.clean_think_tags(llm_response)
                        self.logger.debug(f"LLM清理后响应 (第{attempt}次):\n{llm_response}")
                        parsed_response = self._parse_llm_json(llm_response)
                        self._validate_memory_response(parsed_response)
                        if isinstance(parsed_response.get("new_direction_item"), dict):
                            new_direction_item = parsed_response["new_direction_item"]
                        extracted_item = parsed_response.get("new_memory_item")
                        if isinstance(extracted_item, dict):
                            new_memory_item = extracted_item
                        self.logger.info(f"LLM记忆提炼成功 (第{attempt}次)")
                        break
                    except ValueError as e:
                        last_error = "invalid_response_format"
                        self.logger.warning(f"LLM记忆提炼解析失败: 响应格式错误或无有效JSON片段 (第{attempt}次): {e}")
                    except Exception as e:
                        last_error = "llm_call_failed"
                        self.logger.warning(f"LLM记忆提炼调用失败 (第{attempt}次): {e}")
                if last_error:
                    self.logger.error(f"LLM记忆提炼最终失败: {last_error}")
            except Exception as e:
                self.logger.warning(f"LLM记忆提炼失败，使用规则回退: {e}")

        if new_direction_item:
            if not isinstance(memory.get("attempted_directions"), list):
                memory["attempted_directions"] = []
            memory["attempted_directions"].append(new_direction_item)

        if new_memory_item:
            if isinstance(new_memory_item, dict):
                bank = memory.get("reasoning_bank") or []
                bank.append(new_memory_item)
                memory["reasoning_bank"] = bank

        # 更新全局状态
        gs = memory.get("global_status") or {}
        gs["current_generation"] = int(iteration)
        try:
            current_solution_id = current_entry.get("label", "")
        except Exception:
            current_solution_id = None
        gs["current_solution_id"] = current_solution_id

        try:
            best_solution_id = best_entry.get("label", "")
        except Exception:
            best_solution_id = None
        gs["best_solution_id"] = best_solution_id

        memory["global_status"] = gs

        # 压缩（必要时）并保存
        self.compress_if_needed(memory)
        self.save(memory)
        self.logger.info(
            json.dumps(
                {
                    "memory_update": {
                        "instance": instance_name,
                        "iteration": iteration,
                        "label": current_label,
                        "current_generation": memory.get("global_status", {}).get("current_generation"),
                    }
                },
                ensure_ascii=False,
            )
        )
