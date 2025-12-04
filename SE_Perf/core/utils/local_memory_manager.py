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
        language: str = "Unknown",
        optimization_target: str = "Runtime",
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
  "updated_directions": [
    {
      "direction": "Short strategy description (e.g., Approach: Dynamic Programming)",
      "outcome": "Baseline", 
      "source_ref": "Current_Sol_ID",
      "evidence": "Initial implementation. Runtime: X ms."
    }
  ],
  "new_memory_item": {
    "type": "Success",
    "title": "Concise Title (e.g., DP Approach Initialization)",
  "description": "One sentence summary of the algorithm used.",
  "content": "Explanation of the algorithmic logic applied to this problem.",
  "evidence": [
    {
      "solution_id": "Current_Sol_ID",
      "code_change": "N/A (Initial Solution)",
      "metrics_delta": "N/A (Baseline)",
      "context": "Initial Valid Solution"
    }
  ]
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
                    return TrajPoolManager.format_entry({lbl: entry})
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
Your task is to analyze the changes between the **Source Solution** and the **Current Solution**, explain *why* the performance improved, and update the agent's memory to guide future evolution.

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

- Noise Consideration: If improvement < 3% or the absolute delta is within typical measurement jitter, treat as likely noise.
- Memory Item Limit: You can add 0-3 new memory items to the reasoning bank.

## Output Format
You must output a single JSON object strictly adhering to this schema:

```json
{
  "thought_process": "Briefly explain your reasoning here (max 2 sentences).",
  "updated_directions": [
    {
      "direction": "Short strategy description (e.g., Use Fast I/O)",
      "outcome": "Success",
      "source_ref": "Current_Sol_ID",
      "evidence": "Reduced runtime from X ms to Y ms."
    }
    // ... Include other active directions, keeping the list concise (max 5 items)
  ],
  "new_memory_item": {
    "type": "Success",
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
                    return TrajPoolManager.format_entry({lbl: entry})
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

- Noise Consideration: If improvement < 3% or the absolute delta is within typical measurement jitter, treat as likely noise.
- Memory Item Limit: You can add 0-3 new memory items to the reasoning bank.

## Output Format
You must output a single JSON object strictly adhering to this schema:

```json
{
  "thought_process": "Briefly explain your reasoning here (max 2 sentences).",
  "updated_directions": [
    {
      "direction": "Short strategy description (e.g., Use Fast I/O)",
      "outcome": "Success",
      "source_ref": "Current_Sol_ID",
      "evidence": "Reduced runtime from X ms to Y ms."
    }
    // ... Include other active directions, keeping the list concise (max 5 items)
  ],
  "new_memory_item": {
    "type": "Success",
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
                    return TrajPoolManager.format_entry({lbl: entry})
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
        required_top = ["updated_directions", "new_memory_item"]
        missing_top = [k for k in required_top if k not in data]
        if missing_top:
            raise ValueError(f"响应格式缺少键: {', '.join(missing_top)}")
        if not isinstance(data.get("updated_directions"), list):
            raise ValueError("updated_directions必须为列表")
        item = data.get("new_memory_item")
        if not isinstance(item, dict):
            raise ValueError("new_memory_item必须为对象")
        required_item = ["type", "title", "description", "content", "evidence"]
        missing_item = [k for k in required_item if k not in item]
        if missing_item:
            raise ValueError(f"new_memory_item缺少键: {', '.join(missing_item)}")
        ev = item.get("evidence")
        if not isinstance(ev, list):
            raise ValueError("new_memory_item.evidence必须为列表")
        for e in ev:
            if not isinstance(e, dict):
                raise ValueError("evidence项必须为对象")

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
        # 1. System Prompt: 强调结构化证据的聚合
        system_prompt = """You are the **Chief Knowledge Officer** for an evolutionary coding agent.
Your goal is to maintain a high-quality, token-efficient 'Reasoning Bank' by compressing the agent's current memory.

## Compression Strategy
1. **Consolidate (Merge)**: Group insights that utilize the *same* algorithmic strategy or optimization technique.
2. **Aggregate Evidence**: When merging items, you must **collect their evidence objects** into a single list.
- **Do NOT** summarize the evidence into a string. Keep the JSON object structure (`solution_id`, `code_change`, etc.).
- **Limit Evidence**: If a merged item has more than 3 pieces of evidence, keep only the **Top-3 most distinct or impactful** ones to save space.
3. **Filter**: Remove low-value items (trivial improvements < 1%) unless they represent a unique direction.

## Output Constraints
- **Format**: Output a SINGLE JSON object containing the key `"reasoning_bank"`.
- **Item Limit**: Keep the total number of consolidated items between 3 and 10.
- **Schema**: Each item must strictly follow this structure:
{
    "type": "Success" | "Failure",
    "title": "Concise Strategy Title",
    "description": "One sentence summary",
    "content": "Detailed technical explanation",
    "evidence": [
        {
        "solution_id": "GenX_SolY", 
        "code_change": "Brief snippet (e.g., 'dp[i] % 2 -> dp[i] & 1')", 
        "metrics_delta": "Key impact (e.g., 'Runtime: 150ms -> 120ms')",
        "context": "Brief condition (e.g., 'Effective when MOD is power of 2')"
        }
        // ... Merge evidence from combined items here (Max 3)
    ]
}
"""

        # 2. User Prompt: 注入当前数据
        current_bank_json = json.dumps(memory.get("reasoning_bank", []), indent=2)

        user_prompt = f"""
## Current Reasoning Bank (Overfilled)
{current_bank_json}

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

    def extract_and_update(
        self,
        instance_name: str,
        current_entry: dict[str, Any],
        source_entries: list[dict[str, Any]] | None = None,
        best_entry: dict[str, Any] | None = None,
        problem_description: str | None = None,
        language: str | None = None,
        optimization_target: str | None = None,
        **kwargs: Any,
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
        summary = current_entry.get("summary") or {}
        perf_metrics = current_entry.get("perf_metrics")
        current_label = str(current_entry.get("label") or "")
        operator_name = str(current_entry.get("operator_name") or "")
        problem_description = str(kwargs.get("problem_description") or "")
        language = str(kwargs.get("language") or "Unknown")
        optimization_target = str(kwargs.get("optimization_target") or "Runtime")

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

        # LLM 提炼：更新 Directions + 生成 Reasoning Item
        updated_directions: list[dict[str, Any]] = attempted
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
                        if isinstance(parsed_response.get("updated_directions"), list):
                            updated_directions = parsed_response["updated_directions"]
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

        # 规则回退：若未生成项，则根据 perf_diff 简单追加一条经验
        if new_memory_item is None:
            try:
                perf_delta_str = self._format_metrics_delta(perf_old, perf_new)
                item_type = (
                    "Success" if (perf_old is not None and perf_new is not None and perf_new < perf_old) else "Failure"
                )
                new_memory_item = {
                    "type": item_type,
                    "title": "Performance Change",
                    "description": "Observed performance change between iterations.",
                    "content": summary.get("approach_summary") or "",
                    "evidence": [
                        {
                            "solution_id": current_label,
                            "code_change": "see diff head/tail",
                            "metrics_delta": perf_delta_str,
                            "context": (
                                summary.get("analysis", {}).get("best_strategy", {}).get("high_level")
                                if isinstance(summary.get("analysis"), dict)
                                else None
                            ),
                        }
                    ],
                }
            except Exception:
                pass

        # 写回：更新 Directions（全量替换）与追加 Reasoning 项
        memory["attempted_directions"] = list(updated_directions or [])
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
