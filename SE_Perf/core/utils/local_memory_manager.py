#!/usr/bin/env python3

"""
Local Memory Manager

管理短期工作记忆（Local Memory），用于在迭代优化过程中：
- 维护全局状态（当前代数、最佳性能、最佳解ID、当前方法）
- 记录尝试过的高层方向及其成败（direction board）
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

    "direction_board": [
        {
        "direction": "Use faster input/output instead of standard C++ streams.",
        "description": "For input-heavy C++ problems, replace cin/cout with faster I/O patterns such as scanf/printf or enabling ios::sync_with_stdio(false) and cin.tie(nullptr). This reduces per-call overhead and improves constant factors when reading or writing large volumes of data.",
        "status": "Success",               // Success | Failed | Neutral | Untried
        "success_count": 2,
        "failure_count": 1,
        "evidence": [
            {
            "solution_id": "Gen5_Sol_2",
            "metrics_delta": "Runtime: 150ms -> 120ms (-20%).",
            "code_change": "Replaced cin/cout with scanf/printf for all integer reads.",
            "context": "C++ solution with N up to 2e5 where input reading dominated runtime.",
            "step_outcome": "Success"
            }
        ]
        }
    ],

    "experience_library": [
        {
        "type": "Success",                 // Success | Failure | Neutral
        "title": "Bitwise modulo for power-of-two MOD",
        "description": "When MOD is a power of two, using x & (MOD-1) is faster than x % MOD and is mathematically equivalent.",
        "content": "- Only apply when MOD = 2^k.\n- Replacing division-based modulo with bitwise AND removes expensive division operations in tight loops.\n- This can significantly improve performance in DP transitions or frequency counting loops.\n- Must avoid using this trick when MOD can change or is not guaranteed to be a power of two.",
        "evidence": [
            {
            "solution_id": "Gen5_Sol_2",
            "code_change": "Changed dp[i] % 1024 -> dp[i] & 1023 in the main DP loop.",
            "metrics_delta": "Runtime: 150ms -> 120ms (-20%).",
            "context": "Hot DP loop with fixed MOD=1024, N up to 1e5."
            }
        ]
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
                "direction_board": [],
                "experience_library": [],
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
            return {"global_status": {}, "direction_board": [], "experience_library": []}
        except Exception as e:
            self.logger.warning(f"加载本地记忆库失败: {e}")
            return {"global_status": {}, "direction_board": [], "experience_library": []}

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
        dirs = memory.get("direction_board") or []
        bank = memory.get("experience_library") or []

        lines: list[str] = []
        lines.append("## Global Status")
        lines.append(f"- Generation: {gs.get('current_generation', 'N/A')}")
        lines.append(f"- Current Solution ID: {gs.get('current_solution_id', 'N/A')}")
        lines.append(f"- Best Solution ID: {gs.get('best_solution_id', 'N/A')}")
        lines.append("")
        lines.append("## Strategy Board")
        for d in dirs:
            status = d.get("status", "Unknown")
            succ = d.get("success_count", 0)
            fail = d.get("failure_count", 0)
            lines.append(f"- [{status}] {d.get('direction', '')} (✓{succ} ✗{fail}) — {d.get('description', '')}")
        lines.append("")
        lines.append("## Experience Library (Latest)")
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
  "new_direction_items": [
    {
      "direction": "Short strategy description (e.g., Approach: Dynamic Programming)",
      "description": "One sentence explaining the baseline approach.",
      "status": "Baseline",
      "evidence": [
        {
          "solution_id": "Current_Sol_ID",
          "code_change": "Initial implementation.",
          "metrics_delta": "N/A",
          "context": "Baseline"
        }
      ]
    }
  ],
  "new_memory_items": []
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
        system_prompt = """You are an expert Algorithm Optimization Specialist. You have just observed an evolutionary step where an agent **attempted to optimize** a code solution and the **metrics show an improvement** (or at least not a clear regression).

Your job is NOT to log every tiny change. Your job is to maintain:
- a **high-level strategy board** (`direction_board`), and
- an **experience library** (`experience_library`)
that together guide future evolution.

---

## Goal

Given the previous and current solutions, you must:

1. Decide whether this step is truly a **Success**, or actually **Neutral** (e.g., noise, trivial refactor).
2. If (and only if) there are **strategy-level changes**, extract up to 3 new:
   - **Direction items**: reusable optimization strategies that can be tried again on other solutions.
   - **Memory items**: distilled reasoning patterns that explain *why* certain strategies work.

This memory is local to a single problem and will be shown to the model in later steps to encourage **diverse strategy exploration**, not to duplicate the same ideas.

---

## Definitions

- **Strategy-level change**:
  - Switching algorithms (e.g., brute force → two-pointer, BFS → Dijkstra, naive DP → optimized DP).
  - Changing core data structures (e.g., vector → bitset, list → array, unordered_map → array-based counter).
  - Applying a clear performance trick (e.g., fast I/O, precomputation, caching, reducing passes over the array).
  - Changing memory layout or loop structure in a way that affects asymptotics or constant factors in a hotspot.

- **Non-strategy changes (DO NOT create directions for these)**:
  - Renaming variables, reformatting, reordering independent statements.
  - Small cosmetic refactors that do not change complexity or memory access patterns.
  - Pure measurement noise: identical code with slightly different runtimes.

---

## Very Important Rules

1. **You may return ZERO new directions and ZERO new memories.**
   - This is the correct behavior when no strategy-level change happened.

2. **Do NOT create directions about measurement noise or “no change”.**
   - The following are explicitly forbidden as directions:
     - "No Change, OS Jitter"
     - "Measurement noise"
     - "Same code as previous solution"

3. **Noise vs Success vs Neutral**:
   - If the improvement is less than about 3% or within typical measurement jitter (e.g., < 0.05 seconds), and there is *no* meaningful strategy change, treat the step as **Neutral**.
   - Only mark `"step_outcome": "Success"` when:
     - There is a real metric improvement **and**
     - You can tie it to a strategy-level code change.

4. **Rich, semantic content**:
   - `direction` should look like a clear strategy name that could appear on a “strategy board”.
   - `description` should be 1–3 sentences explaining:
     - what the strategy does,
     - when to use it,
     - and potential trade-offs or risks.
   - `content` in memory items should contain 2–6 bullet points about conditions, mechanism, and risks.

5. **Cardinality constraints**:
   - At most 3 `new_direction_items`.
   - At most 3 `new_memory_items`.
   - Arrays can be empty (`[]`).

---

## Input Data Provided

You will be given:

1. **Optimization Target**: e.g., runtime, memory, integral.
2. **Language**: e.g., C++, Python.
3. **Problem Description**: The algorithmic problem being solved.
4. **Source Solutions**: Parent code(s), summaries and metrics before mutation.
5. **Current Solution**: Mutated code, summary and metrics after mutation.
6. **Best Solution**: The global best solution so far (for context).
7. **Current Directions**: The current snapshot of the strategy board for this problem.

Use the diffs between Source and Current solution to reason about what changed.

---

## Output Format

You must output a single JSON object **strictly** adhering to this schema:

```json
{
  "thought_process": "Briefly explain your reasoning here (max 2 sentences).",
  "step_outcome": "Success | Neutral",

  "new_direction_items": [
    {
      "direction": "High-level strategy name.",
      "description": "1–3 sentences describing what was changed, why it is a reusable strategy, and when it applies.",
      "status": "Success | Neutral",
      "evidence": [
        {
          "solution_id": "Current_Sol_ID",
          "code_change": "Brief summary of the key code edits implementing this strategy.",
          "metrics_delta": "Exact improvement (e.g., Runtime: 150ms -> 120ms, -20%).",
          "context": "Conditions where this applies (e.g., input-heavy C++ problems with N up to 2e5)."
        }
      ]
    }
  ],

  "new_memory_items": [
    {
      "type": "Success | Neutral",
      "title": "Concise title of the reasoning pattern.",
      "description": "One-sentence summary of the insight.",
      "content": "2–6 bullet points or short paragraphs explaining when to apply this, why it works, and any risks.",
      "evidence": [
        {
          "solution_id": "Current_Sol_ID",
          "code_change": "Brief snippet or description of what changed.",
          "metrics_delta": "Exact improvement (e.g., -20ms or -20%).",
          "context": "Problem constraints or other conditions where this insight holds."
        }
      ]
    }
  ]
}
```

Notes:
- If there is no meaningful strategy-level change, set "step_outcome": "Neutral" and both arrays to [].
- Do not invent fake strategies just to fill the JSON.
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

## Current Directions (Strategy Board Snapshot)

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
        system_prompt = """You are an expert Algorithm Optimization Specialist. You have just observed an evolutionary step where an agent **attempted to optimize** a code solution and the **metrics show a regression or incorrectness**.

Your job is NOT to log every tiny change. Your job is to maintain:
- a **high-level strategy board** (`direction_board`), and
- an **experience library** (`experience_library`)
that warn future steps about bad ideas.

---

## Goal

Given the previous and current solutions, you must:

1. Decide whether this step is truly a **Failure**, or actually **Neutral** (e.g., noise, trivial refactor).
2. If (and only if) there are **strategy-level changes that caused the regression**, extract up to 3 new:
   - **Direction items**: strategies that should be marked as Failed or risky in the current context.
   - **Memory items**: warnings or anti-patterns explaining *why* this approach failed and when to avoid it.

---

## Definitions

- **Strategy-level change**:
  - Same as in the Success case: algorithm switch, data structure switch, clear performance trick, major loop or memory layout change.
- **Non-strategy changes (DO NOT create directions for these)**:
  - Formatting, renaming, minor refactors with no impact on complexity or memory access.
  - Pure measurement noise with identical code.

---

## Very Important Rules

1. **You may return ZERO new directions and ZERO new memories.**
   - This is the correct behavior when no strategy-level change caused the regression.

2. **Do NOT create directions about measurement noise or “no change”.**
   - Explicitly forbidden directions:
     - "No Change, OS Jitter"
     - "Measurement noise"
     - "Same code as previous solution"

3. **Noise vs Failure vs Neutral**:
   - If the regression is less than about 3% or within typical measurement jitter (e.g., < 0.05 seconds), and there is *no* meaningful strategy change, treat the step as **Neutral**.
   - Only mark `"step_outcome": "Failure"` when:
     - Runtime, memory, or correctness clearly got worse **and**
     - You can tie it to a strategy-level change (e.g., added redundant checks, switched to a slower algorithm, broke edge cases).

4. **Rich, semantic content**:
   - Directions should describe *what strategy went wrong* (e.g., “aggressive pruning without correctness proof”, “using recursion with unbounded depth”).
   - Memory items should explain *why* the strategy failed and **under what conditions** it is dangerous.

5. **Cardinality constraints**:
   - At most 3 `new_direction_items`.
   - At most 3 `new_memory_items`.
   - Arrays can be empty (`[]`).

---

## Input Data Provided

Same as in the Success case:

1. **Optimization Target**
2. **Language**
3. **Problem Description**
4. **Source Solutions**
5. **Current Solution**
6. **Best Solution**
7. **Current Directions**

---

## Output Format

You must output a single JSON object **strictly** adhering to this schema:

```json
{
  "thought_process": "Briefly explain your reasoning here (max 2 sentences).",
  "step_outcome": "Failure | Neutral",

  "new_direction_items": [
    {
      "direction": "High-level description of the failed strategy.",
      "description": "1–3 sentences explaining what the strategy tried to do and why it is problematic in this context.",
      "status": "Failed | Neutral",
      "evidence": [
        {
          "solution_id": "Current_Sol_ID",
          "code_change": "Brief summary of the key code edits that introduced the bad strategy.",
          "metrics_delta": "Exact regression (e.g., Runtime: 120ms -> 200ms, +66% or caused WA/TLE).",
          "context": "Conditions where this strategy is risky (e.g., deep recursion, large N, dense graph)."
        }
      ]
    }
  ],

  "new_memory_items": [
    {
      "type": "Failure | Neutral",
      "title": "Concise title of the anti-pattern or failure mode.",
      "description": "One-sentence summary of why this approach is dangerous.",
      "content": "2–6 bullet points explaining what went wrong, under what conditions it fails, and how to avoid it.",
      "evidence": [
        {
          "solution_id": "Current_Sol_ID",
          "code_change": "Brief snippet or description of the harmful change.",
          "metrics_delta": "Exact regression (e.g., +80ms or increased memory by 2x, or caused incorrect results).",
          "context": "Problem constraints or inputs that triggered the failure."
        }
      ]
    }
  ]
}
```

Notes:
- If there is no meaningful strategy-level change, set "step_outcome": "Neutral" and both arrays to [].
- Do not mark previously successful strategies as failed just because one noisy run was slower.
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

## Current Directions (Strategy Board Snapshot)

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
        # 仅支持数组形式返回
        if "new_direction_items" not in data:
            raise ValueError("响应格式缺少键: new_direction_items")
        nd = data.get("new_direction_items")
        if nd is not None and not isinstance(nd, list):
            raise ValueError("new_direction_items必须为列表")
        if isinstance(nd, list):
            for it in nd:
                if not isinstance(it, dict):
                    raise ValueError("new_direction_items的元素必须为对象")

        if "new_memory_items" in data:
            nm = data.get("new_memory_items")
            if nm is not None and not isinstance(nm, list):
                raise ValueError("new_memory_items必须为列表")
            if isinstance(nm, list):
                for it in nm:
                    if not isinstance(it, dict):
                        raise ValueError("new_memory_items的元素必须为对象")

    def _normalize_extraction_response(self, resp: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """将LLM响应统一转换为列表形式。"""
        dirs: list[dict[str, Any]] = []
        mems: list[dict[str, Any]] = []
        try:
            single_dir = resp.get("new_direction_item")
            if isinstance(single_dir, dict):
                dirs.append(single_dir)
            multi_dir = resp.get("new_direction_items")
            if isinstance(multi_dir, list):
                dirs.extend([d for d in multi_dir if isinstance(d, dict)])
        except Exception:
            pass
        try:
            single_mem = resp.get("new_memory_item")
            if isinstance(single_mem, dict):
                mems.append(single_mem)
            multi_mem = resp.get("new_memory_items")
            if isinstance(multi_mem, list):
                mems.extend([m for m in multi_mem if isinstance(m, dict)])
        except Exception:
            pass
        return dirs, mems

    def _merge_direction_board(self, memory: dict[str, Any], new_items: list[dict[str, Any]]) -> None:
        """将提炼的方向项直接插入 direction_board。"""
        board: list[dict[str, Any]] = memory.get("direction_board") or []
        for raw in new_items:
            if not isinstance(raw, dict):
                continue
            direction = str(raw.get("direction") or "").strip()
            if not direction:
                continue
            description = str(raw.get("description") or "").strip()
            status = str(raw.get("status") or "Neutral").strip()
            evidence_src = raw.get("evidence") if isinstance(raw.get("evidence"), list) else []
            evidence = [e for e in evidence_src if isinstance(e, dict)]

            board.append(
                {
                    "direction": direction,
                    "description": description,
                    "status": status,
                    "success_count": int(raw.get("success_count") or 0),
                    "failure_count": int(raw.get("failure_count") or 0),
                    "evidence": evidence,
                }
            )
        memory["direction_board"] = board

    def _merge_experience_library(self, memory: dict[str, Any], new_items: list[dict[str, Any]]) -> None:
        """将提炼的经验项直接插入 experience_library。"""
        library: list[dict[str, Any]] = memory.get("experience_library") or []
        for raw in new_items:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            if not title:
                continue
            typ = str(raw.get("type") or "Neutral").strip()
            description = str(raw.get("description") or "").strip()
            content = raw.get("content")
            evidence_src = raw.get("evidence") if isinstance(raw.get("evidence"), list) else []
            evidence = [e for e in evidence_src if isinstance(e, dict)]

            library.append(
                {
                    "type": typ,
                    "title": title,
                    "description": description,
                    "content": content,
                    "evidence": evidence,
                }
            )
        memory["experience_library"] = library

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
                    db = parsed.get("direction_board")
                    if isinstance(db, list):
                        memory["direction_board"] = db
                    el = parsed.get("experience_library")
                    if isinstance(el, list):
                        memory["experience_library"] = el

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
        system_prompt = f"""
You are the **Chief Knowledge Officer** of an evolutionary coding agent.
Your job is to **compress and consolidate** the agent's local memory so that it remains:
- small enough to fit within a token limit (~{token_limit} tokens), and
- rich enough to guide future evolution.

The memory you receive has two main components:
1. `direction_board`: high-level strategies that have been tried on this problem.
2. `experience_library`: distilled experiences and lessons learned.

You must output a **cleaned and consolidated** version of BOTH.

---

## Part 1: Compress `direction_board` (Strategy Board)

Each item in `direction_board` has the schema:
- `direction`: high-level strategy name (natural language).
- `description`: 1–3 sentences explaining what the strategy does and when it applies.
- `status`: "Success" | "Failed" | "Neutral" | "Untried".
- `success_count`: integer.
- `failure_count`: integer.
- `evidence`: list of objects with fields:
- `solution_id`
- `metrics_delta`
- `code_change`
- `context`
- (optionally) `step_outcome`

Your tasks:

1. **Merge semantically similar strategies**
- If multiple entries describe essentially the same idea (e.g., "Use fast I/O", "Replace cin/cout with scanf", "Enable sync_with_stdio(false) for faster input"),
    merge them into a SINGLE consolidated direction.
- Rewrite `direction` as a clear, unique strategy name.
- Rewrite `description` as a compact but informative description (1–3 sentences).

2. **Aggregate counts and status**
- For merged directions, set:
    - `success_count` = sum of `success_count` from all merged items (treat missing counts as 0).
    - `failure_count` = sum of `failure_count` from all merged items.
- Compute `status` as:
    - "Success" if successes clearly dominate and there is at least one meaningful improvement example.
    - "Failed" if failures clearly dominate and there is at least one clear regression or correctness issue.
    - "Neutral" if evidence is weak, conflicting, or mostly within noise/jitter (e.g., < 3% or < 0.05s absolute change).
- Do NOT invent fake counts. Use only what is implied by the input.

3. **Compress evidence**
- Merge evidence lists from all merged entries.
- Select at most **3** evidence items per direction.
- Prefer:
    - larger performance deltas (absolute or percentage change),
    - diverse contexts (different input sizes, patterns, or solution styles),
    - clear code changes that illustrate the strategy.
- Each evidence item must preserve the fields:
    - `solution_id`, `code_change`, `metrics_delta`, `context`
    - (you may keep `step_outcome` if present, but you must not invent it).

4. **Prune low-value directions**
- Remove directions that are:
    - extremely vague (e.g., "optimize code a bit"),
    - pure noise (e.g., strategies about "OS jitter" or "no code change"),
    - fully redundant with another, better described direction.
- Aim to keep roughly **5–8** directions that are genuinely useful for guiding future exploration.

---

## Part 2: Compress `experience_library` (Experience Library)

Each item in `experience_library` has the schema:
- `type`: "Success" | "Failure" | "Neutral".
- `title`: concise name of the experience.
- `description`: one-sentence summary.
- `content`: 2–6 bullet points or short paragraphs giving details.
- `evidence`: list of objects with fields:
- `solution_id`
- `code_change`
- `metrics_delta`
- `context`

Your tasks:

1. **Merge overlapping experiences**
- If multiple items describe the same underlying lesson (e.g., multiple entries about "bitwise modulo for power-of-two MOD"),
    merge them into a single, stronger experience.
- Choose a clear, general `title`.
- Rewrite `description` to summarize the key idea in one sentence.
- Merge and rewrite `content` into 3–7 concise bullet points or short paragraphs:
    - when it applies,
    - why it works or fails,
    - what the main trade-offs or risks are.
- Merge their evidence lists and keep at most **3** of the most representative items.

2. **Determine `type`**
- If the lesson clearly results in better performance or correctness when applied properly, mark as "Success".
- If the lesson is mainly a warning/anti-pattern, mark as "Failure".
- If the evidence is weak, mixed, or mainly about measurement noise, mark as "Neutral".

3. **Filter out trivial or redundant items**
- Discard entries that:
    - only reflect measurement noise with no actionable lesson,
    - have negligible effect (< 1% change) AND no interesting reasoning content,
    - are completely subsumed by a merged, more general experience.

---

## Global Constraints

- You MUST NOT invent new strategies, evidence, or numbers.
- You MAY rewrite text (direction, description, content) for clarity and consolidation.
- You MUST preserve the overall JSON schema, but you can reduce the number of items.
- Try to keep the final memory compact enough to reasonably fit within ~{token_limit} tokens.

---

## Output Format

Output a SINGLE JSON object with the following keys:

```json
{
            "thought_process": "Briefly explain how you compressed and merged the memory (max 2 sentences).",
"direction_board": [
    {
                "direction": "Concise, unique strategy name.",
    "description": "1–3 sentences explaining what the strategy does and when to apply it.",
    "status": "Success | Failed | Neutral | Untried",
    "success_count": 0,
    "failure_count": 0,
    "evidence": [
        {
                    "solution_id": "...",
        "code_change": "...",
        "metrics_delta": "...",
        "context": "..."
        }
    ]
    }
],
"experience_library": [
    {
                "type": "Success | Failure | Neutral",
    "title": "Concise experience title.",
    "description": "One-sentence summary of the lesson.",
    "content": "2–6 bullet points or short paragraphs giving details.",
    "evidence": [
        {
                    "solution_id": "...",
        "code_change": "...",
        "metrics_delta": "...",
        "context": "..."
        }
    ]
    }
]
}
```

- If some list is empty, return an empty array ([]) for that list.
- Return only the JSON object, with no extra commentary or backticks.
        """

        # 2. User Prompt: 注入当前数据
        data_to_compress = {
            "direction_board": memory.get("direction_board", []),
            "experience_library": memory.get("experience_library", []),
        }

        current_memory_json = json.dumps(data_to_compress, indent=2)

        user_prompt = f"""
## Current Reasoning Bank (Uncompressed)
{current_memory_json}

## Task
The current memory is too fragmented and may exceed the token limit.
Please compress and consolidate the direction_board and experience_library above:

- Merge duplicate or overlapping strategies and experiences.
- Aggregate their evidence.
- Recompute status/counts where appropriate.
- Prune low-value or noisy entries.

Output ONLY the valid JSON object.
        """

        return system_prompt, user_prompt

    def _validate_compress_response(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("响应数据必须为JSON对象")
        db = data.get("direction_board")
        if not isinstance(db, list):
            raise ValueError("direction_board必须为列表")
        for item in db:
            if not isinstance(item, dict):
                raise ValueError("direction_board项必须为对象")
            for k in ("direction", "description", "status", "success_count", "failure_count", "evidence"):
                if k not in item:
                    raise ValueError(f"direction_board项缺少键: {k}")
            ev = item.get("evidence")
            if not isinstance(ev, list):
                raise ValueError("direction_board.evidence必须为列表")
            for e in ev:
                if not isinstance(e, dict):
                    raise ValueError("evidence项必须为对象")
        el = data.get("experience_library")
        if not isinstance(el, list):
            raise ValueError("experience_library必须为列表")
        for item in el:
            if not isinstance(item, dict):
                raise ValueError("experience_library项必须为对象")
            for k in ("type", "title", "description", "content", "evidence"):
                if k not in item:
                    raise ValueError(f"experience_library项缺少键: {k}")
            ev = item.get("evidence")
            if not isinstance(ev, list):
                raise ValueError("experience_library.evidence必须为列表")
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
        attempted = memory.get("direction_board") or []

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
        dir_items: list[dict[str, Any]] = []
        mem_items: list[dict[str, Any]] = []
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
                        dir_items = [d for d in parsed_response.get("new_direction_items") or [] if isinstance(d, dict)]
                        mem_items = [m for m in parsed_response.get("new_memory_items") or [] if isinstance(m, dict)]
                        # 合并全部新项到内存结构
                        if dir_items:
                            self._merge_direction_board(memory, dir_items)
                        if mem_items:
                            self._merge_experience_library(memory, mem_items)
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

        # 不再进行单项插入的兼容处理

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
