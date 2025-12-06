#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
from typing import Any

from ..global_memory.bank import GlobalMemoryBank
from ..global_memory.utils.config import GlobalMemoryConfig
from .llm_client import LLMClient
from .se_logger import get_se_logger
from .traj_pool_manager import TrajPoolManager


class GlobalMemoryManager:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        bank_config: GlobalMemoryConfig | None = None,
        bank_config_path: str | None = None,
    ):
        self.llm_client = llm_client
        self.logger = get_se_logger("global_memory", emoji="🌐")
        self.bank: GlobalMemoryBank | None = None
        try:
            if bank_config is not None or bank_config_path is not None:
                self.bank = (
                    GlobalMemoryBank(config=bank_config)
                    if bank_config is not None
                    else GlobalMemoryBank(config_path=bank_config_path)
                )
        except Exception as e:
            self.bank = None
            self.logger.warning(f"初始化GlobalMemoryBank失败: {e}")

    def _get_perf(self, val: Any) -> float:
        try:
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                s = val.strip().lower()
                if s in ("inf", "+inf", "infinity", "+infinity", "nan"):
                    return float("inf")
                return float(s)
            return float("inf")
        except Exception:
            return float("inf")

    def _select_global_best(
        self, pool_data: dict[str, Any], traj_pool_manager: TrajPoolManager
    ) -> dict[str, Any] | None:
        try:
            traj_pool_manager.refresh_best_labels()
            best_candidates: list[tuple[str, str, float, dict]] = []
            for inst_name, entry in pool_data.items():
                if not isinstance(entry, dict):
                    continue
                best_label = traj_pool_manager.get_best_label(str(inst_name))
                if not best_label:
                    continue
                detail = traj_pool_manager.get_trajectory(str(best_label), str(inst_name))
                if not isinstance(detail, dict):
                    continue
                pm = detail.get("perf_metrics")
                val = (pm or {}).get("performance") if isinstance(pm, dict) else detail.get("performance")
                perf_best = self._get_perf(val)
                best_candidates.append((str(inst_name), str(best_label), perf_best, detail))
            if not best_candidates:
                return None
            finite = [b for b in best_candidates if math.isfinite(b[2])]
            chosen = min(finite, key=lambda t: t[2]) if finite else min(best_candidates, key=lambda t: t[2])
            return {
                "instance_name": chosen[0],
                "label": chosen[1],
                "performance": chosen[2],
                "detail": chosen[3],
            }
        except Exception:
            return None

    def update_from_pool(self, traj_pool_manager: TrajPoolManager, k: int = 3) -> int:
        try:
            pool_data = traj_pool_manager.load_pool()
            steps = traj_pool_manager.extract_steps()

            # Fix: Convert dict_keys to list before indexing
            keys_list = list(pool_data.keys())
            if not keys_list:
                return 0
            instance_name = keys_list[0]

            problem_description = pool_data[instance_name]["problem"]
            improvements = [s for s in steps if isinstance(s.get("pct"), (int, float)) and s.get("pct") > 0]
            regressions = [s for s in steps if isinstance(s.get("pct"), (int, float)) and s.get("pct") < 0]
            improvements.sort(key=lambda s: s["pct"], reverse=True)
            regressions.sort(key=lambda s: abs(s["pct"]), reverse=True)
            top_improve = improvements[: max(0, int(k))]
            top_regress = regressions[: max(0, int(k))]

            best_entry = self._select_global_best(pool_data, traj_pool_manager)

            items_to_add: list[dict[str, Any]] = []
            sys_prompt, user_prompt = self._build_prompts(problem_description, top_improve, top_regress, best_entry)
            items_to_add = self._generate_experiences(sys_prompt, user_prompt)
            if not items_to_add:
                for s in top_improve:
                    items_to_add.append(
                        {
                            "type": "Success",
                            "title": f"Improved performance {s['prev_label']} -> {s['curr_label']}",
                            "description": f"Iteration {s['curr_iter']} reduced runtime from {s['perf_prev']} to {s['perf_curr']}",
                            "content": [
                                f"Delta: {s['delta']:.4f}",
                                ("Percent: " + f"{s['pct']:.1f}%") if isinstance(s["pct"], float) else "Percent: N/A",
                                "Keep correctness-first, then micro-optimize.",
                            ],
                            "evidence": [
                                {
                                    "solution_id": s["curr_label"],
                                    "code_change": "N/A",
                                    "metrics_delta": self._fmt_metrics(s["perf_prev"], s["perf_curr"]),
                                    "context": s["instance_name"],
                                }
                            ],
                        }
                    )

                for s in top_regress:
                    items_to_add.append(
                        {
                            "type": "Failure",
                            "title": f"Performance regression {s['prev_label']} -> {s['curr_label']}",
                            "description": f"Iteration {s['curr_iter']} increased runtime from {s['perf_prev']} to {s['perf_curr']}",
                            "content": [
                                f"Delta: {s['delta']:.4f}",
                                ("Percent: " + f"{s['pct']:.1f}%") if isinstance(s["pct"], float) else "Percent: N/A",
                                "Avoid premature micro-optimizations that hurt runtime.",
                            ],
                            "evidence": [
                                {
                                    "solution_id": s["curr_label"],
                                    "code_change": "N/A",
                                    "metrics_delta": self._fmt_metrics(s["perf_prev"], s["perf_curr"]),
                                    "context": s["instance_name"],
                                }
                            ],
                        }
                    )

            if not items_to_add and best_entry:
                detail = best_entry.get("detail") or {}
                txt = TrajPoolManager.format_entry({str(best_entry.get("label")): detail})
                items_to_add.append(
                    {
                        "type": "Success",
                        "title": f"Best solution {best_entry.get('label')}",
                        "description": "Globally best-performing solution.",
                        "content": [txt] if isinstance(txt, str) and txt.strip() else [],
                        "evidence": [
                            {
                                "solution_id": str(best_entry.get("label")),
                                "code_change": "N/A",
                                "metrics_delta": f"Best runtime: {best_entry.get('performance')}",
                                "context": str(best_entry.get("instance_name")),
                            }
                        ],
                    }
                )

            added = 0
            try:
                if self.bank is not None:
                    for item in items_to_add:
                        try:
                            doc = self._render_experience_markdown(item, instance_name)
                            meta = copy.deepcopy(item)
                            # 扁平化处理 evidence 列表
                            evidence_list = meta.pop("evidence", [])
                            if evidence_list and isinstance(evidence_list, list):
                                meta["evidence_json"] = json.dumps(evidence_list, ensure_ascii=False)

                            # 扁平化处理 content 列表
                            content_list = meta.pop("content", [])
                            if content_list and isinstance(content_list, list):
                                meta["content_json"] = json.dumps(content_list, ensure_ascii=False)

                            meta["instance_name"] = instance_name
                            self.bank.add_experience(doc, meta)
                            added += 1
                        except Exception:
                            continue
            except Exception as be:
                self.logger.warning(f"同步到GlobalMemoryBank失败: {be}")
            return added
        except Exception as e:
            self.logger.warning(f"全局记忆更新失败: {e}")
            return 0

    def retrieve(self, queries: list[str], k: int = 3) -> str:
        """根据查询检索相关经验，并格式化返回。

        Args:
            queries (list[str]): 查询列表
            k (int, optional): 每个查询检索的数量. Defaults to 3.

        Returns:
            str: 格式化后的检索结果字符串
        """
        try:
            if self.bank is None:
                return ""

            all_results: list[dict[str, Any]] = []
            seen_ids = set()

            for query in queries:
                # 检索
                try:
                    # retrieve_memories 返回 dict: {"documents": [[doc1, ...]], "metadatas": [[meta1, ...]], "ids": [[id1, ...]], "distances": [[dist1, ...]]}
                    ret = self.bank.retrieve_memories(query, k=k)

                    if not ret or not ret.get("documents") or not ret.get("ids"):
                        continue

                    documents = ret.get("documents", [[]])[0]
                    metadatas = ret.get("metadatas", [[]])[0]
                    ids = ret.get("ids", [[]])[0]

                    for i, doc_id in enumerate(ids):
                        if doc_id in seen_ids:
                            continue
                        seen_ids.add(doc_id)

                        doc_content = documents[i] if i < len(documents) else ""
                        meta = metadatas[i] if i < len(metadatas) else {}

                        all_results.append({"content": doc_content, "metadata": meta, "id": doc_id})

                except Exception as e:
                    self.logger.warning(f"Query检索失败 '{query}': {e}")
                    continue

            # 如果没有检索到结果，返回空字符串
            if not all_results:
                return ""

            # 格式化输出
            lines = []
            for i, item in enumerate(all_results, 1):
                content = item["content"]
                # 尝试解析 metadata 中的信息用于增强展示
                meta = item.get("metadata", {})

                lines.append(content.strip())
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            self.logger.warning(f"全局记忆检索流程异常: {e}")
            return ""

    def _format_step_context(self, s: dict[str, Any]) -> str:
        try:
            prev = {str(s.get("prev_label")): s.get("prev_detail") or {}}
            curr = {str(s.get("curr_label")): s.get("curr_detail") or {}}
            ptxt = TrajPoolManager.format_entry(prev, include_keys={"code", "perf_metrics"})
            ctxt = TrajPoolManager.format_entry(curr, include_keys={"code", "perf_metrics"})
            parts = [
                f"Instance: {s.get('instance_name')}",
                f"Previous Solution :\n{ptxt}",
                f"Current Solution :\n{ctxt}",
            ]
            return "\n\n".join([p for p in parts if isinstance(p, str)])
        except Exception:
            return ""

    def _build_prompts(
        self,
        problem_description: str,
        improve: list[dict[str, Any]],
        regress: list[dict[str, Any]],
        best_entry: dict[str, Any] | None,
    ) -> tuple[str, str]:
        # ---- System Prompt：强调对比、抽象、限量、JSON schema ---- #
        sys_prompt = """
You are an expert experience extractor for algorithm optimization.

You will be given multiple optimization steps for the SAME problem:
- Some steps lead to **Improvement** (better metrics).
- Some steps lead to **Regression** (worse metrics).
- One step corresponds to the **Best Solution** so far.

Each step contains:
- A "Previous Solution" (code + perf_metrics)
- A "Current Solution" (code + perf_metrics)

Your goal is to build a EXPERIENCE LIBRARY that captures generalizable Success / Failure lessons that can be reused on future algorithmic optimization tasks.

## What you should focus on

- Think in terms of **situation → change → effect**:
  - What kind of problem / pattern was the step handling?
  - What strategy or code-level change was applied?
  - How did the metrics change (better or worse)? Why?
- Use **contrastive reasoning**:
  - Compare Improvement vs Regression steps and vs the Best Solution.
  - Identify what consistently works and what consistently fails.
- Focus on **strategy-level changes**, for example:
  - Algorithm choice (e.g., replace O(N^2) nested loops with two pointers or binary search).
  - Data structure choice (e.g., use hash map / heap / prefix sums).
  - IO patterns (e.g., fast IO for Python).
  - Performance patches (e.g., precomputation, caching, avoiding repeated allocations).
- Ignore superficial changes:
  - Variable renaming, formatting, comments, or tiny constant tweaks without clear impact.

## Generalization requirements

- Abstract away from specific instance names, labels, or variable names.
- Describe experiences so that they apply to a FAMILY of problems (e.g., "pair counting with constraints", "DP with large N", "graph traversal with many edges").
- Merge overlapping ideas into a single, more general experience.
- You may also record **anti-patterns** as Failure experiences when regressions reveal what should be avoided.

## Limits and selection

- Output **at most 5–10 experiences** in total.
- Prioritize experiences with:
  - Clear and significant metric impact (large improvement or large regression).
  - Clear and explainable strategy behind the change.
- If some steps look noisy or ambiguous, you may skip them.

## Output format (STRICT)

You MUST return a single JSON object with key `"experiences"` whose value is a list of items.

Each item MUST have the following fields:

- `"type"`: one of `"Success"` or `"Failure"`.
- `"title"`: short, rule-like name for the experience (string).
- `"description"`: one-sentence summary of the experience (string).
- `"content"`: a **list of 1–5 short strings**, each describing a key aspect of:
  - when this applies (or should be avoided),
  - what change to make (or avoid),
  - why it matters for correctness/performance,
  - important caveats.
- `"evidence"`: a **list** of JSON objects, each with:
  - `"solution_id"`: string or empty string if unknown,
  - `"context"`: short text describing the situation (problem type, constraints, language),
  - `"metrics_delta"`: short text describing the metric change (e.g., "runtime 1200ms -> 150ms (-87.5%)").

### JSON example schema (only an example, do not copy literally):

{
  "thought_process": "Briefly explain how you compressed and merged the memory (max 2 sentences).",
  "experiences": [
    {
      "type": "Success",
      "title": "Two-pointer sweep on sorted array instead of nested loops",
      "description": "When counting pairs under a monotone condition, use a single two-pointer sweep on a sorted array instead of O(N^2) loops.",
      "content": [
        "Applicable when the condition on indices or values is monotone, such as A[j] >= k * A[i].",
        "Sort the array once, then move two pointers to maintain the valid window and count pairs.",
        "This typically reduces time from O(N^2) to O(N log N) or O(N) depending on sorting.",
        "Carefully handle boundary conditions when advancing the left pointer."
      ],
      "evidence": [
        {
          "solution_id": "Gen5_Sol_2",
          "context": "Pair counting problem in Python with N up to 2e5 on a sorted array.",
          "metrics_delta": "runtime 1.2s -> 0.15s (-87.5%)"
        }
      ]
    }
  ]
}

Return ONLY the JSON object. Do not add any explanations outside of JSON.
    """.strip()

        # ---- User Prompt：整理上下文，分区清晰 ---- #
        sections: list[str] = []

        # Problem Description
        if problem_description:
            sections.append("## Problem Description")
            sections.append(problem_description)

        # Improvements
        if improve:
            sections.append("## Improvement Steps")
            for idx, s in enumerate(improve, start=1):
                ctx = self._format_step_context(s)
                if ctx:
                    sections.append(f"### Improvement Step {idx}\n{ctx}")

        # Regressions
        if regress:
            sections.append("## Regression Steps")
            for idx, s in enumerate(regress, start=1):
                ctx = self._format_step_context(s)
                if ctx:
                    sections.append(f"### Regression Step {idx}\n{ctx}")

        # Best solution
        if best_entry:
            detail = best_entry.get("detail") or {}
            txt = TrajPoolManager.format_entry(
                {str(best_entry.get("label")): detail},
                include_keys={"code", "perf_metrics"},
            )
            sections.append("## Best Solution")
            if isinstance(txt, str) and txt.strip():
                sections.append(txt)

        # 最后附一个简短 reminder（系统里已经有详细 schema，这里只强化一次）
        guide = """
    ## Extraction Reminder

    Using the Improvement / Regression steps and the Best Solution above, extract at most 5–7 generalizable experiences.
    Remember to:
    - Focus on strategy-level changes and their metric impact.
    - Use type in ["Success", "Failure"].
    - Follow the exact JSON schema described in the system prompt and return a single JSON object with key "experiences".
    """.strip()

        user_prompt = "\n\n".join(sections + [guide])
        return sys_prompt, user_prompt

    def _parse_llm_json(self, txt: str) -> dict[str, Any] | None:
        try:
            if not isinstance(txt, str) or not txt.strip():
                return None
            s = txt.strip()
            if s.startswith("{"):
                return json.loads(s)
            start = s.find("{")
            end = s.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(s[start:end])
            return None
        except Exception:
            return None

    def _valid_experience_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(data, dict):
            msg = "响应数据必须为JSON对象"
            raise ValueError(msg)
        experiences = data.get("experiences")
        if not isinstance(experiences, list):
            msg = "experiences必须为列表"
            raise ValueError(msg)
        for item in experiences:
            if not isinstance(item, dict):
                msg = "experience项必须为对象"
                raise ValueError(msg)
            for k in ("type", "title", "description", "content", "evidence"):
                if k not in item:
                    msg = f"experiences项缺少键: {k}"
                    raise ValueError(msg)
            ev = item.get("evidence")
            if not isinstance(ev, list):
                msg = "evidence必须为列表"
                raise ValueError(msg)
            for e in ev:
                if not isinstance(e, dict):
                    msg = "evidence项必须为对象"
                    raise ValueError(msg)

    def _generate_experiences(self, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        try:
            if self.llm_client is None:
                return items
            last_error: str | None = None
            for attempt in range(1, 4):
                try:
                    resp_text = self.llm_client.call_with_system_prompt(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.7,
                        max_tokens=20000,
                        usage_context="global_memory.extract",
                    )
                    try:
                        resp_text = self.llm_client.clean_think_tags(resp_text)
                    except Exception:
                        pass
                    parsed = self._parse_llm_json(resp_text)
                    self._valid_experience_response(parsed)
                    if isinstance(parsed, dict):
                        exps = parsed.get("experiences") or []
                        if isinstance(exps, list):
                            items = [e for e in exps if isinstance(e, dict)]
                    if items:
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
            return items
        except Exception:
            return items

    def _render_experience_markdown(self, item: dict[str, Any], instance_name: str) -> str:
        lines: list[str] = []

        typ = str(item.get("type") or "")
        title = str(item.get("title") or "")
        desc = str(item.get("description") or "")

        lines.append(f"### {typ} Experience: {title} ")
        lines.append(f"- Instance: {instance_name}")
        lines.append(f"- ({typ}) {title} — {desc}")
        cnt = item.get("content")
        if isinstance(cnt, list) and cnt:
            lines.append("")
            lines.append("#### Content")
            for c in cnt:
                if isinstance(c, str) and c.strip():
                    lines.append(f"- {c}")
        return "\n".join(lines)

    def generate_queries(self, context: dict[str, Any]) -> list[str]:
        """根据上下文生成查询

        Args:
            context (dict[str, Any]): 上下文信息，包含以下键:
                - "language": 目标语言 (例如 "Python")
                - "optimization_target": 优化目标 (例如 "性能")
                - "problem_description": 问题描述 (例如 "实现一个简单的排序算法")
                - "additional_requirements": 额外需求 (例如 "必须使用原地排序")
                - "local_memory": 本地内存中的相关信息 (可选)

        Returns:
            list[str]: 生成的查询字符串列表
        """
        language = str(context.get("language") or "").strip()
        optimization_target = str(context.get("optimization_target") or "").strip()
        problem_description = str(context.get("problem_description") or "").strip()
        additional_requirements = str(context.get("additional_requirements") or "").strip()
        local_memory = str(context.get("local_memory") or "").strip()

        system_prompt, user_prompt = self._build_query_prompt(
            language=language,
            optimization_target=optimization_target,
            problem_description=problem_description,
            additional_requirements=additional_requirements,
            local_memory=local_memory,
        )

        queries = []
        try:
            if self.llm_client is None:
                return ""
            last_error: str | None = None
            for attempt in range(1, 4):
                try:
                    resp_text = self.llm_client.call_with_system_prompt(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.7,
                        max_tokens=20000,
                        usage_context="global_memory.generate_query",
                    )
                    try:
                        resp_text = self.llm_client.clean_think_tags(resp_text)
                    except Exception:
                        pass
                    parsed = self._parse_llm_json(resp_text)
                    self._valid_query_response(parsed)

                    if isinstance(parsed, dict):
                        qs = parsed.get("queries") or []
                        if isinstance(qs, list):
                            queries = [str(q) for q in qs if isinstance(q, str) and q.strip()]

                    if queries:
                        self.logger.info(f"LLM查询生成成功 (第{attempt}次)")
                        break
                except ValueError as e:
                    last_error = "invalid_response_format"
                    self.logger.warning(f"LLM查询生成解析失败: 响应格式错误或无有效JSON片段 (第{attempt}次): {e}")
                except Exception as e:
                    last_error = "llm_call_failed"
                    self.logger.warning(f"LLM查询生成调用失败 (第{attempt}次): {e}")

            if last_error:
                self.logger.error(f"LLM查询生成最终失败: {last_error}")

            return queries

        except Exception:
            return ""

    def _build_query_prompt(
        self,
        language: str,
        optimization_target: str,
        problem_description: str,
        additional_requirements: str,
        local_memory: str,
    ) -> tuple[str, str]:
        """根据上下文构建用于生成 Global Memory 检索查询的提示词"""

        system_prompt = """
You are an assistant that designs search queries for a global memory of past algorithm optimization experiences.

Your job:
- Read the given context (problem, constraints, language, optimization target, additional requirements, local memory).
- Briefly think about what may be hard or bottleneck-prone in this task.
- Output:
1) A short explanation of your reasoning.
2) 2–3 concrete queries to search for useful experiences.

Notes:
- The optimization target can be runtime, memory, or an integral score. Do not focus only on time complexity.
- Queries should be short natural-language descriptions of a scenario + a problem, e.g.:
- "In Python, how to handle fast IO when reading hundreds of thousands of integers"
- "For counting pairs in an array with N up to 2e5, how to avoid O(N^2) nested loops"
- "How to reduce memory usage for DP when the table is O(N^2) and N is 5000"
- Write queries in English. Do not use internal IDs, file names, or variable names.
- It is OK if queries mention language, input scale, data structure, and the optimization goal.

Output format (STRICT):
Return a single JSON object:

{
"thought_process": "<1–3 short paragraphs explaining what you think the key issues/bottlenecks are and why you chose these queries>",
"queries": [
    "<query 1>",
    "<query 2>",
    "<optional query 3>"
]
}

No markdown, no extra keys, ensure valid JSON.
    """.strip()

        parts: list[str] = ["## Current Context"]

        if problem_description.strip():
            parts.append("### Problem Description")
            parts.append(problem_description.strip())

        if optimization_target.strip():
            parts.append("### Optimization Target")
            parts.append(optimization_target.strip())

        if language.strip():
            parts.append("### Target Language")
            parts.append(language.strip())

        if additional_requirements.strip():
            parts.append("### Additional Requirements")
            parts.append(additional_requirements.strip())

        if local_memory.strip():
            parts.append("### Local Memory")
            parts.append(local_memory.strip())

        user_prompt = "\n\n".join(parts).strip()

        return system_prompt, user_prompt

    def _valid_query_response(self, data: dict[str, Any]) -> bool:
        """验证 Global Memory 检索查询响应是否符合预期格式"""
        if not isinstance(data, dict):
            msg = "响应数据必须为JSON对象"
            raise ValueError(msg)
        queries = data.get("queries")
        if not isinstance(queries, list):
            msg = "queries必须为列表"
            raise ValueError(msg)
        return True
