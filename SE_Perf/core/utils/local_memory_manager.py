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

from .se_logger import get_se_logger


class LocalMemoryManager:
    """
    本地记忆管理器（JSON 后端）

    存储结构（示例）：
    {
      "global_status": {
        "current_generation": 5,
        "best_runtime": "120ms",
        "best_solution_id": "Gen3_Sol_4",
        "current_approach": "Dynamic Programming with Bitmask"
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
          "related_operator": "Refinement",
          "source_ref": {"generation": 3, "solution_id": "Sol_5", "parent_id": "Gen_2_Sol_2"},
          "evidence": {
            "code_change": "Changed dp[i] % 1024 -> dp[i] & 1023",
            "metrics_delta": "Runtime: 150ms -> 120ms (-20%)",
            "context": "Effective when MOD=1024"
          }
        }
      ]
    }
    """

    def __init__(
        self,
        memory_path: str | Path,
        llm_client: Any | None = None,
        token_limit: int = 1500,
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
                    "best_runtime": None,
                    "best_solution_id": None,
                    "current_approach": None,
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
        lines.append(f"- Best Runtime: {gs.get('best_runtime', 'N/A')}")
        lines.append(f"- Best Solution ID: {gs.get('best_solution_id', 'N/A')}")
        lines.append(f"- Current Approach: {gs.get('current_approach', 'N/A')}")
        lines.append("")
        lines.append("## Attempted Directions")
        for d in dirs[:8]:
            lines.append(f"- [{d.get('outcome', 'Unknown')}] {d.get('direction', '')} — {d.get('evidence', '')}")
        lines.append("")
        lines.append("## Reasoning Bank (Latest)")
        for item in bank[-5:]:
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
        perf_old: float | None,
        perf_new: float | None,
        code_diff: str,
        current_directions: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """
        构造记忆提炼的 System/User 提示词。
        """
        outcome = "SUCCESS" if (perf_old is not None and perf_new is not None and perf_new < perf_old) else "FAILURE"
        metrics_line = self._format_metrics_delta(perf_old, perf_new)
        sys = """You are the Memory Manager for an evolutionary coding agent.

Return a JSON object with keys `updated_directions` and `new_memory_item`.
Do not include commentary outside JSON.
"""
        user = (
            "## Context\n"
            f"- {metrics_line} [Outcome: {outcome}]\n\n"
            "## Inputs\n"
            "1. Code Diff:\n" + (code_diff or "N/A") + "\n\n"
            "2. Current Directions (JSON):\n" + json.dumps(current_directions or [], ensure_ascii=False) + "\n\n"
            "## Task\n"
            "1. Update Directions (replace list).\n"
            "2. Create Reasoning Item (or null).\n"
        )
        return sys, user

    def _parse_llm_json(self, text: str) -> dict[str, Any]:
        """提取并解析 LLM 返回的 JSON 内容。"""
        content = (text or "").strip()
        if not content:
            return {}
        # 直接解析或从三引号中提取
        try:
            return json.loads(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                frag = content[start : end + 1]
                try:
                    return json.loads(frag)
                except Exception:
                    return {}

    def compress_if_needed(self, memory: dict[str, Any]) -> None:
        """
        当记忆体量超过阈值时进行触发式压缩：
        - 保留 Top-3 Success（按提升幅度排序）与 Top-2 Failure（按回归幅度排序）
        - 合并相似条目（简化为按标题去重）
        """
        try:
            if self._estimate_chars(memory) <= self.token_limit:
                return
            bank = memory.get("reasoning_bank") or []
            if not isinstance(bank, list) or not bank:
                return

            def _score(item: dict[str, Any]) -> float:
                md = str(item.get("evidence", {}).get("metrics_delta", ""))
                # 简单解析百分比，失败则按 0 处理
                try:
                    if "%" in md:
                        p = md.split("(")[-1].split("%")[0]
                        return float(p)
                except Exception:
                    pass
                return 0.0

            success = [x for x in bank if str(x.get("type")) == "Success"]
            failure = [x for x in bank if str(x.get("type")) == "Failure"]
            success_sorted = sorted(success, key=_score, reverse=True)[:3]
            failure_sorted = sorted(failure, key=_score)[:2]
            kept = success_sorted + failure_sorted

            # 按标题去重
            seen_titles: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for it in kept + [x for x in bank if x not in kept]:
                title = str(it.get("title") or "").strip()
                if title and title in seen_titles:
                    continue
                seen_titles.add(title)
                deduped.append(it)

            memory["reasoning_bank"] = deduped
        except Exception as e:
            self.logger.warning(f"压缩记忆失败: {e}")

    def extract_and_update(
        self,
        instance_name: str,
        iteration: int,
        summary: dict[str, Any],
        patch_content: str,
        perf_metrics: dict[str, Any] | None = None,
        previous_code: str | None = None,
        current_label: str | None = None,
        operator_name: str | None = None,
    ) -> None:
        """
        根据一次迭代的总结与性能数据，进行记忆提炼并更新本地记忆库。

        Args:
            instance_name: 实例名称。
            iteration: 当前迭代号（用于全局状态）。
            summary: 轨迹总结字典（含 approach_summary/analysis/best_strategy 等）。
            patch_content: 当前迭代的最终代码或占位（失败时可能为 "FAILED_NO_PATCH"）。
            perf_metrics: 性能指标字典（来自 result.json 提取的精简字段）。
            previous_code: 上一次解的代码（用于生成简化 diff）。
            current_label: 当前条目的标签（如 "sol1" / "iterN"）。
            operator_name: 算子名称（用于记录来源）。
        """
        memory = self.load()
        attempted = memory.get("attempted_directions") or []

        # 计算性能差异（old vs new）
        perf_old = None
        perf_new = None
        try:
            if perf_metrics:
                # 优先使用 performance_before / final_performance
                po = perf_metrics.get("performance_before")
                pn = perf_metrics.get("final_performance")
                perf_old = float(po) if po is not None else None
                perf_new = float(pn) if pn is not None else None
        except Exception:
            perf_old = None
            perf_new = None

        # 构造简化 diff（仅文本级，避免依赖外部库）
        code_diff = ""
        try:
            prev = (previous_code or "").splitlines()
            curr = (patch_content or "").splitlines()
            # 仅包含头尾片段提高可读性
            head = "\n".join(curr[:20])
            tail = "\n".join(curr[-10:]) if len(curr) > 20 else ""
            code_diff = "# New Code (head)\n" + head + ("\n# New Code (tail)\n" + tail if tail else "")
            if prev:
                phead = "\n".join(prev[:10])
                code_diff = "# Old Code (head)\n" + phead + "\n\n" + code_diff
        except Exception:
            code_diff = patch_content or ""

        # LLM 提炼：更新 Directions + 生成 Reasoning Item
        updated_dirs: list[dict[str, Any]] = attempted
        new_item: dict[str, Any] | None = None
        if self.llm_client:
            try:
                sys_prompt, user_prompt = self._build_extraction_prompts(perf_old, perf_new, code_diff, attempted)
                resp = self.llm_client.call_with_system_prompt(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    max_tokens=3000,
                    usage_context="local_memory_manager",
                )
                parsed = self._parse_llm_json(resp)
                if isinstance(parsed.get("updated_directions"), list):
                    updated_dirs = parsed["updated_directions"]
                nmi = parsed.get("new_memory_item")
                if isinstance(nmi, dict):
                    new_item = nmi
            except Exception as e:
                self.logger.warning(f"LLM记忆提炼失败，使用规则回退: {e}")

        # 规则回退：若未生成项，则根据 perf_diff 简单追加一条经验
        if new_item is None:
            try:
                perf_line = self._format_metrics_delta(perf_old, perf_new)
                itype = (
                    "Success" if (perf_old is not None and perf_new is not None and perf_new < perf_old) else "Failure"
                )
                new_item = {
                    "type": itype,
                    "title": "Performance Change",
                    "description": "Observed performance change between iterations.",
                    "content": summary.get("approach_summary") or "",
                    "related_operator": operator_name or "unknown",
                    "source_ref": {"generation": iteration, "solution_id": current_label, "instance": instance_name},
                    "evidence": {
                        "code_change": "see diff head/tail",
                        "metrics_delta": perf_line,
                        "context": summary.get("analysis", {}).get("best_strategy", {}).get("high_level")
                        if isinstance(summary.get("analysis"), dict)
                        else None,
                    },
                }
            except Exception:
                pass

        # 写回：更新 Directions（全量替换）与追加 Reasoning 项
        memory["attempted_directions"] = list(updated_dirs or [])
        if isinstance(new_item, dict):
            bank = memory.get("reasoning_bank") or []
            bank.append(new_item)
            memory["reasoning_bank"] = bank

        # 更新全局状态
        gs = memory.get("global_status") or {}
        gs["current_generation"] = int(iteration)
        try:
            approach = summary.get("solution_name") or summary.get("strategy") or summary.get("approach_summary")
        except Exception:
            approach = None
        gs["current_approach"] = approach

        # 维护最佳性能
        try:
            current_best = gs.get("best_runtime")
            cb_val = float(current_best) if current_best is not None else float("inf")
        except Exception:
            cb_val = float("inf")
        try:
            if perf_new is not None and float(perf_new) < cb_val:
                gs["best_runtime"] = float(perf_new)
                gs["best_solution_id"] = current_label or f"iter_{iteration}"
        except Exception:
            pass
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
                        "best_runtime": memory.get("global_status", {}).get("best_runtime"),
                    }
                },
                ensure_ascii=False,
            )
        )
