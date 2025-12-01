#!/usr/bin/env python3
"""
SE Operators Base Classes

定义了所有算子的基类和核心接口。
算子是模块化的、可重用的组件，用于执行特定的轨迹操作，如生成、交叉或过滤。
"""

from __future__ import annotations

import abc
import re
from typing import Any

from core.utils.llm_client import LLMClient
from core.utils.se_logger import get_se_logger
from core.utils.traj_pool_manager import TrajPoolManager


class BaseOperator(abc.ABC):
    """
    SE算子基类，定义通用功能和新的 `run` 接口。
    所有算子都应继承自此类。
    """

    def __init__(self, config: dict[str, Any]):
        """
        初始化算子。

        Args:
            config: 包含 `operator_models` 等模型配置信息。
        """
        self.config = config
        self.llm_client: LLMClient | None = None
        self.logger = get_se_logger(f"operator.{self.get_name()}", emoji="🔧")

    def _setup_model(self) -> None:
        """设置LLM客户端实例。"""
        if self.llm_client is not None:
            return
        model_config_data = self.config.get("operator_models", self.config.get("model", {}))
        self.llm_client = LLMClient(model_config_data)
        self.logger.info(f"LLM客户端已初始化: {model_config_data.get('name')}")

    def _call_llm_api(self, prompt: str, system_prompt: str = "") -> str:
        """
        调用LLM API。

        Args:
            prompt: 用户提示。
            system_prompt: 系统提示。

        Returns:
            LLM生成的响应文本。
        """
        self._setup_model()
        history = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": prompt})

        try:
            model_cfg = self.config.get("operator_models", self.config.get("model", {}))
            temp = model_cfg.get("temperature", 0.3)
            max_out = model_cfg.get("max_output_tokens")
            self.logger.debug(f"LLM系统提示词:\n{system_prompt}")
            self.logger.debug(f"LLM用户提示词:\n{prompt}")
            message = self.llm_client.call_llm(history, temperature=temp, max_tokens=max_out)
            self.logger.debug(f"LLM原始响应:\n{message}")
            if message:
                message = self.llm_client.clean_think_tags(message)
            self.logger.debug(f"LLM清理后响应:\n{message}")
            return message or ""
        except Exception as e:
            self.logger.error(f"LLM API调用失败: {e}")
            return ""

    def _extract_code_block_py(self, text: str) -> str | None:
        """从LLM输出中提取 ```py ... ``` 代码块内容。"""
        if not isinstance(text, str) or not text:
            return None
        pattern = re.compile(r"```(?:py|python)\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
        m = pattern.search(text)
        if m:
            return m.group(1).strip() or None
        return None

    def _extract_code_text(self, text: str) -> str | None:
        """优先提取代码块，否则返回原始文本并尝试清理。"""
        if not isinstance(text, str) or not text.strip():
            return None
        block = self._extract_code_block_py(text)
        if isinstance(block, str) and block.strip():
            return block.strip()

        raw_code = text.strip()
        if raw_code.startswith("```") and raw_code.endswith("```"):
            try:
                raw_code = re.sub(r"^```(?:py|python)?\s*\n?", "", raw_code, flags=re.IGNORECASE)
                raw_code = re.sub(r"\n?```$", "", raw_code)
            except Exception:
                pass
        return raw_code.strip() or None

    def _require_py_block_with_retry(
        self,
        build_prompt_fn,
        max_retries: int = 2,
        temperature_override: float | None = None,
    ) -> str | None:
        """要求LLM以```py代码块```输出，若未满足则重试。"""
        self._setup_model()
        original_model_cfg = self.config.get("operator_models", self.config.get("model", {}))
        base_temp = original_model_cfg.get("temperature", 0.3)
        temp_to_use = base_temp if temperature_override is None else temperature_override

        for attempt in range(max_retries + 1):
            try:
                prompt, system_prompt = build_prompt_fn(attempt)
                enforce_tail = "\n\nSTRICT FORMAT: Wrap the entire solution inside a fenced code block starting with ```py and ending with ```."
                import_blocks = """\n\nAllowed Imports Scope: You may only import libraries within the scope defined below.
```python
import re
from re import match, search, sub, split, findall, finditer
import sys
from sys import maxsize, stdin
import json
from json import loads
import math
from math import floor, ceil, factorial, sqrt, isqrt, inf, log2, log10, sin, cos, tan, pi, e, comb, perm, gcd, lcm
import copy
import pickle
import heapq
from heapq import heappush, heappop, heapify, heappushpop, nlargest, nsmallest
import bisect
from bisect import bisect_left, bisect_right
import string
from string import ascii_letters, ascii_lowercase, ascii_uppercase, digits, whitespace, punctuation, hexdigits
import random
import operator
import itertools
from itertools import combinations, permutations, product, groupby, chain, accumulate, zip_longest
import functools
from functools import lru_cache, cache, reduce
import collections
from collections import OrderedDict, defaultdict, Counter, deque
from typing import Set, Dict, List, Optional, Tuple
import sortedcontainers # pip install sortedcontainers
from sortedcontainers import SortedList, SortedDict, SortedSet
```"""
                optimization_target = """
CORE TASK
Your task is to iteratively improve a given program in python for the problem described below, aiming to increase its **runtime**.

GUIDING PRINCEPLES
Your core philosophy is **CORRECTNESS FIRST, THEN PERFORMANCE**.
1.  **Correctness Priority**: Your primary goal is to produce correct outputs for all required cases. Ensure any changes maintain or improve correctness *before* optimizing for performance.
2.  **Performance Focus**: Improve performance only *after* correctness is assured. Prefer algorithmic improvements over micro-optimizations.
3.  **Context Utilization**: You MUST leverage all provided information (evolution history in the chat, current metrics, artifacts etc.) to make informed optimization decisions.
4.  **Substantial Impact**: Focus on meaningful improvements that significantly impact the fitness score.
5.  **Code Quality**: Keep the code readable, robust, and maintainable. Avoid unnecessary refactors.
6.  **Diversity**: Explore alternative algorithms, data structures, or techniques (e.g., built-in operators, packages) when appropriate.         
                """
                system_prompt_use = (system_prompt or "") + enforce_tail + import_blocks + optimization_target

                history = [{"role": "system", "content": system_prompt_use}, {"role": "user", "content": prompt}]
                max_out = original_model_cfg.get("max_output_tokens")
                enable_thinking = None if attempt == 0 else False

                self.logger.info(f"第{attempt + 1}次尝试，温度={temp_to_use}")
                self.logger.debug(f"LLM系统提示词(重试第{attempt + 1}次)")
                self.logger.debug(f"LLM用户提示词(重试第{attempt + 1}次)")

                message = self.llm_client.call_llm(
                    history,
                    temperature=temp_to_use,
                    max_tokens=max_out,
                    enable_thinking=enable_thinking,
                )
                self.logger.debug(f"LLM原始响应(重试第{attempt + 1}次):\n{message}")
                if message:
                    message = self.llm_client.clean_think_tags(message)
                # self.logger.debug(f"LLM清理后响应(重试第{attempt + 1}次):\n{message}")

                code = self._extract_code_block_py(message or "")
                if code:
                    return code

                self.logger.warning("未检测到```py代码块，进行重试")
            except Exception as e:
                self.logger.error(f"格式化代码块生成失败: {e}")
                continue
        return None

    def _format_entry(self, approaches_data: dict[str, Any]) -> str:
        if not isinstance(approaches_data, dict) or not approaches_data:
            return ""

        def _parse_key_num(k: Any) -> int | None:
            if isinstance(k, str):
                if k.isdigit():
                    try:
                        return int(k)
                    except Exception:
                        return None
                m = re.search(r"(\d+)$", k)
                if m:
                    try:
                        return int(m.group(1))
                    except Exception:
                        return None
            return None

        candidates: list[int] = []
        mapping: dict[int, tuple[str, Any]] = {}
        for key, val in approaches_data.items():
            if key == "problem":
                continue
            key_num = _parse_key_num(key)
            iter_num = None
            if isinstance(val, dict):
                it = val.get("iteration")
                try:
                    if it is not None:
                        iter_num = int(it)
                except Exception:
                    iter_num = None
            use_num = iter_num if isinstance(iter_num, int) else key_num if isinstance(key_num, int) else -1
            candidates.append(use_num)
            mapping[use_num] = (str(key), val)

        if not candidates:
            return ""
        latest_iteration = max(candidates)
        latest_key, latest_data = mapping.get(latest_iteration, ("", {}))

        def indent_str(level: int) -> str:
            return "  " * level

        def fmt_value(val: Any, level: int) -> str:
            if val is None:
                return "null"
            if isinstance(val, (int, float)):
                return str(val)
            if isinstance(val, bool):
                return "true" if val else "false"
            if isinstance(val, str):
                if "\n" in val:
                    lines = val.splitlines()
                    pad = indent_str(level + 1)
                    return "|\n" + "\n".join(f"{pad}{line}" for line in lines)
                return val
            if isinstance(val, dict):
                lines: list[str] = []
                for k, v in val.items():
                    key_line = f"{indent_str(level)}{k}:"
                    if isinstance(v, (dict, list)) or (isinstance(v, str) and "\n" in v):
                        lines.append(key_line)
                        lines.append(fmt_value(v, level + 1))
                    else:
                        lines.append(f"{key_line} {fmt_value(v, 0)}")
                return "\n".join(lines)
            if isinstance(val, list):
                lines: list[str] = []
                for item in val:
                    if isinstance(item, (dict, list)) or (isinstance(item, str) and "\n" in item):
                        lines.append(f"{indent_str(level)}-")
                        lines.append(fmt_value(item, level + 1))
                    else:
                        lines.append(f"{indent_str(level)}- {fmt_value(item, 0)}")
                return "\n".join(lines)
            return str(val)

        chosen_label = latest_data.get("label") if isinstance(latest_data, dict) else None
        header = str(chosen_label or latest_key).strip()
        body = fmt_value(latest_data, 0)
        return f"{header}\n{body}".strip() if header else body

    @abc.abstractmethod
    def get_name(self) -> str:
        """获取算子名称。"""
        pass

    @abc.abstractmethod
    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager: TrajPoolManager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        """
        执行算子的核心逻辑。

        Args:
            step_config: 当前步骤的配置，来自YAML文件。
            traj_pool_manager: 轨迹池管理器实例。
            workspace_dir: 工作区目录。

        Returns:
            一个包含算子执行结果的字典。
            例如: {"generated_code": "..."} 或 {"filtered_labels": ["..."]}
        """
        pass


class TemplateOperator(BaseOperator):
    """
    模板算子基类，用于为下一次 PerfAgent 运行生成初始代码。
    这类算子的 `run` 方法通常返回一个包含 `generated_code` 键的字典。
    """

    @abc.abstractmethod
    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager: TrajPoolManager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        pass


class EnhanceOperator(BaseOperator):
    """
    增强算子基类，用于为下一次 PerfAgent 运行生成增强历史配置。
    这类算子的 `run` 方法通常返回一个包含 `enhance_history_filter_json` 键的字典。
    """

    @abc.abstractmethod
    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager: TrajPoolManager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        pass
