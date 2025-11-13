#!/usr/bin/env python3

"""
SE Operators Base Classes

基于Aeon generators设计理念，为SE项目提供模块化算子系统。
支持两种基础算子类型：
- TemplateOperator: 返回 initial_code_dir（初始代码目录）
- EnhanceOperator: 返回enhance_history_filter_json（历史增强配置）
"""

import abc
import concurrent.futures
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from core.utils.se_logger import get_se_logger

from core.utils.llm_client import LLMClient


class BaseOperator(abc.ABC):
    """SE算子基类，定义通用功能和接口"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化算子

        Args:
            config: 包含operator_models等配置信息
        """
        self.config = config
        self.model = None  # LLM模型实例（旧路径）
        self.llm_client: Optional[LLMClient] = None  # 统一的 OpenAI LLM 客户端
        self.logger = get_se_logger(f"operator.{self.get_name()}", emoji="🔧")

    def _setup_model(self) -> None:
        """设置LLM客户端实例，改用统一的 OpenAI 接口客户端"""
        if self.llm_client is not None:
            return

        # 使用 operator_models 配置（如果存在），否则回退到 model 配置
        model_config_data = self.config.get("operator_models", self.config.get("model", {}))
        # 初始化统一 LLM 客户端
        self.llm_client = LLMClient(model_config_data)
        self.logger.info(f"LLM客户端已初始化: {model_config_data.get('name')}")

    def _call_llm_api(self, prompt: str, system_prompt: str = "") -> str:
        """
        调用LLM API（复用Aeon generators的调用方式）

        Args:
            prompt: 用户提示
            system_prompt: 系统提示

        Returns:
            LLM生成的响应文本
        """
        self._setup_model()

        # 构建消息历史
        history = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": prompt})

        try:
            temp = (
                self.config.get("operator_models", self.config.get("model", {})).get("temperature", 0.3)
            )
            max_out = (
                self.config.get("operator_models", self.config.get("model", {})).get("max_output_tokens")
            )
            message = self.llm_client.call_llm(history, temperature=temp, max_tokens=max_out)
            # 按需调用 LLMClient 的清理方法，移除 <think> 标签内容
            if message:
                message = self.llm_client.clean_think_tags(message)
            return message if message else ""
        except Exception as e:
            self.logger.error(f"LLM API调用失败: {e}")
            return ""

    def _extract_code_block_py(self, text: str) -> Optional[str]:
        """
        从LLM输出中提取 ```py \n ... \n ``` 代码块，仅返回块内内容。

        返回:
            提取到的代码字符串，若未找到返回 None。
        """
        if not isinstance(text, str) or not text:
            return None
        # 支持三引号中含有语言标记的fence，如```python, ```py
        pattern = re.compile(r"```(?:py|python)\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
        m = pattern.search(text)
        if m:
            code = m.group(1).strip()
            return code if code else None
        return None

    def _extract_code_text(self, text: str) -> Optional[str]:
        """
        优先提取 ```py/```python fenced 代码块内容；若未检测到 fence，则返回纯文本。
        若文本首尾误含 fence（但未匹配成功），尽力剥离首尾 fence 后返回。

        返回:
            纯代码文本；若无法提取或文本为空，返回 None。
        """
        if not isinstance(text, str) or not text.strip():
            return None
        # 先尝试严格提取 fence 中的代码
        block = self._extract_code_block_py(text)
        if isinstance(block, str) and block.strip():
            return block.strip()
        # 否则接受纯文本作为代码，并尽力剥离首尾 fence
        raw_code = text.strip()
        if raw_code.startswith("```") and raw_code.endswith("```"):
            try:
                raw_code = re.sub(r"^```(?:py|python)?\s*\n?", "", raw_code, flags=re.IGNORECASE)
                raw_code = re.sub(r"\n?```$", "", raw_code)
            except Exception:
                pass
        return raw_code if raw_code.strip() else None

    def _require_py_block_with_retry(
        self,
        build_prompt_fn,
        max_retries: int = 2,
        temperature_override: Optional[float] = None,
    ) -> Optional[str]:
        """
        要求LLM以```py代码块```输出，若未满足则重试。

        参数:
            build_prompt_fn: 可调用，返回 (prompt, system_prompt) 二元组；每次重试可根据计数调整。
            max_retries: 最大重试次数（不含首轮）。
            temperature_override: 可选，覆盖温度使模型更顺从格式。

        返回:
            仅代码块内容的字符串；若失败返回 None。
        """
        self._setup_model()

        # 保存原始温度配置，必要时临时提高/降低
        original_model_cfg = self.config.get("operator_models", self.config.get("model", {}))
        base_temp = original_model_cfg.get("temperature", 0.3)
        temp_to_use = base_temp if temperature_override is None else temperature_override

        for attempt in range(max_retries + 1):
            try:
                prompt, system_prompt = build_prompt_fn(attempt)

                # 强化格式要求的系统提示追加
                enforce_tail = "\n\nSTRICT FORMAT: Wrap the entire solution inside a fenced code block starting with ```py and ending with ```."
                allowed_import_scope = (
                    "\n\n### Allowed Imports Scope\n"
                    "You may only import libraries within the scope defined below.\n"
                    "```python\n"
                    "import re\n"
                    "from re import match, search, sub, split, findall, finditer\n"
                    "import sys\n"
                    "from sys import maxsize, stdin\n"
                    "import json\n"
                    "from json import loads\n"
                    "import math\n"
                    "from math import floor, ceil, factorial, sqrt, isqrt, inf, log2, log10, sin, cos, tan, pi, e, comb, perm, gcd, lcm\n"
                    "import copy\n"
                    "import pickle\n"
                    "import heapq\n"
                    "from heapq import heappush, heappop, heapify, heappushpop, nlargest, nsmallest\n"
                    "import bisect\n"
                    "from bisect import bisect_left, bisect_right\n"
                    "import string\n"
                    "from string import ascii_letters, ascii_lowercase, ascii_uppercase, digits, whitespace, punctuation, hexdigits\n"
                    "import random\n"
                    "import operator\n"
                    "import itertools\n"
                    "from itertools import combinations, permutations, product, groupby, chain, accumulate, zip_longest\n"
                    "import functools\n"
                    "from functools import lru_cache, cache, reduce\n"
                    "import collections\n"
                    "from collections import OrderedDict, defaultdict, Counter, deque\n"
                    "from typing import Set, Dict, List, Optional, Tuple\n"
                    "import sortedcontainers # pip install sortedcontainers\n"
                    "from sortedcontainers import SortedList, SortedDict, SortedSet\n"
                    "```\n"
                )

                system_prompt_use = (system_prompt or "") + enforce_tail + allowed_import_scope

                # 手动构造消息并调用底层客户端（保持与 _call_llm_api 一致）
                history = []
                if system_prompt_use:
                    history.append({"role": "system", "content": system_prompt_use})
                history.append({"role": "user", "content": prompt})

                max_out = original_model_cfg.get("max_output_tokens")
                # 首次尝试保持默认（可能使用思考模式）；若未提取到代码块，下一次强制关闭思考模式
                enable_thinking = None if attempt == 0 else False # 直接关闭思考模式
                
                self.logger.info(f"第{attempt}次尝试，温度={temp_to_use}，最大输出token={max_out}，思考模式={enable_thinking}")
                self.logger.info(f"请求内容: {prompt}")
                
                message = self.llm_client.call_llm(
                    history,
                    temperature=temp_to_use,
                    max_tokens=max_out,
                    enable_thinking=enable_thinking,
                )
                self.logger.info(f"响应内容: {message}")
                if message:
                    message = self.llm_client.clean_think_tags(message)

                code = self._extract_code_block_py(message or "")
                if code:
                    return code

                # 若未提取到，调整温度或在下一次尝试加重格式说明
                self.logger.warning("未检测到```py代码块，进行重试")
                
            except Exception as e:
                self.logger.error(f"格式化代码块生成失败: {e}")
                # 继续下一次尝试
                continue

        return None

    def _discover_instances(self, workspace_dir: Path, current_iteration: int) -> List[Dict[str, Any]]:
        """
        发现可处理的实例列表

        Args:
            workspace_dir: 工作目录路径
            current_iteration: 当前迭代号

        Returns:
            实例信息列表，每个元素包含: {
                'instance_name': str,
                'instance_dir': Path,
                'trajectory_file': Path,
                'previous_iteration': int
                'problem_description': str
            }
        """
        instances = []
        previous_iteration = current_iteration - 1

        if previous_iteration < 1:
            self.logger.warning(f"无效的前一迭代号: {previous_iteration}")
            return instances

        # 查找前一迭代的输出目录
        prev_iter_dir = workspace_dir / f"iteration_{previous_iteration}"
        if not prev_iter_dir.exists():
            self.logger.warning(f"前一迭代目录不存在: {prev_iter_dir}")
            return instances

        # 查找前一迭代中的所有实例目录
        for instance_dir in prev_iter_dir.iterdir():
            if not instance_dir.is_dir() or instance_dir.name.startswith("."):
                continue

            # 查找.tra轨迹文件
            tra_files = list(instance_dir.glob("*.tra"))
            if not tra_files:
                continue

            # 使用第一个找到的.tra文件
            trajectory_file = tra_files[0]

            # 提取问题陈述，问题陈述文件在 instance dir 下的 instance_name.problem
            problem_file = list(instance_dir.glob("*.problem"))[0]
            if not problem_file:
                continue
            with open(problem_file, "r", encoding="utf-8") as f:
                problem_description = f.read().strip()

            instances.append(
                {
                    "instance_name": instance_dir.name,
                    "instance_dir": instance_dir,
                    "trajectory_file": trajectory_file,
                    "previous_iteration": previous_iteration,
                    "problem_description": problem_description,
                }
            )

        self.logger.info(f"发现 {len(instances)} 个可处理的实例")
        return instances

    def _load_trajectory_data(self, trajectory_file: Path) -> Dict[str, Any]:
        """
        加载轨迹数据（复用Aeon generators的数据加载逻辑）

        Args:
            trajectory_file: 轨迹文件路径

        Returns:
            轨迹数据字典
        """
        try:
            with open(trajectory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载轨迹文件失败 {trajectory_file}: {e}")
            return {}

    def _load_traj_pool(self, workspace_dir: Path, instance_name: Optional[str] = None) -> Dict[str, Any]:
        """
        加载工作目录下的 traj.pool 文件。

        参数:
            workspace_dir: 工作空间根目录路径（包含 iteration_* 目录的上层目录）
            instance_name: 若提供，则返回该实例的池数据；否则返回整个池数据映射

        返回:
            - 当提供 instance_name 时：返回该实例的字典数据（格式通常为 {"1": {...}, "2": {...}, ...}）
            - 未提供 instance_name 时：返回 {instance_name: instance_data} 的完整映射
            - 发生错误或未找到：返回 {}
        """
        traj_pool_file = workspace_dir / "traj.pool"

        if not traj_pool_file.exists():
            self.logger.warning(f"traj.pool文件不存在: {traj_pool_file}")
            return {}

        try:
            with open(traj_pool_file, "r", encoding="utf-8") as f:
                pool_data = json.load(f)

            if not isinstance(pool_data, dict):
                self.logger.error(f"traj.pool 格式不正确（期望为字典）: {traj_pool_file}")
                return {}

            if instance_name is None:
                # 返回完整的池数据映射
                return pool_data

            instance_data = pool_data.get(instance_name)
            if isinstance(instance_data, dict):
                return instance_data

            self.logger.warning(f"轨迹池中未找到实例 {instance_name}")
            return {}

        except Exception as e:
            self.logger.error(f"加载traj.pool失败 {traj_pool_file}: {e}")
            return {}

    def _process_single_instance(self, instance_info: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """
        处理单个实例，优先生成新的初始代码文本；若生成失败，回退到上一迭代提交代码。

        Args:
            instance_info: 实例信息字典

        Returns:
            (instance_name, initial_code_text) 或 None 表示处理失败
        """
        instance_name = instance_info["instance_name"]
        try:
            # 加载轨迹数据与问题陈述
            trajectory_data = self._load_trajectory_data(instance_info["trajectory_file"])
            problem_statement = instance_info.get("problem_description", "")

            # 通过子类逻辑生成新的初始代码文本，并用统一提取助手得到纯代码
            generated_code = self._generate_content(instance_info, problem_statement, trajectory_data)
            code_text = self._extract_code_text(generated_code)
            if isinstance(code_text, str) and code_text.strip():
                return (instance_name, code_text)
            # 若子类未按要求输出或为空，则通过统一重试助手强制要求```py```封装
            def _builder(_attempt: int):
                # 子类通常使用内部拼接prompt；此处无法复用其私有方法，保底采用问题陈述+轨迹摘要短提示
                base_prompt = (
                    "You previously returned content without a proper ```py fenced code block. "
                    "Please regenerate the initial Python solution and wrap it strictly within ```py ... ``` with ONLY Python code.\n\n"
                    f"Problem:\n{problem_statement}\n"
                )
                # 引导简短生成，避免超长
                system_prompt = (
                    "You are a coding agent. Return ONLY Python code within a fenced block ```py ... ```. "
                    "No prose, no backticks outside the fence."
                )
                return base_prompt, system_prompt

            retry_code = self._require_py_block_with_retry(_builder, max_retries=2)
            if isinstance(retry_code, str) and retry_code.strip():
                return (instance_name, retry_code)

            # 生成失败则回退到上一迭代提交代码
            self.logger.warning(f"{instance_name}: 生成初始代码失败，尝试回退到上一迭代提交代码")
            fallback_code = self._extract_initial_code(
                instance_info["instance_dir"], instance_name, instance_info["trajectory_file"]
            )
            if isinstance(fallback_code, str) and fallback_code.strip():
                return (instance_name, fallback_code)

            self.logger.warning(f"跳过 {instance_name}: 未能生成或提取初始代码")
            return None

        except Exception as e:
            self.logger.error(f"处理实例 {instance_name} 时出错: {e}")
            return None

    @abc.abstractmethod
    def get_name(self) -> str:
        """获取算子名称"""
        pass

    @abc.abstractmethod
    def _generate_content(
        self, instance_info: Dict[str, Any], problem_statement: str, trajectory_data: Dict[str, Any]
    ) -> str:
        """
        生成内容（子类实现核心逻辑）

        Args:
            instance_info: 实例信息
            problem_statement: 问题陈述
            trajectory_data: 轨迹数据

        Returns:
            生成的内容字符串
        """
        pass

    @abc.abstractmethod
    def process(self, workspace_dir: str, current_iteration: int, num_workers: int = 1) -> Optional[Dict[str, str]]:
        """
        处理算子逻辑的主入口方法

        Args:
            workspace_dir: 工作目录路径
            current_iteration: 当前迭代号
            num_workers: 并发worker数量

        Returns:
            算子返回的参数字典，如 {'instance_templates_dir': 'path'} 或 None表示失败
        """
        pass


class TemplateOperator(BaseOperator):
    """
    模板算子基类，用于为下一次 PerfAgent 运行生成初始代码目录
    返回 initial_code_dir 参数
    """

    def _create_output_dir(self, workspace_dir: Path, current_iteration: int) -> Path:
        """
        创建输出目录

        Args:
            workspace_dir: 工作目录路径
            current_iteration: 当前迭代号

        Returns:
            输出目录路径
        """
        # 输出到当前迭代的初始代码目录
        output_dir = workspace_dir / f"iteration_{current_iteration}" / "initial_code"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"创建输出目录: {output_dir}")
        return output_dir

    # 不再生成 YAML 系统模板，保留方法以兼容旧子类但不使用
    def _create_yaml_content(self, strategy_content: str) -> str:
        """
        创建YAML格式的系统提示内容（复用Aeon generators的格式）

        Args:
            strategy_content: 策略内容文本

        Returns:
            YAML格式的配置内容
        """
        # 创建YAML结构
        yaml_content = {"prompts": {"additional_requirements": strategy_content}}

        return yaml.dump(yaml_content, default_flow_style=False, allow_unicode=True, width=1000)

    # 不再生成 YAML 系统模板，保留方法以兼容旧子类但不使用
    def _save_instance_template(self, instance_name: str, content: str, output_dir: Path) -> None:
        """
        保存实例模板文件

        Args:
            instance_name: 实例名称
            content: 生成的内容
            output_dir: 输出目录
        """
        yaml_content = self._create_yaml_content(content)
        output_file = output_dir / f"{instance_name}.yaml"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        self.logger.debug(f"保存模板文件: {output_file}")

    def _extract_initial_code(self, instance_dir: Path, instance_name: str, trajectory_file: Path) -> Optional[str]:
        """
        从上一迭代提取提交代码，作为初始代码。

        优先读取 <instance_dir>/<instance_name>.pred；若无则从 .tra JSON 的 info/metadata.submission 中提取。
        """
        try:
            pred_file = instance_dir / f"{instance_name}.pred"
            if pred_file.exists():
                code = pred_file.read_text(encoding="utf-8")
                if isinstance(code, str) and code.strip():
                    return code
        except Exception as e:
            self.logger.warning(f"读取 .pred 失败 {pred_file if 'pred_file' in locals() else ''}: {e}")

        try:
            if trajectory_file and trajectory_file.exists():
                with open(trajectory_file, "r", encoding="utf-8") as tf:
                    traj_json = json.load(tf)
                info = traj_json.get("info") or traj_json.get("metadata") or {}
                submission = info.get("submission") or ""
                if isinstance(submission, str) and submission.strip():
                    return submission
        except Exception as e:
            self.logger.warning(f"从轨迹读取提交代码失败 {trajectory_file}: {e}")

        return None

    def _save_initial_code(self, instance_name: str, code_text: str, output_dir: Path) -> Optional[Path]:
        """
        将初始代码写入输出目录（使用 .py 扩展名）。返回写入文件路径或 None。
        """
        try:
            if not code_text or not isinstance(code_text, str) or not code_text.strip():
                return None
            output_file = output_dir / f"{instance_name}.py"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(code_text)
            self.logger.debug(f"保存初始代码: {output_file}")
            return output_file
        except Exception as e:
            self.logger.warning(f"写入初始代码失败: {e}")
            return None

    @abc.abstractmethod
    def get_strategy_prefix(self) -> str:
        """获取策略前缀标识（如 'ALTERNATIVE SOLUTION STRATEGY'）"""
        pass

    def process(self, workspace_dir: str, current_iteration: int, num_workers: int = 1) -> Optional[Dict[str, str]]:
        """
        处理模板算子逻辑（仅生成初始代码目录）

        Args:
            workspace_dir: 工作目录路径
            current_iteration: 当前迭代号
            num_workers: 并发worker数量

        Returns:
            {'initial_code_dir': 'path'} 或 None 表示失败
        """
        workspace_path = Path(workspace_dir)

        self.logger.info(f"开始处理 {self.get_name()} 算子")
        self.logger.info(f"工作目录: {workspace_path}")
        self.logger.info(f"当前迭代: {current_iteration}")
        self.logger.info(f"并发数: {num_workers}")

        # 发现实例
        instances = self._discover_instances(workspace_path, current_iteration)
        if not instances:
            self.logger.warning("未找到可处理的实例")
            return None

        # 创建输出目录（初始代码目录）
        output_dir = self._create_output_dir(workspace_path, current_iteration)

        # 并行处理实例
        processed_count = 0
        failed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_instance = {
                executor.submit(self._process_single_instance, instance_info): instance_info
                for instance_info in instances
            }

            # 收集结果
            for future in concurrent.futures.as_completed(future_to_instance):
                instance_info = future_to_instance[future]
                instance_name = instance_info["instance_name"]
                try:
                    result = future.result()
                    if result is not None:
                        # 返回值为 (instance_name, initial_code_text)
                        if isinstance(result, tuple) and len(result) >= 2:
                            name, code_text = result[0], result[1]
                            self._save_initial_code(name, code_text, output_dir)
                        else:
                            failed_count += 1
                            self.logger.warning(f"处理实例失败: 返回值格式不正确 {result}")
                        processed_count += 1
                        self.logger.debug(f"成功处理实例: {instance_name}")
                    else:
                        failed_count += 1
                        self.logger.warning(f"处理实例失败: {instance_name}")
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"处理实例 {instance_name} 时出现异常: {e}")

        self.logger.info(f"处理完成: 成功 {processed_count}, 失败 {failed_count}")

        if processed_count == 0:
            self.logger.error("没有成功处理任何实例")
            return None

        # 返回 initial_code_dir 参数
        try:
            any_code_files = any((output_dir.glob("*.py")))
        except Exception:
            any_code_files = False
        if not any_code_files:
            self.logger.error("未生成任何初始代码文件")
            return None
        return {"initial_code_dir": str(output_dir)}


class EnhanceOperator(BaseOperator):
    """
    增强算子基类，用于生成历史增强配置
    返回 enhance_history_filter_json 参数
    """

    def process(self, workspace_dir: str, current_iteration: int, num_workers: int = 1) -> Optional[Dict[str, str]]:
        """
        处理增强算子逻辑（未开发）

        Args:
            workspace_dir: 工作目录路径
            current_iteration: 当前迭代号
            num_workers: 并发worker数量

        Returns:
            {'enhance_history_filter_json': 'path'} 或 None表示失败
        """
        # TODO: 此类型算子还未开发完成
        self.logger.warning("EnhanceOperator 类型算子还未开发完成")
        return None
