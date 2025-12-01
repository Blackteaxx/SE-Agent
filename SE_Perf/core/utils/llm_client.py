#!/usr/bin/env python3
"""
LLM客户端模块
为SE框架提供统一的LLM调用接口
"""

from typing import Any, Dict, List, Optional  # noqa: UP035
import re
import json
import time

from openai import OpenAI

from core.utils.se_logger import get_se_logger


class LLMClient:
    """LLM客户端，支持多种模型和API端点"""

    def __init__(self, model_config: Dict[str, Any]):
        """
        初始化LLM客户端

        Args:
            model_config: 模型配置字典，包含name, api_base, api_key等
        """
        self.config = model_config
        self.logger = get_se_logger("llm_client", emoji="🤖")

        # 验证必需的配置参数
        required_keys = ["name", "api_base", "api_key"]
        missing_keys = [key for key in required_keys if key not in model_config]
        if missing_keys:
            raise ValueError(f"缺少必需的配置参数: {missing_keys}")

        # 请求控制参数（带默认值）
        self.request_timeout: float = float(self.config.get("request_timeout", 600.0))
        self.max_retries: int = int(self.config.get("max_retries", 3))
        self.retry_delay: float = float(self.config.get("retry_delay", 1.5))

        # 初始化OpenAI客户端，遵循api_test.py的工作模式，并设置超时
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["api_base"],
            timeout=self.request_timeout,
        )

        self.logger.info(f"初始化LLM客户端: {self.config['name']}")

    def clean_think_tags(self, text: str) -> str:
        """移除 <think>...</think> 标签及其中内容，并去除首尾空白"""
        try:
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        except Exception:
            return text

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """
        调用LLM并返回响应内容

        Args:
            messages: 消息列表，每个消息包含role和content
            temperature: 温度参数，控制输出随机性
            max_tokens: 最大输出token数，None表示使用配置默认值

        Returns:
            LLM响应的文本内容

        Raises:
            Exception: LLM调用失败时抛出异常
        """
        # 使用配置中的 max_output_tokens 作为默认值
        if max_tokens is None:
            max_tokens = self.config.get("max_output_tokens", 4000)

        attempt = 0
        last_err: Optional[Exception] = None
        while attempt < self.max_retries:
            try:
                self.logger.debug(
                    f"调用LLM: {len(messages)} 条消息, temp={temperature}, max_tokens={max_tokens}"
                )

                # 规范化模型名：仅移除 openai/ 前缀，其他保持原样
                raw_name = self.config.get("name", "")
                if isinstance(raw_name, str) and raw_name.startswith("openai/"):
                    model_name = raw_name.split("/", 1)[1]
                else:
                    model_name = raw_name

                # 使用 OpenAI 客户端调用，传入必需的参数
                # 生成可选的 extra_body，用于控制是否启用思考模板
                extra_body: Dict[str, Any] = {}
                if enable_thinking is not None:
                    extra_body = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}

                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **({"extra_body": extra_body} if extra_body else {}),
                )

                # 提取响应内容
                content = response.choices[0].message.content

                # 记录使用情况
                if getattr(response, "usage", None):
                    self.logger.debug(
                        f"Token使用: 输入={getattr(response.usage, 'prompt_tokens', '未知')}, "
                        f"输出={getattr(response.usage, 'completion_tokens', '未知')}, "
                        f"总计={getattr(response.usage, 'total_tokens', '未知')}"
                    )

                return content

            except Exception as e:
                last_err = e
                attempt += 1
                self.logger.warning(
                    f"LLM调用失败: {e}; attempt={attempt}/{self.max_retries}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    break

        assert last_err is not None
        raise last_err

    def call_with_system_prompt(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: Optional[int] = None
    ) -> str:
        """
        使用系统提示词和用户提示词调用LLM

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 温度参数
            max_tokens: 最大输出token数

        Returns:
            LLM响应的文本内容
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        return self.call_llm(messages, temperature, max_tokens)

    @classmethod
    def from_se_config(cls, se_config: Dict[str, Any], use_operator_model: bool = False) -> "LLMClient":
        """
        从SE框架配置创建LLM客户端

        Args:
            se_config: SE框架配置字典
            use_operator_model: 是否使用operator_models配置而不是主模型配置

        Returns:
            LLM客户端实例
        """
        if use_operator_model and "operator_models" in se_config:
            model_config = se_config["operator_models"]
        else:
            model_config = se_config["model"]

        return cls(model_config)


class TrajectorySummarizer:
    """专门用于轨迹总结的LLM客户端包装器"""

    def __init__(self, llm_client: LLMClient):
        """
        初始化轨迹总结器

        Args:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client
        self.logger = get_se_logger("traj_summarizer", emoji="📊")

    def summarize_trajectory(
        self, trajectory_content: str, patch_content: str, iteration: int, problem_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        使用LLM总结轨迹内容

        Args:
            trajectory_content: .tra文件内容
            patch_content: .patch/.pred文件内容 (预测结果)
            iteration: 迭代次数
            problem_description: 问题描述（可选，将并入提示词）

        Returns:
            轨迹总结字典
        """
        from .traj_summarizer import TrajSummarizer

        summarizer = TrajSummarizer()

        # 获取提示词
        system_prompt = summarizer.get_system_prompt()
        user_prompt = summarizer.format_user_prompt(trajectory_content, patch_content, problem_description)

        self.logger.info(f"开始LLM轨迹总结 (迭代{iteration})")
        self.logger.debug(f"LLM系统提示词 (迭代{iteration}):\n{system_prompt}")
        self.logger.debug(f"LLM用户提示词 (迭代{iteration}):\n{user_prompt}")

        # 重试机制：解析失败或调用失败时重试，总次数3次
        last_error: Optional[str] = None
        for attempt in range(1, 4):
            try:
                response = self.llm_client.call_with_system_prompt(
                    system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.6, max_tokens=10000
                )
                self.logger.debug(f"LLM原始响应 (迭代{iteration}, 第{attempt}次):\n{response}")
                # 去除思考内容
                response = self.llm_client.clean_think_tags(response)
                self.logger.debug(f"LLM清理后响应 (迭代{iteration}, 第{attempt}次):\n{response}")

                # 仅执行字符串到JSON的解析，格式不正确/解析失败会抛异常
                summary = summarizer.parse_response(response)

                self.logger.info(f"LLM轨迹总结成功 (迭代{iteration}, 第{attempt}次)")
                return summary

            except json.JSONDecodeError as e:
                last_error = "json_decode_error"
                self.logger.warning(
                    f"LLM轨迹总结解析失败: JSON解析错误 (迭代{iteration}, 第{attempt}次): {e}"
                )
            except ValueError as e:
                last_error = "invalid_json_format"
                self.logger.warning(
                    f"LLM轨迹总结解析失败: 无有效JSON片段 (迭代{iteration}, 第{attempt}次): {e}"
                )
            except Exception as e:
                last_error = "llm_call_failed"
                self.logger.warning(f"LLM轨迹总结调用失败 (迭代{iteration}, 第{attempt}次): {e}")

        # 所有重试失败，返回备用总结
        if last_error:
            self.logger.error(f"LLM轨迹总结最终失败 (迭代{iteration}): {last_error}")
        return summarizer.create_fallback_summary(trajectory_content, patch_content, iteration)
