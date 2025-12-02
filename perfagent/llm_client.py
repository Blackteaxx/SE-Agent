"""
LLM 客户端，支持多种模型和API端点
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from .utils.log import get_se_logger


class LLMClient:
    """LLM客户端，支持多种模型和API端点"""

    def __init__(
        self,
        model_config: dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.5,
        io_log_path: str | Path | None = None,
        log_inputs_outputs: bool = True,
        log_sanitize: bool = True,
        request_timeout: float = 60.0,
    ):
        """
        初始化LLM客户端

        Args:
            model_config: 模型配置字典，包含name, api_base, api_key等
            max_retries: 最大重试次数
            retry_delay: 每次重试的等待秒数
            io_log_path: LLM 输入/输出日志文件路径
            log_inputs_outputs: 是否记录原始输入输出
        """
        self.config = model_config
        # 统一使用文件日志（带 emoji），与 IO 日志同目录
        self.io_log_path = Path(io_log_path) if io_log_path else Path("./logs/llm_io.log")
        # Logger 名称增加任务名后缀（取日志目录名），避免并发任务冲突
        task_suffix = self.io_log_path.parent.name or "default"
        logger_name = f"perfagent.llm_client.{task_suffix}"
        get_se_logger(logger_name, self.io_log_path, emoji="🤖", also_stream=False)
        self.logger = logging.getLogger(logger_name)
        self.token_log_path = os.getenv("SE_TOKEN_LOG_PATH")
        self._token_lock = threading.Lock()

        # 优先使用配置中的增强参数
        self.max_retries = int(model_config.get("max_retries", max_retries))
        self.retry_delay = float(model_config.get("retry_delay", retry_delay))
        self.log_inputs_outputs = bool(model_config.get("log_inputs_outputs", log_inputs_outputs))
        self.log_sanitize = bool(model_config.get("log_sanitize", log_sanitize))
        self.request_timeout = float(model_config.get("request_timeout", request_timeout))

        # 验证必需的配置参数
        required_keys = ["name", "api_base", "api_key"]
        missing_keys = [key for key in required_keys if key not in model_config]
        if missing_keys:
            raise ValueError(f"缺少必需的配置参数: {missing_keys}")

        # 初始化OpenAI客户端，遵循api_test.py的工作模式
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["api_base"],
            timeout=self.request_timeout,
        )

        self.logger.info(f"初始化LLM客户端: {self.config['name']}")

    def _format_content_for_log(self, content: str | None, indent: int = 2) -> str:
        """将文本内容格式化为多行日志，保留真实换行并缩进。

        Args:
            content: 文本内容
            indent: 缩进空格数量

        Returns:
            友好的多行字符串，带缩进并保留换行
        """
        prefix = " " * indent
        if content is None:
            return f"{prefix}content: (None)"
        text = str(content)
        if text == "":
            return f"{prefix}content: (empty)"
        lines = text.splitlines() or [text]
        formatted = [f"{prefix}content:"]
        formatted.extend(f"{prefix}  {line}" for line in lines)
        return "\n".join(formatted)

    def _format_messages_for_log(self, messages: list[dict[str, str]], indent: int = 0) -> str:
        """将消息列表格式化为多行日志，保留真实换行并缩进内容。"""
        base_prefix = " " * indent
        out_lines = [f"{base_prefix}messages:"]
        for i, m in enumerate(messages, start=1):
            role = m.get("role", "unknown")
            out_lines.append(f"{base_prefix}  [{i}] role: {role}")
            out_lines.append(self._format_content_for_log(m.get("content"), indent=indent + 4))
        return "\n".join(out_lines)

    def call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int | None = None,
        usage_context: str | None = None,
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
        # 使用配置中的max_output_tokens作为默认值
        if max_tokens is None:
            max_tokens = self.config.get("max_output_tokens", 4000)

        attempt = 0
        last_err: Exception | None = None
        while attempt < self.max_retries:
            try:
                self.logger.debug(f"调用LLM: {len(messages)} 条消息, temp={temperature}, max_tokens={max_tokens}")

                # 记录原始输入
                if self.log_inputs_outputs:
                    try:
                        # 日志脱敏：移除可能的密钥与端点信息
                        model_name = self.config.get("name", "unknown")
                        api_base = self.config.get("api_base", "")
                        safe_api_base = "<redacted>" if self.log_sanitize and api_base else api_base
                        req_lines = [
                            "LLM Request:",
                            f"model: {model_name}",
                            f"temperature: {temperature}",
                            f"max_tokens: {max_tokens}",
                            f"api_base: {safe_api_base}",
                            self._format_messages_for_log(messages, indent=0),
                        ]
                        self.logger.info("\n".join(req_lines))
                    except Exception as log_e:
                        self.logger.error(f"记录请求失败: {log_e}")

                # 使用基本的OpenAI客户端调用，遵循api_test.py的工作模式
                # 不使用额外参数，避免服务器错误
                # 禁止思考，仅返回直接回答
                response = self.client.chat.completions.create(
                    model="/".join(self.config["name"].split("/")[1:]),
                    messages=messages,
                    temperature=temperature,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
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
                    try:
                        if self.token_log_path:
                            entry = {
                                "ts": time.time(),
                                "context": usage_context or "perfagent",
                                "model": "/".join(self.config["name"].split("/")[1:]),
                                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                                "total_tokens": getattr(response.usage, "total_tokens", None),
                                "messages_chars": sum(
                                    len(str(m.get("content", ""))) for m in messages if isinstance(m, dict)
                                ),
                            }
                            with self._token_lock:
                                with open(self.token_log_path, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    except Exception:
                        pass

                # 记录原始输出
                if self.log_inputs_outputs:
                    try:
                        usage = getattr(response, "usage", None)
                        usage_dict = None
                        if usage:
                            usage_dict = {
                                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                                "completion_tokens": getattr(usage, "completion_tokens", None),
                                "total_tokens": getattr(usage, "total_tokens", None),
                            }
                        # 响应日志脱敏：不记录原始头信息
                        resp_lines = [
                            "LLM Response:",
                            self._format_content_for_log(content, indent=0),
                            "usage: " + json.dumps(usage_dict, ensure_ascii=False),
                        ]
                        self.logger.info("\n".join(resp_lines))
                    except Exception as log_e:
                        self.logger.error(f"记录响应失败: {log_e}")

                return content

            except Exception as e:
                last_err = e
                self.logger.error(f"LLM调用失败: {e}; attempt={attempt + 1}/{self.max_retries}")
                if self.log_inputs_outputs:
                    self.logger.error(f"LLM 调用异常: {e}")
                attempt += 1
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    break

        # 重试后仍失败，抛出最后一个错误
        assert last_err is not None
        raise last_err

    def call_with_system_prompt(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int | None = None
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
