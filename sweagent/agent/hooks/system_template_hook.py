"""
System Template Hook - 为特定实例动态加载自定义的system_template

提供基于实例ID的自定义系统提示词注入功能。
从指定目录加载实例特定的YAML配置文件，动态修改agent的system_template。
"""

from pathlib import Path
from typing import Any

import yaml

from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.utils.log import get_logger


class SystemTemplateHook(AbstractAgentHook):
    """Hook that dynamically loads instance-specific system templates.

    This hook loads custom system_template configurations from YAML files
    based on instance IDs, allowing for per-instance customization of agent behavior.
    """

    def __init__(self, instance_id: str, instance_templates_dir: Path):
        """Initialize the system template hook.

        Args:
            instance_id: The instance identifier
            instance_templates_dir: Directory containing instance-specific template files
        """
        self.instance_id = instance_id
        self.instance_templates_dir = Path(instance_templates_dir)
        self.logger = get_logger("system-template", emoji="📝")
        self.template_loaded = False

    def on_init(self, *, agent) -> None:
        """Called when agent is initialized. Load and apply custom template if available.

        Args:
            agent: The agent instance being initialized
        """
        template_file = self.instance_templates_dir / f"{self.instance_id}.yaml"

        if not template_file.exists():
            self.logger.info(f"没有找到实例 {self.instance_id} 的自定义模板文件: {template_file}")
            return

        try:
            self.logger.info(f"正在加载实例 {self.instance_id} 的自定义模板: {template_file}")

            with open(template_file, encoding="utf-8") as f:
                template_config = yaml.safe_load(f)

            # 提取system_template
            system_template = self._extract_system_template(template_config)

            if system_template:
                # 更新agent的system_template
                if hasattr(agent, "templates") and hasattr(agent.templates, "system_template"):
                    original_template = agent.templates.system_template
                    agent.templates.system_template = system_template
                    self.template_loaded = True
                    self.logger.info(f"成功为实例 {self.instance_id} 应用自定义system_template")
                    self.logger.debug(f"原始模板长度: {len(original_template)}, 新模板长度: {len(system_template)}")
                else:
                    self.logger.warning("Agent没有templates.system_template属性，无法应用自定义模板")
            else:
                self.logger.warning(f"在模板文件中没有找到有效的system_template: {template_file}")

        except Exception as e:
            self.logger.error(f"加载实例 {self.instance_id} 的自定义模板时出错: {e}")

    def _extract_system_template(self, config: dict[str, Any]) -> str:
        """Extract system_template from loaded YAML configuration.

        Args:
            config: Loaded YAML configuration

        Returns:
            The system_template string if found, empty string otherwise
        """
        # 尝试多种可能的路径结构
        paths_to_try = [
            ["agent", "templates", "system_template"],
            ["templates", "system_template"],
            ["system_template"],
            ["agent", "system_template"],
        ]

        for path in paths_to_try:
            current = config
            try:
                for key in path:
                    current = current[key]
                if isinstance(current, str) and current.strip():
                    self.logger.debug(f"在路径 {' -> '.join(path)} 找到system_template")
                    return current.strip()
            except (KeyError, TypeError):
                continue

        self.logger.debug("在所有可能路径中都没有找到system_template")
        return ""

    def on_model_query(self, *, messages, instance_id: str, **kwargs) -> None:
        """Called before model query. Log template usage if loaded.
            A test interface.

        Args:
            messages: The conversation messages
            instance_id: The instance identifier
            **kwargs: Additional arguments
        """
        if self.template_loaded and len(messages) == 1:  # 只在第一次查询时记录
            self.logger.info(f"实例 {self.instance_id} 正在使用自定义system_template")
