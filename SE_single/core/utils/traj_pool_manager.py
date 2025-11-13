#!/usr/bin/env python3
"""
Trajectory Pool管理器
用于管理多迭代执行中每个实例的轨迹总结
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from core.utils.se_logger import get_se_logger


class TrajPoolManager:
    """轨迹池管理器"""

    def __init__(self, pool_path: str, llm_client=None, num_workers: int | None = None):
        """
        初始化轨迹池管理器

        Args:
            pool_path: traj.pool文件路径
            llm_client: LLM客户端实例，用于轨迹总结
            num_workers: 并行生成总结的并发数（来自SE配置）
        """
        self.pool_path = Path(pool_path)
        self.llm_client = llm_client
        # 并发控制（来自SE配置）；为空则使用默认策略
        self.num_workers = num_workers
        self.logger = get_se_logger("traj_pool", emoji="🏊")

    def initialize_pool(self) -> None:
        """初始化轨迹池文件"""
        try:
            # 确保目录存在
            self.pool_path.parent.mkdir(parents=True, exist_ok=True)

            # 如果文件不存在，创建空的轨迹池
            if not self.pool_path.exists():
                initial_pool = {}
                with open(self.pool_path, "w", encoding="utf-8") as f:
                    json.dump(initial_pool, f, ensure_ascii=False, indent=2)
                self.logger.info(f"初始化轨迹池: {self.pool_path}")
            else:
                self.logger.info(f"轨迹池已存在: {self.pool_path}")

        except Exception as e:
            self.logger.error(f"初始化轨迹池失败: {e}")
            raise

    def load_pool(self) -> dict[str, Any]:
        """加载轨迹池数据"""
        try:
            if not self.pool_path.exists():
                self.logger.warning("轨迹池文件不存在，返回空池")
                return {}

            with open(self.pool_path, encoding="utf-8") as f:
                pool_data = json.load(f)

            self.logger.debug(f"加载轨迹池: {len(pool_data)} 个实例")
            return pool_data

        except Exception as e:
            self.logger.error(f"加载轨迹池失败: {e}")
            return {}

    def save_pool(self, pool_data: dict[str, Any]) -> None:
        """保存轨迹池数据"""
        try:
            with open(self.pool_path, "w", encoding="utf-8") as f:
                json.dump(pool_data, f, ensure_ascii=False, indent=2)

            self.logger.debug(f"保存轨迹池: {len(pool_data)} 个实例")

        except Exception as e:
            self.logger.error(f"保存轨迹池失败: {e}")
            raise

    def summarize_trajectory(
        self, trajectory_content: str, patch_content: str, iteration: int, problem_description: str | None = None
    ) -> dict[str, Any]:
        """
        总结轨迹内容

        Args:
            trajectory_content: .tra文件内容
            patch_content: .patch/.pred文件内容或"FAILED_NO_PATCH"
            iteration: 迭代次数

        Returns:
            轨迹总结字典
        """
        from .llm_client import TrajectorySummarizer
        from .traj_summarizer import TrajSummarizer

        summarizer = TrajSummarizer()

        # 检查是否为失败实例
        is_failed = patch_content == "FAILED_NO_PATCH"

        try:
            # 如果有LLM客户端，使用LLM进行总结
            if self.llm_client is not None:
                traj_summarizer = TrajectorySummarizer(self.llm_client)
                summary = traj_summarizer.summarize_trajectory(
                    trajectory_content, patch_content, iteration, problem_description=problem_description
                )
                # 为失败实例添加特殊标记
                if is_failed:
                    summary["strategy_status"] = "FAILED"
                    summary["failure_reason"] = (
                        "No patch/prediction generated (likely due to cost limit or early termination)"
                    )
                self.logger.debug(f"LLM轨迹总结 (迭代{iteration}): {summary.get('approach_summary', 'N/A')}")
                return summary
            else:
                # 没有LLM客户端时使用备用总结
                self.logger.info(f"未配置LLM客户端，使用备用总结 (迭代{iteration})")
                summary = summarizer.create_fallback_summary(trajectory_content, patch_content, iteration)
                self.logger.debug(f"备用轨迹总结 (迭代{iteration}): {summary.get('approach_summary', 'N/A')}")
                return summary

        except Exception as e:
            self.logger.error(f"轨迹总结失败: {e}")
            # 返回错误总结
            return {
                "error": "summarization_failed",
                "details": str(e),
                "iteration": iteration,
                "fallback_summary": f"Failed to summarize trajectory for iteration {iteration}",
            }

    def add_iteration_summary(
        self,
        instance_name: str,
        iteration: int,
        trajectory_content: str,
        patch_content: str,
        problem_description: str = None,
    ) -> None:
        """
        为指定实例添加迭代总结

        Args:
            instance_name: 实例名称
            iteration: 迭代次数
            trajectory_content: .tra文件内容
            patch_content: .patch/.pred文件内容 (预测结果)
            problem_description: 问题描述（可选）
        """
        try:
            # 加载现有池数据
            pool_data = self.load_pool()

            # 确保实例存在
            if instance_name not in pool_data:
                pool_data[instance_name] = {}

            # 如果是第一次添加这个实例，添加problem字段
            if "problem" not in pool_data[instance_name] and problem_description:
                pool_data[instance_name]["problem"] = problem_description

            # 生成轨迹总结
            summary = self.summarize_trajectory(trajectory_content, patch_content, iteration, problem_description)

            # 添加到池中
            pool_data[instance_name][str(iteration)] = summary

            # 保存池数据
            self.save_pool(pool_data)

            self.logger.info(f"添加轨迹总结: {instance_name} 迭代{iteration}")

        except Exception as e:
            self.logger.error(f"添加轨迹总结失败: {e}")
            raise

    def add_iteration_summaries_batch(self, items: list[dict[str, Any]], iteration: int) -> None:
        """
        批量添加迭代总结，并一次性写入轨迹池，避免并发写入产生竞争。

        Args:
            items: 每个元素包含 {"instance_name", "summary", "problem_description"}
            iteration: 迭代次数
        """
        try:
            pool_data = self.load_pool()

            for item in items:
                instance_name = item.get("instance_name")
                summary = item.get("summary")
                problem_description = item.get("problem_description")

                if not instance_name:
                    continue

                # 确保实例存在
                if instance_name not in pool_data:
                    pool_data[instance_name] = {}

                # 如果是第一次添加这个实例，添加problem字段
                if "problem" not in pool_data[instance_name] and problem_description:
                    pool_data[instance_name]["problem"] = problem_description

                # 添加总结
                pool_data[instance_name][str(iteration)] = summary

            # 一次性写入
            self.save_pool(pool_data)

            self.logger.info(f"批量添加轨迹总结: {len(items)} 个实例, 迭代{iteration}")

        except Exception as e:
            self.logger.error(f"批量添加轨迹总结失败: {e}")
            raise

    def summarize_and_add_iteration_batch(
        self, instance_data_list: list[Any], iteration: int, num_workers: int | None = None
    ) -> int:
        """
        并行生成多个实例的轨迹总结，并一次性写入到轨迹池文件。

        Args:
            instance_data_list: 由 TrajExtractor.extract_instance_data 返回的列表，
                元素形如 (instance_name, problem_description, trajectory_content, patch_content)
            iteration: 迭代次数
            num_workers: 并发数（优先于构造函数中的 num_workers）

        Returns:
            成功处理的实例数量
        """
        if not instance_data_list:
            return 0

        # 并行生成总结，避免并发写文件；统一批量写入
        def _summarize_item(item: Any) -> dict[str, Any]:
            instance_name, problem_description, trajectory_content, patch_content = item
            summary = self.summarize_trajectory(
                trajectory_content,
                patch_content,
                iteration,
                problem_description=problem_description or None,
            )
            return {
                "instance_name": instance_name,
                "summary": summary,
                "problem_description": problem_description or None,
            }

        try:
            # 优先使用传入的 num_workers，其次使用构造函数中的 num_workers；都没有时使用默认值
            cfg_workers = num_workers if num_workers is not None else self.num_workers
            if cfg_workers is not None:
                try:
                    max_workers = max(1, int(cfg_workers))
                except (TypeError, ValueError):
                    max_workers = 1
            else:
                max_workers = max(1, min(8, (os.cpu_count() or 4) * 2))

            self.logger.debug(f"并行轨迹总结并发数: {max_workers} (配置: {cfg_workers})")
            summary_items: list[dict[str, Any]] = []
            from concurrent.futures import as_completed

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_summarize_item, item) for item in instance_data_list]
                for f in as_completed(futures):
                    try:
                        summary_items.append(f.result())
                    except Exception as e:
                        self.logger.error(f"并行轨迹总结任务失败: {e}")

            # 批量写入，避免并发写文件导致竞争
            self.add_iteration_summaries_batch(summary_items, iteration=iteration)
            self.logger.info(f"成功并行生成并写入 {len(summary_items)} 个实例总结 (迭代{iteration})")
            return len(summary_items)

        except Exception as e:
            self.logger.error(f"并行生成与批量写入轨迹总结失败: {e}")
            raise

    def get_instance_summary(self, instance_name: str) -> dict[str, str] | None:
        """
        获取指定实例的所有迭代总结

        Args:
            instance_name: 实例名称

        Returns:
            实例的迭代总结字典，key为迭代次数，value为总结
        """
        try:
            pool_data = self.load_pool()
            return pool_data.get(instance_name)

        except Exception as e:
            self.logger.error(f"获取实例总结失败: {e}")
            return None

    def get_pool_stats(self) -> dict[str, Any]:
        """获取轨迹池统计信息"""
        try:
            pool_data = self.load_pool()

            total_instances = len(pool_data)
            total_iterations = sum(len(iterations) for iterations in pool_data.values())

            stats = {
                "total_instances": total_instances,
                "total_iterations": total_iterations,
                "instances": list(pool_data.keys()),
            }

            self.logger.debug(f"轨迹池统计: {stats}")
            return stats

        except Exception as e:
            self.logger.error(f"获取轨迹池统计失败: {e}")
            return {"total_instances": 0, "total_iterations": 0, "instances": []}
