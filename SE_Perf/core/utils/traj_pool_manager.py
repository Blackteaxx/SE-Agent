#!/usr/bin/env python3
"""
Trajectory Pool Manager (Label-based)

管理一个以“标签”为键的轨迹池。每个轨迹都是一个独立的实体，包含了执行摘要、
性能数据、代码路径等元信息。
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from core.utils.se_logger import get_se_logger


class TrajPoolManager:
    """
    轨迹池管理器 (基于标签)。
    负责加载、保存、查询和修改存储在 traj.pool 文件中的轨迹数据。
    轨迹池是一个以字符串标签为键的字典。
    """

    def __init__(self, pool_path: str, llm_client=None, num_workers: int | None = None):
        """
        初始化轨迹池管理器。

        Args:
            pool_path: traj.pool 文件路径。
            llm_client: LLM 客户端实例，用于轨迹总结。
            num_workers: 并行生成总结的并发数。
        """
        self.pool_path = Path(pool_path)
        self.llm_client = llm_client
        # 并发控制（来自SE配置）；为空则使用默认策略
        self.num_workers = num_workers
        self.logger = get_se_logger("traj_pool", emoji="🏊")

    def initialize_pool(self) -> None:
        """初始化轨迹池文件。如果文件不存在，则创建一个空的 JSON 对象。"""
        try:
            # 确保目录存在
            self.pool_path.parent.mkdir(parents=True, exist_ok=True)

            # 如果文件不存在，创建空的轨迹池
            if not self.pool_path.exists():
                with open(self.pool_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
                self.logger.info(f"初始化空的轨迹池: {self.pool_path}")
            else:
                self.logger.info(f"轨迹池已存在: {self.pool_path}")
        except Exception as e:
            self.logger.error(f"初始化轨迹池失败: {e}")
            raise

    def load_pool(self) -> dict[str, Any]:
        """从文件加载整个轨迹池。"""
        try:
            if not self.pool_path.exists():
                self.logger.warning("轨迹池文件不存在，返回空池")
                return {}
            with open(self.pool_path, encoding="utf-8") as f:
                pool_data = json.load(f)
            self.logger.debug(f"加载了 {len(pool_data)} 条轨迹")
            return pool_data
        except Exception as e:
            self.logger.error(f"加载轨迹池失败: {e}")
            return {}

    def save_pool(self, pool_data: dict[str, Any]) -> None:
        """将轨迹池数据完整保存到文件。"""
        try:
            with open(self.pool_path, "w", encoding="utf-8") as f:
                json.dump(pool_data, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"保存了 {len(pool_data)} 条轨迹到轨迹池")
        except Exception as e:
            self.logger.error(f"保存轨迹池失败: {e}")
            raise

    def get_instance(self, instance_name: str) -> dict[str, Any] | None:
        pool_data = self.load_pool()
        return pool_data.get(instance_name)

    def get_trajectory(self, label: str, instance_name: str | None = None) -> dict[str, Any] | None:
        pool_data = self.load_pool()
        if instance_name:
            entry = pool_data.get(instance_name)
            if isinstance(entry, dict):
                if label in entry and isinstance(entry[label], dict):
                    return entry[label]
                for subkey, subval in entry.items():
                    if subkey == "problem":
                        continue
                    if isinstance(subval, dict) and subval.get("label") == label:
                        return subval
            return None
        for inst_name, entry in pool_data.items():
            if isinstance(entry, dict):
                if label in entry and isinstance(entry[label], dict):
                    return entry[label]
                for subkey, subval in entry.items():
                    if subkey == "problem":
                        continue
                    if isinstance(subval, dict) and subval.get("label") == label:
                        return subval
        return None

    def get_all_trajectories(self) -> dict[str, Any]:
        """获取池中所有的轨迹。"""
        return self.load_pool()

    def get_all_labels(self, instance_name: str | None = None) -> list[str]:
        pool_data = self.load_pool()
        labels: list[str] = []
        target_instance = instance_name if instance_name else next(iter(pool_data.keys()), None)

        if target_instance:
            entry = pool_data.get(target_instance) or {}
            if isinstance(entry, dict):
                for subkey, subval in entry.items():
                    if subkey == "problem":
                        continue
                    if isinstance(subval, dict):
                        if subval.get("label"):
                            labels.append(str(subval.get("label")))
                        else:
                            labels.append(str(subkey))

        return labels

    def add_or_update_instance(self, instance_name: str, entry: dict[str, Any]) -> None:
        pool_data = self.load_pool()
        inst_key = str(instance_name)
        existing = pool_data.get(inst_key) or {}
        # 保持顶层 problem
        problem_text = entry.get("problem") or existing.get("problem")
        merged = {**existing}
        if problem_text is not None:
            merged["problem"] = problem_text
        # 将本次迭代的 label 作为子键，保存条目内容
        iter_label = entry.get("label")
        if not iter_label:
            raise ValueError("缺少 label 用于实例条目的子键")
        detail = entry.copy()
        detail.pop("problem", None)
        merged[str(iter_label)] = detail
        pool_data[inst_key] = merged
        self.save_pool(pool_data)
        self.logger.info(f"已更新实例 '{instance_name}' 的条目: {iter_label}")

    def add_trajectory(self, label: str, traj_info: dict[str, Any], instance_name: str | None = None) -> None:
        pool_data = self.load_pool()
        inst_name = str(instance_name or traj_info.get("instance_name") or "")
        if not inst_name:
            raise ValueError("缺少 instance_name，无法添加轨迹")
        entry = {
            "problem": traj_info.get("problem_description") or traj_info.get("problem_statement"),
            "label": label,
            "summary": traj_info.get("summary") or {},
            "performance": traj_info.get("performance"),
            "source_dir": traj_info.get("source_dir"),
            "code": traj_info.get("patch_content") or traj_info.get("content"),
            "content": traj_info.get("content"),
            "iteration": traj_info.get("iteration"),
        }
        self.add_or_update_instance(inst_name, entry)

    def relabel(self, old_label: str, new_label: str, instance_name: str | None = None) -> None:
        pool_data = self.load_pool()
        if instance_name:
            if instance_name not in pool_data:
                raise ValueError(f"实例 '{instance_name}' 不存在，无法重命名标签。")
            # 查找该实例的所有子键，更新匹配的旧标签子键为新标签
            inst_entry = pool_data[instance_name]
            if old_label in inst_entry:
                old_entry = inst_entry.get(old_label)
                if str(new_label) not in inst_entry:
                    new_entry = old_entry.copy() if isinstance(old_entry, dict) else old_entry
                    if isinstance(new_entry, dict):
                        new_entry["label"] = new_label
                        src = new_entry.get("source_entry_labels")
                        if isinstance(src, list):
                            if old_label not in src:
                                src.append(old_label)
                            new_entry["source_entry_labels"] = src
                        else:
                            new_entry["source_entry_labels"] = [old_label]
                    inst_entry[str(new_label)] = new_entry
                inst_entry["label"] = new_label
            else:
                # 若未找到子键，回退到设置顶层当前标签
                inst_entry["label"] = new_label
        else:
            target_inst = None
            for inst_name, entry in pool_data.items():
                if isinstance(entry, dict) and entry.get("label") == old_label:
                    target_inst = inst_name
                    break
            if target_inst is None:
                raise ValueError(f"标签 '{old_label}' 不存在，无法重命名。")
            # 更新顶层当前标签映射，同时若存在子键也更新子键名
            inst_entry = pool_data[target_inst]
            if old_label in inst_entry:
                old_entry = inst_entry.get(old_label)
                if str(new_label) not in inst_entry:
                    new_entry = old_entry.copy() if isinstance(old_entry, dict) else old_entry
                    if isinstance(new_entry, dict):
                        new_entry["label"] = new_label
                        src = new_entry.get("source_entry_labels")
                        if isinstance(src, list):
                            if old_label not in src:
                                src.append(old_label)
                            new_entry["source_entry_labels"] = src
                        else:
                            new_entry["source_entry_labels"] = [old_label]
                    inst_entry[str(new_label)] = new_entry
            inst_entry["label"] = new_label
        self.save_pool(pool_data)
        self.logger.info(f"已重命名标签 '{old_label}' 为 '{new_label}'。")

    def delete_trajectories(self, labels: list[str], instance_name: str | None = None) -> None:
        pool_data = self.load_pool()
        deleted_count = 0
        if instance_name:
            if instance_name in pool_data:
                inst_entry = pool_data[instance_name]
                # 删除匹配的子键，不删除整个实例
                for lb in labels:
                    if lb in inst_entry:
                        del inst_entry[lb]
                        deleted_count += 1
        else:
            to_delete = []
            for inst_name, entry in pool_data.items():
                if isinstance(entry, dict):
                    for lb in labels:
                        if lb in entry:
                            to_delete.append((inst_name, lb))
            for inst_name, lb in to_delete:
                try:
                    del pool_data[inst_name][lb]
                    deleted_count += 1
                    self.logger.debug(f"已从实例 '{inst_name}' 删除子条目 '{lb}'。")
                except Exception:
                    pass
        if deleted_count > 0:
            self.save_pool(pool_data)
        self.logger.info(f"从轨迹池中删除了 {deleted_count} 条轨迹。")

    def summarize_trajectory(
        self,
        trajectory_content: str,
        patch_content: str,
        iteration: int,
        label: str,
        problem_description: str | None = None,
    ) -> dict[str, Any]:
        """
        使用 LLM (或备用方法) 总结单条轨迹的内容。

        Args:
            trajectory_content: .tra 文件内容。
            patch_content: .patch/.pred 文件内容或 "FAILED_NO_PATCH"。
            iteration: 迭代号 (用于上下文)。
            label: 轨迹标签 (用于日志)。
            problem_description: 问题描述。

        Returns:
            轨迹总结字典。
        """
        from .llm_client import TrajectorySummarizer
        from .traj_summarizer import TrajSummarizer

        summarizer = TrajSummarizer()

        # 检查是否为失败实例
        is_failed = patch_content == "FAILED_NO_PATCH"

        try:
            if self.llm_client:
                traj_summarizer = TrajectorySummarizer(self.llm_client)
                summary = traj_summarizer.summarize_trajectory(
                    trajectory_content, patch_content, iteration, problem_description=problem_description
                )
                # 为失败实例添加特殊标记
                if is_failed:
                    summary["strategy_status"] = "FAILED"
                    summary["failure_reason"] = "No patch/prediction generated"
                self.logger.debug(f"LLM 轨迹总结 (标签 '{label}'): {summary.get('approach_summary', 'N/A')}")
                return summary
            else:
                self.logger.info(f"未配置 LLM 客户端，使用备用总结 (标签 '{label}')")
                summary = summarizer.create_fallback_summary(trajectory_content, patch_content, iteration)
                self.logger.debug(f"备用轨迹总结 (标签 '{label}'): {summary.get('approach_summary', 'N/A')}")
                return summary
        except Exception as e:
            self.logger.error(f"轨迹总结失败 (标签 '{label}'): {e}")
            return {
                "error": "summarization_failed",
                "details": str(e),
                "iteration": iteration,
                "label": label,
            }

    def summarize_and_add_trajectories(
        self, trajectories_to_process: list[dict[str, Any]], num_workers: int | None = None
    ) -> int:
        """
        并行生成多条轨迹的总结，并一次性将它们作为新条目添加到轨迹池中。

        Args:
            trajectories_to_process: 待处理轨迹信息的列表。每个元素是一个字典，包含:
                - "label": str
                - "instance_name": str
                - "problem_description": str
                - "trajectory_content": str
                - "patch_content": str
                - "iteration": int
                - "performance": float | None
                - "source_dir": str
            num_workers: 并发数。

        Returns:
            成功处理并添加的轨迹数量。
        """
        if not trajectories_to_process:
            return 0

        def _summarize_item(item: dict[str, Any]) -> dict[str, Any] | None:
            """线程工作函数：总结轨迹并构建完整的 TrajectoryInfo 对象。"""
            try:
                summary = self.summarize_trajectory(
                    trajectory_content=item["trajectory_content"],
                    patch_content=item["patch_content"],
                    iteration=item["iteration"],
                    label=item["label"],
                    problem_description=item.get("problem_description"),
                )

                # 在总结对象中附加来源标签，便于后续分析
                try:
                    src_labels = item.get("source_entry_labels")
                    if src_labels is not None:
                        summary["source_entry_labels"] = list(src_labels)
                except Exception:
                    pass

                # 构建完整的轨迹信息对象
                traj_info = {
                    "label": item["label"],
                    "instance_name": item["instance_name"],
                    "iteration": item["iteration"],
                    "performance": item.get("performance"),
                    "source_dir": item.get("source_dir"),
                    "summary": summary,
                    "problem_description": item.get("problem_description"),
                    "code": item["patch_content"],
                    "source_entry_labels": item.get("source_entry_labels"),
                    "operator_name": item.get("operator_name"),
                }
                return traj_info
            except Exception as e:
                self.logger.error(f"并行轨迹总结任务失败 (标签 '{item.get('label')}'): {e}")
                return None

        try:
            cfg_workers = num_workers if num_workers is not None else self.num_workers
            max_workers = (
                max(1, int(cfg_workers)) if cfg_workers is not None else max(1, min(8, (os.cpu_count() or 4) * 2))
            )
            self.logger.debug(f"并行轨迹总结并发数: {max_workers}")

            newly_completed_trajectories: dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_label = {
                    executor.submit(_summarize_item, item): item["label"] for item in trajectories_to_process
                }
                for future in as_completed(future_to_label):
                    label = future_to_label[future]
                    try:
                        result = future.result()
                        if result:
                            inst_name = result.get("instance_name")
                            if inst_name:
                                newly_completed_trajectories[inst_name] = result
                    except Exception as e:
                        self.logger.error(f"获取总结结果失败 (标签 '{label}'): {e}")

            if not newly_completed_trajectories:
                self.logger.warning("没有成功生成任何轨迹总结。")
                return 0

            written = 0
            for inst_name, res in newly_completed_trajectories.items():
                try:
                    entry = {
                        "problem": res.get("problem_description"),
                        "label": res.get("label"),
                        "summary": res.get("summary"),
                        "performance": res.get("performance"),
                        "source_dir": res.get("source_dir"),
                        "code": res.get("code"),
                        "iteration": res.get("iteration"),
                        "source_entry_labels": res.get("source_entry_labels"),
                        "operator_name": res.get("operator_name"),
                    }
                    self.add_or_update_instance(inst_name, entry)
                    written += 1
                except Exception as we:
                    self.logger.error(f"写入轨迹池失败: 实例 '{inst_name}' 标签 '{res.get('label')}': {we}")
            self.logger.info(f"成功并行生成并向轨迹池添加了 {written} 条实例-迭代条目。")
            return written

        except Exception as e:
            self.logger.error(f"并行生成与批量写入轨迹总结失败: {e}")
            raise

    def get_pool_stats(self) -> dict[str, Any]:
        """获取轨迹池的统计信息。"""
        try:
            pool_data = self.load_pool()
            stats = {
                "total_trajectories": len(pool_data),
                "labels": self.get_all_labels(),
            }
            self.logger.debug(f"轨迹池统计: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"获取轨迹池统计失败: {e}")
            return {"total_trajectories": 0, "labels": []}
