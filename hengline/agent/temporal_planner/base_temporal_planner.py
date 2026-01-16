# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: 时序规划，负责将剧本按5秒粒度切分，估算动作时长
@Author: HengLine
@Time: 2025/10 - 2025/12
"""
from abc import abstractmethod
from datetime import datetime
from typing import Dict, Any

from hengline.agent.base_agent import BaseAgent
from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType, EstimationSource
from hengline.logger import info, warning, error


class BaseTemporalPlanner(BaseAgent):
    """时序规划"""

    def __init__(self):
        # 统计信息
        self.stats = {
            "total_estimations": 0,
            "successful_estimations": 0,
            "failed_estimations": 0,
            "avg_processing_time": 0.0
        }

    @abstractmethod
    def estimate_all_elements(self, script_data: UnifiedScript) -> Dict[str, DurationEstimation]:
        """
        估算所有元素（抽象方法）
        子类必须实现此方法，可以调用现有的工厂方法
        """
        pass

    def plan_timeline(self, script_data: UnifiedScript) -> Dict[str, DurationEstimation]:
        """
        规划剧本的时序分段
        
        Args:
            script_data: 结构化的剧本

        Returns:
            分段计划列表
        """
        start_time = datetime.now()
        info(f"开始估算剧本时长")

        try:
            # 2. 使用工厂方法估算所有元素
            estimations = self.estimate_all_elements(script_data)

            # 3. 验证和修复
            validated = self._validate_estimations(estimations)

            # 4. 计算统计信息
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["total_estimations"] = len(validated)
            self.stats["successful_estimations"] = len([e for e in validated.values() if e.confidence > 0.5])
            self.stats["failed_estimations"] = len([e for e in validated.values() if e.confidence <= 0.5])
            self.stats["avg_processing_time"] = processing_time / len(validated) if validated else 0

            info(f"完成估算 - 成功: {self.stats['successful_estimations']}, "
                 f"失败: {self.stats['failed_estimations']}, "
                 f"用时: {processing_time:.2f}秒")

            return validated

        except Exception as e:
            error(f"剧本估算失败: {str(e)}")
            raise

    def _validate_estimations(self, estimations: Dict[str, DurationEstimation]) -> Dict[str, DurationEstimation]:
        """
        验证估算结果的合理性
        """
        validated = {}

        for element_id, estimation in estimations.items():
            validated[element_id] = self._validate_single_estimation(estimation)

        return validated

    def _validate_single_estimation(self, estimation: DurationEstimation) -> DurationEstimation:
        """
        验证单个估算结果
        """
        # 检查时长是否合理
        if estimation.estimated_duration <= 0:
            warning(f"元素 {estimation.element_id} 时长无效: {estimation.estimated_duration}")
            estimation.estimated_duration = self._get_default_duration(estimation.element_type)
            estimation.confidence = max(estimation.confidence - 0.3, 0.1)
            estimation.source = EstimationSource.FALLBACK

        # 检查置信度
        if estimation.confidence < 0.1:
            estimation.confidence = 0.1
        elif estimation.confidence > 1.0:
            estimation.confidence = 1.0

        # 设置合理的范围
        base_duration = estimation.estimated_duration
        estimation.min_duration = base_duration * 0.7
        estimation.max_duration = base_duration * 1.5

        # 根据元素类型调整范围
        if estimation.element_type == ElementType.SCENE:
            estimation.max_duration = min(estimation.max_duration, 20.0)  # 场景最长20秒
        elif estimation.element_type == ElementType.SILENCE:
            estimation.max_duration = min(estimation.max_duration, 8.0)  # 沉默最长8秒

        return estimation

    def _get_default_duration(self, element_type: ElementType) -> float:
        """获取默认时长（用于错误恢复）"""
        defaults = {
            ElementType.SCENE: 4.0,
            ElementType.DIALOGUE: 2.5,
            ElementType.ACTION: 1.5,
            ElementType.SILENCE: 2.0,
            ElementType.TRANSITION: 1.0
        }
        return defaults.get(element_type, 2.0)

    def _create_fallback_estimation(self, element: Any) -> DurationEstimation:
        """
        创建降级估算（当主要估算方法失败时）
        """
        fallback_duration = self._get_fallback_duration(element)

        estimation = DurationEstimation(
            element_id=element.element_id,
            element_type=element.element_type,
            original_duration=element.original_duration,
            estimated_duration=fallback_duration,
            estimator_source=EstimationSource.FALLBACK,
            confidence=0.3,  # 低置信度
            emotional_weight=1.0,
            visual_complexity=1.0
        )

        return estimation

    def _get_fallback_duration(self, element: Any) -> float:
        """
        获取降级估算时长（基于简单规则）
        """
        if element.element_type == ElementType.SCENE:
            # 基于描述长度
            word_count = len(element.description.split())
            return min(word_count * 0.05, 10.0)

        elif element.element_type == ElementType.DIALOGUE:
            # 基于词数
            word_count = len(element.description.split())
            return word_count * 0.4 if word_count > 0 else 1.5

        elif element.element_type == ElementType.SILENCE:
            # 固定沉默时长
            return 2.5

        elif element.element_type == ElementType.ACTION:
            # 基于描述长度
            word_count = len(element.description.split())
            return min(word_count * 0.3, 5.0)

        return 2.0  # 默认
