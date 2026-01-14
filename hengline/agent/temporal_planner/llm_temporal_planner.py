# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: LLM + 规则约束实现的时序规划（负责将剧本按5秒粒度切分，估算动作时长，使用LangChain实现状态记忆）
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from datetime import datetime
from typing import List, Dict

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.base_temporal_planner import TemporalPlanner
from hengline.agent.temporal_planner.estimator.timeline_planner_factory import TimelinePlannerFactory
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, TimeSegment, ElementType, TimelinePlan, PacingAnalysis, ContinuityAnchor
from hengline.logger import error, debug, info, warning
from hengline.prompts.temporal_planner_prompt import PromptConfig


class LLMTemporalPlanner(TemporalPlanner):
    """ LLM 时长估算 """

    def __init__(self, llm_client, config: PromptConfig = None):
        """初始化时序规划智能体"""
        self.llm = llm_client
        self.config = config or PromptConfig()
        self.factory = TimelinePlannerFactory
        # 性能跟踪
        self.processing_stats = {
            "start_time": 0,
            "end_time": 0,
            "element_estimation_time": 0,
            "segmentation_time": 0,
            "analysis_time": 0
        }

        # 缓存
        self.element_cache = {}
        self.segment_cache = {}

    def plan_timeline(self, script_data: UnifiedScript) -> TimelinePlan | None:
        """
        规划剧本的时序分段

        Args:
            script_data: 结构化的剧本

        Returns:
            分段计划列表
        """
        self.processing_stats["start_time"] = datetime.now().second
        try:
            """创建完整的时序规划"""
            debug("开始创建时序规划...")

            # 1. 估算所有元素的时长
            debug("步骤1: 估算元素时长")
            start_est = datetime.now().second
            estimations = self.factory.estimate_script(script_data)
            self.processing_stats["element_estimation_time"] = (datetime.now().second - start_est)

            # 2. 组织估算结果
            debug("步骤2: 组织估算结果")
            organized = self._organize_estimations(estimations, script_data)

            # 3. 生成5秒片段
            debug("步骤3: 生成5秒片段")
            start_seg = datetime.now().second
            segments = self._generate_segments(organized)
            self.processing_stats["segmentation_time"] = (datetime.now().second - start_seg)
            debug(f"自适应分片完成: {len(segments)} 个片段")

            # 4. 分析节奏
            debug("步骤4: 分析节奏")
            start_ana = datetime.now().second
            pacing_analysis = self._analyze_pacing(segments, estimations)
            self.processing_stats["analysis_time"] = (datetime.now().second - start_ana)

            # 5. 生成连续性锚点
            debug("步骤5: 生成连续性锚点")
            continuity_anchors = self._generate_continuity_anchors(segments)
            debug(f"连续性锚点生成: {len(continuity_anchors)} 个锚点")

            # 6. 生成最终计划
            debug("步骤6: 生成最终计划")
            final_plan = self._create_final_plan(
                segments=segments,
                estimations=estimations,
                pacing_analysis=pacing_analysis,
                continuity_anchors=continuity_anchors,
                script_data=script_data
            )

            # 7. 显示错误摘要
            error_summary = self.factory.get_error_summary()
            if error_summary["estimators_with_errors"] > 0:
                warning(f"\n发现错误: {error_summary['estimators_with_errors']}个估算器有错误")

            info(f"时序规划完成！ 片段数: {len(segments)}")

            return final_plan

        except Exception as e:
            error(f"处理失败: {str(e)}")
            raise

    def _organize_estimations(self, estimations: Dict[str, DurationEstimation],
                              script_data: UnifiedScript) -> Dict[str, List]:
        """组织估算结果"""
        organized = {
            "scenes": [],
            "dialogues": [],
            "actions": []
        }

        # 按原始顺序组织场景
        for scene in script_data.scenes:
            scene_id = scene.scene_id
            if scene_id in estimations:
                organized["scenes"].append(estimations[scene_id])

        # 按原始顺序组织对话
        for dialogue in script_data.dialogues:
            dialogue_id = dialogue.dialogue_id
            if dialogue_id in estimations:
                organized["dialogues"].append(estimations[dialogue_id])

        # 按原始顺序组织动作
        for action in script_data.actions:
            action_id = action.action_id
            if action_id in estimations:
                organized["actions"].append(estimations[action_id])

        return organized

    def _generate_segments(self, organized: Dict[str, List]) -> List[TimeSegment]:
        """生成5秒片段（简化实现）"""
        segments = []
        current_segment = {
            "segment_id": "seg_001",
            "time_range": (0.0, 5.0),
            "duration": 5.0,
            "elements": [],
            "visual_summary": "",
            "start_anchor": {},
            "end_anchor": {}
        }

        current_time = 0.0

        # 简单的片段生成逻辑（实际需要更复杂的算法）
        for element_type, elements in organized.items():
            for element in elements:
                if current_time + element.estimated_duration <= 5.0:
                    current_segment["elements"].append(element)
                    current_time += element.estimated_duration
                else:
                    # 完成当前片段
                    segments.append(current_segment)

                    # 开始新片段
                    current_time = element.estimated_duration
                    segment_num = len(segments) + 1
                    current_segment = {
                        "segment_id": f"seg_{segment_num:03d}",
                        "time_range": (segments[-1]["time_range"][1], segments[-1]["time_range"][1] + 5.0),
                        "duration": 5.0,
                        "elements": [element],
                        "visual_summary": "",
                        "start_anchor": {},
                        "end_anchor": {}
                    }

        # 添加最后一个片段
        if current_segment["elements"]:
            segments.append(current_segment)

        return segments

    def _analyze_pacing(self, segments: List[TimeSegment], estimations: Dict[str, DurationEstimation]) -> PacingAnalysis:
        """分析节奏"""
        intensities = []

        for segment in segments:
            intensity = 0.0
            for element in segment["elements"]:
                if element.element_type == ElementType.ACTION:
                    intensity += 1.2
                elif element.element_type in [ElementType.DIALOGUE, ElementType.SILENCE]:
                    intensity += 0.8
                elif element.element_type == ElementType.SCENE:
                    intensity += 0.4

            intensities.append(min(intensity, 3.0))

        # 简单的节奏分析
        if len(intensities) < 3:
            pace_type = "平缓"
        elif max(intensities) > 2.0:
            pace_type = "紧张累积型"
        else:
            pace_type = "平稳叙述型"

        return {
            "pace_type": pace_type,
            "intensity_curve": intensities,
            "avg_intensity": sum(intensities) / len(intensities) if intensities else 0,
            "peak_intensity": max(intensities) if intensities else 0
        }

    def _generate_continuity_anchors(self, segments: List[TimeSegment]) -> List[ContinuityAnchor]:
        """生成连续性锚点"""
        anchors = []

        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # 简单的锚点生成
            anchor = {
                "anchor_id": f"anchor_{i + 1:03d}",
                "from_segment": current.segment_id,
                "to_segment": next_seg.segment_id,
                "constraints": [
                    {"type": "visual_continuity", "priority": "medium"},
                    {"type": "character_consistency", "priority": "high"}
                ],
                "transition_notes": "保持角色状态和环境一致性"
            }

            anchors.append(anchor)

        return anchors

    def _create_final_plan(self, segments: List[TimeSegment], estimations: Dict[str, DurationEstimation],
                           pacing_analysis: PacingAnalysis, continuity_anchors: List[ContinuityAnchor],
                           script_data: UnifiedScript) -> TimelinePlan:
        """创建最终计划"""
        total_duration = self._calculate_total_duration(segments)

        return TimelinePlan(
            timeline_segments=segments,
            duration_estimations=estimations,
            pacing_analysis=pacing_analysis,
            continuity_anchors=continuity_anchors,
            total_duration=total_duration,
            segments_count=len(segments),
            elements_count=len(estimations),
            estimations={k: v.to_dict() for k, v in estimations.items()},
            script_summary={
                "scenes_count": len(script_data.scenes),
                "dialogues_count": len(script_data.dialogues),
                "actions_count": len(script_data.actions)
            },
            processing_stats={
                **self.processing_stats,
                "end_time": datetime.now().isoformat(),
                "total_time": (datetime.now().second - self.processing_stats["start_time"])
            }
        )

    def _calculate_total_duration(self, segments: List[TimeSegment]) -> float:
        """计算总时长"""
        if not segments:
            return 0.0

        total = sum(segment.duration for segment in segments)
        return round(total, 2)
