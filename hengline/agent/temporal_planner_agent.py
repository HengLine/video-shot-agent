# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: 时序规划智能体，负责将剧本按5秒粒度切分，估算动作时长，使用LangChain实现状态记忆
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import copy
import json
from datetime import datetime
from typing import Dict, List, Any

from hengline.agent.script_parser.script_parser_models import UnifiedScript, Dialogue, Scene, Action
from hengline.agent.temporal_planner.hybrid_temporal_planner import HybridTemporalPlanner
from hengline.agent.temporal_planner.splitter.segment_splitter import SegmentSplitter
from hengline.agent.temporal_planner.splitter.splitter_analyzer import RhythmAnalyzer
from hengline.agent.temporal_planner.splitter.splitter_anchor import AnchorGenerator
from hengline.agent.temporal_planner.splitter.splitter_builder import ElementSequenceBuilder
from hengline.agent.temporal_planner.splitter.splitter_validator import SplitterValidator
from hengline.agent.temporal_planner.temporal_planner_model import TimelinePlan, DurationEstimation, TimeSegment, ElementType, ScriptElement
from hengline.logger import debug, error, info
from utils.log_utils import print_log_exception


class TemporalPlannerAgent:
    """时序规划智能体

    输入：统一格式的剧本解析结果
    输出：精确的5秒时间分片方案

    核心任务：
    1. 为每个剧本元素（对话、动作、描述）估算合理时长
    2. 智能分割为5秒粒度的视频片段
    3. 确保时间分配的合理性和连贯性
    4. 标记关键时间节点和情绪转折点

    """

    def __init__(self, llm):
        """初始化时序规划智能体"""
        self.temporal_planner = HybridTemporalPlanner(llm)

        # 初始化所有核心模块
        self.sequence_builder = ElementSequenceBuilder()
        self.rhythm_analyzer = RhythmAnalyzer()
        self.segment_splitter = SegmentSplitter()
        self.anchor_generator = AnchorGenerator()
        self.validator = SplitterValidator()

    def plan_process(self, structured_script: UnifiedScript, save_estimations: bool = False) -> TimelinePlan | None:
        """
        规划剧本的时序分段
        
        Args:
            structured_script: 结构化的剧本

        Returns:
            分段计划列表
        """
        debug("开始根据规则规划时序")
        try:
            # 时长估算字典，key为元素ID
            duration_estimations = self.temporal_planner.plan_timeline(structured_script)

            # duration_estimations保存为JSON
            if save_estimations:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"duration_estimations_{timestamp}.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {k: v.to_dict() for k, v in duration_estimations.items()},
                        f,
                        ensure_ascii=False,
                        indent=4
                    )

            #  创建TimelinePlan
            timeline_plan = self._create_timeline_plan(
                structured_script.scenes,
                structured_script.dialogues,
                structured_script.actions,
                duration_estimations
            )

            return timeline_plan

        except Exception as e:
            print_log_exception()
            error(f"执行时序规划异常: {e}")
            return None

    def _create_timeline_plan(
            self,
            scenes: List[Scene],
            dialogues: List[Dialogue],
            actions: List[Action],
            duration_estimations: Dict[str, DurationEstimation]
    ) -> TimelinePlan:
        """创建完整的时间线规划"""

        # 1. 构建元素序列
        debug("步骤1: 构建元素序列...")
        elements_sequence = self.sequence_builder.build_sequence(
            scenes, dialogues, actions, duration_estimations
        )

        # 2. 分析节奏
        debug("步骤2: 分析节奏模式...")
        pacing_analysis = self.rhythm_analyzer.analyze_pacing(elements_sequence)

        # 3. 分割为5秒片段
        debug("步骤3: 分割为5秒片段...")
        segments = self.segment_splitter.split_into_segments(
            elements_sequence, duration_estimations
        )

        # 4. 生成连贯性锚点
        debug("步骤4: 生成连贯性锚点...")
        continuity_anchors = self.anchor_generator.generate_anchors(
            segments, elements_sequence
        )

        # 5. 计算总时长
        total_duration = sum(segment.duration for segment in segments)

        # 6. 提取元素顺序
        element_order = [e.element_id for e in elements_sequence]

        # 7. 创建时间线规划
        timeline_plan = TimelinePlan(
            timeline_segments=segments,
            duration_estimations=duration_estimations,
            pacing_analysis=pacing_analysis,
            continuity_anchors=continuity_anchors,
            total_duration=total_duration,
            segments_count=len(segments),
            element_order=element_order,
            validation_report={}  # 将在下一步填充
        )

        # 8. 验证规划
        debug("步骤5: 验证规划合理性...")
        validation_report = self.validator.validate_timeline(timeline_plan)
        timeline_plan.validation_report = validation_report

        # 9. 为智能体3准备状态时间线
        self._prepare_for_continuity_agent(timeline_plan, elements_sequence)

        info(f"规划完成！共创建{len(segments)}个片段，总时长{total_duration}秒")

        return timeline_plan

    def _prepare_for_continuity_agent(
            self,
            timeline_plan: TimelinePlan,
            elements_sequence: List[ScriptElement]
    ):
        """为连续性守护智能体准备详细的状态时间线"""

        print("步骤6: 为连续性守护智能体准备状态时间线...")

        # 初始化状态跟踪系统
        character_states = {}
        prop_states = {}
        environment_states = {}

        # 按时间顺序处理每个片段
        for segment in timeline_plan.timeline_segments:
            segment_states = self._analyze_segment_states(segment, elements_sequence)

            # 收集角色状态时间线
            self._update_character_timeline(
                timeline_plan, segment, segment_states["characters"]
            )

            # 收集道具状态时间线
            self._update_prop_timeline(
                timeline_plan, segment, segment_states["props"]
            )

            # 收集环境状态时间线
            self._update_environment_timeline(
                timeline_plan, segment, segment_states["environment"]
            )

        # 添加连续性检查点
        self._add_continuity_checkpoints(timeline_plan)

        print(f"  已生成角色状态时间线: {len(timeline_plan.character_state_timeline)}个角色")
        print(f"  已生成道具状态时间线: {len(timeline_plan.prop_state_timeline)}个道具")

    def _analyze_segment_states(
            self,
            segment: TimeSegment,
            all_elements: List[ScriptElement]
    ) -> Dict[str, Dict[str, Any]]:
        """分析片段的状态信息"""

        segment_states = {
            "characters": {},
            "props": {},
            "environment": {},
            "spatial_relations": []
        }

        # 处理片段中的每个元素
        for element_id in segment.element_coverage:
            element = next((e for e in all_elements if e.element_id == element_id), None)
            if not element:
                continue

            if element.element_type == ElementType.SCENE:
                self._process_scene_states(element.original_data, segment_states)
            elif element.element_type == ElementType.ACTION:
                self._process_action_states(element.original_data, segment_states)
            elif element.element_type == ElementType.DIALOGUE:
                self._process_dialogue_states(element.original_data, segment_states)

        # 为片段生成开始和结束状态快照
        segment.start_state_snapshot = self._create_segment_start_snapshot(
            segment, segment_states
        )
        segment.end_state_snapshot = self._create_segment_end_snapshot(
            segment, segment_states
        )

        return segment_states

    def _process_scene_states(self, scene: Scene, segment_states: Dict[str, Any]):
        """处理场景状态信息"""
        # 环境状态
        segment_states["environment"].update({
            "location": scene.location,
            "time_of_day": getattr(scene, 'time_of_day', '未知'),
            "weather": getattr(scene, 'weather', '未知'),
            "mood": getattr(scene, 'mood', '未知'),
            "key_visuals": scene.key_visuals,
            "lighting_conditions": self._infer_lighting_from_scene(scene)
        })

        # 道具状态
        for prop in scene.props:
            prop_id = f"prop_{prop.name}_{hash(str(prop))}"
            segment_states["props"][prop_id] = {
                "name": prop.name,
                "state": prop.state,
                "position": prop.position,
                "owner": prop.owner,
                "description": prop.description,
                "is_interactive": bool(prop.owner) or "interaction" in prop.state.lower()
            }

    def _process_action_states(self, action: Action, segment_states: Dict[str, Any]):
        """处理动作状态信息"""
        # 角色动作
        if action.actor and not action.actor.startswith("prop_"):
            char_name = action.actor
            if char_name not in segment_states["characters"]:
                segment_states["characters"][char_name] = {
                    "actions": [],
                    "current_posture": "unknown",
                    "facial_expression": "neutral",
                    "interactions": [],
                    "emotional_state": "neutral"
                }

            segment_states["characters"][char_name]["actions"].append({
                "type": action.type,
                "description": action.description,
                "target": action.target,
                "duration": action.duration
            })

            # 更新姿势和表情
            self._update_character_from_action(char_name, action, segment_states["characters"][char_name])

        # 道具动作
        elif action.actor and action.actor.startswith("prop_"):
            prop_name = action.actor.replace("prop_", "")
            prop_id = f"prop_{prop_name}"

            if prop_id not in segment_states["props"]:
                segment_states["props"][prop_id] = {
                    "name": prop_name,
                    "state": "unknown",
                    "position": "unknown",
                    "is_interactive": True
                }

            # 更新道具状态
            segment_states["props"][prop_id].update({
                "state": action.description,
                "last_action": action.description
            })

    def _process_dialogue_states(self, dialogue: Dialogue, segment_states: Dict[str, Any]):
        """处理对话状态信息"""
        char_name = dialogue.speaker

        if char_name not in segment_states["characters"]:
            segment_states["characters"][char_name] = {
                "actions": [],
                "current_posture": "sitting" if "蜷在" in dialogue.parenthetical else "standing",
                "facial_expression": self._map_emotion_to_expression(dialogue.emotion),
                "interactions": [],
                "emotional_state": dialogue.emotion,
                "is_speaking": True
            }
        else:
            segment_states["characters"][char_name].update({
                "facial_expression": self._map_emotion_to_expression(dialogue.emotion),
                "emotional_state": dialogue.emotion,
                "is_speaking": True
            })

        # 添加对话交互
        if dialogue.target:
            segment_states["characters"][char_name]["interactions"].append({
                "type": "dialogue",
                "target": dialogue.target,
                "content_preview": dialogue.content[:30],
                "emotion": dialogue.emotion
            })

    def _update_character_from_action(self, char_name: str, action: Action, character_state: Dict[str, Any]):
        """根据动作更新角色状态"""
        action_desc = action.description.lower()

        # 姿势更新
        if "蜷" in action_desc or "坐" in action_desc:
            character_state["current_posture"] = "sitting"
        elif "站" in action_desc or "起身" in action_desc:
            character_state["current_posture"] = "standing"
        elif "躺" in action_desc:
            character_state["current_posture"] = "lying"

        # 表情更新
        if "瞳孔收缩" in action.description:
            character_state["facial_expression"] = "surprised"
        elif "泪水" in action.description or "哽咽" in action.description:
            character_state["facial_expression"] = "sad"
        elif "微笑" in action.description:
            character_state["facial_expression"] = "smiling"
        elif "紧张" in action.description:
            character_state["facial_expression"] = "tense"

        # 情绪状态更新
        if action.type == "physiological":
            if "呼吸停滞" in action.description:
                character_state["emotional_state"] = "shocked"
            elif "喉头滚动" in action.description:
                character_state["emotional_state"] = "nervous"

    def _map_emotion_to_expression(self, emotion: str) -> str:
        """将情绪描述映射到面部表情"""
        emotion_map = {
            "微颤": "tense",
            "低声": "serious",
            "哽咽": "sad",
            "紧张": "tense",
            "激动": "excited",
            "平静": "neutral",
            "沉思": "thoughtful"
        }
        return emotion_map.get(emotion, "neutral")

    def _infer_lighting_from_scene(self, scene: Scene) -> Dict[str, Any]:
        """从场景推断照明条件"""
        lighting = {
            "source": "artificial",
            "intensity": "medium",
            "color_temperature": "warm",
            "shadows": "medium"
        }

        description = scene.description.lower()

        if "昏暗" in description:
            lighting["intensity"] = "low"
            lighting["shadows"] = "strong"
        elif "明亮" in description:
            lighting["intensity"] = "high"
            lighting["shadows"] = "soft"

        if "电视" in description and "播放" in description:
            lighting["dynamic_source"] = "tv_screen"
            lighting["color_variation"] = True

        if "大雨" in getattr(scene, 'weather', ''):
            lighting["ambient_reflection"] = "wet_surfaces"

        return lighting

    def _create_segment_start_snapshot(
            self,
            segment: TimeSegment,
            segment_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建片段开始状态快照"""
        return {
            "timestamp": segment.start_time,
            "characters": self._simplify_character_states(segment_states["characters"]),
            "props": {k: self._simplify_prop_state(v) for k, v in segment_states["props"].items()},
            "environment": segment_states["environment"],
            "narrative_context": segment.narrative_arc
        }

    def _create_segment_end_snapshot(
            self,
            segment: TimeSegment,
            segment_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建片段结束状态快照"""
        # 对于结束状态，我们可以基于动作推断变化
        end_states = copy.deepcopy(self._create_segment_start_snapshot(segment, segment_states))

        # 应用片段内的状态变化
        for char_name, char_state in segment_states["characters"].items():
            if char_name in end_states["characters"]:
                # 更新最后的表情和姿势
                if "actions" in char_state and char_state["actions"]:
                    last_action = char_state["actions"][-1]
                    self._apply_action_to_snapshot(last_action, end_states, char_name)

        return end_states

    def _simplify_character_states(self, character_states: Dict[str, Any]) -> Dict[str, Any]:
        """简化角色状态以便存储"""
        simplified = {}

        for char_name, state in character_states.items():
            simplified[char_name] = {
                "posture": state.get("current_posture", "unknown"),
                "expression": state.get("facial_expression", "neutral"),
                "emotion": state.get("emotional_state", "neutral"),
                "is_speaking": state.get("is_speaking", False),
                "key_actions": [a["type"] for a in state.get("actions", [])[:3]]
            }

        return simplified

    def _simplify_prop_state(self, prop_state: Dict[str, Any]) -> Dict[str, Any]:
        """简化道具状态"""
        return {
            "name": prop_state.get("name", ""),
            "state": prop_state.get("state", ""),
            "position": prop_state.get("position", ""),
            "is_interactive": prop_state.get("is_interactive", False)
        }

    def _apply_action_to_snapshot(self, action: Dict[str, Any], snapshot: Dict[str, Any], char_name: str):
        """将动作应用到状态快照"""
        action_type = action.get("type", "")
        description = action.get("description", "")

        if action_type == "posture" and "坐直" in description:
            snapshot["characters"][char_name]["posture"] = "sitting_upright"
        elif action_type == "facial" and "泪水" in description:
            snapshot["characters"][char_name]["expression"] = "crying"
            snapshot["characters"][char_name]["emotion"] = "sad"

    def _update_character_timeline(
            self,
            timeline_plan: TimelinePlan,
            segment: TimeSegment,
            character_states: Dict[str, Any]
    ):
        """更新角色状态时间线"""
        for char_name, state in character_states.items():
            if char_name not in timeline_plan.character_state_timeline:
                timeline_plan.character_state_timeline[char_name] = []

            timeline_plan.character_state_timeline[char_name].append({
                "segment_id": segment.segment_id,
                "timestamp": segment.start_time,
                "duration": segment.duration,
                "state_summary": {
                    "posture": state.get("current_posture", "unknown"),
                    "expression": state.get("facial_expression", "neutral"),
                    "emotion": state.get("emotional_state", "neutral"),
                    "is_speaking": state.get("is_speaking", False),
                    "key_action_count": len(state.get("actions", []))
                },
                "actions": state.get("actions", []),
                "interactions": state.get("interactions", [])
            })

    def _update_prop_timeline(
            self,
            timeline_plan: TimelinePlan,
            segment: TimeSegment,
            prop_states: Dict[str, Any]
    ):
        """更新道具状态时间线"""
        for prop_id, state in prop_states.items():
            if prop_id not in timeline_plan.prop_state_timeline:
                timeline_plan.prop_state_timeline[prop_id] = []

            timeline_plan.prop_state_timeline[prop_id].append({
                "segment_id": segment.segment_id,
                "timestamp": segment.start_time,
                "state": state.get("state", ""),
                "position": state.get("position", ""),
                "owner": state.get("owner", ""),
                "is_interactive": state.get("is_interactive", False),
                "last_action": state.get("last_action", "")
            })

    def _update_environment_timeline(
            self,
            timeline_plan: TimelinePlan,
            segment: TimeSegment,
            environment_state: Dict[str, Any]
    ):
        """更新环境状态时间线（如果需要的话）"""
        # 可以在这里添加环境时间线跟踪
        pass

    def _add_continuity_checkpoints(self, timeline_plan: TimelinePlan):
        """为连续性守护智能体添加检查点"""

        continuity_checkpoints = []

        # 在关键片段转换处添加检查点
        for i in range(len(timeline_plan.timeline_segments) - 1):
            current = timeline_plan.timeline_segments[i]
            next_seg = timeline_plan.timeline_segments[i + 1]

            # 检查是否需要特殊连续性关注
            needs_checkpoint = self._needs_continuity_checkpoint(current, next_seg)

            if needs_checkpoint:
                checkpoint = {
                    "checkpoint_id": f"checkpoint_{i + 1:03d}",
                    "position": f"between_{current.segment_id}_and_{next_seg.segment_id}",
                    "timestamp": current.end_time,
                    "critical_elements": self._identify_critical_elements(current, next_seg),
                    "continuity_risks": self._identify_continuity_risks(current, next_seg),
                    "recommended_checks": [
                        "character_appearance",
                        "prop_positions",
                        "lighting_consistency",
                        "spatial_relations"
                    ]
                }
                continuity_checkpoints.append(checkpoint)

        # 将检查点添加到时间线计划
        timeline_plan.validation_report["continuity_checkpoints"] = continuity_checkpoints

        print(f"  已添加{len(continuity_checkpoints)}个连续性检查点")

    def _needs_continuity_checkpoint(self, current: TimeSegment, next_seg: TimeSegment) -> bool:
        """判断是否需要连续性检查点"""
        # 场景标签变化显著时
        current_tags = current.visual_consistency_tags
        next_tags = next_seg.visual_consistency_tags
        if len(current_tags.intersection(next_tags)) < 2:
            return True

        # 叙事弧变化时
        if current.narrative_arc != next_seg.narrative_arc:
            return True

        # 包含重要动作变化时
        if "动作密集" in current.visual_content and "对话密集" in next_seg.visual_content:
            return True

        return False

    def _identify_critical_elements(self, current: TimeSegment, next_seg: TimeSegment) -> List[str]:
        """识别关键连续性元素"""
        critical_elements = []

        # 检查共同的角色
        current_chars = self._extract_characters_from_description(current.visual_content)
        next_chars = self._extract_characters_from_description(next_seg.visual_content)
        common_chars = [c for c in current_chars if c in next_chars]

        if common_chars:
            critical_elements.extend([f"character_{c}" for c in common_chars])

        # 检查环境元素
        env_elements = ["lighting", "camera_angle", "background"]
        critical_elements.extend([f"environment_{e}" for e in env_elements])

        return critical_elements

    def _identify_continuity_risks(self, current: TimeSegment, next_seg: TimeSegment) -> List[str]:
        """识别连续性风险"""
        risks = []

        # 照明变化风险
        if "夜晚" in str(current.visual_consistency_tags) and "白天" in str(next_seg.visual_consistency_tags):
            risks.append("lighting_change_too_abrupt")

        # 角色跳跃风险
        if "林然" in current.visual_content and "林然" in next_seg.visual_content:
            if "坐" in current.visual_content and "站" in next_seg.visual_content:
                risks.append("character_posture_jump")

        return risks

    def _extract_characters_from_description(self, description: str) -> List[str]:
        """从描述中提取角色名"""
        # 简单的提取逻辑，实际中可能需要更复杂的NLP
        characters = []
        known_chars = ["林然", "陈默"]

        for char in known_chars:
            if char in description:
                characters.append(char)

        return characters