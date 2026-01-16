"""
@FileName: splitter_anchor.py
@Description: 连续性锚点生成模型
@Author: HengLine
@Time: 2026/1/16 0:57
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from typing import List, Dict, Set, Optional

from hengline.agent.temporal_planner.splitter.splitter_detector import KeyframeDetector
from hengline.agent.temporal_planner.splitter.splitter_parser import ActionParser
from hengline.agent.temporal_planner.temporal_planner_model import ContinuityAnchor


class AnchorType(Enum):
    """锚点类型分类"""
    CHARACTER_APPEARANCE = "character_appearance"  # 角色外观
    CHARACTER_POSITION = "character_position"  # 角色位置
    CHARACTER_POSTURE = "character_posture"  # 角色姿势
    CHARACTER_EXPRESSION = "character_expression"  # 面部表情
    PROP_STATE = "prop_state"  # 道具状态
    PROP_POSITION = "prop_position"  # 道具位置
    ENVIRONMENT = "environment"  # 环境
    SPATIAL_RELATION = "spatial_relation"  # 空间关系
    VISUAL_MATCH = "visual_match"  # 视觉匹配
    TRANSITION = "transition"  # 过渡要求


class AnchorPriority(Enum):
    """锚点优先级"""
    CRITICAL = 10  # 必须遵守，否则明显不连贯
    HIGH = 7  # 重要，观众会注意到
    MEDIUM = 5  # 建议遵守，提升质量
    LOW = 3  # 可选项，细微优化


@dataclass
class CharacterStateVector:
    """角色状态向量（供智能体3状态跟踪）"""
    character_name: str
    timestamp: float

    # 外观特征
    appearance: Dict[str, str] = field(default_factory=dict)  # 服装、发型等

    # 姿势状态
    posture: str = "unknown"  # 坐、站、躺等
    posture_details: Dict[str, str] = field(default_factory=dict)  # 详细姿势

    # 空间位置
    location: str = "unknown"  # 具体位置描述
    relative_position: Dict[str, str] = field(default_factory=dict)  # 相对位置

    # 面部状态
    facial_expression: str = "neutral"  # 面部表情
    gaze_direction: str = "unknown"  # 视线方向

    # 交互状态
    interacting_with: List[str] = field(default_factory=list)  # 交互对象
    interaction_type: str = "none"  # 交互类型

    # 情绪状态
    emotional_state: str = "neutral"  # 情绪
    emotional_intensity: float = 0.5  # 情绪强度 0-1


@dataclass
class PropStateVector:
    """道具状态向量"""
    prop_name: str
    timestamp: float

    # 基本状态
    current_state: str = "unknown"  # 状态描述
    position: str = "unknown"  # 位置描述
    owner: str = "none"  # 持有者

    # 视觉特征
    visual_description: str = ""  # 视觉描述
    is_visible: bool = True  # 是否可见

    # 交互状态
    is_interacted: bool = False  # 是否被交互
    interaction_details: Dict[str, str] = field(default_factory=dict)


class ContinuityAnchorGenerator:
    """增强版连续性锚点生成器"""

    def __init__(self):
        # 状态跟踪器
        self.character_states: Dict[str, List[CharacterStateVector]] = {}
        self.prop_states: Dict[str, List[PropStateVector]] = {}

        # 时间索引缓存
        self.time_index_cache: Dict[float, Dict[str, Any]] = {}

        # 锚点生成规则
        self.rules = self._initialize_rules()

        # 关键帧检测器
        self.keyframe_detector = KeyframeDetector()

        # 动作解析器
        self.action_parser = ActionParser()

    def _initialize_rules(self) -> Dict[str, Dict]:
        """初始化锚点生成规则"""
        return {
            "character_appearance": {
                "trigger": ["服装变化", "发型变化", "配饰变化"],
                "priority": AnchorPriority.CRITICAL,
                "mandatory": True
            },
            "character_position": {
                "trigger": ["位置移动", "场景切换"],
                "priority": AnchorPriority.HIGH,
                "mandatory": True
            },
            "character_posture": {
                "trigger": ["姿势变化", "动作序列"],
                "priority": AnchorPriority.MEDIUM,
                "mandatory": True
            },
            "prop_state": {
                "trigger": ["道具状态变化", "道具交互"],
                "priority": AnchorPriority.HIGH,
                "mandatory": True
            },
            "spatial_relation": {
                "trigger": ["相对位置变化", "距离变化"],
                "priority": AnchorPriority.MEDIUM,
                "mandatory": False
            }
        }

    def generate_anchors_for_scene(self,
                                   scene_data: Dict,
                                   timeline_segments: List[Dict]) -> List[ContinuityAnchor]:
        """为整个场景生成连续性锚点"""

        anchors = []

        # 1. 初始化状态跟踪
        self._initialize_states(scene_data)

        # 2. 按时间顺序处理时间片段
        sorted_segments = sorted(timeline_segments, key=lambda x: x["time_range"][0])

        for i in range(len(sorted_segments) - 1):
            current_seg = sorted_segments[i]
            next_seg = sorted_segments[i + 1]

            # 3. 生成片段间锚点
            segment_anchors = self._generate_segment_anchors(
                current_seg,
                next_seg,
                scene_data
            )
            anchors.extend(segment_anchors)

            # 4. 更新状态跟踪
            self._update_states(current_seg, scene_data)

        # 5. 生成全局一致性锚点
        global_anchors = self._generate_global_anchors(scene_data, sorted_segments)
        anchors.extend(global_anchors)

        return anchors

        # ==================== 更新现有的状态跟踪方法 ====================

    def _update_states(self, segment: Dict, scene_data: Dict):
        """更新状态跟踪"""
        segment_end_time = segment["time_range"][1]

        # 更新角色状态
        for char_name in self._get_characters_in_segment(segment, scene_data):
            predicted_state = self._predict_next_character_state(
                char_name,
                segment_end_time - 0.1,  # 稍微提前一点
                {"time_range": (segment_end_time, segment_end_time + 5.0)}  # 模拟下一片段
            )

            if predicted_state:
                if char_name not in self.character_states:
                    self.character_states[char_name] = []
                self.character_states[char_name].append(predicted_state)

        # 更新道具状态
        for prop_name in self._get_props_in_segment(segment, scene_data):
            predicted_state = self._predict_next_prop_state(
                prop_name,
                segment_end_time - 0.1,
                {"time_range": (segment_end_time, segment_end_time + 5.0)}
            )

            if predicted_state:
                if prop_name not in self.prop_states:
                    self.prop_states[prop_name] = []
                self.prop_states[prop_name].append(predicted_state)

    def _get_characters_in_segment(self, segment: Dict, scene_data: Dict) -> Set[str]:
        """获取片段中涉及的角色（增强版）"""
        characters = set()

        # 从动作中提取角色
        for action in scene_data.get("actions", []):
            action_id = action.get("action_id", "")

            # 检查这个动作是否在当前片段中
            if action_id in segment.get("element_coverage", []):
                actor = action.get("actor", "")
                if actor and actor != "手机" and actor != "旧羊毛毯":  # 排除道具
                    characters.add(actor)

        # 从对话中提取角色
        for dialogue in scene_data.get("dialogues", []):
            dialogue_id = dialogue.get("dialogue_id", "")

            if dialogue_id in segment.get("element_coverage", []):
                speaker = dialogue.get("speaker", "")
                target = dialogue.get("target", "")

                if speaker:
                    characters.add(speaker)
                if target:
                    characters.add(target)

        return characters

    def _initialize_states(self, scene_data: Dict):
        """初始化状态跟踪"""

        # 初始化角色状态
        for character in scene_data.get("characters", []):
            char_name = character["name"]
            self.character_states[char_name] = []

            # 创建初始状态
            initial_state = CharacterStateVector(
                character_name=char_name,
                timestamp=0.0,
                appearance=self._extract_appearance(character),
                posture="unknown",
                location="unknown",
                facial_expression="neutral",
                emotional_state=character.get("state", {}).get("emotional", "neutral")
            )
            self.character_states[char_name].append(initial_state)

        # 初始化道具状态
        for scene in scene_data.get("scenes", []):
            for prop in scene.get("props", []):
                prop_name = prop["name"]
                self.prop_states[prop_name] = []

                initial_prop_state = PropStateVector(
                    prop_name=prop_name,
                    timestamp=0.0,
                    current_state=prop.get("state", "unknown"),
                    position=prop.get("position", "unknown"),
                    owner=prop.get("owner", "none"),
                    visual_description=prop.get("description", "")
                )
                self.prop_states[prop_name].append(initial_prop_state)

    def _generate_segment_anchors(self,
                                  current_seg: Dict,
                                  next_seg: Dict,
                                  scene_data: Dict) -> List[ContinuityAnchor]:
        """生成两个相邻片段间的锚点"""

        anchors = []

        # 1. 角色连续性锚点
        character_anchors = self._generate_character_continuity_anchors(
            current_seg, next_seg, scene_data
        )
        anchors.extend(character_anchors)

        # 2. 道具连续性锚点
        prop_anchors = self._generate_prop_continuity_anchors(
            current_seg, next_seg, scene_data
        )
        anchors.extend(prop_anchors)

        # 3. 环境一致性锚点
        env_anchors = self._generate_environment_anchors(
            current_seg, next_seg, scene_data
        )
        anchors.extend(env_anchors)

        # 4. 空间关系锚点
        spatial_anchors = self._generate_spatial_anchors(
            current_seg, next_seg, scene_data
        )
        anchors.extend(spatial_anchors)

        # 5. 视觉匹配锚点
        visual_anchors = self._generate_visual_match_anchors(
            current_seg, next_seg, scene_data
        )
        anchors.extend(visual_anchors)

        return anchors

    def _generate_character_continuity_anchors(self,
                                               current_seg: Dict,
                                               next_seg: Dict,
                                               scene_data: Dict) -> List[ContinuityAnchor]:
        """生成角色连续性锚点"""

        anchors = []
        current_time = current_seg["time_range"][1]

        # 获取当前片段中涉及的角色
        involved_characters = self._get_characters_in_segment(current_seg, scene_data)

        for char_name in involved_characters:
            # 获取角色当前状态
            current_state = self._get_character_state_at_time(char_name, current_time)

            if not current_state:
                continue

            # 检查状态是否有重要变化需要锚点
            next_state = self._predict_next_character_state(char_name, current_time, next_seg)

            if next_state:
                # 生成外观锚点
                if self._needs_appearance_anchor(current_state, next_state):
                    anchor = ContinuityAnchor(
                        anchor_id=f"char_app_{char_name}_{current_seg['segment_id']}_{next_seg['segment_id']}",
                        anchor_type=AnchorType.CHARACTER_APPEARANCE,
                        priority=AnchorPriority.CRITICAL,
                        from_segment=current_seg["segment_id"],
                        to_segment=next_seg["segment_id"],
                        temporal_constraint=f"从{current_time}秒到{next_seg['time_range'][0]}秒",
                        description=f"角色{char_name}的外观必须保持一致：{current_state.appearance}",
                        sora_prompt=f"保持{char_name}的外观不变：{self._appearance_to_sora_prompt(current_state.appearance)}",
                        mandatory=True,
                        state_change={"from": current_state.appearance, "to": next_state.appearance}
                    )
                    anchors.append(anchor)

                # 生成位置锚点
                if self._needs_position_anchor(current_state, next_state):
                    anchor = ContinuityAnchor(
                        anchor_id=f"char_pos_{char_name}_{current_seg['segment_id']}_{next_seg['segment_id']}",
                        anchor_type=AnchorType.CHARACTER_POSITION,
                        priority=AnchorPriority.HIGH,
                        from_segment=current_seg["segment_id"],
                        to_segment=next_seg["segment_id"],
                        temporal_constraint=f"在{next_seg['time_range'][0]}秒时",
                        description=f"角色{char_name}在下一片段开始时位置应为：{current_state.location}",
                        sora_prompt=f"{char_name}位于{current_state.location}",
                        mandatory=True
                    )
                    anchors.append(anchor)

                # 生成姿势锚点
                if self._needs_posture_anchor(current_state, next_state):
                    anchor = ContinuityAnchor(
                        anchor_id=f"char_post_{char_name}_{current_seg['segment_id']}_{next_seg['segment_id']}",
                        anchor_type=AnchorType.CHARACTER_POSTURE,
                        priority=AnchorPriority.MEDIUM,
                        from_segment=current_seg["segment_id"],
                        to_segment=next_seg["segment_id"],
                        temporal_constraint="片段过渡时",
                        description=f"角色{char_name}的姿势连续性：{current_state.posture} → {next_state.posture}",
                        sora_prompt=f"{char_name}保持{current_state.posture}姿势",
                        mandatory=False
                    )
                    anchors.append(anchor)

        return anchors

    def _generate_prop_continuity_anchors(self,
                                          current_seg: Dict,
                                          next_seg: Dict,
                                          scene_data: Dict) -> List[ContinuityAnchor]:
        """生成道具连续性锚点"""

        anchors = []
        current_time = current_seg["time_range"][1]

        # 获取当前片段中涉及的道具
        involved_props = self._get_props_in_segment(current_seg, scene_data)

        for prop_name in involved_props:
            # 获取道具当前状态
            current_state = self._get_prop_state_at_time(prop_name, current_time)

            if not current_state:
                continue

            # 检查道具状态变化
            next_state = self._predict_next_prop_state(prop_name, current_time, next_seg)

            if next_state and current_state.current_state != next_state.current_state:
                # 道具状态变化需要锚点
                anchor = ContinuityAnchor(
                    anchor_id=f"prop_{prop_name}_{current_seg['segment_id']}_{next_seg['segment_id']}",
                    anchor_type=AnchorType.PROP_STATE,
                    priority=AnchorPriority.HIGH,
                    from_segment=current_seg["segment_id"],
                    to_segment=next_seg["segment_id"],
                    temporal_constraint="跨片段时",
                    description=f"道具{prop_name}状态变化：{current_state.current_state} → {next_state.current_state}",
                    sora_prompt=f"道具{prop_name}的状态为{next_state.current_state}",
                    mandatory=True,
                    state_change={
                        "prop_name": prop_name,
                        "from_state": current_state.current_state,
                        "to_state": next_state.current_state,
                        "position": next_state.position
                    }
                )
                anchors.append(anchor)

            # 道具位置锚点（即使状态不变）
            if current_state.position != "unknown":
                anchor = ContinuityAnchor(
                    anchor_id=f"prop_pos_{prop_name}_{current_seg['segment_id']}_{next_seg['segment_id']}",
                    anchor_type=AnchorType.PROP_POSITION,
                    priority=AnchorPriority.MEDIUM,
                    from_segment=current_seg["segment_id"],
                    to_segment=next_seg["segment_id"],
                    temporal_constraint="在下一片段开始时",
                    description=f"道具{prop_name}应保持在{current_state.position}",
                    sora_prompt=f"{prop_name}在{current_state.position}",
                    mandatory=False
                )
                anchors.append(anchor)

        return anchors

    def _generate_environment_anchors(self,
                                      current_seg: Dict,
                                      next_seg: Dict,
                                      scene_data: Dict) -> List[ContinuityAnchor]:
        """生成环境一致性锚点"""

        anchors = []

        # 获取场景信息
        current_scene = self._get_scene_for_segment(current_seg, scene_data)
        next_scene = self._get_scene_for_segment(next_seg, scene_data)

        if current_scene and next_scene and current_scene["scene_id"] == next_scene["scene_id"]:
            # 同一场景内，需要环境一致性
            anchor = ContinuityAnchor(
                anchor_id=f"env_{current_scene['scene_id']}_{current_seg['segment_id']}_{next_seg['segment_id']}",
                anchor_type=AnchorType.ENVIRONMENT,
                priority=AnchorPriority.CRITICAL,
                from_segment=current_seg["segment_id"],
                to_segment=next_seg["segment_id"],
                temporal_constraint="整个场景内",
                description=f"场景环境保持一致：{current_scene['location']}, {current_scene['time_of_day']}, {current_scene.get('weather', '')}",
                sora_prompt=f"保持场景一致性：{current_scene['location']}, {current_scene['time_of_day']}, {current_scene.get('weather', '')}",
                mandatory=True,
                visual_reference=current_scene.get("description", "")
            )
            anchors.append(anchor)

        return anchors



    def _generate_spatial_anchors(self,
                                  current_seg: Dict,
                                  next_seg: Dict,
                                  scene_data: Dict) -> List[ContinuityAnchor]:
        """生成空间关系锚点"""

        anchors = []

        # 检查角色间的相对位置
        characters_in_current = self._get_characters_in_segment(current_seg, scene_data)

        if len(characters_in_current) >= 2:
            # 多个角色在同一场景，需要空间关系锚点
            char_list = "、".join(characters_in_current)
            anchor = ContinuityAnchor(
                anchor_id=f"spatial_{current_seg['segment_id']}_{next_seg['segment_id']}",
                anchor_type=AnchorType.SPATIAL_RELATION,
                priority=AnchorPriority.MEDIUM,
                from_segment=current_seg["segment_id"],
                to_segment=next_seg["segment_id"],
                temporal_constraint="角色同时出现时",
                description=f"角色间的空间关系应保持一致：{char_list}",
                sora_prompt=f"保持角色空间关系：{char_list}的相对位置",
                mandatory=False
            )
            anchors.append(anchor)

        return anchors

    def _get_character_state_at_time(self, character_name: str, timestamp: float) -> Optional[CharacterStateVector]:
        """获取指定时间点的角色状态"""

        if character_name not in self.character_states:
            return None

        # 获取该角色的状态历史
        state_history = self.character_states[character_name]

        if not state_history:
            return None

        # 查找最接近指定时间点的状态
        closest_state = None
        min_time_diff = float('inf')

        for state in state_history:
            time_diff = abs(state.timestamp - timestamp)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_state = state

        # 如果时间差太大，可能需要插值或预测
        if min_time_diff > 2.0:  # 2秒阈值
            return self._interpolate_character_state(character_name, timestamp)

        return closest_state

    def _predict_next_character_state(self,
                                      character_name: str,
                                      current_time: float,
                                      next_segment: Dict) -> Optional[CharacterStateVector]:
        """预测角色在下一片段开始时的状态"""

        current_state = self._get_character_state_at_time(character_name, current_time)
        if not current_state:
            return None

        # 分析下一片段中该角色的动作
        next_actions = self._get_character_actions_in_segment(character_name, next_segment)

        # 创建预测状态
        predicted_state = CharacterStateVector(
            character_name=character_name,
            timestamp=next_segment["time_range"][0],  # 下一片段开始时间
            appearance=current_state.appearance.copy(),
            posture=current_state.posture,
            location=current_state.location,
            facial_expression=current_state.facial_expression,
            gaze_direction=current_state.gaze_direction,
            emotional_state=current_state.emotional_state,
            emotional_intensity=current_state.emotional_intensity
        )

        # 根据动作预测状态变化
        for action in next_actions:
            self._apply_action_to_state(predicted_state, action)

        return predicted_state

    def _get_props_in_segment(self, segment: Dict, scene_data: Dict) -> List[str]:
        """获取片段中涉及的道具列表"""

        props = set()

        # 1. 从片段覆盖的元素中查找道具
        for element_id in segment.get("element_coverage", []):
            # 检查是否是道具相关动作
            if element_id.startswith("act_"):
                # 在动作列表中查找
                for action in scene_data.get("actions", []):
                    if action["action_id"] == element_id:
                        # 检查动作是否涉及道具
                        prop_name = self._extract_prop_from_action(action)
                        if prop_name:
                            props.add(prop_name)

        # 2. 从场景道具列表中获取
        scene_id = self._get_scene_id_for_segment(segment, scene_data)
        if scene_id:
            for scene in scene_data.get("scenes", []):
                if scene["scene_id"] == scene_id:
                    for prop in scene.get("props", []):
                        props.add(prop["name"])

        return list(props)

    def _get_prop_state_at_time(self, prop_name: str, timestamp: float) -> Optional[PropStateVector]:
        """获取指定时间点的道具状态"""

        if prop_name not in self.prop_states:
            return None

        prop_history = self.prop_states[prop_name]

        if not prop_history:
            return None

        # 查找最接近的时间点
        closest_state = None
        min_time_diff = float('inf')

        for state in prop_history:
            time_diff = abs(state.timestamp - timestamp)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_state = state

        return closest_state

    def _predict_next_prop_state(self,
                                 prop_name: str,
                                 current_time: float,
                                 next_segment: Dict) -> Optional[PropStateVector]:
        """预测道具在下一片段开始时的状态"""

        current_state = self._get_prop_state_at_time(prop_name, current_time)
        if not current_state:
            return None

        # 查找下一片段中涉及该道具的动作
        next_actions = self._get_prop_actions_in_segment(prop_name, next_segment)

        # 创建预测状态
        predicted_state = PropStateVector(
            prop_name=prop_name,
            timestamp=next_segment["time_range"][0],  # 下一片段开始时间
            current_state=current_state.current_state,
            position=current_state.position,
            owner=current_state.owner,
            visual_description=current_state.visual_description,
            is_visible=current_state.is_visible,
            is_interacted=current_state.is_interacted,
            interaction_details=current_state.interaction_details.copy()
        )

        # 根据动作更新道具状态
        for action in next_actions:
            self._apply_action_to_prop_state(predicted_state, action)

        return predicted_state

    # ==================== 新增辅助方法 ====================

    def _interpolate_character_state(self, character_name: str, timestamp: float) -> Optional[CharacterStateVector]:
        """通过插值获取角色状态"""

        if character_name not in self.character_states:
            return None

        states = self.character_states[character_name]
        if len(states) < 2:
            return states[-1] if states else None

        # 找到前后两个状态点
        before_state = None
        after_state = None

        for state in states:
            if state.timestamp <= timestamp:
                before_state = state
            elif state.timestamp > timestamp:
                after_state = state
                break

        if not before_state and not after_state:
            return None

        if not after_state:
            return before_state  # 只有之前的状态

        if not before_state:
            return after_state  # 只有之后的状态

        # 线性插值（简化版）
        # 在实际应用中，可能需要更复杂的插值逻辑
        time_ratio = (timestamp - before_state.timestamp) / (after_state.timestamp - before_state.timestamp)

        interpolated_state = CharacterStateVector(
            character_name=character_name,
            timestamp=timestamp,
            appearance=self._interpolate_dict(before_state.appearance, after_state.appearance, time_ratio),
            posture=after_state.posture if time_ratio > 0.5 else before_state.posture,
            location=after_state.location if time_ratio > 0.5 else before_state.location,
            facial_expression=after_state.facial_expression if time_ratio > 0.5 else before_state.facial_expression,
            gaze_direction=after_state.gaze_direction if time_ratio > 0.5 else before_state.gaze_direction,
            emotional_state=after_state.emotional_state if time_ratio > 0.5 else before_state.emotional_state,
            emotional_intensity=self._interpolate_float(
                before_state.emotional_intensity,
                after_state.emotional_intensity,
                time_ratio
            )
        )

        return interpolated_state

    def _get_character_actions_in_segment(self, character_name: str, segment: Dict) -> List[Dict]:
        """获取角色在片段中的动作列表"""
        # 这个方法需要访问实际的剧本数据
        # 这里返回模拟数据
        actions = []

        # 模拟：根据角色名返回可能的动作
        if character_name == "林然":
            actions = [
                {"type": "gaze", "target": "手机", "description": "盯着手机"},
                {"type": "physiological", "description": "喉头滚动"},
                {"type": "interaction", "target": "手机", "description": "按下接听键"}
            ]

        return actions

    def _get_prop_actions_in_segment(self, prop_name: str, segment: Dict) -> List[Dict]:
        """获取道具在片段中的动作列表"""
        actions = []

        # 模拟：根据道具名返回可能的动作
        if prop_name == "手机":
            actions = [
                {"type": "device_alert", "state": "震动亮屏"},
                {"type": "interaction", "actor": "林然", "description": "被接听"}
            ]
        elif prop_name == "旧羊毛毯":
            actions = [
                {"type": "prop_fall", "state": "滑落"}
            ]

        return actions

    def _extract_prop_from_action(self, action: Dict) -> Optional[str]:
        """从动作描述中提取道具名"""
        description = action.get("description", "")

        # 关键词匹配
        prop_keywords = {
            "手机": ["手机", "电话", "听筒"],
            "旧羊毛毯": ["毛毯", "毯子", "羊毛毯"],
            "凉茶": ["茶", "茶杯", "水杯"],
            "旧相册": ["相册", "照片"],
            "电视": ["电视", "电视机"]
        }

        for prop_name, keywords in prop_keywords.items():
            for keyword in keywords:
                if keyword in description:
                    return prop_name

        return None

    def _apply_action_to_state(self, state: CharacterStateVector, action: Dict):
        """将动作效果应用到角色状态"""
        action_type = action.get("type", "")
        description = action.get("description", "")

        if action_type == "posture":
            if "蜷在" in description or "蜷坐" in description:
                state.posture = "蜷坐"
            elif "坐直" in description:
                state.posture = "坐直"
            elif "站着" in description:
                state.posture = "站立"

        elif action_type == "gaze":
            target = action.get("target", "")
            state.gaze_direction = f"看向{target}"

        elif action_type == "facial":
            if "瞳孔收缩" in description:
                state.facial_expression = "震惊"
            elif "泪水" in description:
                state.facial_expression = "悲伤"
            elif "微笑" in description:
                state.facial_expression = "微笑"

        elif action_type == "interaction":
            target = action.get("target", "")
            state.interacting_with.append(target)
            state.interaction_type = action_type

        elif action_type == "physiological":
            if "呼吸停滞" in description:
                state.emotional_state = "紧张"
                state.emotional_intensity = 0.9
            elif "喉头滚动" in description:
                state.emotional_state = "紧张"
                state.emotional_intensity = 0.7

    def _apply_action_to_prop_state(self, state: PropStateVector, action: Dict):
        """将动作效果应用到道具状态"""
        action_type = action.get("type", "")
        description = action.get("description", "")

        if action_type == "device_alert":
            state.current_state = "震动亮屏"
            state.is_interacted = False

        elif action_type == "prop_fall":
            state.current_state = "滑落"
            if "肩头" in description:
                state.position = "地板上"

        elif action_type == "interaction":
            state.current_state = "被使用中"
            state.is_interacted = True
            state.interaction_details["actor"] = action.get("actor", "")

    def _get_scene_id_for_segment(self, segment: Dict, scene_data: Dict) -> Optional[str]:
        """获取片段所属的场景ID"""
        # 简化的实现：假设所有元素都属于同一个场景
        # 在实际中，可能需要更复杂的逻辑
        if scene_data.get("scenes"):
            return scene_data["scenes"][0]["scene_id"]
        return None

    def _interpolate_dict(self, dict1: Dict, dict2: Dict, ratio: float) -> Dict:
        """字典插值"""
        if ratio < 0.5:
            return dict1.copy()
        else:
            return dict2.copy()

    def _interpolate_float(self, val1: float, val2: float, ratio: float) -> float:
        """浮点数插值"""
        return val1 + (val2 - val1) * ratio

    def _generate_visual_match_anchors(self,
                                       current_seg: Dict,
                                       next_seg: Dict,
                                       scene_data: Dict) -> List[ContinuityAnchor]:
        """生成视觉匹配锚点"""

        anchors = []

        # 检查是否有需要视觉匹配的关键帧
        keyframes = self.keyframe_detector.detect_keyframes(current_seg, next_seg)

        for keyframe in keyframes:
            if keyframe.get("needs_match", False):
                anchor = ContinuityAnchor(
                    anchor_id=f"visual_match_{keyframe['id']}_{current_seg['segment_id']}_{next_seg['segment_id']}",
                    anchor_type=AnchorType.VISUAL_MATCH,
                    priority=AnchorPriority.HIGH,
                    from_segment=current_seg["segment_id"],
                    to_segment=next_seg["segment_id"],
                    temporal_constraint=f"在{keyframe.get('time', '过渡点')}",
                    description=f"视觉匹配要求：{keyframe.get('description', '')}",
                    sora_prompt=f"视觉匹配：{keyframe.get('sora_prompt', '')}",
                    visual_reference=keyframe.get("reference", ""),
                    mandatory=keyframe.get("mandatory", True)
                )
                anchors.append(anchor)

        return anchors

    def _generate_global_anchors(self,
                                 scene_data: Dict,
                                 segments: List[Dict]) -> List[ContinuityAnchor]:
        """生成全局一致性锚点"""

        anchors = []

        # 1. 角色全局外观锚点
        for character in scene_data.get("characters", []):
            char_name = character["name"]
            appearance = self._extract_appearance(character)

            if appearance:
                anchor = ContinuityAnchor(
                    anchor_id=f"global_app_{char_name}",
                    anchor_type=AnchorType.CHARACTER_APPEARANCE,
                    priority=AnchorPriority.CRITICAL,
                    from_segment="all",
                    to_segment="all",
                    temporal_constraint="整个场景中",
                    description=f"角色{char_name}的全局外观一致性：{appearance}",
                    sora_prompt=f"{char_name}的外观始终为：{self._appearance_to_sora_prompt(appearance)}",
                    mandatory=True
                )
                anchors.append(anchor)

        # 2. 场景全局环境锚点
        for scene in scene_data.get("scenes", []):
            anchor = ContinuityAnchor(
                anchor_id=f"global_env_{scene['scene_id']}",
                anchor_type=AnchorType.ENVIRONMENT,
                priority=AnchorPriority.CRITICAL,
                from_segment="all",
                to_segment="all",
                temporal_constraint=f"场景{scene['scene_id']}内",
                description=f"场景全局环境：{scene['location']}, {scene['time_of_day']}, {scene.get('weather', '')}",
                sora_prompt=f"场景环境：{scene['location']}, {scene['time_of_day']}, {scene.get('weather', '')}",
                mandatory=True,
                visual_reference=scene.get("description", "")
            )
            anchors.append(anchor)

        return anchors

    def _extract_appearance(self, character: Dict) -> Dict[str, str]:
        """从角色数据中提取外观信息"""
        appearance = {}

        if "appearance" in character and character["appearance"]:
            # 解析外观描述文本
            desc = character["appearance"]
            # 简单提取关键信息
            if "裹着" in desc:
                appearance["clothing"] = "旧羊毛毯"
            if "蜷坐在沙发上" in desc:
                appearance["posture"] = "蜷坐"

        return appearance

    def _appearance_to_sora_prompt(self, appearance: Dict) -> str:
        """将外观字典转换为Sora兼容的提示"""
        parts = []
        for key, value in appearance.items():
            parts.append(f"{key}:{value}")
        return ", ".join(parts)

    def _needs_appearance_anchor(self,
                                 current_state: CharacterStateVector,
                                 next_state: CharacterStateVector) -> bool:
        """检查是否需要外观锚点"""
        # 简化逻辑：外观字典有变化就需要
        return current_state.appearance != next_state.appearance

    def _needs_position_anchor(self,
                               current_state: CharacterStateVector,
                               next_state: CharacterStateVector) -> bool:
        """检查是否需要位置锚点"""
        return (current_state.location != next_state.location and
                current_state.location != "unknown" and
                next_state.location != "unknown")

    def _needs_posture_anchor(self,
                              current_state: CharacterStateVector,
                              next_state: CharacterStateVector) -> bool:
        """检查是否需要姿势锚点"""
        return (current_state.posture != next_state.posture and
                current_state.posture != "unknown")
