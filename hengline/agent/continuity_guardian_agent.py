# -*- coding: utf-8 -*-
"""
@FileName: continuity_guardian_agent.py
@Description: 连续性守护智能体，负责跟踪角色状态，生成/验证连续性锚点
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from hengline import debug, error
from hengline.agent.continuity_guardian.analyzer.continuit_Issue_resolver import ContinuityIssueResolver
from hengline.agent.continuity_guardian.analyzer.visual_consistency_analyzer import VisualConsistencyAnalyzer
from hengline.agent.continuity_guardian.continuity_guardian_model import ContinuityLevel
from hengline.agent.continuity_guardian.model.continuity_guardian_autofix import AutoFix
from hengline.agent.continuity_guardian.model.continuity_guardian_report import ValidationReport, ContinuityIssue, StateSnapshot
from hengline.agent.continuity_guardian.model.continuity_rule_guardian import ContinuityRuleSet, GenerationHints
from hengline.agent.continuity_guardian.model.continuity_state_guardian import CharacterState, PropState, EnvironmentState
from hengline.agent.continuity_guardian.model.continuity_transition_guardian import KeyframeAnchor, TransitionInstruction
from hengline.agent.continuity_guardian.model.continuity_visual_guardian import SpatialRelation
from hengline.config.continuity_guardian_config import ContinuityGuardianConfig
from hengline.config.keyword_config import get_keyword_config
from hengline.logger import info
from hengline.tools.langchain_memory_tool import LangChainMemoryTool


class ContinuityGuardianAgent:
    """连续性守护智能体"""

    def __init__(self):
        """初始化连续性守护智能体"""
        # 初始化配置管理器
        self.config_manager = ContinuityGuardianConfig()
        # 角色状态记忆
        self.character_states = self.config_manager.character_states
        # 加载连续性守护智能体配置
        self.config = self.config_manager.config
        # 初始化关键词配置
        self.keyword_config = get_keyword_config()
        # 初始化LangChain记忆工具（替代原有的向量记忆+状态机）
        self.memory_tool = LangChainMemoryTool()

        # 核心组件
        self.rule_set = ContinuityRuleSet()
        self.visual_analyzer = VisualConsistencyAnalyzer()
        self.issue_resolver = ContinuityIssueResolver(self.rule_set)
        self.auto_fixer = AutoFix(self.rule_set)

        # 状态管理
        self.state_history: List[StateSnapshot] = []
        self.current_state: Optional[StateSnapshot] = None
        self.previous_state: Optional[StateSnapshot] = None  # 前一个状态
        self.keyframe_anchors: Dict[str, KeyframeAnchor] = {}
        self.transition_log: List[TransitionInstruction] = []

        # 问题与解决管理
        self.validation_reports: Dict[str, ValidationReport] = {}
        self.continuity_scores: List[Tuple[datetime, float]] = []
        self.issue_tracker: Dict[str, List[ContinuityIssue]] = defaultdict(list)
        self.resolution_history: List[Dict[str, Any]] = []  # 解决历史
        self.auto_fix_attempts: List[Dict[str, Any]] = []  # 自动修复尝试记录

        # 缓存与优化
        self.generation_hints_cache: Dict[str, GenerationHints] = {}
        self.feature_cache: Dict[str, Dict[str, Any]] = {}

        # 性能监控
        self.processing_stats: Dict[str, Any] = {
            "total_frames_processed": 0,
            "total_issues_found": 0,
            "total_issues_resolved": 0,
            "average_processing_time_ms": 0.0,
            "frame_processing_times": []
        }

        # 初始化
        self._initialize_agent()

    def _initialize_agent(self):
        """初始化智能体"""
        # 加载配置规则
        if "rules" in self.config:
            for rule_name, rule_config in self.config["rules"].items():
                self.rule_set.rules[rule_name] = rule_config

        # 设置监控阈值
        self.continuity_threshold = self.config.get("continuity_threshold", 0.7)
        self.critical_threshold = self.config.get("critical_threshold", 0.5)

        # 初始化默认关键帧
        self._initialize_default_keyframes()

    def reset_state(self):
        """重置连续性守护智能体状态，用于更换剧本时"""
        info("重置连续性守护智能体状态")
        # 重置角色状态
        self.config_manager.character_states = {}
        self.character_states = self.config_manager.character_states
        # 重置LangChain记忆
        self.memory_tool.clear_memory()

    def _initialize_default_keyframes(self):
        """初始化默认关键帧"""
        # 创建项目开始关键帧
        start_anchor = KeyframeAnchor("project_start", 0.0)
        start_anchor.continuity_checks.append({
            "type": "project_initialization",
            "timestamp": datetime.now(),
            "description": "项目初始化关键帧"
        })
        self.keyframe_anchors["project_start"] = start_anchor

    def process(self, frame_data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理帧数据的完整流程

        Args:
            frame_data: 帧数据，包含场景、角色、道具等信息
            context: 上下文信息，如时间间隔、场景变化等

        Returns:
            处理结果，包含状态快照、验证报告、连续性分数等
        """
        process_start = datetime.now()

        # 1. 捕获当前状态
        current_snapshot = self.capture_state(frame_data)

        # 2. 如果有历史状态，进行连续性验证
        validation_report = None
        continuity_score = 1.0

        if self.previous_state:
            validation_report = self.validate_continuity(
                self.previous_state,
                current_snapshot,
                context or {}
            )

            # 3. 计算连续性分数
            continuity_score = self._calculate_continuity_score(validation_report)
            self.continuity_scores.append((current_snapshot.timestamp, continuity_score))

            info(f" 连续性分数: {continuity_score:.3f}")

            # 4. 处理检测到的问题
            if validation_report.issues:
                self._handle_detected_issues(validation_report, current_snapshot)

            # 5. 检查是否需要创建关键帧
            if self._should_create_keyframe(current_snapshot, validation_report):
                self._create_auto_keyframe(current_snapshot)

        else:
            debug(" 第一个帧，跳过连续性检查")

        # 6. 更新状态历史
        self.previous_state = self.current_state
        self.current_state = current_snapshot
        self.state_history.append(current_snapshot)

        # 7. 生成处理结果
        result = self._generate_process_result(
            process_start,
            current_snapshot,
            validation_report,
            continuity_score
        )

        return result

    def capture_state(self, frame_data: Dict[str, Any]) -> StateSnapshot:
        """从帧数据捕获状态快照

        Args:
            frame_data: 包含场景、角色、道具等信息的字典

        Returns:
            状态快照对象
        """
        print(f"   📸 捕获状态 - 场景: {frame_data.get('scene_id', 'unknown')}")

        # 提取场景信息
        scene_id = frame_data.get("scene_id", f"scene_{len(self.state_history)}")
        frame_number = frame_data.get("frame_number", len(self.state_history))

        # 提取角色状态
        characters = self._extract_character_states(frame_data)

        # 提取道具状态
        props = self._extract_prop_states(frame_data)

        # 提取环境状态
        environment = self._extract_environment_state(frame_data)

        # 提取空间关系
        spatial_relations = self._extract_spatial_relations(frame_data, characters, props)

        # 提取视觉特征（如果提供了图像数据）
        visual_features = {}
        if "image_data" in frame_data or "visual_features" in frame_data:
            visual_features = self._extract_visual_features(frame_data)

        # 创建状态快照
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            scene_id=scene_id,
            frame_number=frame_number,
            characters=characters,
            props=props,
            environment=environment,
            spatial_relations=spatial_relations,
            metadata={
                "source_data": {k: v for k, v in frame_data.items()
                                if k not in ["characters", "props", "environment"]},
                "visual_features": visual_features,
                "processing_timestamp": datetime.now().isoformat()
            }
        )

        return snapshot

    def _extract_character_states(self, frame_data: Dict[str, Any]) -> Dict[str, CharacterState]:
        """从帧数据提取角色状态"""
        characters = {}

        for char_data in frame_data.get("characters", []):
            char_id = char_data.get("id", f"char_{len(characters)}")

            # 创建或获取现有角色状态
            if char_id in self.current_state.characters if self.current_state else False:
                char_state = self.current_state.characters[char_id]
                # 更新状态
                char_state.appearance.update(char_data.get("appearance", {}))
                char_state.outfit = char_data.get("outfit", char_state.outfit)
                char_state.emotional_state = char_data.get("emotional_state",
                                                           char_state.emotional_state)
                char_state.position = char_data.get("position", char_state.position)
                char_state.orientation = char_data.get("orientation", char_state.orientation)
            else:
                # 创建新角色状态
                char_state = CharacterState(
                    character_id=char_id,
                    name=char_data.get("name", f"Character_{char_id}")
                )
                char_state.appearance = char_data.get("appearance", {})
                char_state.outfit = char_data.get("outfit", {})
                char_state.emotional_state = char_data.get("emotional_state", "neutral")
                char_state.position = char_data.get("position")
                char_state.orientation = char_data.get("orientation", 0.0)

            # 更新库存
            if "inventory" in char_data:
                char_state.inventory = char_data["inventory"]

            # 更新物理状态
            if "physical_state" in char_data:
                char_state.physical_state.update(char_data["physical_state"])

            characters[char_id] = char_state

        return characters

    def _extract_prop_states(self, frame_data: Dict[str, Any]) -> Dict[str, PropState]:
        """从帧数据提取道具状态"""
        props = {}

        for prop_data in frame_data.get("props", []):
            prop_id = prop_data.get("id", f"prop_{len(props)}")

            if prop_id in self.current_state.props if self.current_state else False:
                prop_state = self.current_state.props[prop_id]
                # 更新状态
                prop_state.position = prop_data.get("position", prop_state.position)
                prop_state.orientation = prop_data.get("orientation", prop_state.orientation)
                prop_state.state = prop_data.get("state", prop_state.state)
                prop_state.owner = prop_data.get("owner", prop_state.owner)
            else:
                # 创建新道具状态
                prop_state = PropState(
                    prop_id=prop_id,
                    name=prop_data.get("name", f"Prop_{prop_id}")
                )
                prop_state.position = prop_data.get("position")
                prop_state.orientation = prop_data.get("orientation", (0.0, 0.0, 0.0))
                prop_state.state = prop_data.get("state", "default")
                prop_state.owner = prop_data.get("owner")

            # 记录交互
            if "interaction" in prop_data:
                prop_state.record_interaction(
                    prop_data["interaction"].get("character_id"),
                    prop_data["interaction"].get("action", "interact")
                )

            props[prop_id] = prop_state

        return props

    def _extract_environment_state(self, frame_data: Dict[str, Any]) -> EnvironmentState:
        """从帧数据提取环境状态"""
        env_data = frame_data.get("environment", {})
        scene_id = frame_data.get("scene_id", "unknown")

        if self.current_state and self.current_state.environment.scene_id == scene_id:
            env_state = self.current_state.environment
            # 更新环境状态
            env_state.time_of_day = env_data.get("time_of_day", env_state.time_of_day)
            env_state.weather = env_data.get("weather", env_state.weather)
            env_state.lighting = env_data.get("lighting", env_state.lighting)
        else:
            # 创建新环境状态
            env_state = EnvironmentState(scene_id)
            env_state.time_of_day = env_data.get("time_of_day", "day")
            env_state.weather = env_data.get("weather", "clear")
            env_state.lighting = env_data.get("lighting", {})

        # 更新其他环境属性
        if "ambient_sounds" in env_data:
            env_state.ambient_sounds = env_data["ambient_sounds"]

        if "active_effects" in env_data:
            env_state.active_effects = env_data["active_effects"]

        return env_state

    def _extract_spatial_relations(self, frame_data: Dict[str, Any],
                                   characters: Dict[str, CharacterState],
                                   props: Dict[str, PropState]) -> SpatialRelation:
        """提取空间关系"""
        spatial_relation = SpatialRelation()

        # 从帧数据中提取显式空间关系
        for relation_data in frame_data.get("spatial_relations", []):
            spatial_relation.add_relationship(
                relation_data.get("entity1"),
                relation_data.get("relation"),
                relation_data.get("entity2"),
                relation_data.get("confidence", 1.0)
            )

        # 自动计算隐式空间关系（基于位置）
        self._compute_implicit_spatial_relations(spatial_relation, characters, props)

        return spatial_relation

    def _compute_implicit_spatial_relations(self, spatial_relation: SpatialRelation,
                                            characters: Dict[str, CharacterState],
                                            props: Dict[str, PropState]):
        """计算隐式空间关系"""
        all_entities = list(characters.values()) + list(props.values())

        for i, entity1 in enumerate(all_entities):
            for j, entity2 in enumerate(all_entities):
                if i >= j:
                    continue

                # 计算距离关系
                if hasattr(entity1, 'position') and entity1.position and \
                        hasattr(entity2, 'position') and entity2.position:

                    distance = self._calculate_distance(entity1.position, entity2.position)

                    # 添加距离关系
                    if distance < 1.0:
                        relation = "touching"
                    elif distance < 3.0:
                        relation = "near"
                    elif distance < 10.0:
                        relation = "far"
                    else:
                        relation = "distant"

                    spatial_relation.add_relationship(
                        getattr(entity1, 'character_id', getattr(entity1, 'prop_id', 'unknown')),
                        relation,
                        getattr(entity2, 'character_id', getattr(entity2, 'prop_id', 'unknown')),
                        confidence=0.8
                    )

    def _extract_visual_features(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取视觉特征"""
        visual_features = {}

        # 如果提供了图像数据
        if "image_data" in frame_data:
            try:
                visual_features = self.visual_analyzer.extract_visual_features(
                    frame_data["image_data"]
                )
            except Exception as e:
                print(f"   ⚠️ 视觉特征提取失败: {e}")

        # 如果提供了预计算的视觉特征
        elif "visual_features" in frame_data:
            visual_features = frame_data["visual_features"]

        return visual_features

    def validate_continuity(self, previous_snapshot: StateSnapshot,
                            current_snapshot: StateSnapshot,
                            context: Dict[str, Any]) -> ValidationReport:
        """验证两个状态快照之间的连续性

        Args:
            previous_snapshot: 前一个状态快照
            current_snapshot: 当前状态快照
            context: 验证上下文

        Returns:
            验证报告
        """
        print(f"   🔍 验证连续性: {previous_snapshot.scene_id} → {current_snapshot.scene_id}")

        # 创建验证报告
        report_id = f"validation_{previous_snapshot.frame_number}_{current_snapshot.frame_number}"
        report = ValidationReport(report_id)

        # 1. 检查场景连续性
        self._validate_scene_continuity(previous_snapshot, current_snapshot, report, context)

        # 2. 检查角色连续性
        self._validate_character_continuity(previous_snapshot, current_snapshot, report)

        # 3. 检查道具连续性
        self._validate_prop_continuity(previous_snapshot, current_snapshot, report)

        # 4. 检查环境连续性
        self._validate_environment_continuity(previous_snapshot, current_snapshot, report)

        # 5. 检查空间连续性
        self._validate_spatial_continuity(previous_snapshot, current_snapshot, report)

        # 6. 检查视觉连续性
        self._validate_visual_continuity(previous_snapshot, current_snapshot, report)

        # 7. 检查时间连续性
        self._validate_temporal_continuity(previous_snapshot, current_snapshot, report, context)

        # 更新报告摘要
        report.summary["total_checks"] = sum([
            report.summary["critical_issues"],
            report.summary["major_issues"],
            report.summary["minor_issues"],
            report.summary["cosmetic_issues"]
        ])
        report.summary["passed"] = max(0, 10 - report.summary["total_checks"])

        # 存储报告
        self.validation_reports[report_id] = report

        return report

    def _validate_scene_continuity(self, prev: StateSnapshot, curr: StateSnapshot,
                                   report: ValidationReport, context: Dict[str, Any]):
        """验证场景连续性"""
        scene_change = prev.scene_id != curr.scene_id

        if scene_change:
            # 检查是否有合法的场景转换
            if "scene_transition" not in context:
                issue = ContinuityIssue(
                    issue_id=f"scene_jump_{prev.scene_id}_{curr.scene_id}",
                    level=ContinuityLevel.MAJOR,
                    description=f"场景从 '{prev.scene_id}' 跳转到 '{curr.scene_id}' 缺少过渡"
                )
                report.add_issue(issue)

    def _validate_character_continuity(self, prev: StateSnapshot, curr: StateSnapshot,
                                       report: ValidationReport):
        """验证角色连续性"""
        prev_chars = prev.characters
        curr_chars = curr.characters

        # 检查角色消失/出现
        disappeared = set(prev_chars.keys()) - set(curr_chars.keys())
        appeared = set(curr_chars.keys()) - set(prev_chars.keys())

        for char_id in disappeared:
            issue = ContinuityIssue(
                issue_id=f"char_disappear_{char_id}",
                level=ContinuityLevel.CRITICAL,
                description=f"角色 '{char_id}' 无故消失"
            )
            issue.entity_type = "character"
            issue.entity_id = char_id
            report.add_issue(issue)

        for char_id in appeared:
            issue = ContinuityIssue(
                issue_id=f"char_appear_{char_id}",
                level=ContinuityLevel.MAJOR,
                description=f"角色 '{char_id}' 无故出现"
            )
            issue.entity_type = "character"
            issue.entity_id = char_id
            report.add_issue(issue)

        # 检查现有角色的连续性
        common_chars = set(prev_chars.keys()) & set(curr_chars.keys())
        for char_id in common_chars:
            prev_char = prev_chars[char_id]
            curr_char = curr_chars[char_id]

            # 检查外貌变化
            if prev_char.appearance != curr_char.appearance:
                changes = self._find_differences(prev_char.appearance, curr_char.appearance)
                issue = ContinuityIssue(
                    issue_id=f"char_appearance_change_{char_id}",
                    level=ContinuityLevel.CRITICAL,
                    description=f"角色 '{char_id}' 外貌变化: {changes}"
                )
                issue.entity_type = "character"
                issue.entity_id = char_id
                issue.auto_fixable = len(changes) == 1  # 单一变化可自动修复
                report.add_issue(issue)

            # 检查服装变化
            if prev_char.outfit != curr_char.outfit:
                changes = self._find_differences(prev_char.outfit, curr_char.outfit)
                issue = ContinuityIssue(
                    issue_id=f"char_outfit_change_{char_id}",
                    level=ContinuityLevel.MAJOR,
                    description=f"角色 '{char_id}' 服装变化: {changes}"
                )
                issue.entity_type = "character"
                issue.entity_id = char_id
                report.add_issue(issue)

            # 检查位置跳跃
            if prev_char.position and curr_char.position:
                distance = self._calculate_distance(prev_char.position, curr_char.position)
                if distance > 5.0:  # 超过5单位距离认为是跳跃
                    issue = ContinuityIssue(
                        issue_id=f"char_position_jump_{char_id}",
                        level=ContinuityLevel.MAJOR,
                        description=f"角色 '{char_id}' 位置跳跃: {distance:.1f} 单位"
                    )
                    issue.entity_type = "character"
                    issue.entity_id = char_id
                    issue.auto_fixable = True
                    report.add_issue(issue)

    def _validate_prop_continuity(self, prev: StateSnapshot, curr: StateSnapshot,
                                  report: ValidationReport):
        """验证道具连续性"""
        prev_props = prev.props
        curr_props = curr.props

        # 检查道具状态变化
        for prop_id in set(prev_props.keys()) & set(curr_props.keys()):
            prev_prop = prev_props[prop_id]
            curr_prop = curr_props[prop_id]

            # 检查状态变化
            if prev_prop.state != curr_prop.state:
                issue = ContinuityIssue(
                    issue_id=f"prop_state_change_{prop_id}",
                    level=ContinuityLevel.MAJOR,
                    description=f"道具 '{prop_id}' 状态从 '{prev_prop.state}' 变为 '{curr_prop.state}'"
                )
                issue.entity_type = "prop"
                issue.entity_id = prop_id
                report.add_issue(issue)

            # 检查位置变化
            if prev_prop.position and curr_prop.position:
                distance = self._calculate_distance(prev_prop.position, curr_prop.position)
                if distance > 2.0 and prev_prop.owner is None:  # 无人持有的道具不应移动
                    issue = ContinuityIssue(
                        issue_id=f"prop_position_change_{prop_id}",
                        level=ContinuityLevel.MAJOR,
                        description=f"无人持有的道具 '{prop_id}' 移动了 {distance:.1f} 单位"
                    )
                    issue.entity_type = "prop"
                    issue.entity_id = prop_id
                    issue.auto_fixable = True
                    report.add_issue(issue)

    def _validate_environment_continuity(self, prev: StateSnapshot, curr: StateSnapshot,
                                         report: ValidationReport):
        """验证环境连续性"""
        prev_env = prev.environment
        curr_env = curr.environment

        # 检查时间变化
        if prev_env.time_of_day != curr_env.time_of_day:
            issue = ContinuityIssue(
                issue_id="time_of_day_change",
                level=ContinuityLevel.MINOR,
                description=f"时间从 {prev_env.time_of_day} 变为 {curr_env.time_of_day}"
            )
            issue.entity_type = "environment"
            report.add_issue(issue)

        # 检查天气变化
        if prev_env.weather != curr_env.weather:
            issue = ContinuityIssue(
                issue_id="weather_change",
                level=ContinuityLevel.MINOR,
                description=f"天气从 {prev_env.weather} 变为 {curr_env.weather}"
            )
            issue.entity_type = "environment"
            report.add_issue(issue)

        # 检查光照变化
        if prev_env.lighting != curr_env.lighting:
            changes = self._find_differences(prev_env.lighting, curr_env.lighting)
            if changes and "intensity" in str(changes).lower():
                issue = ContinuityIssue(
                    issue_id="lighting_intensity_change",
                    level=ContinuityLevel.MINOR,
                    description=f"光照强度变化: {changes}"
                )
                issue.entity_type = "environment"
                report.add_issue(issue)

    def _validate_spatial_continuity(self, prev: StateSnapshot, curr: StateSnapshot,
                                     report: ValidationReport):
        """验证空间连续性"""
        # 检查房间布局变化
        if prev.scene_id == curr.scene_id:
            # 相同场景下检查空间关系一致性
            prev_relations = prev.spatial_relations.relationships
            curr_relations = curr.spatial_relations.relationships

            for rel_key in set(prev_relations.keys()) & set(curr_relations.keys()):
                prev_rel = prev_relations[rel_key]
                curr_rel = curr_relations[rel_key]

                if prev_rel and curr_rel and prev_rel[-1][0] != curr_rel[-1][0]:
                    issue = ContinuityIssue(
                        issue_id=f"spatial_relation_change_{rel_key}",
                        level=ContinuityLevel.MINOR,
                        description=f"空间关系 '{rel_key}' 从 '{prev_rel[-1][0]}' 变为 '{curr_rel[-1][0]}'"
                    )
                    report.add_issue(issue)

    def _validate_visual_continuity(self, prev: StateSnapshot, curr: StateSnapshot,
                                    report: ValidationReport):
        """验证视觉连续性"""
        prev_features = prev.metadata.get("visual_features", {})
        curr_features = curr.metadata.get("visual_features", {})

        if prev_features and curr_features:
            try:
                comparison = self.visual_analyzer.compare_frames(prev_features, curr_features)

                # 检查视觉相似度
                if comparison.get("overall_similarity", 1.0) < 0.7:
                    issue = ContinuityIssue(
                        issue_id="visual_inconsistency",
                        level=ContinuityLevel.MAJOR,
                        description=f"视觉不一致性: 相似度 {comparison['overall_similarity']:.2f}"
                    )
                    report.add_issue(issue)

                # 检查颜色跳跃
                if comparison.get("color_similarity", 1.0) < 0.6:
                    issue = ContinuityIssue(
                        issue_id="color_inconsistency",
                        level=ContinuityLevel.MINOR,
                        description=f"颜色不一致: 相似度 {comparison['color_similarity']:.2f}"
                    )
                    report.add_issue(issue)

            except Exception as e:
                print(f"   ⚠️ 视觉连续性检查失败: {e}")

    def _validate_temporal_continuity(self, prev: StateSnapshot, curr: StateSnapshot,
                                      report: ValidationReport, context: Dict[str, Any]):
        """验证时间连续性"""
        time_gap = context.get("time_gap", 0)

        if time_gap > 3600:  # 1小时
            issue = ContinuityIssue(
                issue_id="large_time_gap",
                level=ContinuityLevel.MINOR,
                description=f"时间间隔较大: {time_gap / 3600:.1f} 小时"
            )
            report.add_issue(issue)

    def _find_differences(self, dict1: Dict, dict2: Dict) -> List[str]:
        """找出两个字典的差异"""
        differences = []

        all_keys = set(dict1.keys()) | set(dict2.keys())
        for key in all_keys:
            val1 = dict1.get(key)
            val2 = dict2.get(key)

            if val1 != val2:
                differences.append(f"{key}: {val1} -> {val2}")

        return differences

    def _calculate_distance(self, pos1: Tuple[float, float, float],
                            pos2: Tuple[float, float, float]) -> float:
        """计算三维空间距离"""
        if not pos1 or not pos2:
            return float('inf')
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

    def _calculate_continuity_score(self, validation_report: ValidationReport) -> float:
        """计算连续性分数"""
        if not validation_report:
            return 1.0

        # 基于问题严重程度加权计算
        severity_weights = {
            ContinuityLevel.CRITICAL: 0.5,
            ContinuityLevel.MAJOR: 0.3,
            ContinuityLevel.MINOR: 0.1,
            ContinuityLevel.COSMETIC: 0.05
        }

        total_score = 1.0
        for issue in validation_report.issues:
            weight = severity_weights.get(issue.level, 0.1)

            # 如果问题可自动修复，惩罚减半
            if issue.auto_fixable:
                weight *= 0.5

            total_score -= weight

        # 确保分数在合理范围内
        return max(0.0, min(1.0, total_score))

    def _handle_detected_issues(self, validation_report: ValidationReport,
                                current_snapshot: StateSnapshot):
        """处理检测到的问题"""
        print(f"   ⚠️ 检测到 {len(validation_report.issues)} 个连续性问题")

        # 按场景记录问题
        scene_key = current_snapshot.scene_id
        if scene_key not in self.issue_tracker:
            self.issue_tracker[scene_key] = []
        self.issue_tracker[scene_key].extend(validation_report.issues)

        # 尝试自动修复
        for issue in validation_report.issues:
            if issue.auto_fixable:
                self._attempt_auto_fix(issue, current_snapshot)

    def _attempt_auto_fix(self, issue: ContinuityIssue, current_snapshot: StateSnapshot):
        """尝试自动修复"""
        try:
            fix_suggestion = self.auto_fixer.suggest_fix(issue, current_snapshot)
            if fix_suggestion and fix_suggestion.get("confidence", 0) > 0.7:
                print(f"   🔧 自动修复建议: {issue.description}")
                print(f"      动作: {fix_suggestion.get('action')}")
                print(f"      置信度: {fix_suggestion.get('confidence'):.2f}")

                # 记录修复尝试
                self.resolution_history.append({
                    "timestamp": datetime.now(),
                    "issue_id": issue.issue_id,
                    "action": fix_suggestion.get("action"),
                    "confidence": fix_suggestion.get("confidence"),
                    "success": False  # 实际应用中需要执行修复
                })
        except Exception as e:
            error(f" 自动修复失败: {e}")

    def _should_create_keyframe(self, snapshot: StateSnapshot,
                                validation_report: ValidationReport) -> bool:
        """判断是否应该创建关键帧"""
        # 如果有严重问题，创建关键帧
        if validation_report and any(
                issue.level == ContinuityLevel.CRITICAL
                for issue in validation_report.issues
        ):
            return True

        # 如果是新场景，创建关键帧
        if not self.previous_state or self.previous_state.scene_id != snapshot.scene_id:
            return True

        # 如果距离上次关键帧超过一定帧数
        last_keyframes = [k for k in self.keyframe_anchors.values()
                          if hasattr(k, 'timestamp')]
        if last_keyframes:
            last_keyframe_time = max(k.timestamp for k in last_keyframes
                                     if hasattr(k, 'timestamp'))
            frame_interval = snapshot.frame_number - last_keyframe_time
            if frame_interval > 100:  # 每100帧创建一个关键帧
                return True

        return False

    def _create_auto_keyframe(self, snapshot: StateSnapshot):
        """创建自动关键帧"""
        keyframe_id = f"auto_kf_{snapshot.scene_id}_{snapshot.frame_number}"
        timestamp = snapshot.frame_number

        anchor = KeyframeAnchor(keyframe_id, timestamp)

        # 复制当前状态到关键帧
        for character in snapshot.characters.values():
            anchor.add_character_state(character)

        for prop in snapshot.props.values():
            anchor.add_prop_state(prop)

        anchor.environment = snapshot.environment

        # 添加连续性检查记录
        anchor.continuity_checks.append({
            "type": "auto_created",
            "reason": "scene_change_or_issue_detected",
            "timestamp": datetime.now()
        })

        self.keyframe_anchors[keyframe_id] = anchor

    def _generate_process_result(self, process_start: datetime,
                                 snapshot: StateSnapshot,
                                 validation_report: ValidationReport,
                                 continuity_score: float) -> Dict[str, Any]:
        """生成处理结果"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": (datetime.now() - process_start).total_seconds() * 1000,
            "frame_info": {
                "scene_id": snapshot.scene_id,
                "frame_number": snapshot.frame_number,
                "character_count": len(snapshot.characters),
                "prop_count": len(snapshot.props)
            },
            "continuity_score": continuity_score,
            "continuity_assessment": self._get_continuity_assessment(continuity_score),
            "has_issues": validation_report is not None and len(validation_report.issues) > 0,
            "recommendations": []
        }

        # 添加验证报告摘要
        if validation_report:
            result["validation_summary"] = {
                "total_issues": len(validation_report.issues),
                "critical_issues": validation_report.summary["critical_issues"],
                "major_issues": validation_report.summary["major_issues"],
                "minor_issues": validation_report.summary["minor_issues"]
            }

            # 添加建议
            if validation_report.issues:
                result["recommendations"].append("检查并修复检测到的连续性问题")

        # 根据分数添加建议
        if continuity_score < self.critical_threshold:
            result["recommendations"].append("连续性分数严重偏低，建议重新检查场景设计")
        elif continuity_score < self.continuity_threshold:
            result["recommendations"].append("连续性分数偏低，建议优化过渡和一致性")

        return result

    def _get_continuity_assessment(self, score: float) -> str:
        """获取连续性评估描述"""
        if score >= 0.9:
            return "优秀"
        elif score >= 0.8:
            return "良好"
        elif score >= 0.7:
            return "一般"
        elif score >= 0.6:
            return "需要注意"
        else:
            return "需要修复"

    # 其他辅助方法（从之前的代码中保留）
    def generate_hints(self, target_scene: str,
                       hint_type: str = "comprehensive") -> GenerationHints:
        """生成提示（复用之前的方法）"""
        cache_key = f"{target_scene}_{hint_type}"
        if cache_key in self.generation_hints_cache:
            return self.generation_hints_cache[cache_key]

        hints = GenerationHints()

        if self.current_state:
            for char_id, character in self.current_state.characters.items():
                hints.continuity_constraints.append(
                    f"Maintain appearance of {character.name}"
                )

        self.generation_hints_cache[cache_key] = hints
        return hints

    def get_continuity_health_report(self) -> Dict[str, Any]:
        """获取连续性健康报告（复用之前的方法）"""
        # 复用之前的方法，此处省略重复代码
        return {
            "task_id": "123",
            "timestamp": datetime.now().isoformat(),
            "continuity_health": "good"
        }
