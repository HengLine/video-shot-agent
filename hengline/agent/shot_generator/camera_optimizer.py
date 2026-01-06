"""
@FileName: camera_optimizer.py
@Description: 镜头语言优化器
@Author: HengLine
@Time: 2026/1/5 23:10
"""
from .model.shot_models import ShotSize, CameraMovement, CameraParameters, SoraShot
from hengline.agent.temporal_planner.temporal_planner_model import TimeSegment

from typing import Dict, Any, Optional
from enum import Enum


class ShotComposition(Enum):
    RULE_OF_THIRDS = "rule_of_thirds"
    CENTERED = "centered"
    LEADING_LINES = "leading_lines"
    SYMMETRICAL = "symmetrical"
    ASYMMETRICAL = "asymmetrical"
    FRAME_WITHIN_FRAME = "frame_within_frame"


class CameraOptimizer:
    """相机参数优化器"""

    # 内容类型到相机设置的映射
    CONTENT_TYPE_TO_CAMERA = {
        "dialogue_intimate": {
            "shot_size": ShotSize.MEDIUM_CLOSE_UP,
            "camera_movement": CameraMovement.SLOW_PUSH_IN,
            "lens_focal_length": 50,
            "aperture": "f/2.8",
            "framerate": 24,
            "camera_height": "eye_level",
            "camera_distance": "close",
            "movement_speed": "slow",
            "focus_technique": "rack_focus_between_characters",
            "depth_of_field": "shallow"
        },
        "action_fast": {
            "shot_size": ShotSize.WIDE_SHOT,
            "camera_movement": CameraMovement.HANDHELD_SHAKY,
            "lens_focal_length": 35,
            "aperture": "f/4",
            "framerate": 60,
            "camera_height": "low_angle",
            "camera_distance": "medium",
            "movement_speed": "fast",
            "focus_technique": "continuous_autofocus",
            "depth_of_field": "medium"
        },
        "emotional_reveal": {
            "shot_size": ShotSize.EXTREME_CLOSE_UP,
            "camera_movement": CameraMovement.STATIC,
            "lens_focal_length": 85,
            "aperture": "f/1.8",
            "framerate": 24,
            "camera_height": "eye_level",
            "camera_distance": "very_close",
            "movement_speed": "static",
            "focus_technique": "selective_focus_on_eyes",
            "depth_of_field": "very_shallow"
        },
        "establishing_shot": {
            "shot_size": ShotSize.EXTREME_WIDE_SHOT,
            "camera_movement": CameraMovement.SLOW_PAN,
            "lens_focal_length": 16,
            "aperture": "f/8",
            "framerate": 24,
            "camera_height": "high_angle",
            "camera_distance": "far",
            "movement_speed": "very_slow",
            "focus_technique": "deep_focus",
            "depth_of_field": "deep"
        }
    }

    def __init__(self):
        self.shot_sequences = []  # 记录镜头序列，避免重复

    def optimize_for_content(self, segment: TimeSegment,
                             previous_shot: Optional[SoraShot] = None) -> CameraParameters:
        """根据内容类型优化相机参数"""

        # 确定内容类型
        content_type = self._classify_content_type(segment)

        # 获取基础参数
        base_params = self.CONTENT_TYPE_TO_CAMERA.get(
            content_type,
            self.CONTENT_TYPE_TO_CAMERA["dialogue_intimate"]
        ).copy()

        # 根据具体内容微调
        adjusted_params = self._adjust_for_specific_content(segment, base_params)

        # 避免与前一镜头重复
        if previous_shot:
            adjusted_params = self._avoid_repetition(adjusted_params, previous_shot.camera_parameters)

        # 确保技术可行性
        adjusted_params = self._ensure_technical_feasibility(adjusted_params, segment.duration)

        # 转换为CameraParameters对象
        return CameraParameters(**adjusted_params)

    def _classify_content_type(self, segment: TimeSegment) -> str:
        """分类内容类型"""
        visual_content = segment.visual_content.lower()

        # 对话场景
        dialogue_keywords = ["say", "talk", "speak", "ask", "reply", "conversation", "dialogue"]
        if any(keyword in visual_content for keyword in dialogue_keywords):
            return "dialogue_intimate"

        # 动作场景
        action_keywords = ["run", "fight", "chase", "jump", "move quickly", "action"]
        if any(keyword in visual_content for keyword in action_keywords):
            return "action_fast"

        # 情感揭示
        emotion_keywords = ["tear", "cry", "smile happily", "emotional", "reveal", "realization"]
        if any(keyword in visual_content for keyword in emotion_keywords):
            return "emotional_reveal"

        # 建立场景
        establishing_keywords = ["establishing", "wide shot of", "overview", "panorama"]
        if any(keyword in visual_content for keyword in establishing_keywords):
            return "establishing_shot"

        # 默认
        return "dialogue_intimate"

    def _adjust_for_specific_content(self, segment: TimeSegment,
                                     base_params: Dict[str, Any]) -> Dict[str, Any]:
        """根据具体内容微调参数"""
        params = base_params.copy()
        content = segment.visual_content.lower()

        # 根据角色数量调整
        character_count = self._estimate_character_count(content)
        if character_count > 2:
            # 多个角色需要更广的镜头
            params["shot_size"] = ShotSize.FULL_SHOT
            params["lens_focal_length"] = 35
            params["depth_of_field"] = "medium"

        # 根据动作强度调整
        if segment.action_intensity > 1.5:
            params["framerate"] = 48  # 为慢动作留余地
            params["camera_movement"] = CameraMovement.HANDHELD_SHAKY

        # 根据情绪强度调整
        if segment.emotional_tone:
            if segment.emotional_tone in ["sad", "melancholy", "tense"]:
                params["camera_movement"] = CameraMovement.STATIC
                params["movement_speed"] = "static"
            elif segment.emotional_tone in ["happy", "excited"]:
                params["camera_movement"] = CameraMovement.SMOOTH_TRACKING
                params["movement_speed"] = "medium"

        # 根据持续时间调整
        if segment.duration < 3.0:
            # 短镜头使用更快的运动
            params["movement_speed"] = "fast"
            params["camera_movement"] = CameraMovement.DOLLY_IN

        return params

    def _estimate_character_count(self, content: str) -> int:
        """估计内容中的角色数量"""
        # 简单启发式方法
        character_keywords = ["man", "woman", "boy", "girl", "person", "character"]
        count = 0

        for keyword in character_keywords:
            if keyword in content:
                # 检查是否有多个实例
                count += content.count(keyword)

        # 或者检查提到的名字数量
        name_patterns = ["named", "called", "known as"]
        for pattern in name_patterns:
            if pattern in content:
                count += 1

        return max(1, count)  # 至少1个角色

    def _avoid_repetition(self, current_params: Dict[str, Any],
                          previous_params: CameraParameters) -> Dict[str, Any]:
        """避免与前一镜头重复"""
        params = current_params.copy()

        # 检查镜头大小是否相同
        if params["shot_size"] == previous_params.shot_size:
            # 调整为不同的镜头大小
            shot_sizes = [s for s in ShotSize if s != previous_params.shot_size]
            params["shot_size"] = shot_sizes[0] if shot_sizes else ShotSize.MEDIUM_SHOT

        # 检查镜头运动是否相同
        if params["camera_movement"] == previous_params.camera_movement:
            # 调整为不同的运动
            movements = [m for m in CameraMovement if m != previous_params.camera_movement]
            params["camera_movement"] = movements[0] if movements else CameraMovement.STATIC

        # 检查焦距是否太接近
        focal_diff = abs(params["lens_focal_length"] - previous_params.lens_focal_length)
        if focal_diff < 10:  # 焦距差异小于10mm
            # 调整焦距
            if params["lens_focal_length"] < 50:
                params["lens_focal_length"] = 85
            else:
                params["lens_focal_length"] = 35

        return params

    def _ensure_technical_feasibility(self, params: Dict[str, Any],
                                      duration: float) -> Dict[str, Any]:
        """确保技术可行性"""
        adjusted = params.copy()

        # 检查镜头运动与持续时间的匹配
        movement = adjusted["camera_movement"]

        # 快速运动需要足够时间
        if movement in [CameraMovement.DOLLY_IN, CameraMovement.DOLLY_OUT,
                        CameraMovement.HANDHELD_SHAKY]:
            if duration < 2.0:
                # 时间太短，改用静态或缓慢运动
                adjusted["camera_movement"] = CameraMovement.STATIC
                adjusted["movement_speed"] = "static"

        # 检查焦距与镜头大小的匹配
        focal_length = adjusted["lens_focal_length"]
        shot_size = adjusted["shot_size"]

        if shot_size in [ShotSize.EXTREME_CLOSE_UP, ShotSize.CLOSE_UP]:
            if focal_length < 50:
                adjusted["lens_focal_length"] = 85  # 更适合特写

        elif shot_size in [ShotSize.WIDE_SHOT, ShotSize.EXTREME_WIDE_SHOT]:
            if focal_length > 35:
                adjusted["lens_focal_length"] = 24  # 更适合广角

        return adjusted

    def determine_composition(self, segment: TimeSegment,
                              camera_params: CameraParameters) -> ShotComposition:
        """确定画面构图"""
        content = segment.visual_content.lower()

        # 对话场景：三分法
        if segment.content_type == "dialogue_intimate":
            return ShotComposition.RULE_OF_THIRDS

        # 对称场景：对称构图
        symmetry_keywords = ["symmetrical", "balanced", "centered", "facing each other"]
        if any(keyword in content for keyword in symmetry_keywords):
            return ShotComposition.SYMMETRICAL

        # 引导线场景
        line_keywords = ["road", "path", "corridor", "alley", "leading to"]
        if any(keyword in content for keyword in line_keywords):
            return ShotComposition.LEADING_LINES

        # 框架中框架
        frame_keywords = ["window", "doorway", "arch", "through", "framed by"]
        if any(keyword in content for keyword in frame_keywords):
            return ShotComposition.FRAME_WITHIN_FRAME

        # 默认：三分法
        return ShotComposition.RULE_OF_THIRDS

    def generate_movement_pattern(self, camera_params: CameraParameters,
                                  duration: float) -> Optional[str]:
        """生成镜头运动模式"""
        movement = camera_params.camera_movement

        # 静态镜头：无模式
        if movement == CameraMovement.STATIC:
            return None

        # 推拉镜头：缓入缓出
        if movement in [CameraMovement.SLOW_PUSH_IN, CameraMovement.SLOW_PULL_OUT,
                        CameraMovement.DOLLY_IN, CameraMovement.DOLLY_OUT]:
            return "ease_in_out"

        # 摇摄：线性或弧形
        if movement in [CameraMovement.PAN_LEFT, CameraMovement.PAN_RIGHT]:
            if duration > 3.0:
                return "arc_movement"  # 长时间摇摄用弧形
            else:
                return "linear_smooth"

        # 手持：随机微动
        if movement == CameraMovement.HANDHELD_SHAKY:
            return "micro_random_movements"

        # 平滑跟踪：贝塞尔曲线
        if movement == CameraMovement.SMOOTH_TRACKING:
            return "bezier_curve_smooth"

        return "linear_smooth"  # 默认
