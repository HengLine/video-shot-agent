"""
@FileName: shot_validator.py
@Description: 
@Author: HengLine
@Time: 2026/1/6 15:42
"""

"""
验证工具
"""
from typing import Dict, List, Any, Optional, Tuple
import re


class ValidationUtils:
    """验证工具类"""

    @staticmethod
    def validate_color_hex(hex_color: str) -> bool:
        """验证十六进制颜色格式"""
        pattern = r'^#(?:[0-9a-fA-F]{3}){1,2}$'
        return bool(re.match(pattern, hex_color))

    @staticmethod
    def validate_time_range(time_range: Tuple[float, float]) -> bool:
        """验证时间范围"""
        if not isinstance(time_range, tuple) or len(time_range) != 2:
            return False

        start, end = time_range

        if not (isinstance(start, (int, float)) and isinstance(end, (int, float))):
            return False

        if start < 0 or end < 0:
            return False

        if start >= end:
            return False

        return True

    @staticmethod
    def validate_shot_duration(duration: float, shot_type: str = "general") -> Tuple[bool, Optional[str]]:
        """验证镜头时长"""
        duration_limits = {
            "extreme_close_up": (0.5, 3.0),
            "close_up": (1.0, 5.0),
            "medium_close_up": (2.0, 8.0),
            "medium_shot": (3.0, 10.0),
            "full_shot": (2.0, 8.0),
            "wide_shot": (4.0, 15.0),
            "extreme_wide_shot": (5.0, 20.0),
            "general": (1.0, 15.0)
        }

        limits = duration_limits.get(shot_type, duration_limits["general"])
        min_duration, max_duration = limits

        if duration < min_duration:
            return False, f"Duration too short for {shot_type}. Minimum: {min_duration}s"
        elif duration > max_duration:
            return False, f"Duration too long for {shot_type}. Maximum: {max_duration}s"
        else:
            return True, None

    @staticmethod
    def validate_prompt_length(prompt: str, max_length: int = 500) -> Tuple[bool, Optional[str]]:
        """验证提示词长度"""
        if len(prompt) > max_length:
            return False, f"Prompt too long ({len(prompt)} chars). Maximum: {max_length}"
        elif len(prompt) < 10:
            return False, "Prompt too short. Please provide more detail."
        else:
            return True, None

    @staticmethod
    def validate_camera_parameters(params: Dict[str, Any]) -> List[str]:
        """验证相机参数"""
        errors = []

        # 检查必要字段
        required_fields = ["shot_size", "camera_movement", "lens_focal_length"]
        for field in required_fields:
            if field not in params:
                errors.append(f"Missing required camera parameter: {field}")

        # 检查焦距有效性
        if "lens_focal_length" in params:
            focal = params["lens_focal_length"]
            if not isinstance(focal, int) or focal < 10 or focal > 1000:
                errors.append(f"Invalid lens focal length: {focal}. Should be between 10-1000mm")

        # 检查帧率有效性
        if "framerate" in params:
            fps = params["framerate"]
            valid_framerates = [24, 25, 30, 48, 50, 60, 120]
            if fps not in valid_framerates:
                errors.append(f"Unusual framerate: {fps}. Common values: {valid_framerates}")

        return errors

    @staticmethod
    def validate_transition(transition: Dict[str, Any]) -> List[str]:
        """验证镜头过渡"""
        errors = []

        # 检查必要字段
        required_fields = ["transition_type", "duration"]
        for field in required_fields:
            if field not in transition:
                errors.append(f"Missing required transition field: {field}")

        # 检查过渡类型有效性
        valid_transitions = ["cut", "dissolve", "fade", "wipe", "match_cut", "jump_cut"]
        if "transition_type" in transition:
            trans_type = transition["transition_type"]
            if trans_type not in valid_transitions:
                errors.append(f"Invalid transition type: {trans_type}. Valid: {valid_transitions}")

        # 检查持续时间有效性
        if "duration" in transition:
            duration = transition["duration"]
            if not isinstance(duration, (int, float)) or duration < 0:
                errors.append(f"Invalid transition duration: {duration}. Must be non-negative number")
            elif duration > 3.0:
                errors.append(f"Transition duration too long: {duration}s. Maximum recommended: 3.0s")

        return errors

    @staticmethod
    def validate_constraint(constraint: Dict[str, Any]) -> Dict[str, Any]:
        """验证约束"""
        errors = []
        warnings = []

        # 检查必要字段
        required_fields = ["constraint_id", "type", "description", "priority"]
        for field in required_fields:
            if field not in constraint:
                errors.append(f"Missing required constraint field: {field}")

        # 检查优先级
        if "priority" in constraint:
            priority = constraint["priority"]
            if not isinstance(priority, int) or priority < 1 or priority > 10:
                errors.append(f"Invalid priority: {priority}. Must be integer 1-10")

        # 检查适用范围
        if "applicable_segments" in constraint:
            segments = constraint["applicable_segments"]
            if not isinstance(segments, list):
                errors.append("applicable_segments must be a list")
            elif len(segments) == 0:
                warnings.append("Constraint has no applicable segments")

        # 检查时间范围
        if "temporal_range" in constraint and constraint["temporal_range"] is not None:
            time_range = constraint["temporal_range"]
            if not ValidationUtils.validate_time_range(time_range):
                errors.append("Invalid temporal_range format. Should be tuple (start, end)")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    @staticmethod
    def validate_segment_continuity(prev_segment: Dict[str, Any],
                                    next_segment: Dict[str, Any]) -> List[str]:
        """验证片段连续性"""
        issues = []

        # 检查时间连续性
        prev_end = prev_segment.get("time_range", (0, 0))[1]
        next_start = next_segment.get("time_range", (0, 0))[0]

        if abs(next_start - prev_end) > 0.1:  # 允许微小间隙
            issues.append(f"Time gap between segments: {prev_end} -> {next_start}")

        # 检查角色一致性
        prev_chars = set(prev_segment.get("character_states", {}).keys())
        next_chars = set(next_segment.get("character_states", {}).keys())

        # 检查突然出现的角色
        new_chars = next_chars - prev_chars
        if new_chars:
            issues.append(f"New characters suddenly appear: {new_chars}")

        # 检查突然消失的角色
        missing_chars = prev_chars - next_chars
        if missing_chars:
            issues.append(f"Characters suddenly disappear: {missing_chars}")

        return issues

    @staticmethod
    def calculate_validation_score(issues: List[str]) -> float:
        """计算验证得分（0-1）"""
        if not issues:
            return 1.0

        # 根据问题类型和数量计算得分
        critical_keywords = ["must", "required", "invalid", "missing", "error"]
        warning_keywords = ["warning", "suggestion", "recommend", "consider"]

        critical_count = sum(1 for issue in issues
                             if any(keyword in issue.lower() for keyword in critical_keywords))
        warning_count = sum(1 for issue in issues
                            if any(keyword in issue.lower() for keyword in warning_keywords))

        total_issues = len(issues)

        # 加权计算：关键问题权重更高
        weighted_score = (warning_count * 0.3 + critical_count * 0.7) / total_issues
        base_score = 1.0 - (total_issues / (total_issues + 5))  # 问题越多，得分越低

        return base_score * (1.0 - weighted_score * 0.5)


def test_validation_utils():
    """测试验证工具"""
    utils = ValidationUtils

    # 测试颜色验证
    print("Color validation:")
    print(f"  #FF5733: {utils.validate_color_hex('#FF5733')}")
    print(f"  invalid: {utils.validate_color_hex('invalid')}")

    # 测试时间范围验证
    print("\nTime range validation:")
    print(f"  (0, 5): {utils.validate_time_range((0, 5))}")
    print(f"  (5, 0): {utils.validate_time_range((5, 0))}")

    # 测试镜头时长验证
    print("\nShot duration validation:")
    valid, message = utils.validate_shot_duration(2.5, "close_up")
    print(f"  2.5s close-up: {valid} - {message}")

    valid, message = utils.validate_shot_duration(0.3, "close_up")
    print(f"  0.3s close-up: {valid} - {message}")

    # 测试提示词长度验证
    print("\nPrompt length validation:")
    short_prompt = "A shot"
    long_prompt = "A" * 600
    normal_prompt = "A cinematic shot of a character in a room with natural lighting"

    print(f"  Short: {utils.validate_prompt_length(short_prompt)}")
    print(f"  Long: {utils.validate_prompt_length(long_prompt)}")
    print(f"  Normal: {utils.validate_prompt_length(normal_prompt)}")

    # 测试相机参数验证
    print("\nCamera parameters validation:")
    bad_params = {"shot_size": "close_up", "lens_focal_length": 5}
    good_params = {"shot_size": "close_up", "camera_movement": "static", "lens_focal_length": 50}

    print(f"  Bad params errors: {utils.validate_camera_parameters(bad_params)}")
    print(f"  Good params errors: {utils.validate_camera_parameters(good_params)}")

    # 测试约束验证
    print("\nConstraint validation:")
    bad_constraint = {"constraint_id": "c1", "priority": 15}
    good_constraint = {
        "constraint_id": "c1",
        "type": "character_appearance",
        "description": "Blue shirt",
        "priority": 8,
        "applicable_segments": ["s001"]
    }

    print(f"  Bad constraint: {utils.validate_constraint(bad_constraint)}")
    print(f"  Good constraint: {utils.validate_constraint(good_constraint)}")

    # 测试验证得分计算
    print("\nValidation score calculation:")
    issues = ["Missing required field", "Consider using wider shot", "Invalid parameter"]
    score = utils.calculate_validation_score(issues)
    print(f"  Score for {len(issues)} issues: {score:.2f}")


if __name__ == "__main__":
    test_validation_utils()