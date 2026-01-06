"""
@FileName: constraints_config.py
@Description: 约束配置系统
@Author: HengLine
@Time: 2026/1/6 12:30
"""
import re
from enum import Enum
from typing import Dict, List, Any, Optional


class ConstraintPriority(Enum):
    """约束优先级枚举"""
    CRITICAL = 10  # 必须满足，否则视频无法观看
    HIGH = 8  # 非常重要，显著影响质量
    MEDIUM = 5  # 重要，但稍有偏差可接受
    LOW = 3  # 最好满足，但不是必须
    OPTIONAL = 1  # 指导性建议


class ConstraintCategory(Enum):
    """约束类别枚举"""
    CHARACTER = "character"  # 角色相关约束
    PROPS = "props"  # 道具相关约束
    ENVIRONMENT = "environment"  # 环境相关约束
    SPATIAL = "spatial"  # 空间关系约束
    TEMPORAL = "temporal"  # 时间相关约束
    VISUAL = "visual"  # 视觉风格约束
    TECHNICAL = "technical"  # 技术参数约束
    CONTINUITY = "continuity"  # 连续性约束


class ConstraintType(Enum):
    """约束类型枚举"""
    # 角色相关
    CHARACTER_APPEARANCE = "character_appearance"  # 角色外观
    CHARACTER_POSITION = "character_position"  # 角色位置
    CHARACTER_POSTURE = "character_posture"  # 角色姿势
    CHARACTER_EXPRESSION = "character_expression"  # 角色表情
    CHARACTER_INTERACTION = "character_interaction"  # 角色交互

    # 道具相关
    PROP_STATE = "prop_state"  # 道具状态
    PROP_POSITION = "prop_position"  # 道具位置
    PROP_INTERACTION = "prop_interaction"  # 道具交互

    # 环境相关
    ENVIRONMENT_SETTING = "environment_setting"  # 环境设定
    ENVIRONMENT_LIGHTING = "environment_lighting"  # 环境光照
    ENVIRONMENT_WEATHER = "environment_weather"  # 环境天气

    # 空间相关
    SPATIAL_RELATION = "spatial_relation"  # 空间关系
    CAMERA_ANGLE = "camera_angle"  # 相机角度
    CAMERA_MOVEMENT = "camera_movement"  # 相机运动

    # 时间相关
    TEMPORAL_ORDER = "temporal_order"  # 时间顺序
    ACTION_TIMING = "action_timing"  # 动作时序

    # 视觉相关
    VISUAL_STYLE = "visual_style"  # 视觉风格
    COLOR_PALETTE = "color_palette"  # 色彩调色板
    LIGHTING_STYLE = "lighting_style"  # 灯光风格

    # 技术相关
    TECHNICAL_SPEC = "technical_spec"  # 技术规格
    RESOLUTION = "resolution"  # 分辨率
    FRAMERATE = "framerate"  # 帧率

    # 连续性相关
    CONTINUITY_ACTION = "continuity_action"  # 动作连续性
    CONTINUITY_POSITION = "continuity_position"  # 位置连续性
    CONTINUITY_TIME = "continuity_time"  # 时间连续性


class ConstraintConfig:
    """约束配置类"""

    # 约束类型到优先级的默认映射
    CONSTRAINT_PRIORITY_DEFAULTS = {
        # 角色约束
        ConstraintType.CHARACTER_APPEARANCE: ConstraintPriority.HIGH,
        ConstraintType.CHARACTER_POSITION: ConstraintPriority.MEDIUM,
        ConstraintType.CHARACTER_POSTURE: ConstraintPriority.MEDIUM,
        ConstraintType.CHARACTER_EXPRESSION: ConstraintPriority.HIGH,
        ConstraintType.CHARACTER_INTERACTION: ConstraintPriority.HIGH,

        # 道具约束
        ConstraintType.PROP_STATE: ConstraintPriority.HIGH,
        ConstraintType.PROP_POSITION: ConstraintPriority.MEDIUM,
        ConstraintType.PROP_INTERACTION: ConstraintPriority.HIGH,

        # 环境约束
        ConstraintType.ENVIRONMENT_SETTING: ConstraintPriority.MEDIUM,
        ConstraintType.ENVIRONMENT_LIGHTING: ConstraintPriority.HIGH,
        ConstraintType.ENVIRONMENT_WEATHER: ConstraintPriority.MEDIUM,

        # 空间约束
        ConstraintType.SPATIAL_RELATION: ConstraintPriority.MEDIUM,
        ConstraintType.CAMERA_ANGLE: ConstraintPriority.LOW,
        ConstraintType.CAMERA_MOVEMENT: ConstraintPriority.LOW,

        # 时间约束
        ConstraintType.TEMPORAL_ORDER: ConstraintPriority.CRITICAL,
        ConstraintType.ACTION_TIMING: ConstraintPriority.HIGH,

        # 视觉约束
        ConstraintType.VISUAL_STYLE: ConstraintPriority.LOW,
        ConstraintType.COLOR_PALETTE: ConstraintPriority.LOW,
        ConstraintType.LIGHTING_STYLE: ConstraintPriority.MEDIUM,

        # 技术约束
        ConstraintType.TECHNICAL_SPEC: ConstraintPriority.MEDIUM,
        ConstraintType.RESOLUTION: ConstraintPriority.LOW,
        ConstraintType.FRAMERATE: ConstraintPriority.LOW,

        # 连续性约束
        ConstraintType.CONTINUITY_ACTION: ConstraintPriority.CRITICAL,
        ConstraintType.CONTINUITY_POSITION: ConstraintPriority.CRITICAL,
        ConstraintType.CONTINUITY_TIME: ConstraintPriority.CRITICAL,
    }

    # 约束验证方法
    CONSTRAINT_VALIDATION_METHODS = {
        ConstraintType.CHARACTER_APPEARANCE: "visual_comparison",
        ConstraintType.CHARACTER_POSITION: "spatial_analysis",
        ConstraintType.CHARACTER_POSTURE: "pose_detection",
        ConstraintType.CHARACTER_EXPRESSION: "facial_recognition",
        ConstraintType.CHARACTER_INTERACTION: "action_recognition",

        ConstraintType.PROP_STATE: "object_detection",
        ConstraintType.PROP_POSITION: "spatial_analysis",
        ConstraintType.PROP_INTERACTION: "interaction_analysis",

        ConstraintType.ENVIRONMENT_SETTING: "scene_classification",
        ConstraintType.ENVIRONMENT_LIGHTING: "lighting_analysis",
        ConstraintType.ENVIRONMENT_WEATHER: "weather_classification",

        ConstraintType.SPATIAL_RELATION: "spatial_reasoning",
        ConstraintType.CAMERA_ANGLE: "camera_parameter_check",
        ConstraintType.CAMERA_MOVEMENT: "motion_analysis",

        ConstraintType.TEMPORAL_ORDER: "temporal_logic",
        ConstraintType.ACTION_TIMING: "action_timing_analysis",

        ConstraintType.VISUAL_STYLE: "style_classification",
        ConstraintType.COLOR_PALETTE: "color_analysis",
        ConstraintType.LIGHTING_STYLE: "lighting_classification",

        ConstraintType.TECHNICAL_SPEC: "technical_check",
        ConstraintType.RESOLUTION: "resolution_check",
        ConstraintType.FRAMERATE: "framerate_check",

        ConstraintType.CONTINUITY_ACTION: "action_continuity_check",
        ConstraintType.CONTINUITY_POSITION: "position_continuity_check",
        ConstraintType.CONTINUITY_TIME: "temporal_continuity_check",
    }

    # 约束冲突解决规则
    CONSTRAINT_CONFLICT_RESOLUTION = {
        # 当约束冲突时，按优先级解决
        "priority_based": {
            "rule": "higher_priority_wins",
            "description": "高优先级约束覆盖低优先级约束"
        },
        # 时间连续性优先
        "temporal_priority": {
            "rule": "temporal_constraints_first",
            "description": "时间相关约束优先于空间约束"
        },
        # 角色一致性优先
        "character_consistency": {
            "rule": "character_constraints_first",
            "description": "角色相关约束优先于环境和道具约束"
        },
        # 视觉连续性优先
        "visual_continuity": {
            "rule": "continuity_constraints_first",
            "description": "连续性约束优先于风格约束"
        }
    }

    # 约束严重性级别
    CONSTRAINT_SEVERITY_LEVELS = {
        "critical": {
            "score_range": (0.0, 0.3),
            "action": "must_fix",
            "description": "严重影响观看体验，必须修复"
        },
        "high": {
            "score_range": (0.3, 0.6),
            "action": "should_fix",
            "description": "显著影响质量，建议修复"
        },
        "medium": {
            "score_range": (0.6, 0.8),
            "action": "consider_fixing",
            "description": "有一定影响，可以考虑修复"
        },
        "low": {
            "score_range": (0.8, 1.0),
            "action": "optional",
            "description": "影响较小，可选修复"
        }
    }

    # 常见约束模式
    CONSTRAINT_PATTERNS = {
        # 角色外观模式
        "character_appearance_pattern": {
            "description": "角色外观一致性约束",
            "fields": ["clothing", "hairstyle", "makeup", "accessories"],
            "validation": "compare_across_shots",
            "tolerance": "exact_match"
        },

        # 道具状态模式
        "prop_state_pattern": {
            "description": "道具状态连续性约束",
            "fields": ["state", "position", "orientation"],
            "validation": "state_machine_check",
            "tolerance": "logical_continuity"
        },

        # 空间关系模式
        "spatial_relation_pattern": {
            "description": "角色间空间关系约束",
            "fields": ["distance", "relative_position", "facing_direction"],
            "validation": "spatial_reasoning",
            "tolerance": "relative_consistency"
        },

        # 动作时序模式
        "action_timing_pattern": {
            "description": "动作发生时间约束",
            "fields": ["start_time", "duration", "sequence_order"],
            "validation": "temporal_logic",
            "tolerance": "time_window"
        },

        # 视觉风格模式
        "visual_style_pattern": {
            "description": "视觉风格一致性约束",
            "fields": ["color_palette", "lighting", "texture"],
            "validation": "style_consistency",
            "tolerance": "gradual_change"
        }
    }

    # 约束关键词映射
    CONSTRAINT_KEYWORDS = {
        "character": {
            "appearance": ["穿着", "服装", "衣服", "发型", "妆容", "配饰", "外貌"],
            "position": ["位置", "站在", "坐在", "位于", "在...旁边"],
            "posture": ["姿势", "坐着", "站着", "躺着", "弯腰", "转身"],
            "expression": ["表情", "微笑", "皱眉", "惊讶", "生气", "悲伤"],
            "interaction": ["互动", "交谈", "对视", "握手", "拥抱", "接触"]
        },
        "props": {
            "state": ["状态", "满的", "空的", "破碎的", "干净的", "脏的"],
            "position": ["位置", "在手中", "在桌上", "在地上", "在包里"],
            "interaction": ["拿着", "使用", "放下", "传递", "扔掉"]
        },
        "environment": {
            "setting": ["环境", "场景", "地点", "室内", "室外", "房间"],
            "lighting": ["光线", "灯光", "阳光", "黑暗", "明亮", "阴影"],
            "weather": ["天气", "下雨", "晴天", "下雪", "刮风", "雾天"]
        },
        "spatial": {
            "relation": ["左边", "右边", "前面", "后面", "上面", "下面", "中间"],
            "camera": ["镜头", "角度", "特写", "远景", "中景", "俯拍", "仰拍"]
        },
        "temporal": {
            "order": ["首先", "然后", "接着", "之后", "同时", "之前"],
            "timing": ["时间", "时长", "秒", "分钟", "快", "慢"]
        }
    }

    @staticmethod
    def get_priority_for_type(constraint_type: ConstraintType) -> ConstraintPriority:
        """获取约束类型的默认优先级"""
        return ConstraintConfig.CONSTRAINT_PRIORITY_DEFAULTS.get(
            constraint_type,
            ConstraintPriority.MEDIUM
        )

    @staticmethod
    def get_validation_method(constraint_type: ConstraintType) -> str:
        """获取约束的验证方法"""
        return ConstraintConfig.CONSTRAINT_VALIDATION_METHODS.get(
            constraint_type,
            "visual_check"
        )

    @staticmethod
    def classify_constraint_by_text(text: str) -> Dict[str, Any]:
        """通过文本分类约束"""
        text_lower = text.lower()

        # 检测约束类型
        constraint_type = None
        priority = ConstraintPriority.MEDIUM
        category = None

        # 检查角色相关关键词
        for field, keywords in ConstraintConfig.CONSTRAINT_KEYWORDS["character"].items():
            if any(keyword in text_lower for keyword in keywords):
                category = ConstraintCategory.CHARACTER

                if field == "appearance":
                    constraint_type = ConstraintType.CHARACTER_APPEARANCE
                    priority = ConstraintPriority.HIGH
                elif field == "position":
                    constraint_type = ConstraintType.CHARACTER_POSITION
                elif field == "posture":
                    constraint_type = ConstraintType.CHARACTER_POSTURE
                elif field == "expression":
                    constraint_type = ConstraintType.CHARACTER_EXPRESSION
                    priority = ConstraintPriority.HIGH
                elif field == "interaction":
                    constraint_type = ConstraintType.CHARACTER_INTERACTION
                    priority = ConstraintPriority.HIGH
                break

        # 检查道具相关关键词
        if not constraint_type:
            for field, keywords in ConstraintConfig.CONSTRAINT_KEYWORDS["props"].items():
                if any(keyword in text_lower for keyword in keywords):
                    category = ConstraintCategory.PROPS

                    if field == "state":
                        constraint_type = ConstraintType.PROP_STATE
                        priority = ConstraintPriority.HIGH
                    elif field == "position":
                        constraint_type = ConstraintType.PROP_POSITION
                    elif field == "interaction":
                        constraint_type = ConstraintType.PROP_INTERACTION
                        priority = ConstraintPriority.HIGH
                    break

        # 检查环境相关关键词
        if not constraint_type:
            for field, keywords in ConstraintConfig.CONSTRAINT_KEYWORDS["environment"].items():
                if any(keyword in text_lower for keyword in keywords):
                    category = ConstraintCategory.ENVIRONMENT

                    if field == "setting":
                        constraint_type = ConstraintType.ENVIRONMENT_SETTING
                    elif field == "lighting":
                        constraint_type = ConstraintType.ENVIRONMENT_LIGHTING
                        priority = ConstraintPriority.HIGH
                    elif field == "weather":
                        constraint_type = ConstraintType.ENVIRONMENT_WEATHER
                    break

        # 检查空间相关关键词
        if not constraint_type:
            for field, keywords in ConstraintConfig.CONSTRAINT_KEYWORDS["spatial"].items():
                if any(keyword in text_lower for keyword in keywords):
                    category = ConstraintCategory.SPATIAL

                    if field == "relation":
                        constraint_type = ConstraintType.SPATIAL_RELATION
                    elif field == "camera":
                        constraint_type = ConstraintType.CAMERA_ANGLE
                        priority = ConstraintPriority.LOW
                    break

        # 检查时间相关关键词
        if not constraint_type:
            for field, keywords in ConstraintConfig.CONSTRAINT_KEYWORDS["temporal"].items():
                if any(keyword in text_lower for keyword in keywords):
                    category = ConstraintCategory.TEMPORAL

                    if field == "order":
                        constraint_type = ConstraintType.TEMPORAL_ORDER
                        priority = ConstraintPriority.CRITICAL
                    elif field == "timing":
                        constraint_type = ConstraintType.ACTION_TIMING
                        priority = ConstraintPriority.HIGH
                    break

        # 默认类型
        if not constraint_type:
            constraint_type = ConstraintType.VISUAL_STYLE
            category = ConstraintCategory.VISUAL
            priority = ConstraintPriority.LOW

        # 检测紧急关键词
        urgent_keywords = ["必须", "一定", "不能", "不可以", "禁止", "确保", "保证"]
        if any(keyword in text_lower for keyword in urgent_keywords):
            priority = ConstraintPriority.CRITICAL

        # 检测高优先级关键词
        high_priority_keywords = ["重要", "关键", "主要", "核心", "重点"]
        if any(keyword in text_lower for keyword in high_priority_keywords):
            if priority.value < ConstraintPriority.HIGH.value:
                priority = ConstraintPriority.HIGH

        return {
            "type": constraint_type,
            "category": category,
            "priority": priority,
            "validation_method": ConstraintConfig.get_validation_method(constraint_type)
        }

    @staticmethod
    def extract_constraint_parameters(text: str) -> Dict[str, Any]:
        """从文本中提取约束参数"""
        params = {
            "entities": [],
            "attributes": {},
            "relations": [],
            "temporal_info": {},
            "spatial_info": {}
        }

        # 提取角色名
        character_patterns = [
            r"角色\s*[：:]\s*([^\s，。]+)",
            r"([^\s，。]+?)\s*(?:穿着|手持|站在|坐在)",
            r"(?:特写|镜头对准)\s*([^\s，。]+)"
        ]

        for pattern in character_patterns:
            matches = re.findall(pattern, text)
            params["entities"].extend(matches)

        # 提取道具名
        prop_patterns = [
            r"(?:拿着|手持|使用|带着)\s*([^\s，。]+?)\s*(?:的|在|着|了)",
            r"([^\s，。]+?)\s*(?:杯子|书|手机|包|钥匙|武器|工具)"
        ]

        for pattern in prop_patterns:
            matches = re.findall(pattern, text)
            params["entities"].extend(matches)

        # 提取属性
        attribute_patterns = {
            "color": r"(红色|蓝色|绿色|黄色|黑色|白色|灰色|橙色|紫色|粉色)",
            "position": r"(站在|坐在|位于|在...左边|在...右边|在...前面|在...后面)",
            "state": r"(满的|空的|半满|破碎|干净|脏的|新的|旧的)",
            "expression": r"(微笑|皱眉|惊讶|生气|悲伤|中性|开心)"
        }

        for attr_name, pattern in attribute_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                params["attributes"][attr_name] = matches[0]

        # 提取时间信息
        temporal_patterns = {
            "duration": r"(\d+(?:\.\d+)?)\s*秒",
            "start_time": r"从\s*(\d+(?:\.\d+)?)\s*秒开始",
            "end_time": r"到\s*(\d+(?:\.\d+)?)\s*秒结束"
        }

        for temp_name, pattern in temporal_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                params["temporal_info"][temp_name] = float(matches[0])

        # 提取空间关系
        spatial_patterns = [
            r"距离\s*([^\s，。]+)\s*(?:近|远)",
            r"在\s*([^\s，。]+)\s*(?:的左边|的右边|的前面|的后面|的上面|的下面)"
        ]

        for pattern in spatial_patterns:
            matches = re.findall(pattern, text)
            params["relations"].extend(matches)

        # 去重
        params["entities"] = list(set(params["entities"]))
        params["relations"] = list(set(params["relations"]))

        return params

    @staticmethod
    def generate_constraint_id(constraint_type: ConstraintType,
                               segment_id: str,
                               entity: str = None) -> str:
        """生成约束ID"""
        base_id = f"{constraint_type.value}_{segment_id}"

        if entity:
            # 清理实体名，用于ID
            clean_entity = re.sub(r'[^\w]', '_', entity.lower())
            return f"{base_id}_{clean_entity}"

        return base_id

    @staticmethod
    def resolve_constraint_conflict(constraint1: Dict[str, Any],
                                    constraint2: Dict[str, Any]) -> Dict[str, Any]:
        """解决约束冲突"""
        # 比较优先级
        priority1 = constraint1.get("priority", ConstraintPriority.MEDIUM)
        priority2 = constraint2.get("priority", ConstraintPriority.MEDIUM)

        if isinstance(priority1, ConstraintPriority):
            priority1 = priority1.value
        if isinstance(priority2, ConstraintPriority):
            priority2 = priority2.value

        if priority1 > priority2:
            winner = constraint1
            loser = constraint2
        elif priority2 > priority1:
            winner = constraint2
            loser = constraint1
        else:
            # 优先级相同，应用特定规则
            type1 = constraint1.get("type")
            type2 = constraint2.get("type")

            # 时间连续性优先
            if type1 in [ConstraintType.CONTINUITY_TIME, ConstraintType.TEMPORAL_ORDER]:
                winner = constraint1
                loser = constraint2
            elif type2 in [ConstraintType.CONTINUITY_TIME, ConstraintType.TEMPORAL_ORDER]:
                winner = constraint2
                loser = constraint1
            # 角色一致性优先
            elif "character" in str(type1).lower():
                winner = constraint1
                loser = constraint2
            elif "character" in str(type2).lower():
                winner = constraint2
                loser = constraint1
            else:
                # 默认：第一个约束胜出
                winner = constraint1
                loser = constraint2

        return {
            "winner": winner,
            "loser": loser,
            "reason": f"Priority {winner.get('priority')} > {loser.get('priority')}",
            "conflict_type": "priority_based"
        }

    @staticmethod
    def calculate_constraint_satisfaction_score(constraints: List[Dict[str, Any]],
                                                satisfied_ids: List[str]) -> float:
        """计算约束满足度得分"""
        if not constraints:
            return 1.0

        # 按优先级加权计算
        total_weight = 0
        satisfied_weight = 0

        for constraint in constraints:
            priority = constraint.get("priority", ConstraintPriority.MEDIUM)
            if isinstance(priority, ConstraintPriority):
                weight = priority.value
            else:
                weight = priority

            constraint_id = constraint.get("id", "")

            total_weight += weight
            if constraint_id in satisfied_ids:
                satisfied_weight += weight

        if total_weight == 0:
            return 1.0

        return satisfied_weight / total_weight

    @staticmethod
    def get_severity_for_score(score: float) -> Dict[str, Any]:
        """根据得分获取严重性级别"""
        for level_name, level_info in ConstraintConfig.CONSTRAINT_SEVERITY_LEVELS.items():
            min_score, max_score = level_info["score_range"]
            if min_score <= score < max_score:
                return {
                    "level": level_name,
                    "action": level_info["action"],
                    "description": level_info["description"]
                }

        return {
            "level": "unknown",
            "action": "review",
            "description": "未知严重性级别"
        }

    @staticmethod
    def create_constraint_template(constraint_type: ConstraintType,
                                   segment_id: str,
                                   description: str,
                                   custom_priority: Optional[ConstraintPriority] = None) -> Dict[str, Any]:
        """创建约束模板"""
        classification = ConstraintConfig.classify_constraint_by_text(description)
        params = ConstraintConfig.extract_constraint_parameters(description)

        # 确定实体（如果有）
        entity = None
        if params["entities"]:
            entity = params["entities"][0]

        # 生成ID
        constraint_id = ConstraintConfig.generate_constraint_id(
            constraint_type, segment_id, entity
        )

        # 确定优先级
        if custom_priority is not None:
            priority = custom_priority
        else:
            priority = classification["priority"]

        # 创建Sora指令
        sora_instruction = ConstraintConfig.generate_sora_instruction(
            description, constraint_type
        )

        return {
            "id": constraint_id,
            "type": constraint_type,
            "category": classification["category"],
            "priority": priority,
            "description": description,
            "sora_instruction": sora_instruction,
            "validation_method": classification["validation_method"],
            "parameters": params,
            "is_enforced": True,
            "applicable_segments": [segment_id],
            "metadata": {
                "generated_by": "constraint_config",
                "confidence": 0.8,
                "version": "1.0"
            }
        }

    @staticmethod
    def generate_sora_instruction(description: str,
                                  constraint_type: ConstraintType) -> str:
        """生成Sora能理解的指令"""
        # 根据约束类型转换指令格式
        if constraint_type in [ConstraintType.CHARACTER_APPEARANCE,
                               ConstraintType.CHARACTER_EXPRESSION]:
            # 角色相关：直接描述
            return f"Character must have: {description}"

        elif constraint_type in [ConstraintType.PROP_STATE,
                                 ConstraintType.PROP_POSITION]:
            # 道具相关：明确状态
            return f"Prop requirement: {description}"

        elif constraint_type == ConstraintType.CAMERA_ANGLE:
            # 相机角度：技术描述
            # 转换中文描述为英文技术术语
            translations = {
                "特写": "close-up shot",
                "近景": "medium close-up",
                "中景": "medium shot",
                "全景": "full shot",
                "远景": "wide shot",
                "大远景": "extreme wide shot",
                "俯拍": "overhead shot",
                "仰拍": "low angle shot",
                "侧拍": "profile shot"
            }

            instruction = description
            for chinese, english in translations.items():
                if chinese in description:
                    instruction = instruction.replace(chinese, english)

            return f"Camera angle: {instruction}"

        elif constraint_type in [ConstraintType.CONTINUITY_ACTION,
                                 ConstraintType.CONTINUITY_POSITION]:
            # 连续性：明确要求
            return f"Continuity requirement: {description}"

        else:
            # 默认：直接使用描述
            return description

    @staticmethod
    def validate_constraint_syntax(constraint: Dict[str, Any]) -> Dict[str, Any]:
        """验证约束语法"""
        errors = []
        warnings = []

        # 检查必要字段
        required_fields = ["id", "type", "description", "priority"]
        for field in required_fields:
            if field not in constraint:
                errors.append(f"Missing required field: {field}")

        # 检查类型有效性
        if "type" in constraint:
            try:
                ConstraintType(constraint["type"])
            except ValueError:
                errors.append(f"Invalid constraint type: {constraint['type']}")

        # 检查优先级有效性
        if "priority" in constraint:
            priority = constraint["priority"]
            if isinstance(priority, int):
                if not (1 <= priority <= 10):
                    errors.append(f"Priority must be between 1-10, got: {priority}")
            elif isinstance(priority, ConstraintPriority):
                pass  # 有效
            else:
                errors.append(f"Invalid priority type: {type(priority)}")

        # 检查描述长度
        if "description" in constraint:
            desc = constraint["description"]
            if len(desc) < 5:
                warnings.append("Description is very short, may be unclear")
            if len(desc) > 500:
                warnings.append("Description is very long, consider simplifying")

        # 检查Sora指令
        if "sora_instruction" in constraint:
            instruction = constraint["sora_instruction"]
            if len(instruction) > 200:
                warnings.append("Sora instruction may be too long for optimal generation")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestion": "Fix errors before using constraint" if errors else "Constraint is valid"
        }


# 测试函数
def test_constraint_config():
    """测试约束配置"""
    config = ConstraintConfig

    # 测试文本分类
    text = "林然必须穿着蓝色衬衫，坐在沙发左侧"
    classification = config.classify_constraint_by_text(text)
    print(f"Constraint classification: {classification}")

    # 测试参数提取
    params = config.extract_constraint_parameters(text)
    print(f"\nExtracted parameters: {params}")

    # 测试ID生成
    constraint_id = config.generate_constraint_id(
        ConstraintType.CHARACTER_APPEARANCE,
        "s001",
        "林然"
    )
    print(f"\nGenerated constraint ID: {constraint_id}")

    # 测试约束模板创建
    template = config.create_constraint_template(
        ConstraintType.CHARACTER_APPEARANCE,
        "s001",
        "林然穿着蓝色衬衫",
        custom_priority=ConstraintPriority.HIGH
    )
    print(f"\nConstraint template:")
    for key, value in template.items():
        if key != "parameters":  # 简化输出
            print(f"  {key}: {value}")

    # 测试语法验证
    validation = config.validate_constraint_syntax(template)
    print(f"\nConstraint validation: {validation['is_valid']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")

    # 测试冲突解决
    constraint1 = {"id": "c1", "priority": ConstraintPriority.HIGH, "type": ConstraintType.CHARACTER_APPEARANCE}
    constraint2 = {"id": "c2", "priority": ConstraintPriority.MEDIUM, "type": ConstraintType.ENVIRONMENT_SETTING}

    conflict_result = config.resolve_constraint_conflict(constraint1, constraint2)
    print(f"\nConflict resolution:")
    print(f"  Winner: {conflict_result['winner']['id']}")
    print(f"  Reason: {conflict_result['reason']}")

    # 测试得分计算
    constraints = [
        {"id": "c1", "priority": ConstraintPriority.HIGH},
        {"id": "c2", "priority": ConstraintPriority.MEDIUM},
        {"id": "c3", "priority": ConstraintPriority.LOW}
    ]
    satisfied_ids = ["c1", "c3"]

    score = config.calculate_constraint_satisfaction_score(constraints, satisfied_ids)
    severity = config.get_severity_for_score(score)

    print(f"\nConstraint satisfaction:")
    print(f"  Score: {score:.2f}")
    print(f"  Severity: {severity['level']}")
    print(f"  Action: {severity['action']}")


if __name__ == "__main__":
    test_constraint_config()
