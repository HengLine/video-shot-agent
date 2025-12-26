"""
@FileName: rule_duration_calculator.py
@Description: 规则时长计算器
@Author: HengLine
@Time: 2025/12/20 21:24
"""
import random
import re
from typing import Dict, List, Optional

from hengline.logger import debug
from hengline.agent.script_parser.script_parser_model import Dialogue, Action, Scene
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation
from hengline.config.temporal_planner_config import DurationConfig


class RuleDurationCalculator:
    """基于规则的时长计算器 - 完整实现"""

    def __init__(self, config: DurationConfig):
        self.config = config

    def _find_character(self, character_name: str, characters: List[Dict]) -> Optional[Dict]:
        """
        根据角色名查找角色信息
        """
        if not character_name or not characters:
            return None

        # 精确匹配
        for character in characters:
            if character.get("name") == character_name:
                return character

        # 模糊匹配（处理昵称、简称）
        for character in characters:
            name = character.get("name", "")
            if name and character_name in name or name in character_name:
                return character

        # 尝试从代词推断
        if character_name in ["他", "她", "他们", "她们"]:
            # 返回一个通用角色
            gender = "男" if character_name in ["他", "他们"] else "女"
            return {"name": character_name, "gender": gender, "age": "未知"}

        return None

    def _get_character_factor(self, character: Dict, is_action: bool = False) -> float:
        """
        获取角色影响因子

        Args:
            character: 角色信息
            is_action: 是否是动作（动作和对话的影响因子不同）
        """
        if not character:
            return 1.0

        factor = 1.0

        # 1. 年龄影响
        age = character.get("age")
        if age:
            if age >= 60:
                # 老年人：动作慢，对话可能也慢
                if is_action:
                    factor *= 1.5  # 动作更慢
                else:
                    factor *= 1.2  # 对话稍慢
            elif age <= 12:
                # 儿童：动作快，对话可能快
                if is_action:
                    factor *= 0.8  # 动作更快
                else:
                    factor *= 0.9  # 对话稍快

        # 2. 性别影响（某些文化中可能存在差异）
        gender = character.get("gender", "")
        if gender == "女" and not is_action:
            # 研究表明女性平均语速稍快（但差异很小）
            factor *= 0.95

        # 3. 角色类型影响
        role_hint = character.get("role_hint", "")
        if role_hint == "主角":
            # 主角可能给更多镜头时间
            factor *= 1.1
        elif role_hint == "群众":
            # 群众角色可能时间较少
            factor *= 0.8

        return factor

    def estimate_dialogue(self, text: str, character_count: int) -> float:
        """
        估算对话时长

        参数:
            text: 对话文本
            character_count: 角色数量

        返回:
            估算的时长（秒）
        """
        if not text or not text.strip():
            return 0.0

        # 1. 基础时长：基于字符数
        words = len(text.strip().split())
        characters = len(text.strip())

        # 2. 根据角色数量调整语速
        # 单角色：正常语速，多角色：对话节奏会加快
        speed_factor = 1.0
        if character_count == 1:
            # 单角色独白，语速稍慢
            speed_factor = 0.9
        elif character_count == 2:
            # 双人对话，正常语速
            speed_factor = 1.0
        elif character_count >= 3:
            # 多人对话，语速加快
            speed_factor = 1.2

        # 3. 根据标点符号调整（问号、感叹号会延长停顿）
        punctuation_factor = 1.0
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        ellipsis = text.count('...') + text.count('…')

        if question_marks > 0 or exclamation_marks > 0:
            # 每个问号或感叹号增加0.2秒停顿
            punctuation_factor += (question_marks + exclamation_marks) * 0.02

        if ellipsis > 0:
            # 省略号表示思考或停顿，增加更多时间
            punctuation_factor += ellipsis * 0.05

        # 4. 根据句子复杂度调整（长句子需要更多呼吸时间）
        complexity_factor = 1.0
        sentences = text.replace('?', '.').replace('!', '.').replace('...', '.').replace('…', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length > 20:
                # 长句子需要更多时间
                complexity_factor = 1.1
            elif avg_sentence_length < 5:
                # 短句子可以说得更快
                complexity_factor = 0.9

        # 5. 计算最终时长（基于平均语速：150词/分钟，即2.5词/秒）
        base_duration = words / 2.5  # 基础时长

        # 应用所有调整因子
        estimated_duration = base_duration * speed_factor * punctuation_factor * complexity_factor

        # 6. 确保最小时长（即使是短句子也需要基本的时间）
        min_duration = max(1.0, character_count * 0.5)  # 每个角色至少0.5秒
        estimated_duration = max(estimated_duration, min_duration)

        # 7. 对于特别短的文本，使用字符数作为备用计算
        if words < 3:
            char_based_duration = characters * 0.1  # 每个字符0.1秒
            estimated_duration = max(estimated_duration, char_based_duration)

        debug(
            f"对话时长估算: 文本长度={characters}字符/{words}词, "
            f"角色数={character_count}, 估算时长={estimated_duration:.2f}秒"
        )

        return round(estimated_duration, 2)

    def estimate_action(self, action_type: str, action_description: str, character_count: int) -> float:
        """
        估算动作时长

        参数:
            action_type: 动作类型
            action_description: 动作描述
            character_count: 涉及角色数量

        返回:
            估算的时长（秒）
        """
        if not action_type:
            return 0.0

        # 1. 基础动作时长映射
        base_durations = {
            # 简单动作
            'idle': 1.0,  # 空闲/站立
            'walk': 3.0,  # 行走
            'run': 2.0,  # 奔跑
            'jump': 1.5,  # 跳跃
            'sit': 2.0,  # 坐下
            'stand': 1.5,  # 站起

            # 交互动作
            'pickup': 2.0,  # 拾取物品
            'drop': 1.0,  # 放下物品
            'use': 2.5,  # 使用物品
            'open': 1.5,  # 开门/开箱
            'close': 1.0,  # 关门/关箱

            # 战斗动作
            'attack': 2.0,  # 攻击
            'defend': 1.5,  # 防御
            'cast': 3.0,  # 施法
            'dodge': 1.0,  # 闪避

            # 情感动作
            'laugh': 2.0,  # 大笑
            'cry': 3.0,  # 哭泣
            'nod': 1.0,  # 点头
            'shake_head': 1.0,  # 摇头

            # 复合动作
            'transition': 2.0,  # 场景转换
            'combat': 5.0,  # 战斗
            'conversation': 4.0,  # 对话动作

            'default': 2.0  # 默认动作
        }

        # 2. 获取基础时长
        base_duration = base_durations.get(action_type.lower(), base_durations['default'])

        # 3. 根据动作描述调整
        description_factor = 1.0
        if action_description:
            desc_lower = action_description.lower()
            desc_words = len(desc_lower.split())

            # 复杂的描述通常意味着更复杂的动作
            if desc_words > 10:
                description_factor = 1.3
            elif desc_words > 5:
                description_factor = 1.15
            elif desc_words > 0:
                description_factor = 1.05

            # 特定关键词调整
            intensity_keywords = {
                'quickly': 0.8, 'slowly': 1.3, 'carefully': 1.2,
                'forcefully': 1.1, 'gently': 1.1, 'angrily': 1.1,
                'happily': 1.05, 'sadly': 1.2, 'excitedly': 1.05
            }

            for keyword, factor in intensity_keywords.items():
                if keyword in desc_lower:
                    description_factor *= factor

        # 4. 根据角色数量调整
        character_factor = 1.0
        if character_count == 1:
            character_factor = 1.0
        elif character_count == 2:
            character_factor = 1.3  # 双人动作需要协调
        elif character_count >= 3:
            character_factor = 1.5  # 多人动作更复杂

        # 5. 动作类型特定调整
        type_specific_factor = 1.0
        if action_type.lower() in ['combat', 'battle', 'fight']:
            # 战斗动作通常更长
            type_specific_factor = 1.5
        elif action_type.lower() in ['transition', 'scene_change']:
            # 场景转换需要考虑加载时间
            type_specific_factor = 1.2
        elif action_type.lower() in ['conversation', 'dialogue']:
            # 对话动作通常配合对话内容
            type_specific_factor = 0.8

        # 6. 计算最终时长
        estimated_duration = base_duration * description_factor * character_factor * type_specific_factor

        # 7. 应用最小和最大限制
        min_duration = 0.5  # 任何动作至少0.5秒
        max_duration = 30.0  # 单个动作最多30秒

        estimated_duration = max(min_duration, min(estimated_duration, max_duration))

        # 8. 对于连续动作或复合动作的特殊处理
        if 'and' in (action_description or '').lower() or 'then' in (action_description or '').lower():
            # 如果是复合动作，增加时间
            estimated_duration *= 1.2

        debug(
            f"动作时长估算: 类型={action_type}, 角色数={character_count}, "
            f"基础时长={base_duration:.2f}秒, 估算时长={estimated_duration:.2f}秒"
        )

        return round(estimated_duration, 2)

    def estimate_scene(self, scene_type: str, scene_complexity: int, character_count: int) -> float:
        """
        估算场景基础时长（可选，如果需要的话）

        参数:
            scene_type: 场景类型
            scene_complexity: 场景复杂度（1-10）
            character_count: 角色数量

        返回:
            估算的时长（秒）
        """
        # 基础场景时长映射
        scene_base_durations = {
            'interior': 10.0,
            'exterior': 15.0,
            'battle': 20.0,
            'dialogue': 8.0,
            'cutscene': 30.0,
            'default': 12.0
        }

        base_duration = scene_base_durations.get(scene_type.lower(), scene_base_durations['default'])

        # 根据复杂度调整（1-10，默认5）
        complexity = max(1, min(10, scene_complexity))
        complexity_factor = 0.5 + (complexity / 10)  # 0.6-1.5

        # 根据角色数量调整
        character_factor = 1.0 + (character_count * 0.1)  # 每多一个角色增加10%

        estimated_duration = base_duration * complexity_factor * character_factor

        return round(max(5.0, estimated_duration), 2)  # 场景至少5秒


    def _calculate_pause_time(self, text: str) -> float:
        """
        计算文本中的停顿时间

        基于标点符号计算合理的停顿时间
        """
        if not text:
            return 0.0

        pause_time = 0.0

        # 中文标点统计
        punctuation_counts = {
            "，": text.count("，"),  # 逗号
            "。": text.count("。"),  # 句号
            "！": text.count("！"),  # 感叹号
            "？": text.count("？"),  # 问号
            "…": text.count("…"),  # 省略号
            "——": text.count("——"),  # 破折号
        }

        # 英文标点统计
        punctuation_counts.update({
            ",": text.count(","),
            ".": text.count("."),
            "!": text.count("!"),
            "?": text.count("?"),
            "...": text.count("..."),
            "--": text.count("--"),
        })

        # 计算总停顿时间
        for punct, count in punctuation_counts.items():
            if punct in self.config.pause_config:
                pause_time += count * self.config.pause_config[punct]

        # 句子间的额外停顿
        sentence_endings = punctuation_counts.get("。", 0) + punctuation_counts.get(".", 0)
        if sentence_endings > 1:
            # 每增加一个句子，增加一点思考时间
            pause_time += (sentence_endings - 1) * 0.3

        # 情绪性停顿（基于感叹号和问号）
        emotional_pauses = punctuation_counts.get("！", 0) + punctuation_counts.get("!", 0)
        emotional_pauses += punctuation_counts.get("？", 0) + punctuation_counts.get("?", 0)
        pause_time += emotional_pauses * 0.5

        # 限制范围
        max_pause = len(text) * 0.1  # 最多不超过文本长度的10%作为停顿
        return min(pause_time, max_pause)

    def _estimate_reaction_time(self, emotion: str) -> float:
        """
        估算情绪反应时间

        不同情绪需要不同的反应/消化时间
        """
        base_reaction = self.config.reaction_times.get("平静", 0.5)

        # 情绪特定的反应时间
        emotion_reactions = {
            "惊讶": 1.5,
            "震惊": 2.0,
            "恐惧": 1.8,
            "悲伤": 1.2,
            "愤怒": 1.0,
            "喜悦": 0.8,
            "疑问": 1.2,
            "思考": 1.8,
            "犹豫": 1.5
        }

        reaction_time = emotion_reactions.get(emotion, base_reaction)

        # 添加随机性（±20%）
        variation = random.uniform(-0.2, 0.2)
        reaction_time *= (1 + variation)

        return reaction_time

    def _calculate_dialogue_confidence(self, dialogue: Dialogue, char_count: int) -> float:
        """
        计算对话时长估算的置信度
        """
        confidence = 0.7  # 基础置信度

        # 1. 文本长度影响
        if char_count < 5:
            confidence *= 0.8  # 太短的文本，置信度降低
        elif char_count > 100:
            confidence *= 0.9  # 很长的文本，可能有复杂结构
        else:
            confidence *= 1.0  # 适中长度，置信度正常

        # 2. 情绪明确性影响
        clear_emotions = ["平静", "愤怒", "喜悦", "悲伤"]
        if dialogue.emotion in clear_emotions:
            confidence *= 1.1  # 明确情绪，置信度提高
        elif dialogue.emotion == "未知" or not dialogue.emotion:
            confidence *= 0.9  # 未知情绪，置信度降低

        # 3. 说话者明确性影响
        if dialogue.speaker and dialogue.speaker not in ["未知", "某人", ""]:
            confidence *= 1.05  # 明确说话者，置信度提高

        # 4. 标点完整性影响
        has_proper_punctuation = any(punct in dialogue.text
                                     for punct in ["。", ".", "！", "!", "？", "?"])
        if has_proper_punctuation:
            confidence *= 1.05  # 标点完整，置信度提高

        # 限制在0-1范围内
        return max(0.1, min(1.0, confidence))

    def _calculate_action_confidence(self, action: Action) -> float:
        """
        计算动作时长估算的置信度
        """
        confidence = 0.6  # 动作的基础置信度较低

        # 1. 动作类型明确性
        if action.type and action.type != "unknown":
            confidence *= 1.2  # 明确类型，置信度提高

        # 2. 执行者明确性
        if action.actor and action.actor not in ["某人", "未知", ""]:
            confidence *= 1.1

        # 3. 描述详细度
        desc_length = len(action.description)
        if desc_length > 20:
            confidence *= 1.05  # 详细描述，置信度提高
        elif desc_length < 5:
            confidence *= 0.9  # 描述太短，置信度降低

        # 4. 强度信息
        if action.intensity and 1 <= action.intensity <= 5:
            confidence *= 1.05  # 有强度信息，置信度提高

        return max(0.1, min(1.0, confidence))

    def _get_action_base_time(self, action_type: str, intensity: int) -> float:
        """
        获取动作的基础时长
        """
        # 默认时长
        default_time = 2.0

        if not action_type or action_type not in self.config.action_base_times:
            return default_time

        base_info = self.config.action_base_times[action_type]

        # 处理不同类型的数据结构
        if isinstance(base_info, dict):
            # 根据强度选择
            if intensity >= 4:
                return base_info.get("fast", base_info.get("strong", default_time))
            elif intensity >= 3:
                return base_info.get("normal", base_info.get("complex", default_time))
            else:
                return base_info.get("slow", base_info.get("simple", base_info.get("subtle", default_time)))
        elif isinstance(base_info, (int, float)):
            # 直接是数值
            return float(base_info)
        else:
            return default_time

    def _assess_action_complexity(self, description: str) -> float:
        """
        评估动作复杂度（0-1）
        """
        if not description:
            return 0.0

        complexity_score = 0.0

        # 1. 动作动词数量
        action_verbs = ["走", "跑", "跳", "坐", "站", "看", "说", "笑", "哭",
                        "摸", "推", "拉", "抱", "打", "踢", "转身", "点头", "摇头"]

        verb_count = sum(1 for verb in action_verbs if verb in description)
        complexity_score += min(1.0, verb_count * 0.2) * 0.4

        # 2. 修饰词数量（表示动作细节）
        modifiers = ["轻轻", "慢慢", "快速", "用力", "突然", "缓缓", "小心翼翼",
                     "激烈", "温柔", "猛烈", "迅速", "缓慢"]

        modifier_count = sum(1 for modifier in modifiers if modifier in description)
        complexity_score += min(1.0, modifier_count * 0.3) * 0.3

        # 3. 目标/对象数量
        # 简单的正则匹配对象
        object_patterns = [r'把\s*(\S+)', r'向\s*(\S+)', r'对\s*(\S+)', r'在\s*(\S+)']
        object_count = 0
        for pattern in object_patterns:
            object_count += len(re.findall(pattern, description))

        complexity_score += min(1.0, object_count * 0.4) * 0.3

        return min(1.0, complexity_score)

    def _get_action_emotion_factor(self, description: str) -> float:
        """
        获取动作情感影响因子
        """
        if not description:
            return 1.0

        # 情感关键词影响
        emotional_keywords = {
            "愤怒地": 1.3,
            "激动地": 1.2,
            "悲伤地": 1.4,
            "快乐地": 0.9,
            "恐惧地": 1.3,
            "紧张地": 1.2,
            "轻松地": 0.8,
            "猛烈地": 1.5,
            "温柔地": 1.1,
            "突然": 1.1
        }

        factor = 1.0
        for keyword, keyword_factor in emotional_keywords.items():
            if keyword in description:
                factor *= keyword_factor

        return factor

    def estimate_description(self, description: str, scene: Scene) -> DurationEstimation:
        """
        估算场景描述时长
        """
        if not description:
            return DurationEstimation(
                element_id=f"scene_{scene.scene_id}_desc",
                element_type="description",
                estimated_duration=0.0,
                confidence=0.0
            )

        # 基础时长：每50字符约1秒
        char_count = len(description)
        base_duration = max(self.config.min_description_duration,
                            min(self.config.max_description_duration,
                                char_count / 50))

        # 复杂度影响
        complexity = self._assess_description_complexity(description)
        complexity_factor = 1.0 + complexity * 0.5

        # 情感氛围影响
        mood_factor = self._get_mood_factor(scene.mood)

        # 总时长
        total_duration = base_duration * complexity_factor * mood_factor

        # 置信度计算
        confidence = self._calculate_description_confidence(description, char_count)

        return DurationEstimation(
            element_id=f"scene_{scene.scene_id}_desc",
            element_type="description",
            estimated_duration=total_duration,
            confidence=confidence,
            factors_considered=[
                f"字数: {char_count}",
                f"基础时长: {base_duration:.1f}秒",
                f"复杂度因子: {complexity_factor:.1f}x",
                f"氛围因子: {mood_factor:.1f}x"
            ]
        )

    def _assess_description_complexity(self, description: str) -> float:
        """
        评估描述复杂度（0-1）
        """
        if not description:
            return 0.0

        complexity = 0.0

        # 1. 句子数量
        sentences = re.split(r'[。！？.!?]', description)
        sentence_count = len([s for s in sentences if s.strip()])
        complexity += min(1.0, sentence_count * 0.2) * 0.3

        # 2. 细节描述词
        detail_words = ["细长的", "高大的", "明亮的", "昏暗的", "精致的", "粗糙的",
                        "快速的", "缓慢的", "优雅的", "笨拙的", "鲜艳的", "暗淡的"]
        detail_count = sum(1 for word in detail_words if word in description)
        complexity += min(1.0, detail_count * 0.3) * 0.3

        # 3. 空间关系词
        spatial_words = ["左边", "右边", "前面", "后面", "上面", "下面",
                         "中间", "旁边", "远处", "近处", "之间", "周围"]
        spatial_count = sum(1 for word in spatial_words if word in description)
        complexity += min(1.0, spatial_count * 0.4) * 0.4

        return min(1.0, complexity)

    def _get_mood_factor(self, mood: str) -> float:
        """
        获取氛围影响因子
        """
        mood_factors = {
            "紧张": 1.3,  # 紧张氛围需要更多时间营造
            "恐怖": 1.4,  # 恐怖氛围需要缓慢展开
            "浪漫": 1.2,  # 浪漫氛围可以稍慢
            "欢乐": 0.9,  # 欢乐氛围可以稍快
            "悲伤": 1.3,  # 悲伤氛围需要时间酝酿
            "平静": 1.0,  # 平静氛围正常速度
            "激烈": 0.8,  # 激烈氛围可以快速切换
            "悬疑": 1.2  # 悬疑氛围需要缓慢展开
        }
        return mood_factors.get(mood, 1.0)

    def _calculate_description_confidence(self, description: str, char_count: int) -> float:
        """
        计算描述时长估算的置信度
        """
        confidence = 0.8  # 描述的基础置信度较高

        # 1. 长度影响
        if char_count < 10:
            confidence *= 0.7  # 太短的描述
        elif char_count > 200:
            confidence *= 0.9  # 很长的描述

        # 2. 结构完整性
        has_proper_structure = any(marker in description
                                   for marker in ["。", ".", "，", ","])
        if has_proper_structure:
            confidence *= 1.1

        # 3. 具体性（包含具体名词）
        concrete_nouns = ["桌子", "椅子", "窗户", "门", "灯", "书",
                          "杯子", "手机", "电脑", "花", "树", "车"]
        has_concrete_nouns = any(noun in description for noun in concrete_nouns)
        if has_concrete_nouns:
            confidence *= 1.05

        return max(0.1, min(1.0, confidence))
