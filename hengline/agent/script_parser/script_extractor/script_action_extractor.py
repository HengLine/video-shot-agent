"""
@FileName: acript_action_extractor.py
@Description: 剧本动作提取器
@Author: HengLine
@Time: 2025/12/19 0:17
"""
import re
from typing import List, Tuple, Optional, Dict

from hengline.agent.script_parser.script_parser_model import Action


class ActionExtractor:
    """动作提取器"""

    def __init__(self):
        # 动作分类系统
        self.action_categories = {
            "movement": {
                "walk": ["走", "行走", "散步", "踱步", "迈步", "步行", "走动"],
                "run": ["跑", "奔跑", "冲", "飞奔", "狂奔", "快跑"],
                "sit": ["坐", "坐下", "就坐", "瘫坐", "端坐", "静坐"],
                "stand": ["站", "站立", "起身", "站起来", "伫立", "直立"],
                "lie": ["躺", "卧", "趴", "仰卧", "侧卧"],
                "kneel": ["跪", "跪下", "跪倒"],
            },
            "gesture": {
                "point": ["指", "指向", "指着", "指点"],
                "wave": ["挥", "挥手", "摆动", "摇晃"],
                "nod": ["点头", "颔首"],
                "shake_head": ["摇头", "摆头"],
                "clap": ["拍", "拍手", "鼓掌"],
                "shrug": ["耸肩", "耸耸肩膀"],
            },
            "facial": {
                "look": ["看", "注视", "凝视", "盯着", "观望", "瞥见", "瞧"],
                "smile": ["笑", "微笑", "咧嘴笑", "偷笑", "憨笑"],
                "cry": ["哭", "哭泣", "流泪", "抽泣", "啜泣", "落泪"],
                "frown": ["皱眉", "蹙眉", "皱起眉头"],
                "blink": ["眨眼", "眨眼睛"],
            },
            "interaction": {
                "touch": ["摸", "触摸", "抚摸", "拍拍", "握住", "抓住", "拉住"],
                "push": ["推", "推开", "推搡"],
                "pull": ["拉", "拉扯", "拽", "拖动"],
                "hold": ["抱", "拥抱", "搂住", "环抱"],
                "hit": ["打", "击打", "拍打", "敲打"],
            },
            "verbal": {
                "speak": ["说", "讲", "说道", "开口", "说话"],
                "shout": ["喊", "喊叫", "大叫", "呼喊", "呐喊"],
                "whisper": ["轻声说", "低声说", "耳语", "悄悄说"],
                "ask": ["问", "询问", "问道", "提问"],
                "answer": ["回答", "答道", "回应", "答复"],
            },
        }

        # 强度关键词映射
        self.intensity_keywords = {
            1: ["轻轻", "慢慢", "缓缓", "小心地", "温柔地"],
            2: ["一般", "普通", "正常", ""],  # 无修饰词为强度2
            3: ["用力", "使劲", "努力", "认真地"],
            4: ["猛烈", "激烈", "强烈", "激动地", "愤怒地"],
            5: ["疯狂", "拼命", "全力", "竭尽全力", "歇斯底里"],
        }

    def extract(self, text: str, scene_id: str) -> List[Action]:
        """
        提取动作
        """
        actions = []
        action_counter = 1

        # 按句子分割
        sentences = re.split(r'[。！？]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 3:
                continue

            # 跳过纯对话句子
            if re.search(r'["「].*["」]', sentence):
                continue

            # 提取句子中的动作
            sentence_actions = self._extract_actions_from_sentence(sentence, scene_id, action_counter)

            if sentence_actions:
                actions.extend(sentence_actions)
                action_counter += len(sentence_actions)

        return actions

    def _extract_actions_from_sentence(self, sentence: str, scene_id: str, start_counter: int) -> List[Action]:
        """
        从单个句子中提取动作
        """
        actions = []

        # 查找所有可能的动作动词
        for category, action_types in self.action_categories.items():
            for action_type, keywords in action_types.items():
                for keyword in keywords:
                    if keyword in sentence:
                        # 提取执行者
                        actor = self._extract_actor(sentence, keyword)

                        # 提取目标（如果有）
                        target = self._extract_target(sentence, keyword)

                        # 分析动作强度
                        intensity = self.analyze_intensity(sentence, keyword)

                        # 创建动作对象
                        action_id = f"action_{start_counter + len(actions):03d}"

                        action = Action(
                            action_id=action_id,
                            type=action_type,
                            actor=actor,
                            target=target,
                            description=sentence,
                            intensity=intensity,
                            scene_ref=scene_id,
                            category=category
                        )

                        actions.append(action)

                        # 每个句子最多提取3个主要动作
                        if len(actions) >= 3:
                            return actions

        return actions

    """ 根据动作获取动作类别和类型 """

    def get_action_category_type(self, action_keyword: str) -> (str, str):
        for category, action_types in self.action_categories.items():
            for action_type, keywords in action_types.items():
                if action_keyword in keywords:
                    return category, action_type
        return "unknown", "unknown"

    def _extract_actor(self, sentence: str, action_keyword: str) -> str:
        """
        提取动作执行者
        """
        # 在动作关键词前寻找角色名
        parts = sentence.split(action_keyword)
        if len(parts) >= 1:
            before_action = parts[0]

            # 在动作前寻找可能的角色名
            # 简单策略：取最后一个逗号或句号后的内容
            last_segment = before_action

            # 尝试找到角色名（2-3个中文字符）
            name_pattern = r'([\u4e00-\u9fa5]{2,3})\s*[，。、]?\s*$'
            match = re.search(name_pattern, last_segment)
            if match:
                return match.group(1)

            # 如果找不到，返回"某人"
            return "某人"

        return "未知"

    def _extract_target(self, sentence: str, action_keyword: str) -> str:
        """
        提取动作目标
        """
        # 在动作关键词后寻找目标
        parts = sentence.split(action_keyword)
        if len(parts) >= 2:
            after_action = parts[1]

            # 寻找可能的目标描述
            # 简单策略：取动作后的第一个名词性短语
            target_patterns = [
                r'[向对着]([\u4e00-\u9fa5]{2,6})',
                r'[\u4e00-\u9fa5]{2,6}的([\u4e00-\u9fa5]{2,6})',
                r'([\u4e00-\u9fa5]{2,6})[上去里]',
            ]

            for pattern in target_patterns:
                match = re.search(pattern, after_action)
                if match:
                    return match.group(1)

        return ""

    def analyze_intensity(self, sentence: str, action_keyword: str) -> int:
        """
        分析动作强度
        """
        # 检查强度修饰词
        for intensity, keywords in self.intensity_keywords.items():
            for keyword in keywords:
                if keyword and keyword in sentence:
                    return intensity

        # 默认强度
        return 2


action_extractor = ActionExtractor()


class ScreenplayActionParser:
    """动作解析器 - 标准格式特有"""

    def __init__(self):
        self.action_counter = 0

    def parse_action_line(self, line: str, scene_ref: str) -> Optional[Dict]:
        """
        解析动作描述行

        标准格式：普通段落，描述角色动作和环境
        """
        line_stripped = line.strip()

        # 跳过空行和特殊行
        if not line_stripped or self._is_special_line(line_stripped):
            return None

        # 提取动作类型和执行者
        action_type, actor = self._extract_action_info(line_stripped)

        # 生成动作ID
        self.action_counter += 1
        action_id = f"action_{self.action_counter:03d}"

        return {
            "action_id": action_id,
            "type": action_type,
            "actor": actor,
            "description": line_stripped,
            "intensity": self._assess_intensity(line_stripped),
            "scene_ref": scene_ref,
            "category": "action"
        }

    def _is_special_line(self, line: str) -> bool:
        """检查是否是特殊行（不是动作描述）"""
        # 检查是否是页眉页脚
        if line.isdigit():  # 页码
            return True

        # 检查是否是场景标题、转场等
        special_patterns = [
            r'^(INT\.|EXT\.|INT/EXT\.)',
            r'^(CUT TO|FADE IN|FADE OUT|DISSOLVE)',
            r'^[\u4e00-\u9fa5]{2,4}$',  # 纯中文名
            r'^[A-Z\s]+$',  # 全大写（可能是角色名）
        ]

        for pattern in special_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True

        return False

    def _extract_action_info(self, text: str) -> Tuple[str, str]:
        """提取动作类型和执行者"""
        text_lower = text.lower()

        # 动作类型映射
        action_patterns = {
            "walk": ["walk", "step", "move", "go", "走", "行走", "移动"],
            "run": ["run", "dash", "rush", "跑", "奔跑", "冲"],
            "sit": ["sit", "坐下", "就坐"],
            "stand": ["stand", "rise", "站", "站立", "起身"],
            "look": ["look", "glance", "stare", "watch", "看", "注视", "盯着"],
            "touch": ["touch", "grab", "hold", "pull", "摸", "抓", "握"],
            "speak": ["say", "speak", "talk", "whisper", "说", "说话", "讲"],
        }

        # 寻找动作类型
        action_type = "action"
        for action, keywords in action_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    action_type = action
                    break
            if action_type != "action":
                break

        # 提取执行者（寻找角色名）
        actor = self._extract_actor(text)

        return action_type, actor

    def _extract_actor(self, text: str) -> str:
        """从动作描述中提取执行者"""
        # 常见角色名模式
        # 英文：He/She/John/Mary
        # 中文：他/她/张三/李四

        # 检查代词
        if " he " in text.lower() or "He " in text:
            return "他"
        elif " she " in text.lower() or "She " in text:
            return "她"
        elif "他们" in text:
            return "他们"
        elif "她们" in text:
            return "她们"

        # 检查常见英文名
        common_names = ['John', 'Mary', 'Tom', 'Lisa', 'David', 'Sarah']
        for name in common_names:
            if name in text:
                return name

        # 检查中文名（2-3个字符）
        name_match = re.search(r'[\u4e00-\u9fa5]{2,3}', text)
        if name_match:
            return name_match.group(0)

        return "某人"

    def _assess_intensity(self, text: str) -> int:
        """评估动作强度"""
        text_lower = text.lower()

        intensity_keywords = {
            1: ["slowly", "gently", "softly", "轻轻地", "慢慢地"],
            2: ["normally", "usually", "一般", "正常"],
            3: ["quickly", "firmly", "decidedly", "快速地", "坚决地"],
            4: ["forcefully", "angrily", "violently", "用力地", "愤怒地"],
            5: ["frantically", "desperately", "wildly", "疯狂地", "拼命地"]
        }

        for intensity, keywords in intensity_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return intensity

        return 2  # 默认中等强度