"""
@FileName: script_character_extractor.py
@Description: 角色提取器
@Author: HengLine
@Time: 2025/12/19 0:15
"""
import random
import re
from typing import Optional, Dict, List, Set, Tuple

from hengline.agent.script_parser.script_parser_model import Character


class CharacterExtractor:
    """角色提取器"""

    def __init__(self):
        # 中文姓氏（用于识别角色名）
        self.chinese_surnames = [
            '赵', '钱', '孙', '李', '周', '吴', '郑', '王', '冯', '陈',
            '褚', '卫', '蒋', '沈', '韩', '杨', '朱', '秦', '尤', '许',
            '何', '吕', '施', '张', '孔', '曹', '严', '华', '金', '魏',
            '陶', '姜', '戚', '谢', '邹', '喻', '柏', '水', '窦', '章',
            '云', '苏', '潘', '葛', '奚', '范', '彭', '郎', '鲁', '韦'
        ]

        # 常见角色名模式
        self.name_patterns = [
            r'([\u4e00-\u9fa5]{2,3})（',  # 小明（
            r'([\u4e00-\u9fa5]{2,3})[：:]',  # 小明：
            r'([\u4e00-\u9fa5]{2,3})说',  # 小明说
            r'([\u4e00-\u9fa5]{2,3})道',  # 小明道
            r'([\u4e00-\u9fa5]{2,3})喊道',  # 小明喊道
            r'([\u4e00-\u9fa5]{2,3})轻声说',  # 小明轻声说
            r'([\u4e00-\u9fa5]{2,3})[\s]',  # 小明 后面有空格
        ]

        self.age_keywords = {
            "小孩": (5, 12), "儿童": (5, 12), "少年": (13, 18),
            "青年": (18, 35), "年轻人": (18, 35), "中年": (36, 60),
            "老人": (61, 80), "老年": (61, 80), "爷爷": (61, 80),
            "奶奶": (61, 80), "叔叔": (30, 50), "阿姨": (30, 50),
        }

    def extract(self, text: str) -> List[Character]:
        """
        从文本中提取角色信息
        """
        characters = []

        # 方法1：通过模式匹配
        found_names = set()

        for pattern in self.name_patterns:
            matches = re.findall(pattern, text)
            for name in matches:
                if self._is_valid_name(name) and name not in found_names:
                    found_names.add(name)
                    characters.append(self._create_character(name, text))

        # 方法2：通过代词和称呼推断
        inferred_characters = self._infer_characters_from_context(text, found_names)
        characters.extend(inferred_characters)

        # 方法3：去重和合并
        unique_characters = self.deduplicate_characters(characters)

        return unique_characters

    def deduplicate_characters(self, characters: List[Character]) -> List[Character]:
        """
        去重角色列表
        """
        seen_names = set()
        unique_characters = []

        for char in characters:
            if char.name not in seen_names:
                seen_names.add(char.name)
                unique_characters.append(char)

        return unique_characters

    def _is_valid_name(self, name: str) -> bool:
        """
        判断是否是有效的角色名
        """
        # 长度检查
        if len(name) < 2 or len(name) > 4:
            return False

        # 检查是否包含姓氏
        if name[0] in self.chinese_surnames:
            return True

        # 常见名字模式
        common_names = ['小明', '小红', '小李', '小王', '张华', '李娜',
                        '王伟', '刘芳', '陈明', '杨丽', '赵强', '周涛']

        if name in common_names:
            return True

        # 检查是否是描述性词语
        descriptive_words = ['老人', '小孩', '男人', '女人', '青年', '少年',
                             '医生', '老师', '警察', '司机', '老板']

        if name in descriptive_words:
            return True

        # 默认接受2-3个中文字符
        return 2 <= len(name) <= 3

    def _create_character(self, name: str, context: str) -> Character:
        """
        创建角色对象
        """
        character = Character(
            name=name,
            age=self.infer_age(name, context),
            gender=self.infer_gender(name, context),
            role_hint=self._infer_role(name, context),
            description=self._extract_description(name, context)
        )

        return character

    def infer_age(self, name: str, context: str) -> Optional[int]:
        """
        推断角色年龄
        """
        # 基于名字推断
        if any(word in name for word in ['小', '孩', '童', '少年']):
            return random.randint(8, 18)
        elif any(word in name for word in ['老', '翁', '婆', '爷', '奶']):
            return random.randint(60, 80)
        elif any(word in name for word in ['青', '壮']):
            return random.randint(20, 40)
        elif '中年' in name:
            return random.randint(40, 60)

        for keyword, (min_age, max_age) in self.age_keywords.items():
            if keyword in context:
                return random.randint(min_age, max_age)

        return None

    def infer_gender(self, name: str, context: str) -> str:
        """
        推断角色性别
        """
        # 基于名字推断
        feminine_chars = ['芳', '丽', '娜', '婷', '娟', '娇', '妹', '姐', '娘', '女']
        masculine_chars = ['强', '伟', '刚', '勇', '军', '兵', '哥', '兄', '弟', '男']

        for char in feminine_chars:
            if char in name:
                return "女"

        for char in masculine_chars:
            if char in name:
                return "男"

        # 基于上下文
        if '她' in context or '女士' in context or '女孩' in context:
            return "女"
        elif '他' in context or '先生' in context or '男孩' in context:
            return "男"

        # 常见角色名
        common_female_names = ['小红', '李娜', '刘芳', '杨丽', '王芳']
        common_male_names = ['小明', '王伟', '李强', '张华', '刘勇']

        if name in common_female_names:
            return "女"
        elif name in common_male_names:
            return "男"

        return "未知"

    def _infer_role(self, name: str, context: str) -> str:
        """
        推断角色类型
        """
        # 统计名字在上下文中的出现频率
        occurrences = context.count(name)

        if occurrences > 5:
            return "主角"
        elif occurrences > 2:
            return "配角"
        else:
            return "群众"

    def _extract_description(self, name: str, context: str) -> str:
        """
        提取角色描述
        """
        # 在名字周围的句子中寻找描述
        sentences = re.split(r'[。！？]', context)

        for sentence in sentences:
            if name in sentence:
                # 提取描述性词语
                descriptive_words = []

                # 外貌描述关键词
                appearance_keywords = ['高', '矮', '胖', '瘦', '美', '丑',
                                       '年轻', '年老', '长发', '短发', '眼镜']

                for keyword in appearance_keywords:
                    if keyword in sentence:
                        descriptive_words.append(keyword)

                if descriptive_words:
                    return f"{'、'.join(descriptive_words)}的{name}"

        return ""

    def _infer_characters_from_context(self, text: str, found_names: Set[str]) -> List[Dict]:
        """
        从上下文中推断角色
        """
        characters = []

        # 通过代词推断
        pronoun_patterns = [
            (r'([他她它])[^。！？]{0,20}说', '说话者'),
            (r'([他她它])[^。！？]{0,20}看', '观察者'),
            (r'([他她它])[^。！？]{0,20}走', '行动者'),
        ]

        for pattern, role in pronoun_patterns:
            matches = re.findall(pattern, text)
            for pronoun in matches:
                if pronoun not in ['他', '她']:
                    continue

                # 为代词创建临时角色名
                temp_name = f"{pronoun}某人"
                if temp_name not in found_names:
                    character = {
                        "name": temp_name,
                        "age": None,
                        "gender": "男" if pronoun == "他" else "女",
                        "role_hint": role,
                        "description": f"用'{pronoun}'代指的角色"
                    }
                    characters.append(character)
                    found_names.add(temp_name)

        return characters


character_extractor = CharacterExtractor()


class ScreenplayCharacterParser:
    """角色行解析器 - 标准格式特有"""

    def parse_character_line(self, line: str, current_scene: Optional[Dict]) -> Optional[Dict]:
        """
        解析角色名行

        标准格式：全大写，单独一行
        可能包含表演提示：(V.O.) (O.S.) 等
        """
        line_stripped = line.strip()

        # 检查是否全大写（英文）或纯中文名
        is_all_uppercase = line_stripped.isupper() and line_stripped.isalpha()
        is_chinese_name = bool(re.match(r'^[\u4e00-\u9fa5]{2,4}$', line_stripped))

        if not (is_all_uppercase or is_chinese_name):
            return None

        # 检查是否是转场或场景标题（避免误判）
        if self._is_transition_or_scene(line_stripped):
            return None

        # 提取角色名和表演提示
        character_name, performance_notes = self._extract_name_and_notes(line_stripped)

        if not character_name:
            return None

        # 推断角色信息
        character_data = {
            "name": character_name,
            "original_line": line_stripped,
            "performance_notes": performance_notes,
            "gender": self._infer_gender(character_name),
            "role_hint": self._infer_role_hint(character_name, current_scene)
        }

        return character_data

    def _is_transition_or_scene(self, line: str) -> bool:
        """检查是否是转场或场景标题"""
        transition_keywords = ['CUT TO', 'FADE IN', 'FADE OUT', 'DISSOLVE TO',
                               'SMASH CUT', 'MATCH CUT', 'IRIS']

        scene_keywords = ['INT.', 'EXT.', 'INT/EXT.', '内.', '外.']

        line_upper = line.upper()

        for keyword in transition_keywords + scene_keywords:
            if line_upper.startswith(keyword):
                return True

        return False

    def _extract_name_and_notes(self, line: str) -> Tuple[str, List[str]]:
        """
        提取角色名和表演提示

        格式：
        JOHN (V.O.)
        MARY (O.S.)
        LI MING (同时)
        """
        # 移除括号内容（表演提示）
        performance_notes = []

        # 提取括号内容
        paren_matches = re.findall(r'\(([^)]+)\)', line)
        for match in paren_matches:
            performance_notes.append(match)

        # 移除括号内容得到纯净的角色名
        clean_name = re.sub(r'\s*\([^)]+\)', '', line).strip()

        # 如果是英文名，转为首字母大写
        if clean_name.isupper():
            clean_name = clean_name.title()

        return clean_name, performance_notes

    def _infer_gender(self, name: str) -> str:
        """推断角色性别"""
        # 英文名性别推断（基于常见名字）
        male_names = ['JOHN', 'MIKE', 'DAVID', 'ROBERT', 'JAMES', 'TOM',
                      'JACK', 'WILLIAM', 'RICHARD', 'CHARLES']
        female_names = ['MARY', 'ANNA', 'SUSAN', 'LISA', 'JENNIFER', 'SARAH',
                        'EMILY', 'JESSICA', 'ELIZABETH', 'REBECCA']

        name_upper = name.upper()

        if any(male in name_upper for male in male_names):
            return "男"
        elif any(female in name_upper for female in female_names):
            return "女"

        # 中文名性别推断
        if re.search(r'[\u4e00-\u9fa5]', name):
            feminine_chars = ['芳', '丽', '娜', '婷', '娟', '娇', '妹', '姐']
            masculine_chars = ['强', '伟', '刚', '勇', '军', '兵', '哥', '兄']

            for char in feminine_chars:
                if char in name:
                    return "女"

            for char in masculine_chars:
                if char in name:
                    return "男"

        return "未知"

    def _infer_role_hint(self, name: str, current_scene: Optional[Dict]) -> str:
        """推断角色类型提示"""
        if not current_scene:
            return "角色"

        # 如果这是场景中的第一个角色，可能是主角
        if not current_scene.get("characters"):
            return "主角"

        # 如果名字在场景描述中频繁出现，可能是重要角色
        scene_desc = current_scene.get("description", "")
        if name.upper() in scene_desc.upper():
            return "重要角色"

        return "配角"
