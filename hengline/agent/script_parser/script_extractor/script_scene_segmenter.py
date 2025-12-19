"""
@FileName: script_scene_extractor.py
@Description: 场景分割器
@Author: HengLine
@Time: 2025/12/19 0:14
"""
import re
from typing import List, Dict, Optional

from hengline.agent.script_parser.script_parser_model import Scene
from .script_emotion_extractor import emotion_extractor


class NaturalLanguageSceneSegmenter:
    """场景分割器"""

    def segment(self, text: str) -> List[str]:
        """
        将剧本分割成场景
        """
        scenes = []
        current_scene = []

        # 按段落分割
        paragraphs = text.split('\n')

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 检测是否是新的场景开始
            if self._is_new_scene_start(para, current_scene):
                if current_scene:
                    scenes.append('\n'.join(current_scene))
                    current_scene = []

            current_scene.append(para)

        # 添加最后一个场景
        if current_scene:
            scenes.append('\n'.join(current_scene))

        return scenes

    def _is_new_scene_start(self, paragraph: str, current_scene_lines: List[str]) -> bool:
        """
        判断是否是新的场景开始
        """
        # 规则1：如果当前没有场景内容，第一个段落就是场景开始
        if not current_scene_lines:
            return True

        # 规则2：检测时间/地点变化关键词
        time_location_keywords = [
            # 时间变化
            r'第二天', r'几天后', r'不久', r'转眼', r'此时', r'突然',
            r'次日', r'凌晨', r'清晨', r'上午', r'中午', r'下午',
            r'傍晚', r'夜晚', r'深夜',

            # 地点变化
            r'在.*(?:房间|客厅|卧室|厨房|办公室|学校|医院|街道|路上)',
            r'来到.*', r'走进.*', r'进入.*', r'到达.*',
            r'从.*到.*', r'在.*里', r'在.*中',

            # 场景转换词
            r'切换', r'转场', r'画面一转', r'镜头一转',
        ]

        for pattern in time_location_keywords:
            if re.search(pattern, paragraph):
                return True

        # 规则3：检测空行后的新段落（如果之前有足够内容）
        # 这里基于段落顺序判断，不依赖空行

        # 规则4：检测明显的场景描述开头
        scene_start_patterns = [
            r'^[^，。！？]{5,20}（',  # 角色名+动作描述
            r'^[^，。！？]{2,4}[：:]',  # 角色名+冒号
            r'^【.*】',  # 方括号标记
            r'^##\s+',  # Markdown标题
            r'^第[一二三四五六七八九十\d]+[章节场幕]',
        ]

        for pattern in scene_start_patterns:
            if re.match(pattern, paragraph):
                return True

        return False

    def extract_scene_info(self, scene_text: str, scene_id: str) -> Scene:
        """
        提取场景基本信息
        """
        scene = Scene(
            scene_id=scene_id,
            order_index=1,
            location="",
            time_of_day="",
            mood="",
            summary=None,
            character_refs=[],
            dialogue_refs=[],
            action_refs=[]
        )

        # 从场景文本中提取地点
        scene.location = self.extract_location(scene_text)

        # 提取时间
        scene.time_of_day = self.extract_time(scene_text)

        # 推断氛围
        scene.mood = emotion_extractor.infer_mood(scene_text)

        return scene

    @staticmethod
    def extract_location(text: str) -> str:
        """提取地点信息"""
        location_patterns = [
            r'在([^，。！？]{2,10}?)[里中内]',  # 在XXX里/中/内
            r'到([^，。！？]{2,10}?)去?',  # 到XXX去
            r'来到([^，。！？]{2,10})',  # 来到XXX
            r'([^，。！？]{2,10}?)门口',  # XXX门口
            r'([^，。！？]{2,10}?)前',  # XXX前
        ]

        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                location = match.group(1)
                # 过滤掉常见非地点词
                if location not in ["这里", "那里", "这边", "那边", "外面", "里面", "在这", "在那"]:
                    return location

        # 常见地点关键词
        common_locations = [
            "客厅", "卧室", "厨房", "书房", "办公室", "教室",
            "医院", "餐厅", "咖啡厅", "酒吧", "公园", "街道",
            "车站", "机场", "山上", "海边", "河边", "商场",
            "超市", "电影院", "图书馆", "体育馆", "游乐场",
            "地铁站", "学校", "公司", "工厂", "农场", "村庄", "城市", "乡村", "森林", "沙漠", "岛屿", "桥上",
            "屋顶", "地下室", "阳台", "浴室", "走廊", "大厅", "停车场", "车内", "车外", "船上", "飞机上",
            "火车上", "剧院", "音乐厅", "展览馆", "会议室", "礼堂", "体育场", "游泳池", "健身房", "美容院", "理发店", "宠物店", "花园"
        ]

        for location in common_locations:
            if location in text:
                return location

        return "未知地点"

    @staticmethod
    def extract_time(text: str) -> str:
        """提取时间信息"""
        time_patterns = {
            "清晨": r'清晨|黎明|拂晓|早上|早晨',
            "上午": r'上午|早上|早晨',
            "中午": r'中午|正午|午间',
            "下午": r'下午|午后|晌午',
            "傍晚": r'傍晚|黄昏|日落',
            "夜晚": r'夜晚|晚上|夜里|夜间|深夜|午夜',
            "凌晨": r'凌晨|半夜',
            "白天": r'白天|日间|晴天',
        }

        for time_label, pattern in time_patterns.items():
            if re.search(pattern, text):
                return time_label

        # 检查时间数字
        time_match = re.search(r'(\d{1,2})[点时]', text)
        if time_match:
            hour = int(time_match.group(1))
            if 5 <= hour < 12:
                return "上午"
            elif 12 <= hour < 14:
                return "中午"
            elif 14 <= hour < 18:
                return "下午"
            elif 18 <= hour < 24:
                return "夜晚"
            elif 0 <= hour < 5:
                return "凌晨"

        return "未知时间"


class StructuredSceneSegmenter:
    """结构化分场剧本场景分割器"""

    def __init__(self):
        # 字段名标准化映射
        self.field_standardization = {
            # 中文字段名 -> 标准字段名
            '场景': 'scene',
            '场次': 'scene',
            '地点': 'location',
            '位置': 'location',
            '时间': 'time',
            '时刻': 'time',
            '人物': 'characters',
            '角色': 'characters',
            '内容': 'content',
            '情节': 'content',
            '对话': 'dialogue',
            '台词': 'dialogue',
            '动作': 'action',
            '行动': 'action',
            '氛围': 'mood',
            '情绪': 'mood',

            # 英文字段名 -> 标准字段名
            'Scene': 'scene',
            'Location': 'location',
            'Time': 'time',
            'Characters': 'characters',
            'Content': 'content',
            'Dialogue': 'dialogue',
            'Action': 'action',
            'Mood': 'mood'
        }

        # 多行字段（可以跨多行的字段）
        self.multiline_fields = {'content', 'dialogue', 'action'}

    def segment(self, text: str) -> List[str]:
        """
        分割结构化剧本为独立场景

        特有逻辑：处理结构化场景标记，如：
        - "场景1："、"第二场："、"## 场景三"
        - 字段化结构：场景、地点、时间、人物、内容
        """
        scenes = []
        current_scene_lines = []
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测是否是新的结构化场景开始
            if self._is_structured_scene_start(line):
                # 保存当前场景
                if current_scene_lines:
                    scene_text = '\n'.join(current_scene_lines)
                    if self._is_valid_scene(scene_text):
                        scenes.append(scene_text)
                    current_scene_lines = []

            current_scene_lines.append(line)

        # 处理最后一个场景
        if current_scene_lines:
            scene_text = '\n'.join(current_scene_lines)
            if self._is_valid_scene(scene_text):
                scenes.append(scene_text)

        return scenes

    def _is_structured_scene_start(self, line: str) -> bool:
        """
        检测结构化场景开始

        特有逻辑：识别结构化标记
        """
        # 场景编号模式
        scene_number_patterns = [
            r'^场景\d*[：:]',  # 场景1：、场景：
            r'^第[一二三四五六七八九十\d]+场[：:]',  # 第一场：、第二场：
            r'^SCENE\s*\d*[:：]',  # SCENE 1:、SCENE:
            r'^场次\d*[：:]',  # 场次1：、场次：
        ]

        # 结构化标题模式
        structured_title_patterns = [
            r'^##\s+场景',  # ## 场景一
            r'^【场景',  # 【场景1】
            r'^===.*场景',  # === 场景一 ===
            r'^◆\s*场景',  # ◆ 场景一
        ]

        # 检查是否匹配任何模式
        for pattern in scene_number_patterns + structured_title_patterns:
            if re.match(pattern, line):
                return True

        # 检查是否有多个结构化字段标记
        if self._count_structured_fields(line) >= 2:
            return True

        return False

    def _count_structured_fields(self, line: str) -> int:
        """
        统计结构化字段数量

        特有逻辑：识别结构化字段标记
        """
        field_markers = [
            '场景', '地点', '时间', '人物', '角色',
            '内容', '对话', '动作', '氛围',
            'Scene', 'Location', 'Time', 'Characters',
            'Content', 'Dialogue', 'Action', 'Mood'
        ]

        count = 0
        for marker in field_markers:
            if re.search(fr'{marker}[：:]', line):
                count += 1

        return count

    def _is_valid_scene(self, scene_text: str) -> bool:
        """
        验证是否有效的结构化场景

        特有逻辑：检查是否包含必要的结构化信息
        """
        lines = scene_text.split('\n')

        # 至少需要2行有效内容
        valid_lines = [l for l in lines if l.strip() and len(l.strip()) > 1]
        if len(valid_lines) < 2:
            return False

        # 检查是否有结构化字段
        has_structured_field = False
        for line in lines[:5]:  # 检查前5行
            if re.search(r'[^：:]{1,10}[：:].+', line):
                has_structured_field = True
                break

        return has_structured_field

    def parse(self, scene_text: str) -> Scene:
        """
        解析单个结构化场景

        特有逻辑：处理结构化字段的键值对
        """
        scene = Scene(
            scene_id="",
            order_index=1,
            location="",
            time_of_day="",
            mood="",
            summary=None,
            character_refs=[],
            dialogue_refs=[],
            action_refs=[]
        )

        lines = scene_text.strip().split('\n')
        current_field = None
        field_content = []

        for line in lines:
            line = line.strip()
            if not line:
                # 空行：如果是多行字段则继续，否则结束当前字段
                if current_field in self.multiline_fields:
                    field_content.append("")
                continue

            # 尝试提取字段标记
            field_match = self._extract_field_marker(line)

            if field_match:
                # 保存前一个字段的内容
                if current_field:
                    self._save_field_content(current_field, field_content, scene)

                # 开始新字段
                current_field = field_match['field_name']
                field_content = [field_match.get('field_value', '')]
            elif current_field:
                # 续行内容
                field_content.append(line)
            else:
                # 添加到描述中
                scene.description += " " + line

        # 保存最后一个字段
        if current_field:
            self._save_field_content(current_field, field_content, scene)

        return scene

    def _extract_field_marker(self, line: str) -> Optional[Dict]:
        """
        提取结构化字段标记

        特有逻辑：识别"字段名：值"格式
        """
        # 支持中文和英文冒号
        patterns = [
            r'^([^：:]{1,10})[：:]\s*(.*)$',  # 字段名：值
            r'^([^：:]{1,10})[：:]\s*$',  # 字段名：（无值，值在后续行）
        ]

        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                field_name = match.group(1).strip()
                field_value = match.group(2).strip() if len(match.groups()) > 1 else ""

                # 标准化字段名
                std_field_name = self.field_standardization.get(field_name, field_name.lower())

                return {
                    "original_name": field_name,
                    "field_name": std_field_name,
                    "field_value": field_value
                }

        return None

    def _save_field_content(self, field_name: str, content_lines: List[str], scene: Scene):
        """
        保存字段内容

        特有逻辑：不同字段的特殊处理
        """
        content = '\n'.join(content_lines).strip()

        if field_name == 'scene':
            scene.summary = content

        elif field_name == 'location':
            scene.location = content

        elif field_name == 'time':
            scene.time_of_day = NaturalLanguageSceneSegmenter.extract_time(content)

        elif field_name == 'characters':
            scene.character_refs = self._parse_character_list(content)

        elif field_name == 'content':
            scene.description = content

        elif field_name == 'dialogue':
            scene.description += "。" + content

        elif field_name == 'action':
            scene.description += "。" + content

        elif field_name == 'mood':
            scene.mood = content

    def _parse_character_list(self, content: str) -> List[str]:
        """
        解析角色列表字符串

        特有逻辑：处理各种分隔符的角色列表
        """
        # 多种分隔符
        separators = ['、', ',', '，', ' ', '\n', ';', '；', '/', '|']

        for sep in separators:
            if sep in content:
                characters = [char.strip() for char in content.split(sep) if char.strip()]
                if characters:
                    return characters

        # 如果没有分隔符，整个内容作为一个角色
        return [content] if content.strip() else []


class SceneSluglineParser:
    """场景标题解析器 - 标准格式特有"""

    def parse_slugline(self, line: str) -> Optional[Dict]:
        """
        解析场景标题 (slugline)

        格式：
        INT. LIVING ROOM - NIGHT
        EXT. PARK - DAY
        INT./EXT. CAR - MOVING - DAY
        """
        # 标准化大写
        line_upper = line.strip().upper()

        # 匹配模式
        patterns = [
            # 标准格式：INT. LOCATION - TIME
            r'^(INT\.|EXT\.|INT/EXT\.|INT\./EXT\.)\s+([^-]+?)\s*-\s*(DAY|NIGHT|MORNING|AFTERNOON|EVENING|LATER|CONTINUOUS|SAME)',

            # 简写格式：INT LOCATION TIME
            r'^(INT|EXT|INT/EXT)\s+([^-]+?)\s+(DAY|NIGHT)',

            # 中文格式：内. 客厅 - 夜
            r'^(内\.|外\.|内/外\.)\s+([^-]+?)\s*-\s*(日|夜|白天|夜晚|早晨|黄昏)',
        ]

        for pattern in patterns:
            match = re.match(pattern, line_upper if pattern.startswith('^[A-Z]') else line)
            if match:
                return self._parse_slugline_match(match, line)

        return None

    def _parse_slugline_match(self, match, original_line: str) -> Dict:
        """解析匹配的场景标题"""
        int_ext = match.group(1).replace('.', '').replace('/', '_').lower()
        location = match.group(2).strip()
        time_raw = match.group(3).strip()

        # 标准化地点
        location_clean = self._clean_location(location)

        # 标准化时间
        time_clean = self._normalize_time(time_raw)

        # 推断室内外
        if 'int' in int_ext:
            location_type = '室内'
        elif 'ext' in int_ext:
            location_type = '室外'
        else:
            location_type = '室内/外'

        return {
            "original": original_line,
            "int_ext": int_ext,
            "location": location_clean,
            "time_raw": time_raw,
            "time_of_day": time_clean,
            "location_type": location_type,
            "is_slugline": True
        }

    def _clean_location(self, location: str) -> str:
        """清理地点描述"""
        # 移除多余空格和特殊字符
        location = re.sub(r'\s+', ' ', location)

        # 常见地点缩写扩展
        location_expansions = {
            'BEDRM': '卧室',
            'LIVING RM': '客厅',
            'KITCHEN': '厨房',
            'OFFICE': '办公室',
            'CAR': '车内',
            'STREET': '街道',
            'PARK': '公园',
        }

        # 检查是否有缩写
        for abbr, full in location_expansions.items():
            if abbr in location.upper():
                return full

        return location.title()

    def _normalize_time(self, time_str: str) -> str:
        """标准化时间描述"""
        time_mapping = {
            'DAY': '白天',
            'NIGHT': '夜晚',
            'MORNING': '早晨',
            'AFTERNOON': '下午',
            'EVENING': '傍晚',
            'LATER': '稍后',
            'CONTINUOUS': '连续',
            'SAME': '同时',
            '日': '白天',
            '夜': '夜晚',
            '白天': '白天',
            '夜晚': '夜晚',
            '早晨': '早晨',
            '黄昏': '傍晚'
        }

        time_upper = time_str.upper()
        return time_mapping.get(time_upper, time_str)