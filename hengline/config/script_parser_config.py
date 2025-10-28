"""
@FileName: script_parser_config.py
@Description: 剧本转换智能体配置
@Author: HengLine
@Time: 2025/10/27 17:22
"""

import re
from typing import Optional, Dict, List, Any


class ScriptParserConfig:
    """剧本解析器配置类"""
    
    # 默认场景模式
    DEFAULT_SCENE_PATTERNS = [
        '场景[:：]\s*([^，。；\n]+)[，。；]\s*([^，。；\n]+)',
        '地点[:：]\s*([^，。；\n]+)[，。；]\s*时间[:：]\s*([^，。；\n]+)',
        '([^，。；\n]+)[，。；]\s*([^，。；\n]+)\s*[的]?场景',
    ]
    
    # 对话模式
    DEFAULT_DIALOGUE_PATTERNS = [
        '([^：]+)[:：]\s*(.+)',
        '([^（）]+)[（(]([^)）]+)[)）][:：]\s*(.+)',
    ]
    
    # 动作情绪映射
    DEFAULT_ACTION_EMOTION_MAP = {
        "走": "平静", "行走": "平静", "漫步": "轻松", "散步": "悠闲",
        "笑": "开心", "微笑": "愉悦", "哭": "悲伤", "流泪": "伤心",
        "颤抖": "恐惧", "紧张": "紧张", "冷静": "平静", "思考": "专注",
    }
    
    # 时间关键词映射
    DEFAULT_TIME_KEYWORDS = {
        "早上": "早晨", "早晨": "早晨", "上午": "上午", "中午": "中午",
        "下午": "下午", "晚上": "晚上", "深夜": "深夜", "凌晨": "凌晨",
    }
    
    # 外貌关键词
    DEFAULT_APPEARANCE_KEYWORDS = {
        "西装": "穿着正式西装", "休闲装": "穿着休闲服装", "老人": "年长的",
        "年轻人": "年轻的", "男人": "男性", "女人": "女性",
    }
    
    # 地点关键词
    DEFAULT_LOCATION_KEYWORDS = {
        "咖啡馆": "咖啡馆", "餐厅": "餐厅", "办公室": "办公室",
        "家": "家", "公园": "公园", "街道": "街道",
        "超市": "超市", "商场": "商场", "学校": "学校", 
        "医院": "医院", "车站": "车站", "机场": "机场",
        "酒吧": "酒吧", "电影院": "电影院", "健身房": "健身房", 
        "图书馆": "图书馆", "会议室": "会议室",
        "公寓": "公寓", "房间": "房间", "卧室": "卧室", 
        "客厅": "客厅", "厨房": "厨房", "浴室": "浴室"
    }
    
    # 情绪关键词
    DEFAULT_EMOTION_KEYWORDS = {
        "高兴": ["开心", "高兴", "快乐", "愉快", "欢乐", "兴奋", "太好了", "真棒", "哈哈"],
        "悲伤": ["伤心", "难过", "悲伤", "难过", "哭", "流泪", "痛苦", "可怜", "惨"],
        "愤怒": ["生气", "愤怒", "恼火", "气死了", "混蛋", "该死", "讨厌", "烦"],
        "惊讶": ["啊", "哇", "惊讶", "震惊", "没想到", "真的吗", "什么", "怎么会"],
        "恐惧": ["害怕", "恐惧", "恐怖", "吓死了", "救命", "不要", "危险"],
        "紧张": ["紧张", "忐忑", "不安", "焦虑", "担心", "怎么办", "不会吧"],
        "平静": ["好的", "嗯", "是的", "知道了", "明白", "了解", "好"],
        "疑问": ["为什么", "什么", "哪里", "谁", "怎么", "如何", "是不是", "有没有"]
    }
    
    # 氛围关键词
    DEFAULT_ATMOSPHERE_KEYWORDS = {
        "温馨": ["温暖", "舒适", "柔和", "愉悦", "快乐", "放松"],
        "正式": ["严肃", "庄重", "严谨", "认真"],
        "轻松": ["愉快", "轻松", "休闲", "自在"],
        "紧张": ["紧张", "焦虑", "不安", "担忧"],
        "浪漫": ["浪漫", "甜蜜", "温馨", "幸福"],
        "悲伤": ["难过", "伤心", "悲伤", "痛苦"],
        "愤怒": ["生气", "愤怒", "恼火", "激动"],
        "惊讶": ["惊讶", "震惊", "意外", "突然"]
    }
    
    # 地点正则模式
    @staticmethod
    def get_location_patterns() -> List[re.Pattern]:
        """获取地点识别的正则模式"""
        return [
            re.compile(r'在([^，。；\n]+)[处里内]'),
            re.compile(r'位于([^，。；\n]+)'),
            re.compile(r'来到([^，。；\n]+)'),
            re.compile(r'走进([^，。；\n]+)'),
            re.compile(r'([^，。；\n]+)[内]'),  # 匹配"公寓内"这种格式
        ]
    
    @staticmethod
    def extract_time(time_hint: str) -> str:
        """从时间提示中提取标准时间格式
        
        Args:
            time_hint: 包含时间信息的文本
            
        Returns:
            格式化后的时间字符串
        """
        # 首先检查是否包含深夜/凌晨关键词
        if any(keyword in time_hint for keyword in ['深夜', '凌晨', '夜晚']):
            # 检查是否包含具体时间
            time_match = re.search(r'(\d{1,2})[:：](\d{1,2})', time_hint)
            if time_match:
                hour = int(time_match.group(1))
                minute = time_match.group(2)
                # 深夜时间保持原样，不转换为12小时制
                return f"深夜{hour}:{minute}"
            
            # 检查是否包含数字+时间单位
            hour_match = re.search(r'(\d{1,2})[点时]', time_hint)
            if hour_match:
                hour = int(hour_match.group(1))
                return f"深夜{hour}点"
            return "深夜"

        # 检查是否包含具体时间
        time_match = re.search(r'(\d{1,2})[:：](\d{1,2})', time_hint)
        if time_match:
            hour = int(time_match.group(1))
            minute = time_match.group(2)
            # 根据时间判断时段
            if hour < 6:
                period = "凌晨"
            elif hour < 12:
                period = "上午"
            elif hour < 18:
                period = "下午"
            else:
                period = "晚上"
            
            # 对于非凌晨/深夜时间，转换为12小时制
            if period not in ['凌晨', '深夜'] and hour > 12:
                hour = hour - 12
            
            return f"{period}{hour}:{minute}"

        # 检查是否包含时段关键词
        for keyword, time_period in ScriptParserConfig.DEFAULT_TIME_KEYWORDS.items():
            if keyword in time_hint:
                return time_period

        # 检查是否包含数字+时间单位
        hour_match = re.search(r'(\d{1,2})[点时]', time_hint)
        if hour_match:
            hour = int(hour_match.group(1))
            # 根据时间判断时段
            if hour < 6:
                period = "凌晨"
            elif hour < 12:
                period = "上午"
            elif hour < 18:
                period = "下午"
            else:
                period = "晚上"
            
            # 对于非凌晨/深夜时间，转换为12小时制
            if period not in ['凌晨', '深夜'] and hour > 12:
                hour = hour - 12
            
            return f"{period}{hour}点"

        return "下午3点"  # 默认时间
    
    @staticmethod
    def extract_time_from_text(text: str) -> Optional[str]:
        """从文本中提取时间信息
        
        Args:
            text: 包含时间信息的文本
            
        Returns:
            提取的时间字符串，如果未提取到则返回None
        """
        # 首先检查是否包含深夜/凌晨关键词
        if any(keyword in text for keyword in ['深夜', '凌晨', '夜晚']):
            # 尝试提取具体时间
            time_match = re.search(r'(\d{1,2})[:：](\d{1,2})', text)
            if time_match:
                hour = int(time_match.group(1))
                minute = time_match.group(2)
                return f"深夜{hour}:{minute}"
            
            # 检查是否包含数字+时间单位
            hour_match = re.search(r'(\d{1,2})[点时]', text)
            if hour_match:
                hour = int(hour_match.group(1))
                return f"深夜{hour}点"
            return "深夜"

        # 检查时间段关键词
        for keyword, time_period in ScriptParserConfig.DEFAULT_TIME_KEYWORDS.items():
            if keyword in text:
                # 尝试提取具体时间
                time_match = re.search(r'(\d{1,2})[:：](\d{1,2})', text)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = time_match.group(2)
                    # 根据时间判断时段
                    if hour < 6:
                        period = "凌晨"
                    elif hour < 12:
                        period = "上午"
                    elif hour < 18:
                        period = "下午"
                    else:
                        period = "晚上"
                    
                    # 对于非凌晨/深夜时间，转换为12小时制
                    if period not in ['凌晨', '深夜'] and hour > 12:
                        hour = hour - 12
                    
                    return f"{period}{hour}:{minute}"
                return time_period

        # 检查是否有具体时间
        time_match = re.search(r'(\d{1,2})[:：](\d{1,2})', text)
        if time_match:
            hour = int(time_match.group(1))
            minute = time_match.group(2)
            # 根据时间判断时段
            if hour < 6:
                period = "凌晨"
            elif hour < 12:
                period = "上午"
            elif hour < 18:
                period = "下午"
            else:
                period = "晚上"
            
            # 对于非凌晨/深夜时间，转换为12小时制
            if period not in ['凌晨', '深夜'] and hour > 12:
                hour = hour - 12
            
            return f"{period}{hour}:{minute}"
        
        return None
    
    @staticmethod
    def initialize_patterns(config_path: Optional[str] = None) -> Dict[str, Any]:
        """初始化中文剧本解析需要的模式和关键词
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            包含所有解析模式和关键词的字典
        """
        # 默认配置
        default_config = {
            "scene_patterns": ScriptParserConfig.DEFAULT_SCENE_PATTERNS,
            "dialogue_patterns": ScriptParserConfig.DEFAULT_DIALOGUE_PATTERNS,
            "action_emotion_map": ScriptParserConfig.DEFAULT_ACTION_EMOTION_MAP,
            "time_keywords": ScriptParserConfig.DEFAULT_TIME_KEYWORDS,
            "appearance_keywords": ScriptParserConfig.DEFAULT_APPEARANCE_KEYWORDS,
            "location_keywords": ScriptParserConfig.DEFAULT_LOCATION_KEYWORDS,
            "emotion_keywords": ScriptParserConfig.DEFAULT_EMOTION_KEYWORDS,
            "atmosphere_keywords": ScriptParserConfig.DEFAULT_ATMOSPHERE_KEYWORDS
        }

        # 尝试从配置文件加载
        config_data = default_config.copy()
        if config_path:
            try:
                import yaml
                from hengline.logger import debug, warning
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    # 确保loaded_config不为None
                    if loaded_config is not None:
                        # 合并配置，保留默认值作为回退
                        for key in default_config:
                            if key in loaded_config:
                                config_data[key] = loaded_config[key]
                debug(f"成功从配置文件加载剧本解析配置: {config_path}")
                # 打印配置信息，用于调试
                print(f"配置加载成功: ")
                print(f"  - 场景识别模式: {len(config_data.get('scene_patterns', []))} 个")
                print(f"  - 对话识别模式: {len(config_data.get('dialogue_patterns', []))} 个")
                print(f"  - 动作情绪映射: {len(config_data.get('action_emotion_map', {}))} 个")
                print(f"  - 角色外观关键词: {len(config_data.get('appearance_keywords', {}))} 个")
                print(f"  - 时段关键词: {len(config_data.get('time_keywords', {}))} 个")
                print(f"  - 地点关键词: {len(config_data.get('location_keywords', {}))} 个")
                print(f"  - 情绪关键词: {len(config_data.get('emotion_keywords', {}))} 个")
                print(f"  - 场景氛围关键词: {len(config_data.get('atmosphere_keywords', {}))} 个")
            except Exception as e:
                warning(f"无法加载配置文件 {config_path}，使用默认配置: {str(e)}")

        # 编译正则表达式模式
        scene_patterns = []
        for pattern_str in config_data.get('scene_patterns', []):
            try:
                # 注意：这里需要添加r前缀以确保正则表达式中的转义字符正确处理
                scene_patterns.append(re.compile(pattern_str))
            except re.error as e:
                import warnings
                warnings.warn(f"正则表达式模式编译失败: {pattern_str}, 错误: {str(e)}")

        # 编译对话模式
        dialogue_patterns = []
        for pattern_str in config_data.get('dialogue_patterns', []):
            try:
                dialogue_patterns.append(re.compile(pattern_str))
            except re.error as e:
                import warnings
                warnings.warn(f"对话模式编译失败: {pattern_str}, 错误: {str(e)}")

        # 如果对话模式为空，使用默认模式
        if not dialogue_patterns:
            dialogue_patterns = [
                re.compile(r'([^：]+)[:：]\s*(.+)'),
                re.compile(r'([^（）]+)[（(]([^)）]+)[)）][:：]\s*(.+)')
            ]

        return {
            "scene_patterns": scene_patterns,
            "dialogue_patterns": dialogue_patterns,
            "action_emotion_map": config_data.get('action_emotion_map', {}),
            "time_keywords": config_data.get('time_keywords', {}),
            "appearance_keywords": config_data.get('appearance_keywords', {}),
            "location_keywords": config_data.get('location_keywords', {}),
            "emotion_keywords": config_data.get('emotion_keywords', {}),
            "atmosphere_keywords": config_data.get('atmosphere_keywords', {})
        }
    
    @staticmethod
    def extract_location_from_text(text: str, location_keywords: Optional[Dict[str, str]] = None) -> Optional[str]:
        """从文本中提取地点信息
        
        Args:
            text: 包含地点信息的文本
            location_keywords: 自定义地点关键词字典
            
        Returns:
            提取的地点字符串，如果未提取到则返回None
        """
        # 获取地点正则模式
        location_patterns = ScriptParserConfig.get_location_patterns()

        # 使用从配置加载的地点关键词
        common_locations = list(location_keywords.keys()) if location_keywords else ScriptParserConfig.DEFAULT_LOCATION_KEYWORDS.keys()

        # 首先尝试模式匹配
        for pattern in location_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()

        # 然后检查常见地点关键词
        for location in common_locations:
            if location in text:
                # 尝试提取更具体的地点描述
                location_match = re.search(f'(.{{0,20}}){location}(.{{0,10}})', text)
                if location_match:
                    full_location = location_match.group(0).strip()
                    # 清理多余字符
                    full_location = re.sub(r'[，。；：]', '', full_location)
                    return full_location
                return location

        return None


# 创建配置实例供外部使用
script_parser_config = ScriptParserConfig()
