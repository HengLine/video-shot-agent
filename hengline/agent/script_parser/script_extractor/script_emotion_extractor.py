"""
@FileName: acript_other_extractor.py
@Description: 
@Author: HengLine
@Time: 2025/12/19 14:44
"""
class EmotionExtractor:
    """情绪提取器"""
    def __init__(self):

        # 情绪关键词
        self.emotion_keywords = {
            "愤怒": ["生气", "愤怒", "发火", "怒吼", "咆哮", "气愤"],
            "悲伤": ["悲伤", "难过", "伤心", "哭泣", "流泪", "哽咽", "哀伤", "失落"],
            "高兴": ["开心", "高兴", "欢乐", "笑", "愉快", "欣喜"],
            "惊讶": ["惊讶", "吃惊", "震惊", "诧异", "惊奇"],
            "恐惧": ["害怕", "恐惧", "惊慌", "恐慌", "惊恐"],
            "紧张": ["紧张", "紧绷", "危急", "危险", "惊恐", "恐慌"],
            "欢乐": ["开心", "高兴", "欢乐", "欢笑", "喜悦", "愉快"],
            "浪漫": ["浪漫", "温馨", "甜蜜", "柔情", "温柔"],
            "恐怖": ["恐怖", "可怕", "惊悚", "诡异", "阴森"],
            "悬疑": ["神秘", "疑惑", "好奇", "探究", "怀疑"],
            "激烈": ["激烈", "激烈", "争吵", "冲突", "战斗"],
            "平静": ["平静", "安静", "宁静", "祥和", "舒适", "安宁", "冷静", "淡定", "沉稳"],
        }
        # 检查语气词
        self.tone_words = {
            "感叹": ["啊", "呀", "哇", "哦", "噢", "哟"],
            "疑问": ["吗", "呢", "吧", "啊"],
            "无奈": ["唉", "哎", "哼"],
        }

    def infer_mood(self, text: str) -> str:
        """推断氛围情绪"""
        for mood, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return mood

        return "一般"

    def detect_emotion(self, text: str) -> str:
        """
        检测对话情绪
        """
        # 检查标点符号
        if text.endswith('！'):
            return "激动"
        elif text.endswith('？'):
            return "疑问"

        # 检查情绪关键词
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return emotion

        for emotion, words in self.tone_words.items():
            for word in words:
                if word in text:
                    return emotion

        return "平静"


emotion_extractor = EmotionExtractor()