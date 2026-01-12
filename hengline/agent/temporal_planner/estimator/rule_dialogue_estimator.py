"""
@FileName: dialogue_estimator.py
@Description: 对话时长估算器
@Author: HengLine
@Time: 2026/1/12 15:40
"""


class DialogueDurationEstimator:
    """对话时长估算模型"""

    # 语速基准（词/秒）
    SPEECH_RATE_BASELINE = {
        "normal": 2.5,  # 正常对话
        "slow": 2.0,  # 慢速/情绪化
        "fast": 3.5,  # 快速/紧张
        "very_fast": 4.0,  # 极快
    }

    # 对话类型修正因子
    DIALOGUE_TYPE_FACTORS = {
        "exposition": 1.2,  # 交代信息
        "emotional": 0.8,  # 情绪对话（含停顿）
        "argument": 0.9,  # 争论（稍快）
        "casual": 1.1,  # 闲聊
    }

    def estimate(self, dialogue: Dialogue, context: EstimationContext) -> DurationEstimation:
        """估算单句对话"""

        word_count = len(dialogue.content.split())

        # 1. 检测对话特征
        features = self._analyze_dialogue_features(dialogue.content)

        # 2. 确定语速
        speech_rate = self._determine_speech_rate(dialogue, features, context)

        # 3. 计算基础时长
        base_duration = word_count / speech_rate

        # 4. 添加情绪停顿
        pause_duration = self._estimate_pause_duration(features, context)

        total_duration = base_duration + pause_duration

        return DurationEstimation(
            element_id=dialogue.id,
            element_type=ElementType.DIALOGUE,
            base_duration=total_duration,
            min_duration=total_duration * 0.8,
            max_duration=total_duration * 1.3,
            confidence=self._calculate_confidence(dialogue, features),
            emotional_weight=features.emotional_intensity
        )