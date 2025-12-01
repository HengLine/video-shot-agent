# -*- coding: utf-8 -*-
"""
@FileName: script_parser_agent.py
@Description: LLM 剧本解析功能，通过LLM 将中文剧本转换为结构化动作序列
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from typing import Dict, Any, List

from hengline.logger import debug, error
from utils.log_utils import print_log_exception


class ScriptParser:
    """优化版剧本解析智能体"""

    def parse_script(self, script_text: str, result: Dict[str, Any] = None) -> dict[str, Any] | None:
        """
        优化版剧本解析函数
        将整段中文剧本转换为结构化动作序列
        
        Args:
            script_text: 原始剧本文本

        Returns:
            结构化的剧本动作序列
        """
        debug(f"开始解析剧本: {script_text[:100]}...")

        try:
            # 初始化结果结构
            return self.parse_script_to_json(script_text, result)

        except Exception as e:
            print_log_exception()
            error(f"剧本解析失败: {str(e)}")
            return None

    def parse_script_to_json(self, script_text: str, result: Dict[str, Any] = None) -> Dict[str, Any]:
        """ 需要实现将剧本解析结果转换为JSON字符串的功能
        """
        pass

    def _infer_atmosphere(self, scene: Dict[str, Any]) -> str:
        """
        从剧本中动态推断场景氛围描述，提供简洁明了的氛围描述

        Args:
            scene: 场景信息

        Returns:
            简洁的场景氛围描述
        """
        # 默认氛围
        default_atmosphere = "室内环境"

        # 从场景信息推断氛围
        location = scene.get("location", "")
        time_of_day = scene.get("time_of_day", "")

        # 核心氛围元素
        atmosphere_elements = []

        # 时间氛围 - 选择最关键的时间描述
        if "深夜" in time_of_day or "夜晚" in time_of_day:
            atmosphere_elements.append("深夜")
        elif "黄昏" in time_of_day or "傍晚" in time_of_day:
            atmosphere_elements.append("傍晚")
        elif "早晨" in time_of_day:
            atmosphere_elements.append("清晨")
        elif "白天" in time_of_day:
            atmosphere_elements.append("白天")

        # 地点氛围 - 选择最关键的地点描述
        if "公寓" in location:
            atmosphere_elements.append("公寓内")
        elif "咖啡馆" in location:
            atmosphere_elements.append("咖啡馆")
        elif "办公室" in location:
            atmosphere_elements.append("办公室")
        elif "客厅" in location:
            atmosphere_elements.append("客厅")
        elif "卧室" in location:
            atmosphere_elements.append("卧室")

        # 收集所有动作文本用于综合分析
        actions = scene.get("actions", [])
        all_action_text = " ".join([action.get("action", "") for action in actions])

        # 环境细节 - 只添加最突出的环境元素
        if "电视" in all_action_text and "静音" in all_action_text:
            atmosphere_elements.append("电视静音")

        # 天气细节 - 只添加最明显的天气元素
        if "窗外" in all_action_text:
            if "大雨" in all_action_text or "下雨" in all_action_text:
                atmosphere_elements.append("窗外下雨")
            elif "阳光" in all_action_text:
                atmosphere_elements.append("窗外阳光")

        # 灯光细节 - 只添加最关键的灯光描述
        if "台灯" in all_action_text:
            atmosphere_elements.append("台灯照明")
        elif "灯光" in all_action_text:
            if "柔和" in all_action_text:
                atmosphere_elements.append("灯光柔和")

        # 情绪氛围 - 提取最主要的情绪基调
        emotions_count = {}
        for action in actions:
            emotion = action.get("emotion", "")
            if emotion:
                # 提取主要情绪关键词
                main_emotions = ["紧张", "焦虑", "悲伤", "平静", "欢乐", "震惊", "警觉", "犹豫"]
                for main_emotion in main_emotions:
                    if main_emotion in emotion:
                        emotions_count[main_emotion] = emotions_count.get(main_emotion, 0) + 1

        # 找出出现次数最多的情绪
        if emotions_count:
            dominant_emotion = max(emotions_count.items(), key=lambda x: x[1])[0]
            atmosphere_elements.append(f"{dominant_emotion}氛围")

        # 特殊场景模式识别
        if "裹着毯子" in all_action_text and "沙发" in all_action_text:
            atmosphere_elements.append("温暖舒适")

        # 如果没有收集到任何氛围元素，返回默认值
        if not atmosphere_elements:
            return default_atmosphere

        # 生成简洁的氛围描述
        # 确保不超过4个元素，避免描述过长
        concise_elements = atmosphere_elements[:4]
        return "，".join(concise_elements)

    def _reorder_actions_for_logic(self, actions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        根据逻辑顺序重新排列动作，确保动作序列符合自然发展

        Args:
            actions: 原始动作列表

        Returns:
            重新排序后的动作列表
        """
        if not actions:
            return []

        # 定义动作类型及其优先级
        action_priorities = {
            # 初始状态动作
            "状态类": {"关键词": ["坐在沙发上", "裹着毯子", "调整坐姿"], "优先级": 1},
            # 感知类动作（听到、看到等）
            "感知类": {"关键词": ["听到", "看到", "手机震动"], "优先级": 2},
            # 思考类动作（犹豫、考虑等）
            "思考类": {"关键词": ["犹豫", "思考", "回想"], "优先级": 3},
            # 肢体动作（伸手、起身等）
            "肢体类": {"关键词": ["伸手", "拿起", "接起", "放下", "挂断"], "优先级": 4},
            # 对话类动作
            "对话类": {"关键词": ["说：", "低声说", "轻声说", "对方说"], "优先级": 5},
            # 反应类动作（震惊、发抖等）
            "反应类": {"关键词": ["震惊", "发抖", "颤抖", "身体僵硬"], "优先级": 6},
            # 结束类动作
            "结束类": {"关键词": ["挂断电话", "关机", "深呼吸"], "优先级": 7}
        }

        # 为每个动作分配类型和优先级
        actions_with_priority = []
        for idx, action in enumerate(actions):
            action_text = action.get("action", "") + " " + action.get("dialogue", "")
            priority = 100  # 默认低优先级
            action_type = "其他"

            # 检查每个动作类型的关键词
            for atype, config in action_priorities.items():
                if any(keyword in action_text for keyword in config["关键词"]):
                    priority = config["优先级"]
                    action_type = atype
                    break

            # 特殊处理对话行，确保它们在肢体动作后
            if "dialogue" in action:
                priority = action_priorities["对话类"]["优先级"]
                action_type = "对话类"

            actions_with_priority.append((idx, action, priority, action_type))

        # 按优先级排序，相同优先级的保持原顺序
        actions_with_priority.sort(key=lambda x: (x[2], x[0]))

        # 提取排序后的动作
        reordered_actions = [action for _, action, _, _ in actions_with_priority]

        # 确保第一个动作是关于角色状态的（如坐在沙发上）
        if reordered_actions and not any(keyword in reordered_actions[0].get("action", "") for keyword in ["坐在沙发上", "裹着毯子", "调整坐姿"]):
            # 查找状态类动作
            for i, action in enumerate(reordered_actions):
                if any(keyword in action.get("action", "") for keyword in ["坐在沙发上", "裹着毯子", "调整坐姿"]):
                    # 移动到第一个位置
                    state_action = reordered_actions.pop(i)
                    reordered_actions.insert(0, state_action)
                    break

        # 确保接电话的动作在听到手机响之后
        phone_ring_idx = -1
        answer_phone_idx = -1
        for i, action in enumerate(reordered_actions):
            action_text = action.get("action", "")
            if "手机震动" in action_text or "手机响" in action_text:
                phone_ring_idx = i
            elif "接起电话" in action_text or "拿起电话" in action_text:
                answer_phone_idx = i

        if phone_ring_idx > answer_phone_idx and phone_ring_idx != -1 and answer_phone_idx != -1:
            # 交换位置
            reordered_actions[phone_ring_idx], reordered_actions[answer_phone_idx] = \
                reordered_actions[answer_phone_idx], reordered_actions[phone_ring_idx]

        return reordered_actions