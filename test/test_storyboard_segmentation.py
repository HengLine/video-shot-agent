"""
@FileName: storyboard_segmentation_example.py
@Description: 从剧本中提取动词并生成5-6个分镜的示例
@Author: HengLine
@Time: 2025/10/24 14:21
"""
import sys
sys.path.append('../')
from hengline.tools.action_duration_tool import ActionDurationEstimator
from typing import List, Dict, Any
import json


def extract_verbs_and_actions(script: str) -> List[Dict[str, Any]]:
    """
    从剧本中提取动词和相关动作
    
    Args:
        script: 剧本文本
        
    Returns:
        提取的动作列表
    """
    # 这里是一个简化的实现，实际应用中应该使用更复杂的NLP解析
    # 示例中提取了剧本中的主要动词和动作
    extracted_actions = [
        {
            "character": "小明",
            "action": "走进房间",
            "verb": "走",
            "emotion": "平静"
        },
        {
            "character": "小明",
            "action": "环顾四周",
            "verb": "环顾",
            "emotion": "好奇"
        },
        {
            "character": "小明",
            "action": "坐在沙发上",
            "verb": "坐",
            "emotion": "放松"
        },
        {
            "character": "小明",
            "action": "拿起手机",
            "verb": "拿起",
            "emotion": "平静"
        },
        {
            "character": "小明",
            "action": "查看消息",
            "verb": "查看",
            "emotion": "惊讶",
            "dialogue": "哇，太棒了！"
        },
        {
            "character": "小明",
            "action": "站起来",
            "verb": "站",
            "emotion": "激动"
        },
        {
            "character": "小明",
            "action": "来回踱步",
            "verb": "踱步",
            "emotion": "兴奋"
        },
        {
            "character": "小明",
            "action": "拿起外套",
            "verb": "拿起",
            "emotion": "急切"
        },
        {
            "character": "小明",
            "action": "走向门口",
            "verb": "走向",
            "emotion": "期待"
        },
        {
            "character": "小明",
            "action": "打开门",
            "verb": "打开",
            "emotion": "兴奋"
        },
        {
            "character": "小明",
            "action": "离开房间",
            "verb": "离开",
            "emotion": "喜悦"
        }
    ]
    
    print("从剧本中提取的动词:")
    for action in extracted_actions:
        print(f"- {action['verb']}: {action['action']}")
    
    return extracted_actions


def calculate_action_durations(actions: List[Dict[str, Any]], estimator: ActionDurationEstimator) -> List[Dict[str, Any]]:
    """
    计算每个动作的时长
    
    Args:
        actions: 动作列表
        estimator: 动作时长估算器
        
    Returns:
        包含时长信息的动作列表
    """
    for action in actions:
        # 计算基本动作时长
        action_duration = estimator.estimate(action["action"], emotion=action.get("emotion", ""))
        action["duration"] = action_duration
        
        # 如果有对话，额外计算对话时长
        if "dialogue" in action:
            dialogue_text = f"{action['character']}说：'{action['dialogue']}'"
            dialogue_duration = estimator.estimate(dialogue_text, emotion=action.get("emotion", ""))
            action["dialogue_duration"] = dialogue_duration
            # 取动作和对话的最大时长
            action["total_duration"] = max(action_duration, dialogue_duration)
        else:
            action["total_duration"] = action_duration
    
    return actions


def create_segments_with_5_to_6_shots(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    创建5-6个分镜，确保每个分镜时长合理
    
    Args:
        actions: 包含时长信息的动作列表
        
    Returns:
        分镜列表
    """
    # 计算总时长
    total_duration = sum(action["total_duration"] for action in actions)
    print(f"\n剧本总时长: {total_duration:.2f}秒")
    
    # 计算每个分镜的平均目标时长
    target_shots = 6  # 目标分镜数
    avg_target_duration = total_duration / target_shots
    print(f"平均每个分镜目标时长: {avg_target_duration:.2f}秒")
    
    segments = []
    current_segment = {
        "id": 1,
        "actions": [],
        "duration": 0.0
    }
    
    for action in actions:
        action_duration = action["total_duration"]
        
        # 检查是否应该创建新分镜
        # 策略：如果当前分镜加上这个动作超过目标时长的1.2倍，则创建新分镜
        if current_segment["duration"] + action_duration > avg_target_duration * 1.2 and current_segment["actions"]:
            segments.append(current_segment)
            current_segment = {
                "id": len(segments) + 1,
                "actions": [],
                "duration": 0.0
            }
        
        # 添加动作到当前分镜
        current_segment["actions"].append(action)
        current_segment["duration"] += action_duration
    
    # 添加最后一个分镜
    if current_segment["actions"]:
        segments.append(current_segment)
    
    # 调整分镜数量到5-6个
    # 如果分镜数量少于5，尝试合并一些短分镜
    while len(segments) < 5 and len(segments) >= 2:
        # 找到最短的分镜
        shortest_idx = min(range(len(segments)), key=lambda i: segments[i]["duration"])
        # 将其与前一个分镜合并（如果不是第一个）
        if shortest_idx > 0:
            merge_idx = shortest_idx - 1
        else:
            merge_idx = 1
        
        # 合并分镜
        segments[merge_idx]["actions"].extend(segments[shortest_idx]["actions"])
        segments[merge_idx]["duration"] += segments[shortest_idx]["duration"]
        segments.pop(shortest_idx)
    
    # 如果分镜数量多于6，尝试拆分一些长分镜
    while len(segments) > 6:
        # 找到最长的分镜
        longest_idx = max(range(len(segments)), key=lambda i: segments[i]["duration"])
        longest = segments[longest_idx]
        
        if len(longest["actions"]) > 1:
            # 拆分分镜
            half = len(longest["actions"]) // 2
            new_segment = {
                "id": len(segments) + 1,
                "actions": longest["actions"][half:],
                "duration": sum(a["total_duration"] for a in longest["actions"][half:])
            }
            longest["actions"] = longest["actions"][:half]
            longest["duration"] = sum(a["total_duration"] for a in longest["actions"])
            
            # 插入新分镜
            segments.insert(longest_idx + 1, new_segment)
        else:
            # 无法拆分只有一个动作的分镜，退出循环
            break
    
    # 重新分配分镜ID
    for i, segment in enumerate(segments):
        segment["id"] = i + 1
    
    return segments


def generate_storyboard_from_script(script: str) -> List[Dict[str, Any]]:
    """
    从剧本文本生成分镜
    
    Args:
        script: 剧本文本
        
    Returns:
        分镜列表
    """
    # 初始化动作时长估算器
    estimator = ActionDurationEstimator("../hengline/config/zh/action_duration_config.yaml")
    estimator.clear_cache()
    
    # 1. 提取动词和动作
    actions = extract_verbs_and_actions(script)
    
    # 2. 计算每个动作的时长
    actions_with_duration = calculate_action_durations(actions, estimator)
    
    print("\n动作时长计算结果:")
    for action in actions_with_duration:
        dur_info = f"时长: {action['total_duration']:.2f}秒"
        if "dialogue_duration" in action:
            dur_info += f" (动作: {action['duration']:.2f}s, 对话: {action['dialogue_duration']:.2f}s)"
        print(f"- {action['character']}: {action['action']} {dur_info}")
    
    # 3. 创建5-6个分镜
    segments = create_segments_with_5_to_6_shots(actions_with_duration)
    
    return segments


def print_storyboard_segments(segments: List[Dict[str, Any]]):
    """
    打印分镜信息
    
    Args:
        segments: 分镜列表
    """
    print(f"\n生成了 {len(segments)} 个分镜:")
    
    for segment in segments:
        print(f"\n分镜 {segment['id']}: 时长 = {segment['duration']:.2f}秒")
        print(f"包含 {len(segment['actions'])} 个动作:")
        
        for action in segment['actions']:
            action_desc = f"  - {action['character']}({action['emotion']}): {action['action']} [{action['total_duration']:.2f}秒]"
            if "dialogue" in action:
                action_desc += f" '{action['dialogue']}'"
            print(action_desc)


if __name__ == '__main__':
    # 示例剧本
    sample_script = """
    小明走进房间，环顾四周，然后坐在沙发上。他拿起手机查看消息，表情变得惊讶。
    "哇，太棒了！"他兴奋地说道。
    他站起来来回踱步，显得非常激动。随后他拿起外套，走向门口，打开门离开了房间。
    """
    
    print("=== 剧本分镜生成示例 ===")
    print(f"\n原始剧本:\n{sample_script}")
    
    # 生成分镜
    storyboard_segments = generate_storyboard_from_script(sample_script)
    
    # 打印分镜信息
    print_storyboard_segments(storyboard_segments)
    
    # 生成JSON输出
    storyboard_json = {
        "title": "示例剧本分镜",
        "total_shots": len(storyboard_segments),
        "segments": storyboard_segments
    }
    
    print("\n=== 分镜JSON输出 ===")
    print(json.dumps(storyboard_json, ensure_ascii=False, indent=2))
    
    print("\n=== 生成完成 ===")