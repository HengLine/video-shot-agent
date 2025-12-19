# -*- coding: utf-8 -*-
"""
@FileName: state_machine.py
@Description: 状态机模块，用于管理角色状态的转换
@Author: HengLine
@Time: 2025/11/3
"""
from typing import Dict, List, Any, Tuple

class StateMachine:
    """角色状态机类，管理角色状态之间的转换"""
    
    def __init__(self, config):
        """初始化状态机"""
        # 初始化状态转换规则和权重为None，将在load_config中加载
        self.transition_rules = {}
        self.state_weights = {}
        
        # 加载配置
        self.load_config(config)
    
    def load_config(self, config):
        """
        从配置文件加载状态机的配置项
        """
        try:
            if 'state_machine' in config:
                sm_config = config['state_machine']
                # 加载状态权重和转换规则
                self.state_weights = sm_config.get('state_weights', {})
                self.transition_rules = sm_config.get('transition_rules', {})
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 如果加载失败，使用默认配置
            self._set_default_config()
    
    def _set_default_config(self):
        """
        设置默认配置，确保即使配置文件加载失败也能正常工作
        """
        # 默认状态权重
        self.state_weights = {
            "pose": 0.3,
            "position": 0.25,
            "emotion": 0.3,
            "gaze_direction": 0.1,
            "holding": 0.05
        }
        
        # 默认转换规则
        self.transition_rules = {
            "pose": {
                "站立": ["站立", "坐"],
                "坐": ["坐", "站立"],
                "unknown": ["站立", "坐"]
            },
            "position": {
                "画面中央": ["画面中央", "左侧", "右侧"],
                "左侧": ["左侧", "画面中央"],
                "右侧": ["右侧", "画面中央"],
                "unknown": ["画面中央"]
            },
            "gaze_direction": {
                "前方": ["前方"],
                "unknown": ["前方"]
            },
            "holding": {
                "无": ["无"],
                "unknown": ["无"]
            }
        }
    
    def is_transition_valid(self, from_state: Dict[str, Any], to_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """检查状态转换是否有效
        
        Args:
            from_state: 起始状态
            to_state: 目标状态
            
        Returns:
            (是否有效, 无效转换的字段列表)
        """
        invalid_fields = []
        
        # 检查每个字段的转换是否有效
        for field, rules in self.transition_rules.items():
            from_value = from_state.get(field, "unknown")
            to_value = to_state.get(field, "unknown")
            
            # 如果有特定规则，检查是否有效
            if from_value in rules:
                if to_value not in rules[from_value]:
                    invalid_fields.append(field)
            # 对于off-screen的特殊处理
            elif from_value == "off-screen" and to_value != "off-screen":
                # 从画面外到画面内需要特殊处理
                invalid_fields.append(field)
        
        # 检查情绪转换（情绪转换在config中有单独规则）
        if "emotion" in from_state and "emotion" in to_state:
            # 情绪转换由连续性守护智能体中的方法处理
            pass
        
        return len(invalid_fields) == 0, invalid_fields
    
    def calculate_transition_cost(self, from_state: Dict[str, Any], to_state: Dict[str, Any]) -> float:
        """计算状态转换的成本（用于确定最合理的转换路径）
        
        Args:
            from_state: 起始状态
            to_state: 目标状态
            
        Returns:
            转换成本，值越小表示转换越自然
        """
        total_cost = 0.0
        
        # 计算每个字段的转换成本
        for field, weight in self.state_weights.items():
            if field in from_state and field in to_state:
                if from_state[field] != to_state[field]:
                    # 基础成本
                    cost = 1.0
                    
                    # 根据字段类型调整成本
                    if field == "position":
                        # 位置相邻成本较低
                        if self._are_positions_adjacent(from_state[field], to_state[field]):
                            cost *= 0.5
                    elif field == "pose":
                        # 姿势转换难度
                        if self._is_pose_transition_easy(from_state[field], to_state[field]):
                            cost *= 0.3
                        else:
                            cost *= 1.5
                    
                    total_cost += weight * cost
        
        return total_cost
    
    def generate_valid_transitions(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成从当前状态可以转换到的所有有效状态
        
        Args:
            current_state: 当前状态
            
        Returns:
            有效状态列表
        """
        valid_states = []
        
        # 为每个字段生成可能的值
        possible_values = {}
        for field, rules in self.transition_rules.items():
            current_value = current_state.get(field, "unknown")
            possible_values[field] = rules.get(current_value, [current_value])
        
        # 生成所有可能的组合（这里只生成部分组合以避免状态爆炸）
        # 为简化，只生成最多改变一个字段的状态
        for field in self.transition_rules:
            if field in possible_values:
                for value in possible_values[field]:
                    if current_state.get(field) != value:
                        new_state = current_state.copy()
                        new_state[field] = value
                        valid_states.append(new_state)
        
        return valid_states
    
    def find_optimal_transition(self, from_state: Dict[str, Any], target_state: Dict[str, Any]) -> Dict[str, Any]:
        """找到从起始状态到目标状态的最优中间状态
        
        Args:
            from_state: 起始状态
            target_state: 目标状态
            
        Returns:
            最优的中间状态
        """
        # 首先检查直接转换是否有效
        is_valid, _ = self.is_transition_valid(from_state, target_state)
        if is_valid:
            return target_state
        
        # 生成有效转换
        valid_transitions = self.generate_valid_transitions(from_state)
        
        if not valid_transitions:
            return from_state
        
        # 找到最接近目标状态的有效转换
        min_cost = float('inf')
        best_transition = from_state
        
        for transition in valid_transitions:
            # 计算与目标状态的差距
            gap_cost = self.calculate_transition_cost(transition, target_state)
            # 计算从当前状态的转换成本
            transition_cost = self.calculate_transition_cost(from_state, transition)
            # 总成本
            total_cost = gap_cost + transition_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_transition = transition
        
        return best_transition
    
    def _are_positions_adjacent(self, pos1: str, pos2: str) -> bool:
        """判断两个位置是否相邻
        
        Args:
            pos1: 位置1
            pos2: 位置2
            
        Returns:
            是否相邻
        """
        adjacent_map = {
            "画面中央": ["左侧", "右侧", "前景", "背景"],
            "左侧": ["画面中央", "前景", "背景"],
            "右侧": ["画面中央", "前景", "背景"],
            "前景": ["画面中央", "左侧", "右侧"],
            "背景": ["画面中央", "左侧", "右侧"]
        }
        
        if pos1 in adjacent_map:
            return pos2 in adjacent_map[pos1]
        return False
    
    def _is_pose_transition_easy(self, pose1: str, pose2: str) -> bool:
        """判断姿势转换是否简单
        
        Args:
            pose1: 姿势1
            pose2: 姿势2
            
        Returns:
            是否简单
        """
        # 定义简单转换
        easy_transitions = [
            ("站立", "坐"),
            ("坐", "站立"),
            ("站立", "跪"),
            ("跪", "站立")
        ]
        
        return (pose1, pose2) in easy_transitions or (pose2, pose1) in easy_transitions
    
    def validate_sequence(self, state_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证状态序列的连续性
        
        Args:
            state_sequence: 状态序列
            
        Returns:
            验证结果，包含每个转换的有效性
        """
        results = []
        
        for i in range(len(state_sequence) - 1):
            from_state = state_sequence[i]
            to_state = state_sequence[i + 1]
            
            is_valid, invalid_fields = self.is_transition_valid(from_state, to_state)
            cost = self.calculate_transition_cost(from_state, to_state)
            
            results.append({
                "from_state": from_state,
                "to_state": to_state,
                "is_valid": is_valid,
                "invalid_fields": invalid_fields,
                "transition_cost": cost,
                "suggestion": None
            })
            
            # 如果无效，生成建议
            if not is_valid:
                suggestion = self.find_optimal_transition(from_state, to_state)
                results[-1]["suggestion"] = suggestion
        
        return results
    
    def suggest_state_correction(self, current_state: Dict[str, Any], desired_state: Dict[str, Any]) -> Dict[str, Any]:
        """建议状态修正，使转换更合理
        
        Args:
            current_state: 当前状态
            desired_state: 期望状态
            
        Returns:
            修正后的状态
        """
        # 检查直接转换是否有效
        is_valid, invalid_fields = self.is_transition_valid(current_state, desired_state)
        
        if is_valid:
            return desired_state
        
        # 修正无效字段
        corrected_state = desired_state.copy()
        
        for field in invalid_fields:
            if field in self.transition_rules and field in current_state:
                current_value = current_state[field]
                valid_values = self.transition_rules[field].get(current_value, [current_value])
                
                # 选择最接近期望的值
                if valid_values and desired_state[field] not in valid_values:
                    # 对于姿势和位置，优先选择合理的值
                    if field == "pose":
                        corrected_state[field] = "站立" if "站立" in valid_values else valid_values[0]
                    elif field == "position":
                        corrected_state[field] = "画面中央" if "画面中央" in valid_values else valid_values[0]
                    else:
                        corrected_state[field] = valid_values[0]
        
        return corrected_state