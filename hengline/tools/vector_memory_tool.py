# -*- coding: utf-8 -*-
"""
@FileName: vector_memory.py
@Description: 向量记忆模块，用于角色状态的向量化表示和相似度计算
@Author: HengLine
@Time: 2025/11/3
"""
from typing import Dict, List, Any, Optional
import numpy as np
import hashlib


class VectorMemory:
    """向量记忆类，用于存储和检索角色状态的向量表示"""
    
    def __init__(self, config):
        """初始化向量记忆"""
        # 状态向量存储
        self.state_vectors: Dict[str, Dict[str, np.ndarray]] = {}
        # 向量维度
        self.vector_dim = 5  # 基础维度：姿势、位置、情绪、视线方向、手持物品
        
        # 加载配置
        self.load_config(config)
        
    def load_config(self, config):
        """
        从配置文件加载向量记忆的配置项
        """
        try:
            if 'vector_memory' in config:
                vm_config = config['vector_memory']
                # 加载各种映射配置
                self.pose_mapping = vm_config.get('pose_mapping', {})
                self.position_mapping = vm_config.get('position_mapping', {})
                self.emotion_mapping = vm_config.get('emotion_mapping', {})
                self.gaze_mapping = vm_config.get('gaze_mapping', {})
                self.holding_mapping = vm_config.get('holding_mapping', {})
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 如果加载失败，使用默认配置
            self._set_default_config()
    
    def _set_default_config(self):
        """
        设置默认配置，确保即使配置文件加载失败也能正常工作
        """
        self.pose_mapping = {
            "站立": [1.0, 0.0, 0.0, 0.0],
            "坐": [0.0, 1.0, 0.0, 0.0],
            "躺": [0.0, 0.0, 1.0, 0.0],
            "跪": [0.0, 0.0, 0.0, 1.0],
            "unknown": [0.0, 0.0, 0.0, 0.0],
            "off-screen": [-1.0, -1.0, -1.0, -1.0]
        }
        self.position_mapping = {
            "画面中央": [1.0, 0.0, 0.0, 0.0, 0.0],
            "左侧": [0.0, 1.0, 0.0, 0.0, 0.0],
            "右侧": [0.0, 0.0, 1.0, 0.0, 0.0],
            "前景": [0.0, 0.0, 0.0, 1.0, 0.0],
            "背景": [0.0, 0.0, 0.0, 0.0, 1.0],
            "unknown": [0.0, 0.0, 0.0, 0.0, 0.0],
            "off-screen": [-1.0, -1.0, -1.0, -1.0, -1.0]
        }
        self.emotion_mapping = {
            "平静": [0.5, 0.5],
            "unknown": [0.5, 0.0]
        }
        self.gaze_mapping = {
            "前方": [0.0, 0.0, 1.0],
            "unknown": [0.0, 0.0, 0.0]
        }
        self.holding_mapping = {
            "无": [1.0, 0.0, 0.0, 0.0, 0.0],
            "unknown": [0.0, 0.0, 0.0, 0.0, 0.0]
        }
    
    def _state_to_vector(self, state: Dict[str, Any]) -> np.ndarray:
        """将角色状态转换为向量表示
        
        Args:
            state: 角色状态字典
            
        Returns:
            状态向量
        """
        # 获取各维度特征
        pose_vector = np.array(self.pose_mapping.get(state.get("pose", "unknown"), self.pose_mapping["unknown"]))
        position_vector = np.array(self.position_mapping.get(state.get("position", "unknown"), self.position_mapping["unknown"]))
        emotion_vector = np.array(self.emotion_mapping.get(state.get("emotion", "unknown"), self.emotion_mapping["unknown"]))
        gaze_vector = np.array(self.gaze_mapping.get(state.get("gaze_direction", "unknown"), self.gaze_mapping["unknown"]))
        holding_vector = np.array(self.holding_mapping.get(state.get("holding", "unknown"), self.holding_mapping["unknown"]))
        
        # 拼接向量
        return np.concatenate([pose_vector, position_vector, emotion_vector, gaze_vector, holding_vector])
    
    def _vector_to_state(self, vector: np.ndarray, character_name: str) -> Dict[str, Any]:
        """将向量转换回角色状态（近似转换）
        
        Args:
            vector: 状态向量
            character_name: 角色名称
            
        Returns:
            近似的角色状态字典
        """
        state = {"character_name": character_name}
        
        # 反向映射姿势
        pose_start, pose_end = 0, 4
        pose_subvector = vector[pose_start:pose_end]
        state["pose"] = self._find_closest_key(pose_subvector, self.pose_mapping)
        
        # 反向映射位置
        pos_start, pos_end = 4, 9
        pos_subvector = vector[pos_start:pos_end]
        state["position"] = self._find_closest_key(pos_subvector, self.position_mapping)
        
        # 反向映射情绪
        emotion_start, emotion_end = 9, 11
        emotion_subvector = vector[emotion_start:emotion_end]
        state["emotion"] = self._find_closest_key(emotion_subvector, self.emotion_mapping)
        
        # 反向映射视线方向
        gaze_start, gaze_end = 11, 14
        gaze_subvector = vector[gaze_start:gaze_end]
        state["gaze_direction"] = self._find_closest_key(gaze_subvector, self.gaze_mapping)
        
        # 反向映射手持物品
        holding_start, holding_end = 14, 19
        holding_subvector = vector[holding_start:holding_end]
        state["holding"] = self._find_closest_key(holding_subvector, self.holding_mapping)
        
        return state
    
    def _find_closest_key(self, vector: np.ndarray, mapping: Dict[str, List[float]]) -> str:
        """找到与向量最相似的键
        
        Args:
            vector: 输入向量
            mapping: 键到向量的映射
            
        Returns:
            最相似的键
        """
        min_distance = float('inf')
        closest_key = "unknown"
        
        for key, mapped_vector in mapping.items():
            distance = np.linalg.norm(vector - np.array(mapped_vector))
            if distance < min_distance:
                min_distance = distance
                closest_key = key
        
        return closest_key
    
    def store_state(self, character_name: str, state: Dict[str, Any]):
        """存储角色状态的向量表示
        
        Args:
            character_name: 角色名称
            state: 角色状态
        """
        if character_name not in self.state_vectors:
            self.state_vectors[character_name] = {}
        
        # 生成状态ID
        state_id = self._generate_state_id(state)
        
        # 转换并存储状态向量
        self.state_vectors[character_name][state_id] = self._state_to_vector(state)
    
    def retrieve_similar_states(self, character_name: str, state: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """检索与给定状态相似的历史状态
        
        Args:
            character_name: 角色名称
            state: 查询状态
            top_k: 返回的最大状态数量
            
        Returns:
            相似状态列表，按相似度排序
        """
        if character_name not in self.state_vectors or not self.state_vectors[character_name]:
            return []
        
        query_vector = self._state_to_vector(state)
        similarities = []
        
        # 计算与所有历史状态的相似度
        for state_id, stored_vector in self.state_vectors[character_name].items():
            # 使用余弦相似度
            similarity = np.dot(query_vector, stored_vector) / \
                        (np.linalg.norm(query_vector) * np.linalg.norm(stored_vector) + 1e-10)
            similarities.append((state_id, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回最相似的top_k个状态
        similar_states = []
        for state_id, similarity in similarities[:top_k]:
            # 重建状态
            vector = self.state_vectors[character_name][state_id]
            recovered_state = self._vector_to_state(vector, character_name)
            recovered_state["similarity"] = similarity
            similar_states.append(recovered_state)
        
        return similar_states
    
    def _generate_state_id(self, state: Dict[str, Any]) -> str:
        """生成状态的唯一ID
        
        Args:
            state: 角色状态
            
        Returns:
            状态ID
        """
        state_str = "_".join([str(state.get(k, "unknown")) for k in 
                              ["pose", "position", "emotion", "gaze_direction", "holding"]])
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def get_state_history(self, character_name: str) -> List[Dict[str, Any]]:
        """获取角色的状态历史
        
        Args:
            character_name: 角色名称
            
        Returns:
            状态历史列表
        """
        if character_name not in self.state_vectors:
            return []
        
        history = []
        for state_id, vector in self.state_vectors[character_name].items():
            state = self._vector_to_state(vector, character_name)
            history.append(state)
        
        return history
    
    def clear_history(self, character_name: Optional[str] = None):
        """清除状态历史
        
        Args:
            character_name: 如果指定，只清除特定角色的历史；否则清除所有历史
        """
        if character_name:
            if character_name in self.state_vectors:
                self.state_vectors[character_name] = {}
        else:
            self.state_vectors = {}