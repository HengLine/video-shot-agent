"""
@FileName: action_uration_example.py
@Description: 
@Author: HengLine
@Time: 2025/10/24 14:21
"""
from hengline.tools.action_duration_tool import ActionDurationEstimator

if __name__ == '__main__':
    # 初始化估算器（单例推荐）
    estimator = ActionDurationEstimator("../hengline/config/action_duration_config.yaml")
    estimator.clear_cache()

    # 基础用法
    print(estimator.estimate("缓缓走向窗边"))  # → 3.6

    # 对话 + 情绪（无角色影响）
    print(estimator.estimate("轻声说：'你好。'", emotion="轻声"))  # → 1.5

    # 身体动作 + 角色
    print(estimator.estimate("王爷爷起身", character_type="elder"))  # → 1.95

    # 对话 + 角色（应无影响）
    print(estimator.estimate("王爷爷说：'你好。'", character_type="elder")) # → 1.5

    # 缓存生效（相同输入极快）
    # for _ in range(1000):
    #     estimator.estimate("缓缓走向窗边")  # 第二次起直接返回缓存

