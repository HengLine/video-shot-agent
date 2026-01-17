"""
@FileName: shot_generation_manager.py
@Description: 分镜头生成管理器
@Author: HengLine
@Time: 2026/1/17 22:17
"""
from datetime import datetime
from typing import Dict, List

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.shot.shot_generator_factory import ShotGeneratorFactory
from hengline.agent.temporal_planner.shot.shot_model import ShotGenerationResult
from hengline.agent.temporal_planner.temporal_planner_model import EstimationSource
from hengline.logger import info, error


class ShotGenerationManager:
    """分镜头生成管理器"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.generators = {}
        self.results_history = []

    def generate_shots_for_scene(self, script: UnifiedScript,
                                 generator_type: EstimationSource = EstimationSource.HYBRID) -> ShotGenerationResult:
        """为场景生成分镜头"""
        # 创建或获取生成器
        if generator_type not in self.generators:
            self.generators[generator_type] = ShotGeneratorFactory.create_generator(
                generator_type, self.config
            )

        generator = self.generators[generator_type]

        # 生成分镜头
        result = generator.generate_shots(script)

        # 记录历史
        self.results_history.append({
            "scene_id": script.scene_id,
            "timestamp": datetime.now().isoformat(),
            "generator_type": generator_type,
            "result": result.to_dict()
        })

        return result

    def batch_generate(self, scripts: List[UnifiedScript],
                       generator_type: EstimationSource = EstimationSource.HYBRID) -> Dict[str, ShotGenerationResult]:
        """批量生成分镜头"""
        results = {}

        for script in scripts:
            try:
                result = self.generate_shots_for_scene(script, generator_type)
                results[script.scene_id] = result
                info(f"场景 {script.scene_id} 分镜生成完成: {result.shot_count}镜头")
            except Exception as e:
                error(f"场景 {script.scene_id} 分镜生成失败: {e}")
                results[script.scene_id] = None

        return results

    def compare_generators(self, script: UnifiedScript) -> Dict[EstimationSource, ShotGenerationResult]:
        """比较不同生成器的结果"""
        results = {}

        for gen_type in [EstimationSource.LOCAL_RULE, EstimationSource.AI_LLM]:
            try:
                result = self.generate_shots_for_scene(script, gen_type)
                results[gen_type] = result
            except Exception as e:
                error(f"生成器 {gen_type} 失败: {e}")

        return results
