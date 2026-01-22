"""
@FileName: shot_generation_manager.py
@Description: 分镜头生成管理器
@Author: HengLine
@Time: 2026/1/17 22:17
"""
from datetime import datetime
from typing import Dict

from hengline.agent.script_parser2.script_parser_models import UnifiedScript, Scene
from hengline.agent.temporal_planner.shot.shot_generator_factory import ShotGeneratorFactory
from hengline.agent.temporal_planner.shot.shot_model import SceneShotResult, ScriptShotResult
from hengline.agent.temporal_planner.temporal_planner_model import EstimationSource
from hengline.logger import info, debug


class ShotGenerationManager:
    """分镜头生成管理器"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.generators = {}
        self.history = []

    def generate_for_script(self, script: UnifiedScript,
                            generator_type: EstimationSource = EstimationSource.HYBRID) -> ScriptShotResult:
        """为整个剧本生成分镜头"""
        debug(f"开始为剧本生成分镜头，使用生成器: {generator_type}")

        # 创建生成器
        if generator_type not in self.generators:
            self.generators[generator_type] = ShotGeneratorFactory.create_generator(
                generator_type, self.config
            )

        generator = self.generators[generator_type]

        # 生成分镜头
        start_time = datetime.now()
        result = generator.generate_for_script(script)
        end_time = datetime.now()

        # 记录历史
        self.history.append({
            "timestamp": start_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "generator_type": generator_type,
            "script_scenes": len(script.scenes),
            "total_shots": result.total_shots,
            "total_duration": result.total_duration
        })

        info(f"剧本分镜生成完成: {result.total_shots}个镜头，总时长: {result.total_duration}s")
        return result

    def generate_for_scene(self, scene: Scene, script: UnifiedScript,
                           generator_type: EstimationSource = EstimationSource.HYBRID) -> SceneShotResult:
        """为单个场景生成分镜头"""
        # 创建生成器
        if generator_type not in self.generators:
            self.generators[generator_type] = ShotGeneratorFactory.create_generator(
                generator_type, self.config
            )

        generator = self.generators[generator_type]

        # 生成分镜头
        return generator.generate_for_scene(scene, script)
