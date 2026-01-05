"""
@FileName: material_validator.py
@Description: 物理材料
@Author: HengLine
@Time: 2026/1/4 22:06
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
from collections import defaultdict


class PhysicalState(Enum):
    """物理状态枚举"""
    STABLE = "stable"  # 稳定
    MOVING = "moving"  # 移动中
    FALLING = "falling"  # 下落中
    COLLIDING = "colliding"  # 碰撞中
    DEFORMING = "deforming"  # 形变中
    BROKEN = "broken"  # 破碎


@dataclass
class PhysicsMaterial:
    """物理材料"""
    name: str
    density: float  # 密度 kg/m³
    friction: float  # 摩擦系数
    restitution: float  # 恢复系数 (弹性)
    hardness: float  # 硬度 (0-1)
    toughness: float  # 韧性 (0-1)

    def __post_init__(self):
        """验证材料属性"""
        self.density = max(0.001, self.density)
        self.friction = max(0.0, min(1.0, self.friction))
        self.restitution = max(0.0, min(1.0, self.restitution))
        self.hardness = max(0.0, min(1.0, self.hardness))
        self.toughness = max(0.0, min(1.0, self.toughness))

    def get_mass(self, volume: float) -> float:
        """根据体积计算质量"""
        return self.density * volume

    def combine_with(self, other: 'PhysicsMaterial') -> 'PhysicsMaterial':
        """与另一个材料组合"""
        return PhysicsMaterial(
            name=f"{self.name}_{other.name}_composite",
            density=(self.density + other.density) / 2,
            friction=(self.friction + other.friction) / 2,
            restitution=(self.restitution + other.restitution) / 2,
            hardness=(self.hardness + other.hardness) / 2,
            toughness=(self.toughness + other.toughness) / 2
        )


class MaterialDatabase:
    """材料数据库"""

    def __init__(self):
        self.materials: Dict[str, PhysicsMaterial] = {}
        self._initialize_default_materials()
        self.material_combinations: Dict[str, PhysicsMaterial] = {}
        self.usage_statistics: Dict[str, int] = defaultdict(int)

    def _initialize_default_materials(self):
        """初始化默认材料"""
        default_materials = [
            PhysicsMaterial("rubber", 1100, 0.8, 0.8, 0.3, 0.9),
            PhysicsMaterial("steel", 7850, 0.5, 0.5, 0.9, 0.7),
            PhysicsMaterial("aluminum", 2700, 0.4, 0.4, 0.6, 0.6),
            PhysicsMaterial("wood", 600, 0.4, 0.4, 0.4, 0.5),
            PhysicsMaterial("glass", 2500, 0.1, 0.3, 0.8, 0.2),
            PhysicsMaterial("plastic", 1400, 0.3, 0.5, 0.4, 0.6),
            PhysicsMaterial("concrete", 2400, 0.6, 0.2, 0.7, 0.3),
            PhysicsMaterial("ice", 917, 0.05, 0.1, 0.2, 0.1),
            PhysicsMaterial("water", 1000, 0.0, 0.0, 0.0, 0.0),
            PhysicsMaterial("human_skin", 1100, 0.6, 0.3, 0.2, 0.8),
            PhysicsMaterial("cloth", 300, 0.4, 0.2, 0.1, 0.9),
            PhysicsMaterial("ceramic", 2300, 0.3, 0.2, 0.8, 0.3),
            PhysicsMaterial("foam", 50, 0.7, 0.7, 0.1, 0.9),
            PhysicsMaterial("leather", 900, 0.5, 0.4, 0.4, 0.8),
            PhysicsMaterial("paper", 800, 0.3, 0.1, 0.1, 0.6)
        ]

        for material in default_materials:
            self.materials[material.name] = material

    def get_material_properties(self, material_name: str) -> Optional[PhysicsMaterial]:
        """获取材料属性"""
        # 记录使用统计
        self.usage_statistics[material_name] += 1

        if material_name in self.materials:
            return self.materials[material_name]

        # 尝试查找近似材料
        for name, material in self.materials.items():
            if material_name.lower() in name.lower() or name.lower() in material_name.lower():
                return material

        # 返回默认材料
        return self.materials.get("plastic")

    def add_material(self, material: PhysicsMaterial) -> bool:
        """添加新材料"""
        if material.name in self.materials:
            return False

        self.materials[material.name] = material
        return True

    def remove_material(self, material_name: str) -> bool:
        """移除材料"""
        if material_name in self.materials:
            del self.materials[material_name]
            return True
        return False

    def update_material(self, material_name: str,
                        updates: Dict[str, Any]) -> Optional[PhysicsMaterial]:
        """更新材料属性"""
        if material_name not in self.materials:
            return None

        material = self.materials[material_name]

        # 更新属性
        for key, value in updates.items():
            if hasattr(material, key):
                setattr(material, key, value)

        return material

    def get_material_combination(self, material1_name: str,
                                 material2_name: str) -> PhysicsMaterial:
        """获取两种材料的组合属性"""
        combo_key = f"{material1_name}_{material2_name}"

        if combo_key in self.material_combinations:
            return self.material_combinations[combo_key]

        # 获取两种材料
        mat1 = self.get_material_properties(material1_name)
        mat2 = self.get_material_properties(material2_name)

        if mat1 is None or mat2 is None:
            return self.get_material_properties("plastic")

        # 创建组合材料
        combo_material = mat1.combine_with(mat2)
        self.material_combinations[combo_key] = combo_material

        return combo_material

    def search_materials(self, criteria: Dict[str, Any]) -> List[PhysicsMaterial]:
        """根据条件搜索材料"""
        results = []

        for material in self.materials.values():
            match = True

            # 检查密度范围
            if "density_min" in criteria and material.density < criteria["density_min"]:
                match = False
            if "density_max" in criteria and material.density > criteria["density_max"]:
                match = False

            # 检查摩擦系数范围
            if "friction_min" in criteria and material.friction < criteria["friction_min"]:
                match = False
            if "friction_max" in criteria and material.friction > criteria["friction_max"]:
                match = False

            # 检查恢复系数范围
            if "restitution_min" in criteria and material.restitution < criteria["restitution_min"]:
                match = False
            if "restitution_max" in criteria and material.restitution > criteria["restitution_max"]:
                match = False

            # 检查硬度范围
            if "hardness_min" in criteria and material.hardness < criteria["hardness_min"]:
                match = False
            if "hardness_max" in criteria and material.hardness > criteria["hardness_max"]:
                match = False

            # 检查韧性范围
            if "toughness_min" in criteria and material.toughness < criteria["toughness_min"]:
                match = False
            if "toughness_max" in criteria and material.toughness > criteria["toughness_max"]:
                match = False

            # 检查名称关键词
            if "name_keyword" in criteria:
                keyword = criteria["name_keyword"].lower()
                if keyword not in material.name.lower():
                    match = False

            if match:
                results.append(material)

        return results

    def get_material_similarity(self, material1_name: str,
                                material2_name: str) -> float:
        """计算两种材料的相似度"""
        mat1 = self.get_material_properties(material1_name)
        mat2 = self.get_material_properties(material2_name)

        if mat1 is None or mat2 is None:
            return 0.0

        # 计算属性差异
        density_diff = abs(mat1.density - mat2.density) / max(mat1.density, mat2.density)
        friction_diff = abs(mat1.friction - mat2.friction)
        restitution_diff = abs(mat1.restitution - mat2.restitution)
        hardness_diff = abs(mat1.hardness - mat2.hardness)
        toughness_diff = abs(mat1.toughness - mat2.toughness)

        # 平均差异
        avg_diff = (density_diff + friction_diff + restitution_diff +
                    hardness_diff + toughness_diff) / 5

        # 转换为相似度
        similarity = 1.0 - avg_diff

        return max(0.0, min(1.0, similarity))

    def get_recommended_material(self, requirements: Dict[str, Any]) -> Optional[PhysicsMaterial]:
        """根据需求推荐材料"""
        best_match = None
        best_score = -1.0

        for material in self.materials.values():
            score = self._calculate_material_match_score(material, requirements)

            if score > best_score:
                best_score = score
                best_match = material

        return best_match if best_score > 0 else None

    def _calculate_material_match_score(self, material: PhysicsMaterial,
                                        requirements: Dict[str, Any]) -> float:
        """计算材料匹配分数"""
        score = 0.0
        weight_sum = 0.0

        # 密度要求
        if "target_density" in requirements:
            target = requirements["target_density"]
            tolerance = requirements.get("density_tolerance", 0.1)
            diff = abs(material.density - target) / target
            match = max(0, 1 - diff / tolerance)
            weight = requirements.get("density_weight", 0.2)
            score += match * weight
            weight_sum += weight

        # 摩擦要求
        if "target_friction" in requirements:
            target = requirements["target_friction"]
            tolerance = requirements.get("friction_tolerance", 0.1)
            diff = abs(material.friction - target)
            match = max(0, 1 - diff / tolerance)
            weight = requirements.get("friction_weight", 0.2)
            score += match * weight
            weight_sum += weight

        # 弹性要求
        if "target_restitution" in requirements:
            target = requirements["target_restitution"]
            tolerance = requirements.get("restitution_tolerance", 0.1)
            diff = abs(material.restitution - target)
            match = max(0, 1 - diff / tolerance)
            weight = requirements.get("restitution_weight", 0.2)
            score += match * weight
            weight_sum += weight

        # 硬度要求
        if "target_hardness" in requirements:
            target = requirements["target_hardness"]
            tolerance = requirements.get("hardness_tolerance", 0.1)
            diff = abs(material.hardness - target)
            match = max(0, 1 - diff / tolerance)
            weight = requirements.get("hardness_weight", 0.2)
            score += match * weight
            weight_sum += weight

        # 韧性要求
        if "target_toughness" in requirements:
            target = requirements["target_toughness"]
            tolerance = requirements.get("toughness_tolerance", 0.1)
            diff = abs(material.toughness - target)
            match = max(0, 1 - diff / tolerance)
            weight = requirements.get("toughness_weight", 0.2)
            score += match * weight
            weight_sum += weight

        # 名称关键词要求
        if "name_keyword" in requirements:
            keyword = requirements["name_keyword"].lower()
            if keyword in material.name.lower():
                weight = requirements.get("name_weight", 0.1)
                score += 1.0 * weight
                weight_sum += weight

        if weight_sum > 0:
            return score / weight_sum

        return 0.0

    def get_material_statistics(self) -> Dict[str, Any]:
        """获取材料统计信息"""
        stats = {
            "total_materials": len(self.materials),
            "material_categories": defaultdict(int),
            "density_range": {"min": float('inf'), "max": float('-inf'), "avg": 0},
            "friction_range": {"min": float('inf'), "max": float('-inf'), "avg": 0},
            "restitution_range": {"min": float('inf'), "max": float('-inf'), "avg": 0},
            "most_used_materials": []
        }

        density_sum = 0
        friction_sum = 0
        restitution_sum = 0

        for material in self.materials.values():
            # 更新密度范围
            stats["density_range"]["min"] = min(stats["density_range"]["min"], material.density)
            stats["density_range"]["max"] = max(stats["density_range"]["max"], material.density)
            density_sum += material.density

            # 更新摩擦范围
            stats["friction_range"]["min"] = min(stats["friction_range"]["min"], material.friction)
            stats["friction_range"]["max"] = max(stats["friction_range"]["max"], material.friction)
            friction_sum += material.friction

            # 更新恢复系数范围
            stats["restitution_range"]["min"] = min(stats["restitution_range"]["min"], material.restitution)
            stats["restitution_range"]["max"] = max(stats["restitution_range"]["max"], material.restitution)
            restitution_sum += material.restitution

            # 统计材料类别（根据名称前缀）
            name_parts = material.name.split('_')
            if name_parts:
                category = name_parts[0]
                stats["material_categories"][category] += 1

        # 计算平均值
        if self.materials:
            stats["density_range"]["avg"] = density_sum / len(self.materials)
            stats["friction_range"]["avg"] = friction_sum / len(self.materials)
            stats["restitution_range"]["avg"] = restitution_sum / len(self.materials)

        # 获取最常用的材料
        usage_items = list(self.usage_statistics.items())
        usage_items.sort(key=lambda x: x[1], reverse=True)
        stats["most_used_materials"] = usage_items[:10]

        # 转换defaultdict为普通dict
        stats["material_categories"] = dict(stats["material_categories"])

        return stats

    def export_materials(self, filepath: str) -> bool:
        """导出材料数据到文件"""
        try:
            import json

            export_data = []
            for material in self.materials.values():
                material_dict = {
                    "name": material.name,
                    "density": material.density,
                    "friction": material.friction,
                    "restitution": material.restitution,
                    "hardness": material.hardness,
                    "toughness": material.toughness
                }
                export_data.append(material_dict)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"导出材料数据失败: {e}")
            return False

    def import_materials(self, filepath: str) -> int:
        """从文件导入材料数据"""
        try:
            import json

            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            imported_count = 0
            for material_dict in import_data:
                material = PhysicsMaterial(
                    name=material_dict["name"],
                    density=material_dict["density"],
                    friction=material_dict["friction"],
                    restitution=material_dict["restitution"],
                    hardness=material_dict["hardness"],
                    toughness=material_dict["toughness"]
                )

                if self.add_material(material):
                    imported_count += 1

            return imported_count
        except Exception as e:
            print(f"导入材料数据失败: {e}")
            return 0

    def get_material_card(self, material_name: str) -> str:
        """获取材料卡片（文本描述）"""
        material = self.get_material_properties(material_name)
        if material is None:
            return f"材料 '{material_name}' 不存在"

        card = f"材料卡片: {material.name}\n"
        card += "=" * 40 + "\n"
        card += f"密度: {material.density:.1f} kg/m³\n"
        card += f"摩擦系数: {material.friction:.2f}\n"
        card += f"恢复系数: {material.restitution:.2f}\n"
        card += f"硬度: {material.hardness:.2f}\n"
        card += f"韧性: {material.toughness:.2f}\n"
        card += "-" * 40 + "\n"

        # 特性描述
        characteristics = []
        if material.density < 500:
            characteristics.append("非常轻")
        elif material.density < 1500:
            characteristics.append("中等重量")
        else:
            characteristics.append("很重")

        if material.friction > 0.7:
            characteristics.append("高摩擦力")
        elif material.friction < 0.3:
            characteristics.append("低摩擦力")

        if material.restitution > 0.7:
            characteristics.append("高弹性")
        elif material.restitution < 0.3:
            characteristics.append("低弹性")

        if material.hardness > 0.7:
            characteristics.append("非常硬")
        elif material.hardness < 0.3:
            characteristics.append("柔软")

        if material.toughness > 0.7:
            characteristics.append("非常坚韧")
        elif material.toughness < 0.3:
            characteristics.append("易碎")

        card += f"特性: {', '.join(characteristics)}\n"

        # 推荐用途
        recommendations = []
        if material.name == "rubber":
            recommendations.extend(["轮胎", "密封件", "减震器"])
        elif material.name == "steel":
            recommendations.extend(["结构件", "工具", "机械部件"])
        elif material.name == "wood":
            recommendations.extend(["家具", "建筑", "装饰"])
        elif material.name == "glass":
            recommendations.extend(["窗户", "容器", "光学元件"])
        elif material.name == "foam":
            recommendations.extend(["包装", "隔音", "缓冲"])

        if recommendations:
            card += f"推荐用途: {', '.join(recommendations)}\n"

        return card