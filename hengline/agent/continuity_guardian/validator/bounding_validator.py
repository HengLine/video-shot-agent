"""
@FileName: bounding_validator.py
@Description: 边界检测器
@Author: HengLine
@Time: 2026/1/4 21:51
"""
import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class BoundingVolumeType(Enum):
    """边界体类型"""
    SPHERE = "sphere"
    BOX = "box"
    CAPSULE = "capsule"
    MESH = "mesh"
    CONVEX_HULL = "convex_hull"


class TrajectoryType(Enum):
    """轨迹类型"""
    LINEAR = "linear"
    PARABOLIC = "parabolic"
    CIRCULAR = "circular"
    BEZIER = "bezier"
    RANDOM_WALK = "random_walk"
    OSCILLATING = "oscillating"


@dataclass
class BoundingSphere:
    """边界球"""
    center: Tuple[float, float, float]
    radius: float

    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        """检查点是否在球内"""
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        dz = point[2] - self.center[2]
        distance_sq = dx * dx + dy * dy + dz * dz
        return distance_sq <= self.radius * self.radius

    def intersects_sphere(self, other: 'BoundingSphere') -> bool:
        """检查与另一个球是否相交"""
        dx = other.center[0] - self.center[0]
        dy = other.center[1] - self.center[1]
        dz = other.center[2] - self.center[2]
        distance_sq = dx * dx + dy * dy + dz * dz
        radius_sum = self.radius + other.radius
        return distance_sq <= radius_sum * radius_sum

    def get_volume(self) -> float:
        """获取体积"""
        return (4.0 / 3.0) * math.pi * self.radius ** 3


@dataclass
class BoundingBox:
    """轴对齐边界盒"""
    min_point: Tuple[float, float, float]
    max_point: Tuple[float, float, float]

    def __post_init__(self):
        """初始化后处理"""
        # 确保min_point和max_point有效
        self.min_point = (
            min(self.min_point[0], self.max_point[0]),
            min(self.min_point[1], self.max_point[1]),
            min(self.min_point[2], self.max_point[2])
        )
        self.max_point = (
            max(self.min_point[0], self.max_point[0]),
            max(self.min_point[1], self.max_point[1]),
            max(self.min_point[2], self.max_point[2])
        )

    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        """检查点是否在盒内"""
        return (self.min_point[0] <= point[0] <= self.max_point[0] and
                self.min_point[1] <= point[1] <= self.max_point[1] and
                self.min_point[2] <= point[2] <= self.max_point[2])

    def intersects_box(self, other: 'BoundingBox') -> bool:
        """检查与另一个盒是否相交"""
        # 分离轴定理的简化版本
        return not (self.max_point[0] < other.min_point[0] or
                    self.min_point[0] > other.max_point[0] or
                    self.max_point[1] < other.min_point[1] or
                    self.min_point[1] > other.max_point[1] or
                    self.max_point[2] < other.min_point[2] or
                    self.min_point[2] > other.max_point[2])

    def get_center(self) -> Tuple[float, float, float]:
        """获取中心点"""
        return (
            (self.min_point[0] + self.max_point[0]) / 2,
            (self.min_point[1] + self.max_point[1]) / 2,
            (self.min_point[2] + self.max_point[2]) / 2
        )

    def get_size(self) -> Tuple[float, float, float]:
        """获取尺寸"""
        return (
            self.max_point[0] - self.min_point[0],
            self.max_point[1] - self.min_point[1],
            self.max_point[2] - self.min_point[2]
        )

    def get_volume(self) -> float:
        """获取体积"""
        size = self.get_size()
        return size[0] * size[1] * size[2]

    def expand(self, amount: float):
        """扩展边界盒"""
        self.min_point = (
            self.min_point[0] - amount,
            self.min_point[1] - amount,
            self.min_point[2] - amount
        )
        self.max_point = (
            self.max_point[0] + amount,
            self.max_point[1] + amount,
            self.max_point[2] + amount
        )