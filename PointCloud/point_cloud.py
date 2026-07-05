"""Loads a plain XYZ point cloud and computes bounding box / Ritter bounding sphere,
ported from NGL9Demos/PointCloud."""

from __future__ import annotations

from ncca.ngl import Vec3


class PointCloud:
    def __init__(self) -> None:
        self.points: list[Vec3] = []
        self.bbox_center = Vec3()
        self.bbox_max_dim = 1.0
        self.sphere_center = Vec3()
        self.sphere_radius = 1.0

    @classmethod
    def from_file(cls, path: str) -> "PointCloud":
        cloud = cls()
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                cloud.points.append(
                    Vec3(float(parts[0]), float(parts[1]), float(parts[2]))
                )
        cloud._calculate_bounding_box()
        cloud._calculate_bounding_sphere()
        cloud._unitize()
        return cloud

    def _calculate_bounding_box(self) -> None:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        zs = [p.z for p in self.points]
        min_p = Vec3(min(xs), min(ys), min(zs))
        max_p = Vec3(max(xs), max(ys), max(zs))
        self.bbox_center = Vec3(
            (min_p.x + max_p.x) / 2, (min_p.y + max_p.y) / 2, (min_p.z + max_p.z) / 2
        )
        self.bbox_max_dim = max(max_p.x - min_p.x, max_p.y - min_p.y, max_p.z - min_p.z)
        self._min_p = min_p
        self._max_p = max_p

    def _calculate_bounding_sphere(self) -> None:
        # Ritter's approximate bounding sphere
        p0 = self.points[0]
        farthest_from_p0 = max(self.points, key=lambda p: (p - p0).length_squared())
        farthest_from_that = max(
            self.points, key=lambda p: (p - farthest_from_p0).length_squared()
        )
        center = Vec3(
            (farthest_from_p0.x + farthest_from_that.x) / 2,
            (farthest_from_p0.y + farthest_from_that.y) / 2,
            (farthest_from_p0.z + farthest_from_that.z) / 2,
        )
        radius = float((farthest_from_that - farthest_from_p0).length() / 2)

        for p in self.points:
            d = float((p - center).length())
            if d > radius:
                new_radius = (radius + d) / 2
                k = (new_radius - radius) / d
                center = center + (p - center) * k
                radius = new_radius

        self.sphere_center = center
        self.sphere_radius = radius

    def _unitize(self) -> None:
        scale = float(1.0 / self.bbox_max_dim) if self.bbox_max_dim > 0 else 1.0
        self.points = [(p - self.bbox_center) * scale for p in self.points]
        self.sphere_center = (self.sphere_center - self.bbox_center) * scale
        self.sphere_radius *= scale
        self.bbox_center = Vec3()
