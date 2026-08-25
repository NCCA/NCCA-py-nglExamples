"""Procedural Klein bottle mesh, parametric equations from Paul Bourke,
ported from NGL9Demos/KleinBottle."""

import math

import numpy as np
from ncca.ngl import Vec3, calc_normal


def _eval(u: float, v: float) -> Vec3:
    r = 4.0 * (1.0 - math.cos(u) / 2.0)
    if u < math.pi:
        x = 6.0 * math.cos(u) * (1.0 + math.sin(u)) + r * math.cos(u) * math.cos(v)
        y = 16.0 * math.sin(u) + r * math.sin(u) * math.cos(v)
    else:
        x = 6.0 * math.cos(u) * (1.0 + math.sin(u)) + r * math.cos(v + math.pi)
        y = 16.0 * math.sin(u)
    z = r * math.sin(v)
    return Vec3(x, y, z)


def build_klein_bottle(resolution: int = 40) -> np.ndarray:
    du = (2.0 * math.pi) / resolution
    dv = (2.0 * math.pi) / resolution
    eps = 0.01
    verts: list[float] = []

    for i in range(resolution):
        u = i * du
        for j in range(resolution):
            v = j * dv

            def quad_vertex(uu: float, vv: float) -> tuple[Vec3, Vec3]:
                p = _eval(uu, vv)
                p_u = _eval(uu + eps, vv)
                p_v = _eval(uu, vv + eps)
                n = calc_normal(p, p_u, p_v)
                n = Vec3(-n.x, -n.y, -n.z)
                return p, n

            corners = [
                (u, v),
                (u + du, v),
                (u + du, v + dv),
                (u, v),
                (u + du, v + dv),
                (u, v + dv),
            ]
            for uu, vv in corners:
                p, n = quad_vertex(uu, vv)
                verts.extend([p.x * 0.05, p.y * 0.05, p.z * 0.05, n.x, n.y, n.z])

    return np.array(verts, dtype=np.float32)
