"""Trigonometric and cubic easing/interpolation, reimplemented since ncca.ngl
only provides linear `lerp`. Standard smoothstep-family formulas."""

import math

from ncca.ngl import Vec3


def trig_interp(a: Vec3, b: Vec3, t: float) -> Vec3:
    eased = 0.5 * (1.0 - math.cos(t * math.pi))
    return a + (b - a) * eased


def cubic_interp(a: Vec3, b: Vec3, t: float) -> Vec3:
    eased = t * t * (3.0 - 2.0 * t)
    return a + (b - a) * eased
