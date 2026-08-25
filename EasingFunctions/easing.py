"""The full set of Penner easing functions (as catalogued at https://easings.net).

Each function maps a normalised time t in [0, 1] to an eased value, with
f(0) == 0 and f(1) == 1. Back and elastic variants deliberately overshoot
outside [0, 1] mid-curve. All functions are pure scalar maths so they can be
unit tested headlessly; interpolation between points is done by the caller as
``a + (b - a) * ease(t)``.
"""

import inspect
import math

# --- Sine ---------------------------------------------------------------


def ease_in_sine(t: float) -> float:
    return 1.0 - math.cos((t * math.pi) / 2.0)


def ease_out_sine(t: float) -> float:
    return math.sin((t * math.pi) / 2.0)


def ease_in_out_sine(t: float) -> float:
    return -(math.cos(math.pi * t) - 1.0) / 2.0


# --- Quad ---------------------------------------------------------------


def ease_in_quad(t: float) -> float:
    return t * t


def ease_out_quad(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


def ease_in_out_quad(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - math.pow(-2.0 * t + 2.0, 2) / 2.0


# --- Cubic --------------------------------------------------------------


def ease_in_cubic(t: float) -> float:
    return t * t * t


def ease_out_cubic(t: float) -> float:
    return 1.0 - math.pow(1.0 - t, 3)


def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - math.pow(-2.0 * t + 2.0, 3) / 2.0


# --- Quart --------------------------------------------------------------


def ease_in_quart(t: float) -> float:
    return t * t * t * t


def ease_out_quart(t: float) -> float:
    return 1.0 - math.pow(1.0 - t, 4)


def ease_in_out_quart(t: float) -> float:
    if t < 0.5:
        return 8.0 * t * t * t * t
    return 1.0 - math.pow(-2.0 * t + 2.0, 4) / 2.0


# --- Quint --------------------------------------------------------------


def ease_in_quint(t: float) -> float:
    return t * t * t * t * t


def ease_out_quint(t: float) -> float:
    return 1.0 - math.pow(1.0 - t, 5)


def ease_in_out_quint(t: float) -> float:
    if t < 0.5:
        return 16.0 * t * t * t * t * t
    return 1.0 - math.pow(-2.0 * t + 2.0, 5) / 2.0


# --- Expo ---------------------------------------------------------------


def ease_in_expo(t: float) -> float:
    if t == 0.0:
        return 0.0
    return math.pow(2.0, 10.0 * t - 10.0)


def ease_out_expo(t: float) -> float:
    if t == 1.0:
        return 1.0
    return 1.0 - math.pow(2.0, -10.0 * t)


def ease_in_out_expo(t: float) -> float:
    if t == 0.0:
        return 0.0
    if t == 1.0:
        return 1.0
    if t < 0.5:
        return math.pow(2.0, 20.0 * t - 10.0) / 2.0
    return (2.0 - math.pow(2.0, -20.0 * t + 10.0)) / 2.0


# --- Circ ---------------------------------------------------------------


def ease_in_circ(t: float) -> float:
    return 1.0 - math.sqrt(1.0 - math.pow(t, 2))


def ease_out_circ(t: float) -> float:
    return math.sqrt(1.0 - math.pow(t - 1.0, 2))


def ease_in_out_circ(t: float) -> float:
    if t < 0.5:
        return (1.0 - math.sqrt(1.0 - math.pow(2.0 * t, 2))) / 2.0
    return (math.sqrt(1.0 - math.pow(-2.0 * t + 2.0, 2)) + 1.0) / 2.0


# --- Back (overshoots) --------------------------------------------------

_C1 = 1.70158
_C2 = _C1 * 1.525
_C3 = _C1 + 1.0


def ease_in_back(t: float) -> float:
    return _C3 * t * t * t - _C1 * t * t


def ease_out_back(t: float) -> float:
    return 1.0 + _C3 * math.pow(t - 1.0, 3) + _C1 * math.pow(t - 1.0, 2)


def ease_in_out_back(t: float) -> float:
    if t < 0.5:
        return (math.pow(2.0 * t, 2) * ((_C2 + 1.0) * 2.0 * t - _C2)) / 2.0
    return (
        math.pow(2.0 * t - 2.0, 2) * ((_C2 + 1.0) * (t * 2.0 - 2.0) + _C2) + 2.0
    ) / 2.0


# --- Elastic (overshoots) -----------------------------------------------

_C4 = (2.0 * math.pi) / 3.0
_C5 = (2.0 * math.pi) / 4.5


def ease_in_elastic(t: float) -> float:
    if t == 0.0:
        return 0.0
    if t == 1.0:
        return 1.0
    return -math.pow(2.0, 10.0 * t - 10.0) * math.sin((t * 10.0 - 10.75) * _C4)


def ease_out_elastic(t: float) -> float:
    if t == 0.0:
        return 0.0
    if t == 1.0:
        return 1.0
    return math.pow(2.0, -10.0 * t) * math.sin((t * 10.0 - 0.75) * _C4) + 1.0


def ease_in_out_elastic(t: float) -> float:
    if t == 0.0:
        return 0.0
    if t == 1.0:
        return 1.0
    if t < 0.5:
        return (
            -(math.pow(2.0, 20.0 * t - 10.0) * math.sin((20.0 * t - 11.125) * _C5))
            / 2.0
        )
    return (
        math.pow(2.0, -20.0 * t + 10.0) * math.sin((20.0 * t - 11.125) * _C5)
    ) / 2.0 + 1.0


# --- Bounce -------------------------------------------------------------


def ease_out_bounce(t: float) -> float:
    n1 = 7.5625
    d1 = 2.75
    if t < 1.0 / d1:
        return n1 * t * t
    if t < 2.0 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


def ease_in_bounce(t: float) -> float:
    return 1.0 - ease_out_bounce(1.0 - t)


def ease_in_out_bounce(t: float) -> float:
    if t < 0.5:
        return (1.0 - ease_out_bounce(1.0 - 2.0 * t)) / 2.0
    return (1.0 + ease_out_bounce(2.0 * t - 1.0)) / 2.0


# Ordered name -> function map used to populate the combo box.
EASING_FUNCTIONS = {
    "In Sine": ease_in_sine,
    "Out Sine": ease_out_sine,
    "InOut Sine": ease_in_out_sine,
    "In Quad": ease_in_quad,
    "Out Quad": ease_out_quad,
    "InOut Quad": ease_in_out_quad,
    "In Cubic": ease_in_cubic,
    "Out Cubic": ease_out_cubic,
    "InOut Cubic": ease_in_out_cubic,
    "In Quart": ease_in_quart,
    "Out Quart": ease_out_quart,
    "InOut Quart": ease_in_out_quart,
    "In Quint": ease_in_quint,
    "Out Quint": ease_out_quint,
    "InOut Quint": ease_in_out_quint,
    "In Expo": ease_in_expo,
    "Out Expo": ease_out_expo,
    "InOut Expo": ease_in_out_expo,
    "In Circ": ease_in_circ,
    "Out Circ": ease_out_circ,
    "InOut Circ": ease_in_out_circ,
    "In Back": ease_in_back,
    "Out Back": ease_out_back,
    "InOut Back": ease_in_out_back,
    "In Elastic": ease_in_elastic,
    "Out Elastic": ease_out_elastic,
    "InOut Elastic": ease_in_out_elastic,
    "In Bounce": ease_in_bounce,
    "Out Bounce": ease_out_bounce,
    "InOut Bounce": ease_in_out_bounce,
}

# Module-level constants each family relies on, so the displayed algorithm
# is self-contained.
_FAMILY_CONSTANTS = {
    "Back": "_C1 = 1.70158\n_C2 = _C1 * 1.525\n_C3 = _C1 + 1.0\n",
    "Elastic": "_C4 = (2.0 * math.pi) / 3.0\n_C5 = (2.0 * math.pi) / 4.5\n",
}


def get_source(name: str) -> str:
    """Return the self-contained source code for a named easing function,
    including any module constants and helper functions it uses, for display
    in the UI's algorithm view."""
    fn = EASING_FUNCTIONS[name]
    parts = []
    for family, constants in _FAMILY_CONSTANTS.items():
        if family in name:
            parts.append(constants)
    source = inspect.getsource(fn)
    if "ease_out_bounce(" in source and fn is not ease_out_bounce:
        parts.append(inspect.getsource(ease_out_bounce))
    parts.append(source)
    return "\n".join(parts)
