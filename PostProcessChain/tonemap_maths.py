"""Pure-maths tonemapping curves for the PostProcessChain demo.

Numpy-only (no GL/Qt) so the curves used by ``TonemapFragment.glsl`` can be
unit tested headless. The constants defined here are the single source of
truth: the GLSL fragment shader hardcodes the *same* numbers in comments
next to each use, and the tests in ``tests/test_tonemap_maths.py`` guard
this numpy copy so a drift between the two would first show up as a test
failure, not a silent visual difference between the demo and its maths.

Both curves take *linear HDR* colour (after exposure has already been
applied) and return colour clamped to ``[0, 1]``, ready for the final
gamma-2.2 encode. They operate per-channel; pass a numpy array of any shape
whose last dimension is colour, or a plain scalar/1-D triple.
"""

import numpy as np

# Narkowicz 2015 ACES fitted curve coefficients (the widely used "ACES
# approximation" -- not the full ACES RRT+ODT, but a close, cheap fit).
# https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
ACES_A = 2.51
ACES_B = 0.03
ACES_C = 2.43
ACES_D = 0.59
ACES_E = 0.14


def reinhard(colour: np.ndarray) -> np.ndarray:
    """Simple Reinhard tonemap: ``c / (1 + c)``.

    Monotonic, maps 0 -> 0, and asymptotically approaches 1 as c -> inf
    (never quite reaching or exceeding it for finite, non-negative c).
    """
    c = np.asarray(colour, dtype=np.float64)
    return c / (1.0 + c)


def aces_fitted(colour: np.ndarray) -> np.ndarray:
    """Narkowicz fitted ACES filmic curve, clamped to [0, 1].

    ``(c * (a*c + b)) / (c * (c*c_ + d) + e)`` -- matches the coefficients
    hardcoded in ``shaders/TonemapFragment.glsl``.
    """
    c = np.asarray(colour, dtype=np.float64)
    numerator = c * (ACES_A * c + ACES_B)
    denominator = c * (ACES_C * c + ACES_D) + ACES_E
    return np.clip(numerator / denominator, 0.0, 1.0)


def apply_gamma(colour: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Encode linear colour to display gamma: ``c ** (1 / gamma)``."""
    c = np.asarray(colour, dtype=np.float64)
    return np.power(np.clip(c, 0.0, None), 1.0 / gamma)
