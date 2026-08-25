# CurveDemos

![](CurveDemos.png)

Builds a 4-point Bezier/B-spline curve (`ncca.ngl.BezierCurve`) and draws the
sampled curve (white), its control polygon "hull" (red), and control points
(green dots).

## Controls

- Left-drag : orbit.
- Right-drag : pan.
- Wheel : zoom.
- `space` : reset.

## References

- [A Primer on Bézier Curves (Pomax)](https://pomax.github.io/bezierinfo/) — an interactive walk through Bézier maths: Bernstein form, de Casteljau, splitting, arc length and more.
- [Bézier curves (Paul Bourke)](https://paulbourke.net/geometry/bezier/) — concise notes and code for the curve evaluation used here.
- G. Farin, _Curves and Surfaces for CAGD: A Practical Guide_, 5th ed., Morgan Kaufmann 2002 — the standard text on Bézier/B-spline theory.
