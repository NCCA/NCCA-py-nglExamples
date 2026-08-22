"""Syntax highlighting for the generated PyNGL Python view."""

from __future__ import annotations

import re

from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat


class PythonHighlighter(QSyntaxHighlighter):
    """Colour the small Python subset emitted by the maths graph.

    Parameters
    ----------
    parent : object
        the QTextDocument belonging to the read-only code editor
    """

    def __init__(self, parent: object) -> None:
        super().__init__(parent)

        def make_format(colour: str, bold: bool = False) -> QTextCharFormat:
            text_format = QTextCharFormat()
            text_format.setForeground(QColor(colour))
            if bold:
                text_format.setFontWeight(QFont.Weight.Bold)
            return text_format

        self.rules: list[tuple[re.Pattern[str], QTextCharFormat]] = []
        keyword_format = make_format("#ffb86c", bold=True)
        for keyword in ("def", "for", "from", "if", "import", "in", "return"):
            self.rules.append((re.compile(rf"\b{keyword}\b"), keyword_format))
        maths_format = make_format("#8be9fd", bold=True)
        for name in (
            "Mat2",
            "Mat3",
            "Mat4",
            "Quaternion",
            "Transform",
            "Vec2",
            "Vec3",
            "Vec4",
            "frustum",
            "look_at",
            "ortho",
            "perspective",
        ):
            self.rules.append((re.compile(rf"\b{name}\b"), maths_format))
        self.rules.extend(
            (
                (
                    re.compile(
                        r"(?:'[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\")"
                    ),
                    make_format("#f1fa8c"),
                ),
                (re.compile(r"\b\d+(?:\.\d+)?\b"), make_format("#bd93f9")),
                (re.compile(r"#.*"), make_format("#6272a4")),
            )
        )

    def highlightBlock(self, text: str) -> None:
        """Apply the configured single-line rules to one text block."""
        for pattern, text_format in self.rules:
            for match in pattern.finditer(text):
                start, end = match.span()
                self.setFormat(start, end - start, text_format)
