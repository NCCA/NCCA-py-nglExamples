import re
from typing import List, Tuple

from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat


# Simple GLSL syntax highlighter
class GLSLHighlighter(QSyntaxHighlighter):
    """A simple syntax highlighter for GLSL shader code."""

    def __init__(self, parent):
        """
        Initialize the GLSLHighlighter.

        Parameters
        ----------
            parent
                The parent QObject, typically a QTextDocument.
        """
        super().__init__(parent)
        # common GLSL keywords / types / builtins (extend as needed)
        # fmt: off
        keywords = [ "if","else", "for", "while", "do", "break", "continue", "return",  "struct"]
        types = ["void","float","int","bool","vec2","vec3","vec4","mat3","mat4","sampler2D","samplerCube","in","out","inout","uniform","attribute","varying","const"]
        builtins = [ "sin", "cos", "tan", "pow", "exp", "normalize", "dot","cross", "mix","texture", "texture2D", "textureCube", "gl_Position", "gl_FragColor", "gl_FragCoord", "gl_FragDepth"]
        # fmt: on
        def make_format(color: str, bold: bool = False) -> QTextCharFormat:
            """Create a QTextCharFormat with a given color and optional bold font."""
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            if bold:
                fmt.setFontWeight(QFont.Bold)
            return fmt

        self.rules = []
        # keywords
        kw_fmt = make_format("#FFB86C", bold=True)
        for w in keywords:
            self.rules.append((re.compile(r"\b" + re.escape(w) + r"\b"), kw_fmt))
        # types
        type_fmt = make_format("#8BE9FD", bold=True)
        for t in types:
            self.rules.append((re.compile(r"\b" + re.escape(t) + r"\b"), type_fmt))
        # builtins
        builtin_fmt = make_format("#F1FA8C")
        for b in builtins:
            self.rules.append((re.compile(r"\b" + re.escape(b) + r"\b"), builtin_fmt))
        # numbers
        num_fmt = make_format("#BD93F9")
        self.rules.append((re.compile(r"\b[0-9]+(?:\.[0-9]+)?\b"), num_fmt))
        # preprocessor (#include, #version, etc)
        pre_fmt = make_format("#FF79C6", bold=True)
        self.rules.append((re.compile(r"^\s*#.*", re.MULTILINE), pre_fmt))
        # single-line comments
        comment_fmt = make_format("#6272A4")
        self.rules.append((re.compile(r"//.*"), comment_fmt))
        # block comment start/end handled via QSyntaxHighlighter block handling
        self.comment_start = re.compile(r"/\*")
        self.comment_end = re.compile(r"\*/")
        self.comment_format = comment_fmt

    def highlightBlock(self, text):
        """
        Apply syntax highlighting to a block of text.

        This method is called by Qt whenever a block of text needs to be
        re-highlighted.

        Parameters
        ----------
            text
                The block of text to highlight.
        """
        for pattern, fmt in self.rules:
            for m in pattern.finditer(text):
                start, end = m.span()
                self.setFormat(start, end - start, fmt)

        # block comments (/* ... */)
        start_idx = 0
        if self.previousBlockState() != 1:
            m = self.comment_start.search(text)
            start_idx = m.start() if m else -1
        else:
            start_idx = 0

        while start_idx >= 0:
            # find end
            m_end = self.comment_end.search(text, start_idx)
            if m_end:
                end_idx = m_end.end()
                length = end_idx - (start_idx if start_idx >= 0 else 0)
                self.setFormat(
                    start_idx if start_idx >= 0 else 0, length, self.comment_format
                )
                start_idx = self.comment_start.search(text, end_idx)
                start_idx = start_idx.start() if start_idx else -1
                self.setCurrentBlockState(0)
            else:
                # not closed in this block
                self.setFormat(
                    start_idx if start_idx >= 0 else 0,
                    len(text) - (start_idx if start_idx >= 0 else 0),
                    self.comment_format,
                )
                self.setCurrentBlockState(1)
                break
