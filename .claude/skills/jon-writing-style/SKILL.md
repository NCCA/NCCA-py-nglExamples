---
name: jon-writing-style
description: Write READMEs, code comments, docstrings, and technical prose in Jon Macey's voice instead of the default AI style. Use this skill whenever writing or editing a README.md, adding comments or docstrings to code, drafting a blog post, or producing any documentation Jon will publish under his own name — even if he doesn't explicitly mention style or voice. Also use it when reviewing existing docs for tone.
---

# Jon's Writing Style

Jon Macey teaches programming, graphics, and pipeline development at the NCCA (Bournemouth University). His writing is that of an experienced practitioner talking to students and colleagues: plain, first-person, unfussy, and allergic to marketing language. The default AI documentation voice — exhaustive feature lists, bold-everything, emoji ticks, "comprehensive" and "powerful" — is exactly what to avoid.

Real excerpts from his repos and blog are in [references/examples.md](references/examples.md). Read them before writing anything longer than a couple of paragraphs; matching the rhythm matters more than following rules.

## Voice

Write in first person and British English (colour, behaviour, whilst, organise). Jon says "I" when explaining his choices ("Whilst I will mainly be using WebGPU for my teaching I thought it would be good to also have an OpenGL example") and switches to "we" when walking the reader through steps ("To enable WSL2 we need to activate the WSL feature").

Keep sentences plain and direct. It's fine — good, even — for prose to feel slightly informal and quick rather than polished. Jon uses parenthetical asides for context and dry humour ("the advantages of getting AI to do work for you!"), the occasional exclamation mark when something surprises him, and footnotes for tangential detail in blog posts. He is candid about problems and about his own process, including when things fail or when content is AI-generated.

Link liberally and inline: official docs, GitHub repos, PyPI packages, his own previous posts and course pages. Prefer a link over re-explaining something that's documented elsewhere.

Mention the teaching context where it's the honest explanation of why something exists: "This is all the code used in my design patterns lectures" is a complete and typical repo description.

### Avoid

- Marketing adjectives: comprehensive, powerful, seamless, robust, intelligent, smart, blazing-fast
- Emoji, checkmark lists of platforms, badge walls (badges only where they already exist)
- Boilerplate sections nobody asked for: Contributing, License, Changelog, Support, Security — unless the project actually needs them
- Bold-heavy nested bullet hierarchies; prefer short prose or a simple list
- American spellings
- Over-hedging and filler ("It's worth noting that...", "Simply...")

## READMEs

Jon's READMEs are short. A single sentence is acceptable for a lecture-demo repo (`Maya API demos used in Lectures`). Structure for a typical project:

1. `# Title` — the project name, nothing clever
2. One or two sentences on what it is and why it exists (often mentioning lectures, labs, or a course)
3. How to run it — assume `uv` for Python ("It is expected you will use uv to run all the python applications"), `cargo`/CMake as appropriate elsewhere; show actual commands in fenced code blocks
4. Links: source on GitHub (usually the [NCCA org](https://github.com/NCCA)), PyPI if published, full docs on his site (nccastaff.bournemouth.ac.uk/jmacey) or GitHub Pages
5. Screenshots inline where they help — `![](DemoApp.png)` with no alt-text ceremony

For collections of demos, use his catalogue pattern: a `## Contents` anchor list, then per-category sections each containing a three-column table of `| Preview | Demo | Description |` where the preview is a linked thumbnail image and descriptions are terse noun phrases ("Full-screen triangle technique", "Updating VAO data"). Each demo folder gets its own small README.

Developer notes (building, testing, publishing) go at the end under `## Developer notes` as bare commands with one-line lead-ins, not tutorials.

## Code comments

### C++

Follow the NCCA coding standard. Doxygen triple-slash comments with `@brief`, `@param`, `@note`, separated by long dash banner lines:

```cpp
//----------------------------------------------------------------------------------------------------------------------
/// @brief the doIt command is called everytime the command is executed in the maya shell
/// @param _args the command arguments passed when command is run
/// @note from the maya docs
/// The doIt method should collect whatever information is required to do the task
//----------------------------------------------------------------------------------------------------------------------
MStatus doIt( const MArgList& _args );
```

File headers use `\file`, `\brief`, `\author`, `\version`, `\date`. Function parameters are prefixed with underscore (`_args`, `_width`). Comments explain purpose and behaviour in lower-case, conversational phrasing ("tell of the class is undoable (in this case true)") — they read like Jon explaining the line to a student, and often link to further reading.

### Python

Type hints on signatures. Numpydoc-style docstrings — class docstrings with an `Attributes` section, method docstrings with `Parameters`, both using the indented `name : type` / description layout:

```python
def clear(self, r: int, g: int, b: int, a: int = 255) -> None:
    """
    Sets all pixels to the specified color.

    Parameters
    ----------
        r : int
            red component of the color
        g : int
            green component of the color
    """
```

Trivial methods get no docstring at all. Inline comments are sparse and only where the code is genuinely non-obvious; when present they explain *why* and may link to docs.

## Blog posts / longer prose

Hugo front matter (`type`, `title`, `summary`, `linktitle`, `date`, `tags`). `# Introduction` opening section that links back to related previous posts. Narrative, chronological, experiment-log style: show the actual commands run, the actual errors in fenced blocks, then the fix or the observation. Honest verdicts, including negative ones. Screenshots pasted inline. Close informally, often pointing to what the next post will cover.
