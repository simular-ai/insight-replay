"""LaTeX-answer verifier built on top of math_verify (HuggingFace MathArena's lib).

Used by the `hmmt` dataset where answers are LaTeX expressions
(fractions, radicals, pi, factorials, multi-solution sets) rather than
plain integers.

Pipeline:
  1. Parse gold as LaTeX (wrapped in $...$ so LatexExtractionConfig finds it).
  2. Parse candidate using default config (auto-detects \\boxed{...}/$...$/plain).
  3. If candidate contains \\pm, expand it into both branches and try each.
  4. Try math_verify.verify with float_rounding=3 (~5e-4 tolerance).
  5. Fall back to high-precision numeric comparison with rel_tol=1e-3 to
     catch model outputs truncated near a rounding boundary.

Sanity-checked over all 60 hmmt problems (self-loop) and 72 hand-built cases
covering each answer type and common model output variations.
"""
from __future__ import annotations

import re
import sympy
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig


_GOLD_CFG = [LatexExtractionConfig(), ExprExtractionConfig()]


def parse_gold(s: str):
    """Parse a raw LaTeX gold answer (no $...$ markers needed)."""
    return parse(f"${s}$", extraction_config=_GOLD_CFG)


def parse_pred(s: str):
    """Parse a model prediction (auto-detects \\boxed{...}, $...$, plain)."""
    return parse(s)


def _expand_pm(s: str) -> list[str]:
    """If candidate contains \\pm, return both + and - substituted variants.
    Otherwise return [s]."""
    if "\\pm" not in s:
        return [s]
    return [s.replace("\\pm", "+"), s.replace("\\pm", "-")]


def _numeric_fallback(g_list, c_list, rel_tol: float = 1e-3) -> bool:
    """Compare any sympy expr in gold vs candidate by high-precision numeric
    evaluation with relative tolerance. Catches truncations like 1.876 vs
    1.876796 that just barely miss the float_rounding=3 cutoff."""
    g_exprs = [x for x in g_list if isinstance(x, sympy.Basic)]
    c_exprs = [x for x in c_list if isinstance(x, sympy.Basic)]
    for g in g_exprs:
        for c in c_exprs:
            try:
                gv = float(sympy.N(g, 30))
                cv = float(sympy.N(c, 30))
            except (TypeError, ValueError):
                continue
            denom = max(abs(gv), abs(cv), 1e-12)
            if abs(gv - cv) / denom < rel_tol:
                return True
    return False


def verify_latex(gold_str: str, cand_str: str) -> bool:
    """Return True iff candidate is mathematically equivalent to gold.

    `gold_str`: raw LaTeX (no $...$).
    `cand_str`: model output — may be a full response, a \\boxed{...} fragment,
                or a plain expression.
    """
    if not gold_str or not cand_str:
        return False
    g = parse_gold(gold_str)
    if not g:
        return False
    parsed_cands = []
    for variant in _expand_pm(cand_str):
        p = parse_pred(variant)
        if p:
            parsed_cands.append(p)
    # If candidate had \pm, also try the two branches as a comma-joined set.
    if "\\pm" in cand_str:
        joined = ", ".join(_expand_pm(cand_str))
        p = parse_pred(f"\\boxed{{{joined}}}")
        if p:
            parsed_cands.append(p)
    if not parsed_cands:
        return False
    for c in parsed_cands:
        if verify(g, c, float_rounding=3):
            return True
        if _numeric_fallback(g, c, rel_tol=1e-3):
            return True
    return False


def _find_last_boxed(text: str) -> str | None:
    """Brace-counting extraction of the LAST \\boxed{...} content."""
    needle = "\\boxed{"
    last_start = text.rfind(needle)
    if last_start < 0:
        return None
    i = last_start + len(needle)
    depth = 1
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[last_start + len(needle):i]
        i += 1
    return None


def _strip_lhs_eq(s: str) -> str:
    """If `s` looks like 'var = expr' or 'lhs = rhs', return just rhs.
    Only strips the LAST '=' so chains like 'a = b = c' return 'c'.
    Skips '==', '<=', '>=', '!=' to avoid stripping inside conditions."""
    if "=" not in s:
        return s
    # Reject if the only '=' is part of a comparison
    cleaned = s
    for bad in ("==", "<=", ">=", "!="):
        cleaned = cleaned.replace(bad, "")
    if "=" not in cleaned:
        return s
    return s.rsplit("=", 1)[1].strip()


# After the math block, only trailing whitespace + a tiny amount of
# sentence-ending punctuation is allowed. Don't include ] ) } here — those
# are real LaTeX delimiter chars (`\]`, `)` in inline math, etc.).
_TAIL_OK_RE = re.compile(r"^[\s\.\,\;\:\!\?]*$")


def _last_delim_pair(text: str, open_d: str, close_d: str,
                     max_trailing: int = 50) -> str | None:
    """Walk backwards from end-of-text: skip trailing whitespace/punct, expect
    `close_d` next, then rfind the matching `open_d` BEFORE that close. Returns
    the content between them, stripped. Returns None if not found within
    `max_trailing` chars of trailing or if no matching open delim exists.

    This is more robust than forward finditer over `open_d ... close_d`, which
    can mis-pair when several pairs appear in the same tail."""
    if not text:
        return None
    end = len(text)
    # Skip trailing whitespace/punct
    while end > 0 and _TAIL_OK_RE.match(text[end - 1:end]):
        end -= 1
    if end == 0:
        return None
    if len(text) - end > max_trailing:
        return None
    # Now text[:end] should end with close_d
    if not text[:end].endswith(close_d):
        return None
    inner_end = end - len(close_d)
    open_pos = text.rfind(open_d, 0, inner_end)
    if open_pos < 0:
        return None
    inner = text[open_pos + len(open_d): inner_end].strip()
    return inner if inner else None


def _bare_final_line(text: str) -> str | None:
    """If the final non-empty line of text is a short standalone token
    (number / fraction / simple LaTeX) preceded by a blank line, return it.
    Conservative: rejects English words and anything > 80 chars."""
    lines = text.rstrip().split("\n")
    if not lines:
        return None
    last = lines[-1].strip()
    if not last or len(last) > 80:
        return None
    # Reject if it contains an English word of 3+ letters (probably a sentence)
    if re.search(r"\b[a-zA-Z]{3,}\b", last):
        return None
    # Must be preceded by a blank line, otherwise it's just a sentence-final word
    if len(lines) >= 2 and lines[-2].strip() != "":
        return None
    return last


def extract_boxed(text: str) -> str | None:
    """Extract the final answer from a model response.

    Tries, in order:
      1. \\boxed{...}                — primary, what the prompt asks for
      2. $$...$$ at end              — display math (gemma's habit)
      3. \\[...\\] at end            — alt display math
      4. $...$ at end                — inline math, with 'lhs =' stripped
      5. Bare final line             — short standalone token after a blank line

    Each fallback is restricted to the END of the text to avoid grabbing math
    fragments from the middle of the reasoning. Returns None if nothing found.
    """
    if not text:
        return None
    # 1. \boxed{...}
    boxed = _find_last_boxed(text)
    if boxed is not None:
        return boxed
    # 2. $$...$$ at end
    val = _last_delim_pair(text, "$$", "$$")
    if val is not None:
        return val
    # 3. \[...\] at end
    val = _last_delim_pair(text, "\\[", "\\]")
    if val is not None:
        return val
    # 4. $...$ at end (single dollar). Reject if the content looks like a
    # multi-line math block (single-dollar inline math should be one line).
    val = _last_delim_pair(text, "$", "$")
    if val is not None and "\n" not in val:
        return _strip_lhs_eq(val)
    # 5. Bare final line
    return _bare_final_line(text)


if __name__ == "__main__":
    # Smoke
    cases = [
        ("103", "\\boxed{103}", True),
        ("\\frac{1}{2}", "\\boxed{\\frac{1}{2}}", True),
        ("\\frac{1}{2}", "\\boxed{0.5}", True),
        ("\\frac{1}{2}", "\\boxed{0.6}", False),
        ("\\frac{-1+\\sqrt{17}}{2}, \\frac{-1-\\sqrt{17}}{2}",
         "\\boxed{\\frac{-1\\pm\\sqrt{17}}{2}}", True),
    ]
    for gold, cand, exp in cases:
        got = verify_latex(gold, cand)
        print(f"  {'OK ' if got == exp else 'FAIL'}  gold={gold!r}  cand={cand!r}  got={got}")
    # extract_boxed
    print("extract_boxed:", extract_boxed("foo \\boxed{\\frac{1}{2}} bar"))
    print("extract_boxed nested:", extract_boxed("a \\boxed{a^{x}} z"))
    print("extract_boxed last:", extract_boxed("\\boxed{1} then \\boxed{2}"))
    # Fallback patterns (gemma-style outputs)
    fb_cases = [
        ("foo bar.\n\n$$\\frac{1}{576}$$",                  "\\frac{1}{576}",     "$$ at end"),
        ("Total = 32 ways.\n\nThe final answer is\n\n$$32$$", "32",              "$$X$$ multi-line"),
        ("So we get \\[\\frac{2025}{101}\\]",               "\\frac{2025}{101}",  "\\[..\\] at end"),
        ("Answer is $120$",                                  "120",               "$X$ at end"),
        ("the result is $\\cos\\theta = \\frac{1}{3}$",     "\\frac{1}{3}",       "strip lhs"),
        ("All 5 solutions exhausted.\n\n56",                "56",                 "bare final line"),
        ("middle math $\\pi$ in sentence here.",            None,                 "no answer"),
    ]
    for text, exp, label in fb_cases:
        got = extract_boxed(text)
        ok = "OK " if got == exp else "FAIL"
        print(f"  [{ok}] {label:25s}  expected={exp!r:20s} got={got!r}")
