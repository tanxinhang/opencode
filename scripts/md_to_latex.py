"""Convert paper/submission.md into an IEEE-style LaTeX manuscript."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "submission.md"
OUT = ROOT / "paper" / "main.tex"

CITE_KEYS = {
    1: "hadani2017otfs",
    2: "yuan2023ddisac",
    3: "wu2019irs",
    4: "xu2024risisac",
    5: "tenney1981detection",
    6: "chair1986optimal",
    7: "varshney1997book",
    8: "nurellari2015quantized",
    9: "liu2022fundamental",
    10: "meng2024uavisac",
    11: "wang1993poisson",
    12: "zargari2024riemannian",
    13: "olfati2004consensus",
    14: "raviteja2018interference",
    15: "liu2020joint",
    16: "kumari2018ieee80211ad",
    17: "tun2023risjsac",
    18: "chepuri2016sparse",
    19: "liu2016sensor",
    20: "guo2022nonparametric",
    21: "krause2008robust",
    22: "godrich2012sensor",
}


def _cites_tex(raw: str) -> str:
    """Convert a manual [n] or [n]--[m] citation token to \\cite{...}."""
    range_match = re.fullmatch(r"\[(\d+)\]--\[(\d+)\]", raw)
    if range_match:
        start, end = (int(v) for v in range_match.groups())
        keys = [CITE_KEYS[i] for i in range(start, end + 1) if i in CITE_KEYS]
        return r"\cite{" + ",".join(keys) + "}"
    single = re.fullmatch(r"\[(\d+)\]", raw)
    if single:
        key = CITE_KEYS.get(int(single.group(1)))
        if key:
            return r"\cite{" + key + "}"
    return raw


def escape_inline(text: str, use_cites: bool = True) -> str:
    """Escape LaTeX specials outside $...$ math and `code` spans."""
    cite_placeholders: dict[str, str] = {}
    if use_cites:
        def _protect(match: re.Match) -> str:
            token = f"\x00CITE{len(cite_placeholders)}\x00"
            cite_placeholders[token] = match.group(0)
            return token
        text = re.sub(r"\[(\d+)\]--\[(\d+)\]", _protect, text)
        text = re.sub(r"\[(\d+)\]", _protect, text)
    parts = re.split(r"(\$[^$]*\$|`[^`]*`)", text)
    out = []
    for part in parts:
        if not part:
            continue
        if part.startswith("$") and part.endswith("$"):
            out.append(part.replace(r"\_", "_"))
        elif part.startswith("`") and part.endswith("`"):
            out.append(r"\texttt{" + part[1:-1].replace("_", r"\_") + "}")
        else:
            escaped = part
            escaped = escaped.replace("\\", r"\textbackslash{}")
            escaped = escaped.replace("{", r"\{")
            escaped = escaped.replace("}", r"\}")
            escaped = escaped.replace("_", r"\_")
            escaped = escaped.replace("&", r"\&")
            escaped = escaped.replace("%", r"\%")
            escaped = escaped.replace("#", r"\#")
            escaped = escaped.replace("$", r"\$")
            escaped = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", escaped)
            out.append(escaped)
    joined = "".join(out)
    if use_cites:
        for token, raw in cite_placeholders.items():
            joined = joined.replace(token, _cites_tex(raw))
    return joined


def column_alignment(header: str) -> str:
    cells = [c.strip() for c in header.strip().strip("|").split("|")]
    alignment = []
    for cell in cells:
        left = cell.startswith(":")
        right = cell.endswith(":")
        if left and right:
            alignment.append("c")
        elif right:
            alignment.append("r")
        elif left:
            alignment.append("l")
        else:
            alignment.append("c")
    return "".join(alignment)


def clean_section_title(title: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title).strip()


def fold_numbered_items(lines: list[str]) -> list[str]:
    """Merge multi-line numbered list items into single logical lines."""
    out: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if not match:
            out.append(lines[index])
            index += 1
            continue
        body = [match.group(2)]
        index += 1
        while index < len(lines):
            next_stripped = lines[index].strip()
            if (
                not next_stripped
                or next_stripped.startswith("#")
                or next_stripped.startswith("|")
                or next_stripped.startswith("![")
                or next_stripped.startswith("**")
                or next_stripped.startswith("$$")
                or next_stripped.startswith("```")
                or re.match(r"^\d+\.\s+", next_stripped)
            ):
                break
            body.append(next_stripped)
            index += 1
        out.append(f"{match.group(1)}. " + " ".join(body))
    return out


def main() -> None:
    lines = fold_numbered_items(SRC.read_text(encoding="utf-8").splitlines())
    tex: list[str] = []
    tex.append("% Auto-generated from paper/submission.md by scripts/md_to_latex.py")
    tex.append(r"\documentclass[conference]{IEEEtran}")
    tex.append(r"\usepackage{graphicx}")
    tex.append(r"\usepackage{amsmath,amssymb}")
    tex.append(r"\usepackage{amsthm}")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\newtheorem{theorem}{Theorem}")
    tex.append(r"\newtheorem{lemma}{Lemma}")
    tex.append(r"\newtheorem{corollary}{Corollary}")
    tex.append(r"\newtheorem{proposition}{Proposition}")
    tex.append(r"\newtheorem*{theoremstar}{Theorem}")
    tex.append(r"\newtheorem*{lemmastar}{Lemma}")
    tex.append(r"\begin{document}")

    index = 0
    pending_caption: str | None = None
    in_references = False
    in_abstract = False
    in_appendix = False
    keywords_text: str | None = None
    list_stack: list[str] = []
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if in_references and not stripped.startswith("## "):
            index += 1
            continue
        if stripped.startswith("*Submission draft."):
            while index < len(lines) and lines[index].strip():
                index += 1
            continue
        is_list_item = stripped.startswith("- ") or bool(
            re.match(r"^\d+\.\s", stripped)
        )
        if list_stack and not is_list_item:
            while list_stack:
                tex.append(r"\end{" + list_stack.pop() + "}")

        if stripped.startswith("**Keywords:**"):
            parts = [stripped[len("**Keywords:**"):].strip()]
            while (
                index + 1 < len(lines)
                and lines[index + 1].strip()
            ):
                index += 1
                parts.append(lines[index].strip())
            keywords_text = " ".join(parts)
            index += 1
            continue

        if stripped.startswith("```"):
            code: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "```":
                code.append(lines[index])
                index += 1
            tex.append(r"{\footnotesize")
            tex.append(r"\begin{verbatim}")
            tex.extend(code)
            tex.append(r"\end{verbatim}")
            tex.append(r"}")
            index += 1
            continue

        if stripped.startswith("# ") and not stripped.startswith("## "):
            tex.append(r"\title{" + escape_inline(stripped[2:]) + "}")
            tex.append(r"\author{\IEEEauthorblockN{Author Placeholder}"
                       r"\IEEEauthorblockA{Department, University, City, Country}}")
            tex.append(r"\maketitle")
        elif stripped.startswith("## ") and not stripped.startswith("### "):
            title = stripped[3:]
            if title.startswith("References"):
                if in_abstract:
                    tex.append(r"\end{abstract}")
                    tex.append(r"\begin{IEEEkeywords}")
                    tex.append(
                        keywords_text
                        if keywords_text is not None
                        else "UAV-ISAC, OTFS, RIS, distributed detection, "
                             "selective fusion, exact optimization"
                    )
                    tex.append(r"\end{IEEEkeywords}")
                    in_abstract = False
                in_references = True
                tex.append(r"\bibliographystyle{IEEEtran}")
                tex.append(r"\bibliography{references}")
            elif title.startswith("Appendix "):
                if in_abstract:
                    tex.append(r"\end{abstract}")
                    in_abstract = False
                in_appendix = True
                in_references = False
                tex.append(r"\appendix")
                tex.append(r"\section{" + escape_inline(title.split(".", 1)[-1].strip()) + "}")
            elif title.startswith("Abstract"):
                in_abstract = True
                tex.append(r"\begin{abstract}")
            else:
                if in_abstract:
                    tex.append(r"\end{abstract}")
                    tex.append(r"\begin{IEEEkeywords}")
                    tex.append(
                        keywords_text
                        if keywords_text is not None
                        else "UAV-ISAC, OTFS, RIS, distributed detection, "
                             "selective fusion, exact optimization"
                    )
                    tex.append(r"\end{IEEEkeywords}")
                    in_abstract = False
                tex.append(r"\section{" + escape_inline(clean_section_title(title)) + "}")
        elif stripped.startswith("### "):
            tex.append(r"\subsection{" + escape_inline(clean_section_title(stripped[4:])) + "}")
        elif stripped.startswith("$$"):
            math = []
            if stripped == "$$":
                index += 1
                while index < len(lines) and lines[index].strip() != "$$":
                    math.append(lines[index].strip())
                    index += 1
            else:
                math.append(stripped[2:])
            math = [line.replace(r"\_", "_") for line in math]
            tex.append(r"\begin{equation}")
            tex.append(r"\begin{gathered}")
            tex.extend(math)
            tex.append(r"\end{gathered}")
            tex.append(r"\end{equation}")
        elif stripped.startswith("![") and "](paper_figures/" in stripped:
            path = stripped.split("](", 1)[1].rstrip(")")
            tex.append(r"\begin{figure}[htbp]")
            tex.append(r"\centering")
            tex.append(
                r"\includegraphics[width=0.92\linewidth]{../"
                + path
                + "}"
            )
            if pending_caption:
                caption = re.sub(
                    r"^(Figure|Table)\s+\d+(?:\.\d+)?\.?\s*",
                    "",
                    pending_caption,
                    flags=re.IGNORECASE,
                )
                caption = caption.replace("**", "").strip()
                tex.append(r"\caption{" + escape_inline(caption) + "}")
                pending_caption = None
            tex.append(r"\end{figure}")
        elif stripped.startswith("**Table ") or stripped.startswith("**Figure "):
            pending_caption = stripped[2:-2]
        elif stripped.startswith("|"):
            table_rows: list[list[str]] = []
            alignment = None
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [
                    c.strip()
                    for c in lines[index].strip().strip("|").split("|")
                ]
                if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    alignment = column_alignment(lines[index].strip())
                else:
                    table_rows.append(cells)
                index += 1
            if alignment is None:
                alignment = "c" * len(table_rows[0])
            tex.append(r"\begin{table}[htbp]")
            tex.append(r"\centering")
            tex.append(r"\resizebox{\linewidth}{!}{")
            tex.append(r"\begin{tabular}{" + alignment + "}")
            tex.append(r"\toprule")
            tex.append(" & ".join(
                escape_inline(cell) for cell in table_rows[0]
            ) + r" \\")
            tex.append(r"\midrule")
            for row in table_rows[1:]:
                tex.append(" & ".join(
                    escape_inline(cell) for cell in row
                ) + r" \\")
            tex.append(r"\bottomrule")
            tex.append(r"\end{tabular}")
            tex.append(r"}")
            if pending_caption:
                caption = re.sub(
                    r"^(Figure|Table)\s+\d+(?:\.\d+)?\.?\s*",
                    "",
                    pending_caption,
                    flags=re.IGNORECASE,
                )
                caption = caption.replace("**", "").strip()
                tex.append(r"\caption{" + escape_inline(caption) + "}")
                pending_caption = None
            tex.append(r"\end{table}")
            continue
        elif stripped.startswith("- "):
            if list_stack != ["itemize"]:
                while list_stack:
                    tex.append(r"\end{" + list_stack.pop() + "}")
                tex.append(r"\begin{itemize}")
                list_stack.append("itemize")
            tex.append(r"\item " + escape_inline(stripped[2:]))
        elif re.match(r"^\d+\.\s", stripped):
            if list_stack != ["enumerate"]:
                while list_stack:
                    tex.append(r"\end{" + list_stack.pop() + "}")
                tex.append(r"\begin{enumerate}")
                list_stack.append("enumerate")
            tex.append(r"\item " + escape_inline(re.sub(r"^\d+\.\s", "", stripped)))
        elif in_references and re.match(r"^\[\d+\]\s", stripped):
            index += 1
            continue
        elif (
            stripped.startswith("**Theorem ")
            or stripped.startswith("**Lemma ")
            or stripped.startswith("**Corollary ")
            or stripped.startswith("**Proposition ")
        ):
            if in_appendix:
                if stripped.startswith("**Theorem"):
                    kind = "theoremstar"
                elif stripped.startswith("**Lemma"):
                    kind = "lemmastar"
                elif stripped.startswith("**Proposition"):
                    kind = "proposition"
                else:
                    kind = "corollary"
            else:
                if stripped.startswith("**Theorem"):
                    kind = "theorem"
                elif stripped.startswith("**Lemma"):
                    kind = "lemma"
                elif stripped.startswith("**Proposition"):
                    kind = "proposition"
                else:
                    kind = "corollary"
            header, _, body = stripped.partition(".**")
            header = header[2:].strip()
            if _:
                header = header + "."
            match = re.match(
                r"^(Theorem|Lemma|Corollary|Proposition)\s+([^ (]+)(?:\s+\(([^)]+)\))?\.$",
                header,
            )
            if match and match.group(3):
                tex.append(r"\begin{" + kind + r"}[" + match.group(3) + "]")
            else:
                tex.append(r"\begin{" + kind + "}")
            tex.append(escape_inline(body.strip()))
            while (
                index + 1 < len(lines)
                and lines[index + 1].strip()
                and not lines[index + 1].strip().startswith("**")
            ):
                index += 1
                tex.append(escape_inline(lines[index].strip()))
            tex.append(r"\end{" + kind + "}")
        else:
            paragraph_lines = [stripped]
            while index + 1 < len(lines):
                next_line = lines[index + 1].rstrip()
                next_stripped = next_line.strip()
                if not next_stripped:
                    break
                if (
                    next_stripped.startswith("#")
                    or next_stripped.startswith("|")
                    or next_stripped.startswith("![")
                    or next_stripped.startswith("**Table ")
                    or next_stripped.startswith("**Figure ")
                    or next_stripped.startswith("**Keywords:")
                    or next_stripped.startswith("**Theorem ")
                    or next_stripped.startswith("**Lemma ")
                    or next_stripped.startswith("$$")
                    or next_stripped.startswith("```")
                    or next_stripped.startswith("- ")
                    or re.match(r"^\d+\.\s", next_stripped)
                ):
                    break
                paragraph_lines.append(next_stripped)
                index += 1
            tex.append(escape_inline(" ".join(paragraph_lines)))
        index += 1

    while list_stack:
        tex.append(r"\end{" + list_stack.pop() + "}")
    tex.append(r"\end{document}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
