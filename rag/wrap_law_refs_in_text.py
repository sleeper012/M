"""
将正文中的法条引用（如《xxx》第y条）包裹为 [LAW]规范标题[/LAW]，用于生成训练数据。
模型在训练时看到该格式，学会在生成时输出 [LAW]...[/LAW]，推理后再用 law_reference_validator 校验。
"""

import json
import os
import re
from typing import Any, List, Optional, Tuple

_current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STRUCTURED_LAWS_PATH = os.path.join(_current_dir, "structured_laws_by_category.json")


def _full_to_half(s: str) -> str:
    out = []
    for c in s:
        if "\uff01" <= c <= "\uff5e":
            out.append(chr(ord(c) - 0xfee0))
        elif c == "\u3000":
            out.append(" ")
        else:
            out.append(c)
    return "".join(out)


def _normalize(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    s = _full_to_half(s.strip())
    s = re.sub(r"\s+", "", s)
    s = s.replace("《", "").replace("》", "")
    return s


def _short_source(source: str) -> List[str]:
    """法规名多种写法，用于匹配。"""
    s = (source or "").strip()
    if not s:
        return []
    out = [s]
    if s.startswith("中华人民共和国"):
        out.append(s[7:].strip())
    if s.startswith("中国"):
        out.append(s[2:].strip())
    return out


class LawRefWrapper:
    """把正文里的《法规名》第X条 替换成 [LAW]规范标题[/LAW]。"""

    def __init__(self, structured_laws_path: str = DEFAULT_STRUCTURED_LAWS_PATH):
        self.structured_laws_path = structured_laws_path
        self._index: Dict[str, str] = {}  # normalized_key -> display (用于 [LAW]display[/LAW])
        self._build_index()

    def _build_index(self) -> None:
        if not os.path.exists(self.structured_laws_path):
            return
        with open(self.structured_laws_path, "r", encoding="utf-8") as f:
            by_category = json.load(f)
        if not isinstance(by_category, dict):
            return
        for category, laws in by_category.items():
            if not isinstance(laws, list):
                continue
            for law in laws:
                if not isinstance(law, dict):
                    continue
                title = (law.get("title") or "").strip()
                source = (law.get("source") or "").strip()
                article = (law.get("article_number") or "").strip()
                if not title:
                    title = f"《{source}》{article}"
                display = title
                # 多种 key 都指向同一 display
                for key in [
                    _normalize(title),
                    _normalize(f"{source}{article}"),
                    _normalize(f"《{source}》{article}"),
                ]:
                    if key and key not in self._index:
                        self._index[key] = display
                for short in _short_source(source):
                    k = _normalize(f"{short}{article}")
                    if k and k not in self._index:
                        self._index[k] = display
                    k = _normalize(f"《{short}》{article}")
                    if k and k not in self._index:
                        self._index[k] = display

    def _lookup(self, name: str, article: str) -> Optional[str]:
        """返回规范展示串，未匹配返回 None。"""
        raw = f"《{name}》第{article}条"
        n = _normalize(raw)
        if n in self._index:
            return self._index[n]
        n2 = _normalize(f"{name}第{article}条")
        if n2 in self._index:
            return self._index[n2]
        for short in _short_source(name):
            if _normalize(f"{short}第{article}条") in self._index:
                return self._index[_normalize(f"{short}第{article}条")]
        return None

    def _ranges_law_tags(self, text: str) -> List[Tuple[int, int]]:
        """返回所有 [LAW]...[/LAW] 的 (start, end) 区间，避免在第二遍替换时重复包裹。"""
        ranges = []
        for m in re.finditer(r"\[LAW\].*?\[/LAW\]", text, re.DOTALL):
            ranges.append((m.start(0), m.end(0)))
        return ranges

    def _replace_one_pass(
        self, text: str, pattern: re.Pattern, unmatched_wrap: bool, exclude_ranges: Optional[List[Tuple[int, int]]] = None
    ) -> str:
        exclude_ranges = exclude_ranges or []
        replacements = []
        for m in pattern.finditer(text):
            start, end = m.start(0), m.end(0)
            if any(s <= start < e or s < end <= e for s, e in exclude_ranges):
                continue
            name, article = m.group(1).strip(), m.group(2).strip()
            display = self._lookup(name, article)
            if display:
                replacements.append((start, end, f"[LAW]{display}[/LAW]"))
            elif unmatched_wrap:
                replacements.append((start, end, f"[LAW]{m.group(0)}[/LAW]"))
        for start, end, new in reversed(replacements):
            text = text[:start] + new + text[end:]
        return text

    def wrap(self, text: str, unmatched_wrap: bool = True) -> str:
        """
        把正文中《xxx》第y条 或 【法规n】《xxx》第y条 替换为 [LAW]规范标题[/LAW]；未匹配时若 unmatched_wrap 则替换为 [LAW]原文[/LAW]。
        """
        if not text or not isinstance(text, str):
            return text
        # 第一遍：【法规n】《...》第...条 整段替换
        pattern_with_tag = re.compile(r"【法规\d+】\s*《([^》]+)》\s*第([一二三四五六七八九十百零〇0-9]+)条")
        text = self._replace_one_pass(text, pattern_with_tag, unmatched_wrap, exclude_ranges=[])
        # 第二遍：剩余 《...》第...条，且不替换已落在 [LAW]...[/LAW] 内的内容
        exclude_ranges = self._ranges_law_tags(text)
        pattern_plain = re.compile(r"《([^》]+)》\s*第([一二三四五六七八九十百零〇0-9]+)条")
        text = self._replace_one_pass(text, pattern_plain, unmatched_wrap, exclude_ranges=exclude_ranges)
        return text


def wrap_law_refs_in_text(
    text: str,
    structured_laws_path: str = DEFAULT_STRUCTURED_LAWS_PATH,
    unmatched_wrap: bool = True,
) -> str:
    """便捷：对一段正文做法条包裹，返回新正文。"""
    w = LawRefWrapper(structured_laws_path=structured_laws_path)
    return w.wrap(text, unmatched_wrap=unmatched_wrap)
