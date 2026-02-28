"""
法条链接后处理验证：提取 [LAW]...[/LAW]，在知识库中匹配并修正。

- 精确匹配：规范标题、法规名+条号、全角/半角统一后匹配
- 模糊匹配：法规名包含 + 条号包含（如「刑法」「第二百三十四条」）
- 匹配不到：可配置为改写成规范法条、删除引用、或替换为「未找到对应法条」
"""

import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple

# 默认知识库路径（与 local_model_config 一致）
_current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STRUCTURED_LAWS_PATH = os.path.join(_current_dir, "structured_laws_by_category.json")

def _full_to_half(s: str) -> str:
    """全角转半角（数字、空格等）。"""
    out = []
    for c in s:
        if "\uff01" <= c <= "\uff5e":
            out.append(chr(ord(c) - 0xfee0))
        elif c == "\u3000":
            out.append(" ")
        else:
            out.append(c)
    return "".join(out)


def _normalize_for_match(s: str) -> str:
    """归一化用于匹配：去空格、去《》、全角转半角、统一「第」「条」。"""
    if not s or not isinstance(s, str):
        return ""
    s = s.strip()
    s = _full_to_half(s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("《", "").replace("》", "")
    return s


class LawReferenceValidator:
    """从文本中提取 [LAW]...[/LAW]，在知识库中匹配并返回修正后的文本与审计信息。"""

    def __init__(self, structured_laws_path: str = DEFAULT_STRUCTURED_LAWS_PATH):
        self.structured_laws_path = structured_laws_path
        self._laws_flat: List[Dict[str, Any]] = []
        self._exact_index: Dict[str, Dict[str, Any]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """从 structured_laws_by_category.json 加载并构建精确/归一化索引。"""
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
                law_id = law.get("law_id") or ""
                title = (law.get("title") or "").strip()
                source = (law.get("source") or "").strip()
                article = (law.get("article_number") or "").strip()
                self._laws_flat.append(law)
                # 精确 key：原始 title、source+article、law_id
                for key in [title, f"{source}{article}", law_id]:
                    if key and key not in self._exact_index:
                        self._exact_index[key] = law
                norm_title = _normalize_for_match(title)
                norm_sa = _normalize_for_match(f"{source}{article}")
                for key in [norm_title, norm_sa]:
                    if key and key not in self._exact_index:
                        self._exact_index[key] = law
                # 带《》的 title 也加入
                if title and title not in self._exact_index:
                    self._exact_index[title] = law

    def _match_one(self, raw: str) -> Optional[Dict[str, Any]]:
        """对单条 [LAW] 内原文做匹配，返回命中的 law 或 None。"""
        raw = (raw or "").strip()
        if not raw:
            return None
        norm = _normalize_for_match(raw)
        if norm in self._exact_index:
            return self._exact_index[norm]
        # 尝试「法规名 + 第X条」模糊：提取“第…条”
        art_m = re.search(r"第(.+?)条", raw)
        art_m_norm = re.search(r"第(.+?)条", norm)
        for law in self._laws_flat:
            source = _normalize_for_match((law.get("source") or ""))
            article = (law.get("article_number") or "").strip()
            title = (law.get("title") or "").strip()
            if not source or not article:
                continue
            # 原文包含法规名且条号一致（或归一化后一致）
            if source in norm or source in raw:
                if article in raw or article in norm:
                    return law
                if art_m and art_m_norm:
                    art_raw = art_m.group(1)
                    art_norm = art_m_norm.group(1)
                    if art_norm in _normalize_for_match(article) or article in raw:
                        return law
                    if art_raw == article or art_norm == _normalize_for_match(article):
                        return law
        return None

    def extract_refs(self, text: str) -> List[Tuple[int, int, str]]:
        """提取所有 [LAW]...[/LAW]，返回 [(start, end, content), ...]。"""
        if not text:
            return []
        pattern = re.compile(r"\[LAW\](.*?)\[/LAW\]", re.DOTALL)
        refs = []
        for m in pattern.finditer(text):
            refs.append((m.start(0), m.end(0), m.group(1).strip()))
        return refs

    def validate_and_fix(
        self,
        text: str,
        unmatched_mode: str = "replace_with_not_found",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        对文本中的 [LAW]...[/LAW] 进行校验并修正。

        unmatched_mode:
          - "replace_with_title": 匹配到的替换为规范 title；匹配不到的删除
          - "replace_with_not_found": 匹配到的保留或替换为 title，匹配不到的替换为 [未找到对应法条]
          - "remove": 匹配不到的删除；匹配到的可保留或替换为 title（当前实现：匹配到的替换为 title）

        Returns:
          (corrected_text, list of {raw, matched, action, display})
        """
        if not text:
            return text, []
        refs = self.extract_refs(text)
        if not refs:
            return text, []

        corrections = []
        # 从后往前替换，避免偏移变化
        for start, end, raw in reversed(refs):
            law = self._match_one(raw)
            if law:
                display = (law.get("title") or "").strip() or f"{law.get('source', '')} {law.get('article_number', '')}"
                action = "matched"
                new_block = f"[LAW]{display}[/LAW]"
            else:
                display = ""
                action = "not_found"
                if unmatched_mode == "remove":
                    new_block = ""
                elif unmatched_mode == "replace_with_not_found":
                    new_block = "[LAW]未找到对应法条[/LAW]"
                else:
                    new_block = ""
            corrections.append({"raw": raw, "matched": law is not None, "action": action, "display": display})
            text = text[:start] + new_block + text[end:]
        corrections.reverse()
        return text, corrections


def validate_law_references(
    text: str,
    structured_laws_path: str = DEFAULT_STRUCTURED_LAWS_PATH,
    unmatched_mode: str = "replace_with_not_found",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    便捷函数：对一段文本做法条校验并返回修正后文本与审计列表。

    unmatched_mode: "replace_with_title" | "replace_with_not_found" | "remove"
    """
    validator = LawReferenceValidator(structured_laws_path=structured_laws_path)
    return validator.validate_and_fix(text, unmatched_mode=unmatched_mode)


if __name__ == "__main__":
    # 本地快速测试
    v = LawReferenceValidator()
    test_text = "根据[LAW]联合国宪章第一条[/LAW]以及[LAW]刑法第二百三十四条[/LAW]的规定。"
    corrected, audit = v.validate_and_fix(test_text, unmatched_mode="replace_with_not_found")
    print("原文:", test_text)
    print("修正:", corrected)
    print("审计:", audit)
