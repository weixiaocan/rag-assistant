# modules/loader.py
# -*- coding: utf-8 -*-

import os
import uuid
import hashlib
from typing import Iterable, List, Tuple, Dict, Any, Optional
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition
import unstructured_pytesseract.pytesseract as pytesseract
from collections import defaultdict


# 配置 tesseract OCR（仅在使用 unstructured 的 OCR 模式时需要）
pytesseract.tesseract_cmd = r"D:\ruanjian\tesseract-ocr\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"D:\ruanjian\tesseract-ocr\tessdata"

# ------------------------------------------------------------
# 支持的扩展名
# ------------------------------------------------------------
SUPPORTED_EXTS = {".txt", ".md", ".html", ".htm", ".docx", ".pptx", ".pdf"}

# 过滤页眉、页脚和页码三类元素
DROP_CATEGORIES_FOR_DOCX_PDF = {"Header", "Footer", "PageNumber"}


# ------------------------------------------------------------
# 小工具
# ------------------------------------------------------------
def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip()


def _hash_path(path: str) -> str:
    """稳定的文件级 ID，用于内部引用去重。"""
    return hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]


def _new_uuid() -> str:
    return uuid.uuid4().hex


# ------------------------------------------------------------
# elements → Documents（实现element-document的转换，保留元数据）
# ------------------------------------------------------------
def _elements_to_docs(
    path: str,
    elements,
    *,
    source_id: str,
    file_label: str,
    ext: str,
) -> List[Document]:
    """
    仅将 elements 转为 LangChain Document：
    - 取 el.text（多数元素类型都有可用文本；没有文本的元素自然会被跳过）
    - 元数据仅保留：file_label, source_id, source, path, ext
    """
    basename = os.path.basename(path)
    apath = os.path.abspath(path)

    docs: List[Document] = []
    for el in elements:
        # 统一走 el.text：Title/NarrativeText/ListItem/Table/CodeSnippet/...
        # 这些类型在 unstructured 中都有 text 表示；如 Image 没有文本就会被跳过
        text = _normalize_text(getattr(el, "text", None))
        if not text:
            continue

        md = {
            "file_label": file_label,  # 如 "[1]"，用于可见引用
            "source_id": source_id,    # 稳定 ID（hash），用于内部去重/统计
            "source": basename,        # 文件名
            "path": apath,             # 绝对路径
            "ext": ext,                # 扩展名
        }
        docs.append(Document(page_content=text, metadata=md))

    return docs


# ------------------------------------------------------------
# 解析单个文件（在这里做过滤）
# ------------------------------------------------------------
def _parse_one_file(
    path: str,
    *,
    prefer_strategy: str = "hi_res",
    infer_table_structure: bool = False,
    file_label: str = "[?]",
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    解析一个文件：
    - 用 unstructured.partition.auto.partition 得到 elements
    - 若是 .pdf/.docx：仅按类别丢弃 Header/Footer/PageNumber
    - 其余元素类型全部保留
    - 返回 (docs, manifest)
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"暂不支持的文件类型: {ext}（支持：{', '.join(sorted(SUPPORTED_EXTS))}）")

    kwargs = {}
    if ext == ".pdf":
        kwargs["strategy"] = prefer_strategy  # 'fast' | 'hi_res' | 'ocr_only' | 'auto'
        kwargs["infer_table_structure"] = infer_table_structure

    # 1) 解析
    elements = partition(filename=path, **kwargs)

    # 2) 过滤（只在 docx/pdf 上按分类移除三类；其余类型不过滤）
    if ext in (".pdf", ".docx"):
        filtered = []
        for el in elements:
            cat = getattr(el, "category", None)
            cat_str = str(cat) if cat is not None else None
            if cat_str in DROP_CATEGORIES_FOR_DOCX_PDF:
                continue
            filtered.append(el)
        elements = filtered

    # 3) elements → Documents（不再做任何过滤）
    source_id = _hash_path(path)
    docs = _elements_to_docs(
        path=path,
        elements=elements,
        source_id=source_id,
        file_label=file_label,
        ext=ext,
    )

    # 4) manifest（用于文末引用）
    manifest_item = {
        "file_label": file_label,
        "source_id": source_id,
        "filename": os.path.basename(path),
        "path": os.path.abspath(path),
        "ext": ext,
    }
    return docs, manifest_item


# ------------------------------------------------------------
# 对外 API：批量加载
# ------------------------------------------------------------
def load_multiple_documents(
    paths: List[str],
    *,
    limit: int = 5,
    prefer_strategy: str = "fast",
    infer_table_structure: bool = False,
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """
    解析并组合所有文件 → 返回 (all_docs, manifest)
    - all_docs: element 级 Document 列表（一个文件会产生多个 Document）
    - manifest: 用于渲染文末引用 [{file_label, filename, path, ext, source_id}, ...]
    """
    if len(paths) > limit:
        raise ValueError(f"最多仅支持上传 {limit} 个文件，请减少文件数量后重试。")

    all_docs: List[Document] = []
    manifest: List[Dict[str, Any]] = []

    for i, p in enumerate(paths, start=1):
        file_label = f"[{i}]"
        docs, item = _parse_one_file(
            p,
            prefer_strategy=prefer_strategy,
            infer_table_structure=infer_table_structure,
            file_label=file_label,
        )
        if docs:
            all_docs.extend(docs)
        manifest.append(item)

    return all_docs, manifest



def merge_docs_by_file(docs, keys=("source_id","file_label","source","path","ext"), sep="\n"):
    groups = defaultdict(list); metas={}
    for d in docs:
        k = tuple(d.metadata.get(x) for x in keys)
        groups[k].append(d.page_content)
        metas[k] = {x: d.metadata.get(x) for x in keys}
    out = []
    for k, parts in groups.items():
        text = sep.join(p for p in parts if p)
        if text.strip():
            out.append(Document(page_content=text, metadata=metas[k]))
    return out




# ------------------------------------------------------------
# 切分：透传元数据并补充 chunk 定位信息
# ------------------------------------------------------------
def split_documents(
    documents: List[Document],
    chunk_size: int = 100,
    chunk_overlap: int = 30,
) -> List[Document]:
    """
    按字符切分，每个 chunk 透传：
      - file_label / source_id / source / path / ext
    并补充：
      - chunk_index / chunk_id / char_start / char_end
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: List[Document] = []
    for d in documents:
        spans = splitter.split_text(d.page_content)
        start = 0
        for idx, s in enumerate(spans):
            s = _normalize_text(s)
            if not s:
                continue
            char_start = start
            char_end = start + len(s)
            start = max(0, char_end - chunk_overlap)

            md = dict(d.metadata)  # 透传文件级与引用信息
            md.update({
                "chunk_index": idx,
                "chunk_id": _new_uuid(),
                "char_start": char_start,
                "char_end": char_end,
            })
            chunks.append(Document(page_content=s, metadata=md))
    return chunks


# ------------------------------------------------------------
# 引用：根据 used_source_ids 和 manifest 生成文末引用区
# ------------------------------------------------------------
def build_citation_footer(
    used_source_ids: Iterable[str],
    manifest: List[Dict[str, Any]],
    *,
    title: str = "参考/引用",
) -> str:
    """
    将本次回答涉及到的 source_id 映射为“[序号] 文件名（绝对路径）”列表。
    """
    used = set(used_source_ids or [])
    if not used:
        return ""

    def _label_key(lbl: str) -> int:
        try:
            return int(lbl.strip("[]"))
        except Exception:
            return 10**9

    items = [m for m in manifest if m["source_id"] in used]
    items.sort(key=lambda x: _label_key(x["file_label"]))

    lines = [title]
    for it in items:
        lines.append(f'{it["file_label"]} {it["filename"]}（{it["path"]}）')
    return "\n".join(lines)
