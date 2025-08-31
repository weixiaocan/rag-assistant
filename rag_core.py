# rag_core.py
# -*- coding: utf-8 -*-

import os
from typing import List, Dict, Any, Optional, Tuple

from modules.loader import (
    load_multiple_documents,
    merge_docs_by_file,
    split_documents,
    build_citation_footer  # 新增：用于文末引用
)
from modules.embedder import build_vectorstore
from modules.llm_loader import load_llm

# ✅ 记忆与多轮检索
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ✅ 提示模板（让模型必须在答案末尾列出来源 — 这里我们仍保留，但最终会用脚注兜底）
from langchain.prompts import PromptTemplate


# -----------------------------
# 1) 创建 RAG 管道（构建向量库 + LLM + 检索器 + 记忆）
# -----------------------------
def create_rag_pipeline(
    file_paths: List[str],
    model_key: str,
    *,
    index_dir: str = "vectorstore/index",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    prefer_strategy: str = "fast",
    infer_table_structure: bool = False,
):
    """
    创建并返回一个完整的 RAG 会话（带记忆）。
    返回: (llm, vectorstore, retriever, memory, chain, manifest)
    """
    # 1) 加载 + 切分文档（注意：现在 loader 返回 (docs, manifest)）
    docs, manifest = load_multiple_documents(
        file_paths,
        prefer_strategy=prefer_strategy,
        infer_table_structure=infer_table_structure,
    )
    # 合并同一文件的 docs
    docs = merge_docs_by_file(docs)
    # 进行切分
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 2) 构建/重建向量库
    vectorstore = build_vectorstore(chunks, save_path=index_dir)

    # 3) LLM
    llm = load_llm(model_key)

    # 4) 更智能的检索器：MMR 多样化
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
    )

    # 5) 记忆：对话历史（多轮）
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # 6) 提示（保留要求，但我们会在 get_rag_response 里用脚注做强约束落地）
    system_prompt = """你是一个严谨的检索增强AI助理。请结合“已检索到的资料”回答用户问题。
必须遵守：
- 先给出清晰、结构化的回答。
- 尽量在相关句子后用 [序号] 标注引用（序号由系统分配，可能在答案后自动补全）。
- 不要编造引用，如果没有检索到相关资料，要直说“未检索到足够证据”。
- 若是用户的连续追问，请结合对话历史回答。

对话历史：{chat_history}
用户问题：{question}
已检索到的资料（可能为空）：
{context}

请给出：最佳回答（必要时可在句尾写上标注 [1]、[2]…）
"""
    qa_prompt = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template=system_prompt,
    )

    # 7) 组合成 ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

    return llm, vectorstore, retriever, memory, chain, manifest


# -----------------------------
# 2) 单轮/多轮问答：返回答案 + 明确列出的引用（编号/文件名/原文）
# -----------------------------
def get_rag_response(
    query: str,
    chain: ConversationalRetrievalChain,
    manifest: List[Dict[str, Any]],
    *,
    max_snippet_chars: int = 300,
    append_footer: bool = True,
) -> Dict[str, Any]:
    """
    使用会话链进行问答（支持多轮）。返回：
    {
      "answer": str,                      # 可能已自动拼接“参考/引用”脚注
      "references": [
        {
          "file_label": "[1]",
          "source_id": "...",
          "source": 文件名,
          "path": 绝对路径,
          "ext": 扩展名,
          "score": 可能为空,
          "snippet": 原文片段
        }
      ],
      "footer": "参考/引用\n[1] 文件A（/path/A）\n[2] 文件B（/path/B）"
    }
    """
    result = chain({"question": query})

    answer = result.get("answer") or result.get("result") or ""
    src_docs = result.get("source_documents", []) or []

    # 抽取来源（带 file_label / source_id）
    refs: List[Dict[str, Any]] = []
    seen = set()
    used_source_ids = set()

    for d in src_docs:
        meta = getattr(d, "metadata", {}) or {}

        file_label = meta.get("file_label") or ""     # "[1]" / "[2]"
        source_id = meta.get("source_id") or ""
        source = meta.get("source") or os.path.basename(meta.get("path", "") or "")
        path = meta.get("path", "")
        ext = meta.get("ext", "")
        score = meta.get("score", None)  # 某些向量库会塞相似度
        snippet = (d.page_content or "").strip()
        if max_snippet_chars and len(snippet) > max_snippet_chars:
            snippet = snippet[:max_snippet_chars] + "…"

        key = (file_label, source, snippet)
        if key in seen:
            continue
        seen.add(key)
        if source_id:
            used_source_ids.add(source_id)

        refs.append(
            {
                "file_label": file_label,
                "source_id": source_id,
                "source": source,
                "path": path,
                "ext": ext,
                "score": score,
                "snippet": snippet,
            }
        )

    # 基于 used_source_ids + manifest 生成脚注
    footer_text = build_citation_footer(used_source_ids, manifest) if used_source_ids else ""

    # 将脚注拼到答案后（也可以交给前端分区渲染）
    final_answer = answer
    if append_footer and footer_text:
        final_answer = f"{answer}\n\n{footer_text}"

    return {
        "answer": final_answer,
        "references": refs,
        "footer": footer_text,
    }
