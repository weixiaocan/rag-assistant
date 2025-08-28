# rag_core.py
# -*- coding: utf-8 -*-

import os
from typing import List, Dict, Any, Optional, Tuple

from modules.loader import load_multiple_documents, split_documents
from modules.embedder import build_vectorstore
from modules.llm_loader import load_llm

# ✅ 记忆与多轮检索
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ✅ 提示模板（让模型必须在答案末尾列出来源）
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
):
    """
    创建并返回一个完整的 RAG 会话（带记忆）。
    返回: (llm, vectorstore, retriever, memory, chain)
    """
    # 1) 加载 + 切分文档
    docs = load_multiple_documents(file_paths)
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 2) 构建/重建向量库（你后续可以改成持久化增量更新）
    vectorstore = build_vectorstore(chunks, save_path=index_dir)

    # 3) LLM
    llm = load_llm(model_key)

    # 4) 更智能的检索器：MMR + 评分阈值
    # - search_type="mmr": 多样化结果，减少同质片段
    # - k: 初选候选
    # - fetch_k: 候选池大小（MMR 从中重排）
    # - score_threshold: 过滤低分（需你的 vectorstore 支持相似度分；FAISS+cosine 有）
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.5,          # 多样性/相关性的权衡
        },
    )

    # 5) 记忆：对话历史（用在多轮问答）
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",  # 和下方链条的输出键一致
    )

    # 6) 自定义提示：要求答案后附*按序号*的引用，并保持可追问
    system_prompt = """你是一个严谨的检索增强AI助理。请结合“已检索到的资料”回答用户问题。
必须遵守：
- 先给出清晰、结构化的回答。
- 在回答末尾追加“参考资料”部分，按 [1]、[2]… 列出引用的片段（每条包括文件名与原文片段）。
- 不要编造引用，如果没有检索到相关资料，要直说“未检索到足够证据”。
- 若是用户的连续追问，请结合对话历史回答。

对话历史：{chat_history}
用户问题：{question}
已检索到的资料（可能为空）：
{context}

请给出：
1) 最佳回答
2) 参考资料（以 [编号] 文件名：片段 的形式列出，每条不超过300字）
"""
    qa_prompt = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template=system_prompt,
    )

    # 7) 组合成 ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,                      # ✅ 带记忆
        return_source_documents=True,       # ✅ 回传检索到的原文片段
        verbose=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

    return llm, vectorstore, retriever, memory, chain


# -----------------------------
# 2) 单轮/多轮问答：返回答案 + 明确列出的引用（文件名+原文）
# -----------------------------
def get_rag_response(
    query: str,
    chain: ConversationalRetrievalChain,
    *,
    max_snippet_chars: int = 300,
) -> Dict[str, Any]:
    """
    使用会话链进行问答（支持多轮）。返回：
    {
      "answer": str,
      "references": [
        {"source": 文件名, "path": 绝对路径, "ext": 扩展名, "score": 可能为空, "snippet": 原文片段}
      ]
    }
    """
    result = chain({"question": query})

    answer = result.get("answer") or result.get("result") or ""
    src_docs = result.get("source_documents", []) or []

    # 抽取来源（文件名 + 原文）
    refs: List[Dict[str, Any]] = []
    seen = set()

    for d in src_docs:
        meta = getattr(d, "metadata", {}) or {}
        source = meta.get("source") or os.path.basename(meta.get("path", "") or "")
        path = meta.get("path", "")
        ext = meta.get("ext", "")
        score = meta.get("score", None)  # 部分向量库会放相似度/距离在 metadata 里
        snippet = (d.page_content or "").strip()
        if max_snippet_chars and len(snippet) > max_snippet_chars:
            snippet = snippet[:max_snippet_chars] + "…"

        key = (source, snippet)
        if key in seen:
            continue
        seen.add(key)

        refs.append(
            {
                "source": source,
                "path": path,
                "ext": ext,
                "score": score,
                "snippet": snippet,
            }
        )

    return {
        "answer": answer,
        "references": refs,
    }



