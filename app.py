# app.py
# -*- coding: utf-8 -*-

import os
import streamlit as st
from dotenv import load_dotenv

from rag_core import create_rag_pipeline, get_rag_response

# 读取 .env
load_dotenv()

# ------------------ 模型配置 ------------------
LLM_CANDIDATES = [
    ("gpt-4", "OpenAI GPT-4"),
    ("deepseek-v3", "DeepSeek V3"),
    ("deepseek-r1", "DeepSeek R1"),
]

# ------------------ 页面设置 ------------------
st.set_page_config(page_title="AI 文档问答助手", layout="wide")
st.title("📄 AI 文档问答助手")
st.caption("RAG + 多轮记忆 · 引用精确到文件与原文片段（支持 [1]/[2] 脚注）")

# ------------------ Session State ------------------
if "rag_session" not in st.session_state:
    st.session_state.rag_session = None
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []
if "chat_history_view" not in st.session_state:
    st.session_state.chat_history_view = []
if "manifest" not in st.session_state:
    st.session_state.manifest = []   # 引用清单（file_label/source_id/filename/path/ext）

# ------------------ 侧边栏：文件 & 模型 & 控制 ------------------
with st.sidebar:
    st.header("⚙️ 配置")
    st.subheader("1️⃣ 上传文档")
    uploaded_files = st.file_uploader(
        "请上传 1～5 个文件（PDF / TXT / DOCX / PPTX / MD / HTML）",
        type=["pdf", "txt", "docx", "pptx", "md", "html", "htm"],
        accept_multiple_files=True,
    )

    st.subheader("2️⃣ 选择模型")
    model_map = dict(LLM_CANDIDATES)
    model_key = st.selectbox(
        "请选择一个大语言模型",
        options=[m[0] for m in LLM_CANDIDATES],
        format_func=lambda x: model_map[x],
    )

    col_a, col_b = st.columns(2)
    with col_a:
        build_clicked = st.button("🚀 构建 / 重建知识库", use_container_width=True)
    with col_b:
        clear_clicked = st.button("🧹 清空会话", use_container_width=True)

    if clear_clicked:
        st.session_state.chat_history_view = []
        if st.session_state.rag_session and "memory" in st.session_state.rag_session:
            st.session_state.rag_session["memory"].clear()
        st.success("已清空对话。")

    if build_clicked:
        if not uploaded_files:
            st.error("❗ 请先上传至少一个文档。")
        elif len(uploaded_files) > 5:
            st.error("❌ 最多只能上传 5 个文件。")
        else:
            with st.spinner("⏳ 正在处理与索引文档 ..."):
                os.makedirs("data", exist_ok=True)
                file_paths, names = [], []
                for f in uploaded_files:
                    p = os.path.join("data", f.name)
                    with open(p, "wb") as wf:
                        wf.write(f.getvalue())
                    file_paths.append(p)
                    names.append(f.name)

                try:
                    # ✅ 现在 create_rag_pipeline 返回 6 个对象（多了 manifest）
                    llm, vectorstore, retriever, memory, chain, manifest = create_rag_pipeline(
                        file_paths=file_paths,
                        model_key=model_key,
                    )
                    st.session_state.rag_session = {
                        "llm": llm,
                        "vectorstore": vectorstore,
                        "retriever": retriever,
                        "memory": memory,
                        "chain": chain,
                    }
                    st.session_state.uploaded_file_names = names
                    st.session_state.chat_history_view = []
                    st.session_state.manifest = manifest
                    st.success("✅ 知识库构建成功！可以开始提问啦。")
                except Exception as e:
                    st.session_state.rag_session = None
                    st.session_state.manifest = []
                    st.error(f"构建知识库时发生错误：{e}")

# ------------------ 主区：文件清单 + 问答 ------------------
left, right = st.columns([2, 3])

with left:
    st.subheader("📦 当前知识库")
    if st.session_state.manifest:
        for item in st.session_state.manifest:
            # 用 [1]/[2] 标签展示，和后续引用一致
            st.markdown(f"- {item['file_label']} **{item['filename']}**")
            st.caption(item["path"])
    elif st.session_state.uploaded_file_names:
        # 兼容老状态（理论上不会走到）
        for name in st.session_state.uploaded_file_names:
            st.markdown(f"- 📄 **{name}**")
    else:
        st.info("尚未加载文档。请在左侧上传并构建。")

with right:
    st.subheader("💬 提问区（支持连续追问）")

    disabled = st.session_state.rag_session is None
    with st.form("ask_form", clear_on_submit=True):
        query = st.text_input(
            "请输入您的问题：",
            placeholder="例如：‘这份文档的关键结论是什么？请给出出处。’",
            disabled=disabled,
        )
        submitted = st.form_submit_button("🔍 提交问题", disabled=disabled, use_container_width=True)

    # 展示历史对话
    if st.session_state.chat_history_view:
        st.markdown("#### 🗂️ 历史对话")
        for qa in st.session_state.chat_history_view:
            st.markdown(f"**你：** {qa['q']}")
            st.markdown(f"**助手：** {qa['a']}")
            if qa.get("refs"):
                with st.expander("📚 参考资料（展开查看原文片段）"):
                    for ref in qa["refs"]:
                        label = ref.get("file_label") or ""
                        st.markdown(f"**{label} {ref['source']}**")
                        if ref.get("path"):
                            st.caption(ref["path"])
                        st.code(ref["snippet"])

    # 处理本次提问
    if submitted and query:
        with st.spinner("🤖 正在检索与生成 ..."):
            try:
                chain = st.session_state.rag_session["chain"]
                manifest = st.session_state.manifest

                # ✅ 现在需要把 manifest 传入
                result = get_rag_response(query, chain, manifest)

                # UI 展示
                st.markdown("### 🤖 模型的回答")
                st.write(result["answer"])  # 已自动拼接脚注（footer）

                st.markdown("### 📚 引用来源（可展开查看原文）")
                if result["references"]:
                    # 直接使用我们给的 file_label（与左侧清单一致）
                    for ref in result["references"]:
                        label = ref.get("file_label") or ""
                        with st.expander(f"{label} {ref['source']}"):
                            if ref.get("path"):
                                st.caption(ref["path"])
                            st.code(ref["snippet"])
                else:
                    st.info("未检索到可用的直接引用。")

                # 记录到对话历史
                st.session_state.chat_history_view.append(
                    {"q": query, "a": result["answer"], "refs": result["references"]}
                )
            except Exception as e:
                st.error(f"回答生成失败：{e}")
