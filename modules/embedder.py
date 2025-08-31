# modules/embedder.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import os

def build_vectorstore(chunks: List, save_path="vectorstore/index"):
    """
    将分割后的文档 chunks 嵌入为向量，并保存为本地 FAISS 向量库
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese"
    )

    # 构建 FAISS 向量库
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # 如果目录不存在则创建
    os.makedirs(save_path, exist_ok=True)

    # 保存向量库（会生成 index.faiss 和 index.pkl）
    vectorstore.save_local(save_path)

    return vectorstore
