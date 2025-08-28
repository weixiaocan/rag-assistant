# 📚 RAG-Assistant

一个基于 **LangChain + HuggingFace Embeddings + FAISS + Streamlit** 的轻量级 **RAG（检索增强生成）文档问答系统**。 
支持多种文档解析（PDF、Word、PPT、HTML、Markdown、TXT），目标是构建一个支持多种文档上传，结合大模型完成语义问答的 AI 工具。。
扩展方向：1、尝试本地运行大模型；2、支持各种文件样式均可以上传和解析；3、向量数据库覆盖和更新问题；4、对话的可记忆性；7、如何提高检索质量和回答得更好
---

## ✨ 功能特性
- ✅ 支持多种文档格式：`.pdf` / `.docx` / `.pptx` / `.html` / `.md` / `.txt`  
- ✅ 文档解析：目前版本优先使用 **轻量解析**（PyMuPDF, docx2txt, python-pptx, BeautifulSoup），下一个版本更新支持 `unstructured`  
- ✅ 向量化检索：基于 `shibing624/text2vec-base-chinese` 模型 + FAISS 向量库  
- ✅ 问答能力：支持 GPT-4、DeepSeek 等大模型（可扩展）  
- ✅ 多轮对话：内置 `ConversationBufferMemory`，支持上下文追问  
- ✅ 引用追踪：答案末尾展示引用文档片段，保证可溯源性  

---

## 🛠️ 安装步骤

### 1. 克隆仓库
```bash
git clone https://github.com/weixiaocan/rag-assistant.git
cd rag-assistant

### 2. 创建虚拟环境（推荐 venv）
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac

### 3. 安装依赖
pip install -r requirements.txt

### 4. 配置环境变量
OPENAI_API_KEY=你的OpenAI密钥
DEEPSEEK_API_KEY=你的DeepSeek密钥（如果需要）

### 5.运行 Demo
streamlit run app.py

