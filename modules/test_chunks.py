from unstructured.partition.pdf import partition_pdf
import unstructured_pytesseract.pytesseract as pytesseract
import os
from langchain.schema import Document
from collections import Counter

# -----------------------------
# 配置 OCR
# -----------------------------
pdf_path = r"D:\材料\Ai Engineer Resume.pdf"
pytesseract.tesseract_cmd = r"D:\ruanjian\tesseract-ocr\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"D:\ruanjian\tesseract-ocr\tessdata"

# -----------------------------
# 1. 解析 PDF
# -----------------------------
elements = partition_pdf(
    filename=pdf_path,
    content_type="application/pdf",
    strategy="hi_res"   # 高精度，能拿到坐标
)

print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# -----------------------------
# 2. 转换为 Document
# -----------------------------
basename = os.path.basename(pdf_path)
apath = os.path.abspath(pdf_path)
ext = os.path.splitext(pdf_path)[-1].lower()

docs = []
for i, el in enumerate(elements, 1):
    text = (getattr(el, "text", None) or "").strip()
    if not text:
        continue
    docs.append(
        Document(
            page_content=text,
            metadata={
                "source": basename,
                "path": apath,
                "ext": ext,
                "category": str(el.category),  # 元素类型，如 Title / NarrativeText / Table ...
                "element_index": i,
            }
        )
    )

print(f"转换完成: {len(docs)} 个 Document")

# -----------------------------
# # 3. 元素类型统计
# # -----------------------------
# types = Counter(e.category for e in elements)
# print(f"元素类型: {dict(types)}")

# # -----------------------------
# # 4. 显示部分 Document
# # -----------------------------
# for i, d in enumerate(docs[:5], 1):  # 只看前5个
#     print(f"[Document {i}]")
#     print("内容:", d.page_content[:200])  # 只截前200字符
#     print("元数据:", d.metadata)
#     print("=" * 80)



from collections import defaultdict
from langchain.schema import Document

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

merged_docs = merge_docs_by_file(docs)  # docs 是 elements→Document 的列表


# -----------------------------
# 5. 测试最简单的固定大小分块
# -----------------------------
from langchain_text_splitters import CharacterTextSplitter


splitter1 = CharacterTextSplitter(chunk_size=100, chunk_overlap=50,separator="\n" )
chunks1 = splitter1.split_documents(merged_docs)
print(f"字符分块: {len(chunks1)} 个 chunk")

# 展示一个分块结果
print("\n示例分块结果:")
for c in chunks1:
    print(c.page_content)
    # print(c.metadata)
    print("-" * 60)

# 6.测试递归字符分块

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 定义递归字符分块器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 按照从大到小的优先级切分
)

# 执行分块
chunks = splitter.split_documents(merged_docs)

print(f"递归字符分块: {len(chunks)} 个 chunk")

# 展示一个分块结果
print("\n示例分块结果:")
for c in chunks:
    print(c.page_content)
    # print(c.metadata)
    print("-" * 60)

# 7、测试语义分块
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 定义语义分块器
splitter = SemanticChunker(embeddings,breakpoint_threshold_amount=0.8,min_chunk_size=50 )

# 执行分块
chunks = splitter.split_documents(merged_docs)

print(f"语义分块: {len(chunks)} 个 chunk")

# 展示一个分块结果
print("\n示例分块结果:")
for c in chunks:
    print(c.page_content)
    # print(c.metadata)
    print("-" * 60)