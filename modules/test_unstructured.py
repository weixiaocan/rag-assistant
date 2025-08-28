from unstructured.partition.auto import partition
import unstructured_pytesseract.pytesseract as pytesseract
import os
from unstructured.partition.pdf import partition_pdf

# PDF文件路径
pdf_path = r"D:\材料\第二笔付款-发票.pdf"
pytesseract.tesseract_cmd = r"D:\ruanjian\tesseract-ocr\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"D:\ruanjian\tesseract-ocr\tessdata"

# 使用Unstructured加载并解析PDF文档
elements = partition_pdf(
    filename=pdf_path,
    content_type="application/pdf",
    strategy="hi_res"
)

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)
