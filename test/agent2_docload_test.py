# 在项目根目录下运行：python -m scripts.test_load（或直接在终端跑以下代码）

import sys
sys.path.insert(0, ".")   # 确保能 import scripts 目录

from scripts.build_knowledge_base import load_document

# ── 测试 1：加载 PDF ──────────────────────────────────────────
print("=== 测试 PDF1 加载 ===")
pdf_docs = load_document(r"E:\study\Python_Project\EduAgent\samples\sample1.pdf")   # 替换为你的 PDF 路径

# todo Previous trailer cannot be read: ("invalid literal for int() with base 10: b'/Root'",)、
#   如果遇到以上报错，说明pdf没有按正常流程转化成pdf，重新转一次即可。

print(pdf_docs)

print("=== 测试 PDF2 加载 ===")
pdf_docs = load_document(r"E:\study\Python_Project\EduAgent\samples\sample3.pdf")   # 替换为你的 PDF 路径
print(pdf_docs)

print("=== 测试 MD 加载 ===")
md_docs = load_document(r"E:\study\Python_Project\EduAgent\samples\sample2.md")   # 替换为你的 PDF 路径
print(md_docs)
