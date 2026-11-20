import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env.local')

from docx import Document
import tempfile, os

# 创建一个简单的测试 Word 文件
doc = Document()
doc.add_paragraph('第1题 什么是 Spring IOC？')
doc.add_paragraph('答：Spring IOC 是控制反转容器，负责管理 Bean 的生命周期')
doc.add_paragraph('')
doc.add_paragraph('第2题 下列说法正确的是')
doc.add_paragraph('答：A')
doc.add_paragraph('')
doc.add_paragraph('第3题 用python代码定义方法打印hello world')
# todo ===== 关键改动：逐行写入，保留缩进 =====
answer_lines = [
    "答：",
    "```python",
    "def print_hello():",
    '    print("Hello World")',
    "",
    "print_hello()",
    "```",
]
for line in answer_lines:
    doc.add_paragraph(line)

tmp = tempfile.mktemp(suffix='.docx')
doc.save(tmp)
doc.save("test.docx")

from backend.agents.exam.nodes import _sync_parse_word
result = _sync_parse_word(tmp)
print("整体的result:", result)
for q in result:
    print(f'题{q["question_no"]}: {repr(q["student_answer"][:60])}')
os.remove(tmp)


# 整体的 result:
#   [
#       {
#           'question_no': 1,
#           'header_text': '第1题 什么是 Spring IOC？',
#           'student_answer': 'Spring IOC 是控制反转容器，负责管理 Bean 的生命周期'
#       },
#       {
#           'question_no': 2,
#           'header_text': '第2题 下列说法正确的是',
#           'student_answer': 'A'},
#       {
#           'question_no': 3,
#           'header_text': '第3题 用python代码定义方法打印hello world',
#           'student_answer': 'def print_hello():\n    print("Hello World")\n\nprint_hello()',
#           'is_code': True
#       }
#   ]