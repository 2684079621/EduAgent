import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env.local')

from backend.agents.qa.graph import build_qa_graph
graph = build_qa_graph()
print('图编译成功')
print('节点列表：', list(graph.nodes.keys()))
