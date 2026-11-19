from backend.agents.resume.graph import build_resume_graph
g = build_resume_graph()
nodes = [n for n in g.get_graph().nodes if n not in ("__start__", "__end__")]
print("图编译成功:", g is not None)
print("节点数:", len(nodes))
print("节点:", nodes)
