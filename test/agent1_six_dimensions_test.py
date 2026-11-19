import asyncio, time
import backend.agents.resume.nodes as N
from backend.agents.resume.state import DimensionScore

class FakeLLM:
    """假模型：sleep 0.1s 模拟网络，返回固定分数（或抛错模拟失败）。"""
    def __init__(self, score, fail=False):
        self.score, self.fail = score, fail
    async def ainvoke(self, messages):
        await asyncio.sleep(0.1)
        if self.fail:
            raise RuntimeError("模拟维度失败")
        return DimensionScore(score=self.score)

state = {"raw_text": "简历原文 " * 50,
         "structured": {"name": "张三", "skills_list": ["Java", "Redis"], "projects": [{"name": "电商系统"}]}}

# todo 如果不想用假模型，就直接使用以下 ####################################
aa = asyncio.run(N.run_six_dimensions_node(state))
print(aa)
# todo 以上，使用真模型 ####################################

# ① 全部返回 80 分 → 验证并行（计时）+ 加权
# todo 在node中用到了 get_structured_llm，但是又不想调用真实模型，通过 猴子补丁 将里面的这个get_structured_llm模型替换成了FakeLLM(80)
#   get_structured_llm 有两个参数 agent_type, schema，通过匿名函数，表明不论这两个参数传什么，都返回 FakeLLM(80)
N.get_structured_llm = lambda agent_type, schema: FakeLLM(80)
t = time.perf_counter()
# todo 在 run_six_dimensions_node 方法中，用到了get_structured_llm，也就是FakeLLM(80)，而FakeLLM(80)的ainvoke 每次只会返回 DimensionScore(score=80)
r = asyncio.run(N.run_six_dimensions_node(state))
print("① 维度数:", len(r["dimension_scores"]))
print("   加权综合分:", r["weighted_score"])
print("   耗时:", round(time.perf_counter() - t, 2), "秒")

# ② 全部失败 → 验证降级
# todo 一直失败，就会触发 50分的最低分。
N.get_structured_llm = lambda agent_type, schema: FakeLLM(0, fail=True)
r2 = asyncio.run(N.run_six_dimensions_node(state))
print("② 全失败降级 → 加权分:", r2["weighted_score"], "| issues:", r2["dimension_scores"][0]["issues"])
