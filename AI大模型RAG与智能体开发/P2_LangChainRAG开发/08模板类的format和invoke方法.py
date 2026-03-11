from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate,ChatPromptTemplate

"""
PromptTemplate -> StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
FewShotPromptTemplate -> StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
ChatPromptTemplate -> BaseChatPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
"""

prompt_template = PromptTemplate.from_template("我的邻居叫{lastname}，他最喜欢{hobby}")

res1 = prompt_template.format(lastname="小明",hobby="打羽毛球")
print(res1,type(res1))

res2 = prompt_template.invoke({"lastname":"小红","hobby":"游泳"})
print(res2,type(res2))
