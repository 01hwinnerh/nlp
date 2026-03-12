from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models.tongyi import ChatTongyi

str_parser = StrOutputParser()
model = ChatTongyi(model="qwen3-max")

my_func = RunnableLambda(lambda ai_msg: {"name":ai_msg.content})

first_prompt = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，请起名，仅告诉我名字无需其余任何多余内容"
)

second_prompt = PromptTemplate.from_template(
    "根据姓名{name}，再给我几个类似的名字，并且告诉我这个名字一般属于男孩还是女孩，为什么？"
)

# chain = first_prompt | model | my_func | second_prompt | model | str_parser     #两个模型之间的链接中间需要新的prompt，不然后面的模型不知道需要干什么
chain = first_prompt | model | (lambda ai_msg: {"name":ai_msg.content}) | second_prompt | model | str_parser        #可以直接插入匿名函数，但建议写成函数传入更清晰
for chunk in chain.stream({"lastname":"张","gender":"女儿"}):
    print(chunk,end="",flush=True)
