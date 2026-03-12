from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

str_parser = StrOutputParser()
json_parser = JsonOutputParser()

model = ChatTongyi(model="qwen3-max")

first_prompt = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，请起名，仅告诉我名字无需其余任何多余内容"
    "封装成Json格式给我，要求key是name，value是名字"
)

second_prompt = PromptTemplate.from_template(
    "根据姓名{name}，再给我几个类似的名字，并且告诉我这个名字一般属于男孩还是女孩，为什么？"
)

chain = first_prompt | model | json_parser | second_prompt | model | str_parser     #两个模型之间的链接中间需要新的prompt，不然后面的模型不知道需要干什么
for chunk in chain.stream({"lastname":"张","gender":"女儿"}):
    print(chunk,end="",flush=True)
