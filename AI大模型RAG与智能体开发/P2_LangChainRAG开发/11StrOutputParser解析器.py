from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

parser = StrOutputParser()
model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，请起名，仅告诉我名字无需其余任何多余内容"
)

chain = prompt | model | parser | model     #两个模型不能直接链接起来，因为模型输出的是AiMesseage，这不能当做模型的输入，但可以通过StrOutputParser解析器转换为字符串再输入
for chunk in chain.stream({"lastname":"张","gender":"女儿"}):
    print(chunk.content,end="",flush=True)
