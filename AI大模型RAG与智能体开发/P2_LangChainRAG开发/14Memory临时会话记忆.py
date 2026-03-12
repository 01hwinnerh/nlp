from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

str_parser = StrOutputParser()
model = ChatTongyi(model="qwen3-max")

# prompt = PromptTemplate.from_template(
#     "你需要根据历史会话消息回应用户问题。对话历史：{chat_history}，用户提问：{input}，请回答"
# )

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","你需要根据历史会话消息回应用户问题。对话历史："),
        MessagesPlaceholder("chat_history"),
        ("human","请回到如下问题：{input}")
    ]
)

def print_prompt(full_prompt):
    print("="*20,full_prompt.to_string(),"="*20)
    return full_prompt

# base_chain = prompt | print_prompt | model | str_parser         #函数自动将prompt这一阶段的输出作为自己的输入了
base_chain = chat_prompt_template | print_prompt | model | str_parser         #chat_prompt更加规范标准

chat_history_store = {}     #可以存放多个会话ID的历史会话记录

def get_history(session_id):
    if session_id not in chat_history_store:
        chat_history_store[session_id] = InMemoryChatMessageHistory()
    return chat_history_store[session_id]

conversion_chain = RunnableWithMessageHistory(
    base_chain,     #被附加历史功能的原始chain
    get_history,    #获取历史会话的函数
    input_messages_key="input",      #声明用户输入消息在模板中的占位符
    history_messages_key="chat_history"     #声明历史消息在模板中的占位符
)

if __name__ == '__main__':
    #固定格式，添加Langchain的配置，为当前程序配置所属的session_id
    session_config = {
        "configurable":{
            "session_id":"user_001"
        }
    }

    res1 = conversion_chain.invoke({"input":"小明有两只猫"},session_config)
    print("第一次执行：",res1)
    res2 = conversion_chain.invoke({"input":"小刚有一只狗"},session_config)
    print("第二次执行：",res2)
    res3 = conversion_chain.invoke({"input":"小红有三只鸟"},session_config)
    print("第三次执行：",res3)
    res4 = conversion_chain.invoke({"input":"总共有几个人，平均一人几只宠物？"},session_config)
    print("第四次执行：",res4)


