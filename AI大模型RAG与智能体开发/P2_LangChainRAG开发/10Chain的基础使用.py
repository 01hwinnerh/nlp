from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

#from_template只能输入单一字符串（纯文本 prompt)
#from_messages可以输入多轮对话消息列表（role + content）
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个边塞诗人，非常擅长写诗"),
        MessagesPlaceholder("history"),
        ("human","请再来一首唐诗")
    ]
)

history_data = [
    ("human", "你来写一首唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗好诗，再来一个"),
    ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦"),
]

model = ChatTongyi(model = "qwen3-max")

#组成链，要求每一个组件都是Runnable接口的子类
chain = chat_prompt_template | model        #一个组件的输出是下一个组件的输入

#通过链去调用invoke或stream
res = chain.invoke({"history":history_data})
print(res.content)

#通过stream流式输出
for chunk in chain.stream({"history":history_data}):
    print(chunk.content,end="",flush=True)




