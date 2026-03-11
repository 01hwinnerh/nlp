from huggingface_hub.cli.models import models_ls
from langchain_community.llms.tongyi import Tongyi
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

prompt_text = chat_prompt_template.invoke({"history":history_data})
print(prompt_text)      #.string可以看到文本，但是模型无法识别 system/human/ai 角色，效果变差

from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model = "qwen3-max")
res = model.invoke(input=prompt_text)
print(res.content)

res = model.stream(prompt_text)     #不需要input = 也可以运行
for chunk in res:
    print(chunk.content,end="",flush=True)
