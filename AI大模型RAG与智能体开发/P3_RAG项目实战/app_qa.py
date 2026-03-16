import time
from rag import RagService
import streamlit as st
import config_data as config

st.title("智能客服")
st.divider()

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮你的？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    # 显示用户输入的消息，并保存到聊天历史中
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 用于缓存 AI 流式回复的每个片段，以便后续拼接成完整回复
    ai_res_list = []

    with st.spinner("AI思考中..."):
        # 调用 RAG 链的流式接口，获取生成结果的迭代器
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)


        # 定义一个捕获函数：在流式输出的同时，将每个 chunk 存入缓存列表
        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)  # 保存 chunk 到列表
                yield chunk  # 同时将其传递给 write_stream 实现实时显示


        # 使用 Streamlit 的 write_stream 方法实现流式显示
        st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))

        # 将所有缓存的 chunk 拼接成完整字符串，并保存到聊天历史中，用于后续多轮对话
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})