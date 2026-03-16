"""
基于 Streamlit 完成 WEB 网页上传文档服务

当 web 页面元素发生更改，整个代码重新执行一遍，即无法保存状态，所以需要使用 streamlit中的 session_state （一个字典），他会一直存在，不会因为页面元素更改而改变
"""
import time

import streamlit as st
from knowledge_base import KnowledgeBaseService
#添加网页标题
st.title("知识库更新服务")

#file_uploader,用于上传文件的命令
uploaded_file = st.file_uploader(
    label="请上传txt文件",
    type=["txt"],
    accept_multiple_files=False,
)

#session_state是一个字典，其中存储的信息不会丢失
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_size = uploaded_file.size / 1024 #KB

    st.subheader(f"文件名：{file_name}")
    st.write(f"文件类型：{file_type}，文件大小：{file_size:.2f} KB")

    #getvalue->bytes(字节数组)->decode（“utf-8）
    text = uploaded_file.getvalue().decode("utf-8")

    with st.spinner("载入知识库中。。。"):   #在spinner内的代码执行过程中，会有一个转圈动画
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text,file_name)
        st.write(result)


