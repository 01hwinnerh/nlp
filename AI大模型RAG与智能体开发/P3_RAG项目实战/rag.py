from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class RagService(object):

    def __init__(self):

        #自己定义的向量检索器
        self.vector_service = VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name))

        self.prompt_template = ChatPromptTemplate(
            [
                ("system","以我提供的一直参考资料为主，简介和专业的回答用户问题。参考资料如下：{context}。"),
                ("system", "提供的历史记录如下："),
                MessagesPlaceholder("history"),
                ("user","我的问题如下：{input}"),
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""
        #双下划线 = 强私有，防继承冲突；单下划线 = 弱私有，仅提示
        retriever = self.vector_service.get_retriever()

        def print_prompt(prompt):
            print(prompt.to_string())
            print("=" * 20)
            return prompt

        def format_func(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value):
            return value["input"]

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["history"] = value["input"]["history"]
            new_value["context"] = value["context"]
            return new_value

        chain = (
            {"input":RunnablePassthrough(),"context": RunnableLambda(format_for_retriever) | retriever | format_func} | RunnableLambda(format_for_prompt_template) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        conversion_chain = RunnableWithMessageHistory(
            chain,  # 被附加历史功能的原始chain
            get_history,  # 获取历史会话的函数
            input_messages_key="input",  # 声明用户输入消息在模板中的占位符
            history_messages_key="history"  # 声明历史消息在模板中的占位符
        )

        return conversion_chain

if __name__ == '__main__':
    session_config = {
        "configurable":{
            "session_id":"user_001"
        }
    }
    res = RagService().chain.invoke({"input":"我体重180斤，尺码推荐"},session_config)
    print(res)


