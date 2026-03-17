"""
用于将输入文本转向量，然后向量存储库中找到匹配的回答
返回的是一个向量检索器，方便入链
"""
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

import config_data as config


class VectorStoreService(object):

    def __init__(self,embedding):
        """

        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name = config.collection_name,
            embedding_function = self.embedding,
            persist_directory = config.persist_dir,
        )

    def get_retriever(self):
        """返回向量检索器，方便加入 chain """
        return self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})


if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()
    res = retriever.invoke("我体重100斤，尺码养护")
    print(res)
