"""
知识库的基础构建代码
"""
import os
import config_data as config

import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def check_md5(md5_str:str):
    """检查是否当前字符串对应的 md5 已存在"""
    if not os.path.exists(config.md5_path):
        open(config.md5_path,'w',encoding="utf-8").close()
        return False
    else:
        for line in open(config.md5_path,'r',encoding='utf-8').readlines():
            line = line.strip()
            if md5_str == line:
                return True
        return False


def save_md5(md5_str:str):
    """将当前字符串 md5 保存起来"""
    with open(config.md5_path,'a',encoding='utf-8') as f:
        f.write(md5_str + '\n')


def str_to_md5(input_str:str,encoding='utf-8'):
    """将当前字符串转为 md5 字符串"""
    """先将字符串转换为bytes字节数组"""
    str_bytes = input_str.encode(encoding=encoding)

    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()

    return md5_hex



class KnowledgeBaseService(object):

    def __init__(self):

        os.makedirs(config.persist_dir,exist_ok=True)

        self.chroma = Chroma(  #存储向量的实例，Chroma对象
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_dir,

        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )
#将传入的字符串进行向量化，再存入数据库中
    def upload_by_str(self,data,filename):
        md5_hex = str_to_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        if len(data) >config.max_split_char_number:
            knowledge_chunks = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data] #因为split_text方法返回的就是个列表

        metadata = {
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"Fang Hao",
        }

        self.chroma.add_texts(       #自动按切好的块存入向量库，一块对应一个metadata
            knowledge_chunks,
            metadatas = [metadata for _ in knowledge_chunks],       #for _ in xxxx说明这个循环只要xxxx中的元素个数，不在乎内容
        )

        save_md5(md5_hex)
        return "[成功]内容已经成功载入向量库"



if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("周杰伦","testfile")
    print(r)