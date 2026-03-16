
md5_path = "./md5.text"

#Chroma
collection_name = "rag"
persist_dir = "./chroma_db"

#Spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n","\n",",",".","!","?"," ","，","。","！","？"]
max_split_char_number = 1000


#
similarity_threshold = 2        #表示的是每次返回的 TOP K 检索
embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

session_config = {
    "configurable": {
        "session_id": "user_001"
    }
}