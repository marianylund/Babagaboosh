import logging
import sys
from dotenv import load_dotenv
import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext
)
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
load_dotenv()

# gpt_4_context = ServiceContext.from_defaults(
#     llm=OpenAI(model="gpt-3.5-turbo-1106", temperature=0.3),
#     context_window=2048
# )

# documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(
#     documents, service_context=gpt_4_context
# )


gpt_4_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo-1106", temperature=0.3),
    context_window=2048, chunk_size=1000
)
# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(
        documents, service_context=gpt_4_context
    )
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context, service_context=gpt_4_context)

# query_engine = index.as_query_engine(similarity_top_k=2)
# response = query_engine.query("How does the author define Peepeepoooo?")
    
chat_engine = index.as_chat_engine(chat_mode="best", verbose=True, similarity_top_k=2)
#response = chat_engine.chat("How does the author define Peepeepoooo?")
#print(response)
chat_engine.chat_repl()