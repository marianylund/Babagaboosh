from llama_index.llms import MockLLM
from llama_index import MockEmbedding

import tiktoken
from llama_index.callbacks import CallbackManager, TokenCountingHandler

from llama_index import ServiceContext, set_global_service_context
from llama_index import VectorStoreIndex, SimpleDirectoryReader


llm = MockLLM(max_tokens=256)
embed_model = MockEmbedding(embed_dim=1536)

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo-1106").encode
)

cost_per_token = 0.0020 / 1000 # $0.0020 per 1k tokens very approximate, see more at https://openai.com/pricing/

callback_manager = CallbackManager([token_counter])

set_global_service_context(
    ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, callback_manager=callback_manager
    )
)

documents = SimpleDirectoryReader("./data/").load_data()

index = VectorStoreIndex.from_documents(documents)

print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)

# reset counts
token_counter.reset_counts()

query_engine = index.as_query_engine()
response = query_engine.query("query")

print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
    "Total LLM Token Cost: ",
    token_counter.total_llm_token_count * cost_per_token,
    "$\n",
)