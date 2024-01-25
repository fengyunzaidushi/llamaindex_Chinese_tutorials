#!/usr/bin/env python
# coding: utf-8

# https://blog.csdn.net/weixin_42608414/article/details/135266719

from llama_index.readers.web import TrafilaturaWebReader
 
docs = TrafilaturaWebReader().load_data(
         ["https://baike.baidu.com/item/ChatGPT/62446358",
          "https://baike.baidu.com/item/恐龙/139019"]
)

from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import IndexNode
 
#创建文档切割器
node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
node_parser

base_nodes = node_parser.get_nodes_from_documents(docs)
 
len(base_nodes)

base_nodes[10].text

from llama_index.embeddings import resolve_embed_model
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
# from llama_index.llms import Gemini
 
 
#创建BAAI的embedding
embed_model = resolve_embed_model("local:BAAI/bge-small-zh-v1.5")
 
#创建OpenAI的llm
llm = OpenAI(model="gpt-3.5-turbo",system_prompt="总是用中文回答。")
 
#创建谷歌gemini的llm
#llm = Gemini()
 
#创建service_context 
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model
)

#创建index
base_index = VectorStoreIndex(base_nodes, service_context=service_context)
#创建检索器
base_retriever = base_index.as_retriever(similarity_top_k=2)
 
#检索相关文档
retrievals = base_retriever.retrieve(
    "恐龙是冷血动物吗？"
)

from llama_index.response.notebook_utils import #display_source_node
 
for n in retrievals:
    #display_source_node(n, source_length=1500)

sub_chunk_sizes = [128, 256, 512]
sub_node_parsers = [
    SimpleNodeParser.from_defaults(chunk_size=c,chunk_overlap=0) for c in sub_chunk_sizes
]
 
sub_node_parsers

all_nodes = []
for base_node in base_nodes:
    for n in sub_node_parsers:
        sub_nodes = n.get_nodes_from_documents([base_node])
        sub_inodes = [
            IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
        ]
        all_nodes.extend(sub_inodes)
 
    #添加父节点文档
    original_node = IndexNode.from_text_node(base_node, base_node.node_id)
    all_nodes.append(original_node)

all_nodes[:2]

all_nodes_dict = {n.node_id: n for n in all_nodes}
 
#查看特定节点
all_nodes_dict['548ce39b-1cc9-405f-935f-4532e72ea7d3']

from llama_index.retrievers import RecursiveRetriever
 
vector_index_chunk = VectorStoreIndex(
    all_nodes, service_context=service_context
)
 
vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=2)
 
retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True,
)

nodes = retriever_chunk.retrieve(
   "恐龙是冷血动物吗？"
)
for node in nodes:
    #display_source_node(node, source_length=2000)

from llama_index.query_engine import RetrieverQueryEngine

query_engine_chunk = RetrieverQueryEngine.from_args(
    retriever_chunk, service_context=service_context
)

#openai llm 的回答
response = query_engine_chunk.query(
    "恐龙是冷血动物吗？"
)
print(str(response))

