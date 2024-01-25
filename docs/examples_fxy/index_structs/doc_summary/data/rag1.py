#!/usr/bin/env python
# coding: utf-8

#('pip3 install trafilatura -q')

from llama_index.readers.web import TrafilaturaWebReader
 
docs = TrafilaturaWebReader().load_data(
         ["https://baike.baidu.com/item/ChatGPT/62446358",
          "https://baike.baidu.com/item/恐龙/139019"]
)

print(docs[0].text[:1000])

response = query_engine.query("Rest of your query... \nRespond in Italian")

llm = OpenAI(system_prompt="Always respond in Italian.")

from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import IndexNode
 
#创建文档切割器
node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
node_parser

base_nodes = node_parser.get_nodes_from_documents(docs)
 
len(base_nodes)

base_nodes[0].

