

import os
import logging
import sys
from llama_index.indices.document_summary import (
            DocumentSummaryIndexLLMRetriever,
        )
from llama_index.query_engine import RetrieverQueryEngine
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext

from llama_index.prompts.base import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI
from llama_index.indices.document_summary import (
            DocumentSummaryIndexEmbeddingRetriever,
        )
from read import get_filelisform


def load_pdf(pdf_file):
    from pathlib import Path
    from llama_index import download_loader

    PDFReader = download_loader("PDFReader")

    loader = PDFReader()
    documents = loader.load_data(file=Path('./data/自律修炼手册.pdf'))

def get_docs(directory,format):
    pdf_files = get_filelisform(directory,format)
    # Load all wiki documents
    city_docs = []
    for file in pdf_files:
        docs = SimpleDirectoryReader(
            input_files=[file]
        ).load_data()
        title = file.split(':')[0]
        docs[0].doc_id = title
        city_docs.extend(docs)
    return city_docs,pdf_files


def get_summary(city_docs,pdf_files,chatgpt, persist_dir,context_window=4096):
    service_context = ServiceContext.from_defaults(llm=chatgpt,
                                                   chunk_size=1024,
                                                   context_window = context_window)

    new_summary_tmpl_str = (
        "Context information from multiple sources is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the information from multiple sources and not prior knowledge, "
        "answer the query below. Always respond in Chinese.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    summary_template = PromptTemplate(new_summary_tmpl_str)


    if not os.path.exists(persist_dir):
        # default mode of building the index
        response_synthesizer = get_response_synthesizer(service_context=service_context,
            response_mode="tree_summarize", use_async=True,summary_template=summary_template
        )


        #/opt/miniconda3/envs/py310_chat/lib/python3.10/site-packages/llama_index/response_synthesizers/base.py
        #/opt/miniconda3/envs/py310_chat/lib/python3.10/site-packages/llama_index/response_synthesizers/tree_summarize.py
        doc_summary_index = DocumentSummaryIndex.from_documents(
            city_docs,
            service_context=service_context,
            response_synthesizer=response_synthesizer,
            show_progress=True,
        )

        res = doc_summary_index.get_document_summary(pdf_files[0])
        print(f'res {res}')
        doc_summary_index.storage_context.persist(persist_dir)

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    # doc_summary_index = load_index_from_storage(storage_context=storage_context)
    doc_summary_index = load_index_from_storage(storage_context=storage_context,service_context=service_context)

    # res = doc_summary_index.get_document_summary(pdf_files[0])
    # print()
    # print(len(pdf_files),res)
    # questions= res.split('\n')[-3:]
    questions=["自律关键点在什么","Aurora Gold Mine的矿床有哪些？","如何在当众表达中凸显价值感、吸引注意力和保持连接度？","波士顿的气候如何影响当地居民和环境","What is the current status of technology and biotechnology in Toronto?"]

    ### LLM-based Retrieval

    # /mnt/sda/github/12yue/llama_index/llama_index/prompts/default_prompts.py line 406

    # NEW_CHOICE_SELECT_PROMPT_TMPL = (
    #     "A list of documents is shown below. Each document has a number next to it along "
    #     "with a summary of the document. A question is also provided. \n"
    #     "Respond with the numbers of the summary documents. \n"
    #     "you should consult to answer the question, in order of relevance, as well \n"
    #     "as the relevance score. The relevance score is a number from 1-10 based on "
    #     "how relevant you think the document is to the question.\n"
    #     "Do not include any documents that are not relevant to the question. \n"
    #     "The number of answers corresponds to the number of documents provided \n"
    #     "Here respond in English. \n"
    #     "Example format 1: \n"
    #     "Document 1:\n<summary of document 1>\n\n"
    #     "Document 2:\n<summary of document 2>\n\n"
    #     "...\n\n"
    #     "Document 10:\n<summary of document 10>\n\n"
    #     "Question: <question>\n"
    #     "Answer:\n"
    #     "Document: 9, Relevance: 7\n"
    #     "Document: 3, Relevance: 4\n"
    #     "Document: 7, Relevance: 3\n\n"
    #     "###"
    #     "Example format 2: \n"
    #     "Document 1:\n<summary of document 1>\n\n"
    #     "Question: <question>\n"
    #     "Answer:\n"
    #     "Document: 1, Relevance: 8\n\n"
    #      "###"
    #     "Let's try this now: \n\n"
    #     "{context_str}\n"
    #     "Question: {query_str}\n"
    #     "Answer:\n"
    # )
    # choice_select_prompt = PromptTemplate(NEW_CHOICE_SELECT_PROMPT_TMPL)
    # retriever = DocumentSummaryIndexLLMRetriever(
    #     doc_summary_index,
    #     choice_select_prompt=choice_select_prompt,
    #     choice_batch_size=5,
    #     choice_top_k=1,
    #     # format_node_batch_fn=None,
    #     # parse_choice_select_answer_fn=None,
    #     service_context=service_context
    # )
    #
    # retrieved_nodes = retriever.retrieve(questions[0])
    # print(len(retrieved_nodes))
    # print(retrieved_nodes[0].score)
    # print(retrieved_nodes[0].node.get_text())
    #
    # # configure response synthesizer
    # response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
    #
    # # assemble query engine
    # query_engine = RetrieverQueryEngine(
    #     retriever=retriever,
    #     response_synthesizer=response_synthesizer,
    # )
    #
    # # query
    # response = query_engine.query(questions[0])
    # print(response)

    ### High-level Querying
    # query_engine = doc_summary_index.as_query_engine(
    #     response_mode="tree_summarize", use_async=True
    # )
    # question = questions[0]
    # print(questions)
    # response = query_engine.query(question)
    #
    # print(response)

    ### Embedding-based Retrieval
    retriever = DocumentSummaryIndexEmbeddingRetriever(
        doc_summary_index,
        # similarity_top_k=1,
    )
    retrieved_nodes = retriever.retrieve(questions[0])
    lth = len(retrieved_nodes)
    print(f'lth {lth} nodes')
    res = retrieved_nodes[0].node.get_text()
    print(f'res {res}')
    # use retriever as part of a query engine

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    # query
    response = query_engine.query(questions[0])
    print(response)


if __name__ == '__main__':
    api_base1 = os.environ['openai_api_base1']
    api_key1 = os.environ['openai_api_key1']

    # LLM (gpt-3.5-turbo)
    models= ["gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-3.5-turbo-16k", "gpt-4-0613"]
    system_prompt = "Always answer in Chinese unless otherwise stated"
    chatgpt = OpenAI(temperature=0, model=models[2], api_base=api_base1, api_key=api_key1,
                     system_prompt=system_prompt)
    chatgpt2 = OpenAI(temperature=0, model=models[0], api_base=api_base1, api_key=api_key1)
    persist_dir = "index-pdf-zilv"
    # persist_dir = "index-pdf-gold"


    #/mnt/sda/github/12yue/llama_index/llama_index/indices/document_summary/retrievers.py line83
    #'Document 1, Relevance: 10' Doc: 9, Relevance: 9
    city_docs,pdf_files = get_docs('./data','手册.pdf')
    context_window= 16385
    get_summary(city_docs,pdf_files,chatgpt, persist_dir,context_window=context_window)

# #### LLM-based Retrieval
#
# from llama_index.indices.document_summary import (
#     DocumentSummaryIndexLLMRetriever,
# )
#
# retriever = DocumentSummaryIndexLLMRetriever(
#     doc_summary_index,
#     # choice_select_prompt=None,
#     # choice_batch_size=10,
#     # choice_top_k=1,
#     # format_node_batch_fn=None,
#     # parse_choice_select_answer_fn=None,
#     # service_context=None
# )
#
# retrieved_nodes = retriever.retrieve("What are the sports teams in Toronto?")
#
# print(len(retrieved_nodes))
#
# print(retrieved_nodes[0].score)
# print(retrieved_nodes[0].node.get_text())
#
# # use retriever as part of a query engine
# from llama_index.query_engine import RetrieverQueryEngine
#
# # configure response synthesizer
# response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
#
# # assemble query engine
# query_engine = RetrieverQueryEngine(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer,
# )
#
# # query
# response = query_engine.query("What are the sports teams in Toronto?")
# print(response)
#
# # #### Embedding-based Retrieval
#
# from llama_index.indices.document_summary import (
#     DocumentSummaryIndexEmbeddingRetriever,
# )
#
# retriever = DocumentSummaryIndexEmbeddingRetriever(
#     doc_summary_index,
#     # similarity_top_k=1,
# )
#
# retrieved_nodes = retriever.retrieve("What are the sports teams in Toronto?")
#
# len(retrieved_nodes)
#
# print(retrieved_nodes[0].node.get_text())
#
# # use retriever as part of a query engine
# from llama_index.query_engine import RetrieverQueryEngine
#
# # configure response synthesizer
# response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
#
# # assemble query engine
# query_engine = RetrieverQueryEngine(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer,
# )
#
# # query
# response = query_engine.query("What are the sports teams in Toronto?")
# print(response)

