
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


import nest_asyncio


nest_asyncio.apply()

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI

pdf_files = get_filelisform('./data','.pdf')
pdf_files

# Load all wiki documents
city_docs = []
for file in pdf_files:
    docs = SimpleDirectoryReader(
        input_files=[file]
    ).load_data()
    title = file.split(':')[0]
    docs[0].doc_id = title
    city_docs.extend(docs)


# LLM (gpt-3.5-turbo)
system_prompt="Always respond in Chinese"
chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo",api_base=api_base1,api_key=api_key1,system_prompt=system_prompt)
service_context = ServiceContext.from_defaults(llm=chatgpt, chunk_size=1024)

# default mode of building the index
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)
doc_summary_index = DocumentSummaryIndex.from_documents(
    city_docs,
    service_context=service_context,
    response_synthesizer=response_synthesizer,
    show_progress=True,
)

doc_summary_index.storage_context.persist("index")