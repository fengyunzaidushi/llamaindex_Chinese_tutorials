#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/data_connectors/html_tag_reader.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # HTML Tag Reader

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ### Download HTML file

get_ipython().run_cell_magic('bash', '', 'wget -e robots=off --no-clobber --page-requisites \\\n  --html-extension --convert-links --restrict-file-names=windows \\\n  --domains docs.ray.io --no-parent --accept=html \\\n  -P data/ https://docs.ray.io/en/master/ray-overview/installation.html\n')

from llama_index.readers import HTMLTagReader

reader = HTMLTagReader(tag="section", ignore_no_id=True)
docs = reader.load_data(
    "data/docs.ray.io/en/master/ray-overview/installation.html"
)

for doc in docs:
    print(doc.metadata)

