#!/usr/bin/env python
# coding: utf-8

# # Deplot Reader Demo
# 

#('pip install llama-hub')

from llama_hub.file.image_deplot.base import ImageTabularChartReader
from llama_index import SummaryIndex
from llama_index.response.notebook_utils import #display_response
from pathlib import Path

loader = ImageTabularChartReader(keep_image=True)

# ## Load Protected Waters Chart
# 
# This chart shows the percentage of marine territorial waters that are protected for each country.

documents = loader.load_data(file=Path("./marine_chart.png"))

print(documents[0].text)

summary_index = SummaryIndex.from_documents(documents)
response = summary_index.as_query_engine().query(
    "What is the difference between the shares of Greenland and the share of"
    " Mauritania?"
)

#display_response(response, show_source=True)

# ## Load Pew Research Chart
# 
# Here we load in a Pew Research chart showing international views of the US/Biden.
# 
# Source: https://www.pewresearch.org/global/2023/06/27/international-views-of-biden-and-u-s-largely-positive/

documents = loader.load_data(file=Path("./pew1.png"))

print(documents[0].text)

summary_index = SummaryIndex.from_documents(documents)
response = summary_index.as_query_engine().query(
    "What percentage says that the US contributes to peace and stability?"
)

#display_response(response, show_source=True)

