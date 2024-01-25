#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/multi_modal/multi_modal_pdf_tables.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Multi-Modal on PDF's with tables.

# 
# One common challenge with RAG (`Retrieval-Augmented Generation`) involves handling PDFs that contain tables. Parsing tables in various formats can be quite complex.
# 
# However, Microsoft's newly released model, [`Table Transformer`](https://huggingface.co/microsoft/table-transformer-detection), offers a promising solution for detecting tables within images.
# 

# 
# The experiment is divided into the following parts and we compared those 4 options for extracting table information from PDFs:
# 
# 1. Retrieving relevant images (PDF pages) and sending them to GPT4-V to respond to queries.
# 2. Regarding every PDF page as an image, let GPT4-V do the image reasoning for each page. Build Text Vector Store index for the image reasonings. Query the answer against the `Image Reasoning Vectore Store`.
# 3. Using `Table Transformer` to crop the table information from the retrieved images and then sending these cropped images to GPT4-V for query responses.
# 4. Applying OCR on cropped table images and send the data to GPT4/ GPT-3.5 to answer the query.

# #### Setup

#('pip install llama-index qdrant_client pyMuPDF tools frontend git+https://github.com/openai/CLIP.git easyocr')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import io
from PIL import Image, ImageDraw
import numpy as np
import csv
import pandas as pd

from torchvision import transforms

from transformers import AutoModelForObjectDetection
import torch
import openai
import os
import fitz

device = "cuda" if torch.cuda.is_available() else "cpu"

OPENAI_API_TOKEN = "sk-<your-openai-api-token>"
openai.api_key = OPENAI_API_TOKEN

# Download Llama2 paper for the experiments.

#('wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "llama2.pdf"')

# Here we convert each of the Llama2 paper pdf page to images for indexing.

pdf_file = "llama2.pdf"

# Split the base name and extension
output_directory_path, _ = os.path.splitext(pdf_file)

if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)

# Open the PDF file
pdf_document = fitz.open(pdf_file)

# Iterate through each page and convert to an image
for page_number in range(pdf_document.page_count):
    # Get the page
    page = pdf_document[page_number]

    # Convert the page to an image
    pix = page.get_pixmap()

    # Create a Pillow Image object from the pixmap
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Save the image
    image.save(f"./{output_directory_path}/page_{page_number + 1}.png")

# Close the PDF file
pdf_document.close()

# Display the images.

from PIL import Image
import matplotlib.pyplot as plt
import os

image_paths = []
for img_path in os.listdir("./llama2"):
    image_paths.append(str(os.path.join("./llama2", img_path)))

def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(3, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break

plot_images(image_paths[9:12])

# ## Experiment-1: Retrieving relevant images (PDF pages) and sending them to GPT4-V to respond to queries.

# We will now index the images with `qdrant` vector store using our `MultiModalVectorStoreIndex` abstractions.

import qdrant_client
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index import VectorStoreIndex, StorageContext
from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.schema import ImageDocument

from llama_index.response.notebook_utils import #display_source_node
from llama_index.schema import ImageNode

from llama_index.multi_modal_llms.openai import OpenAIMultiModal

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=1500
)

# Build the Multi-Modal retriever.

# Read the images
documents_images = SimpleDirectoryReader("./llama2/").load_data()

# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_index")

text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

# Create the MultiModal index
index = MultiModalVectorStoreIndex.from_documents(
    documents_images,
    storage_context=storage_context,
)

retriever_engine = index.as_retriever(image_similarity_top_k=2)

from llama_index.indices.multi_modal.retriever import (
    MultiModalVectorIndexRetriever,
)

query = "Compare llama2 with llama1?"
assert isinstance(retriever_engine, MultiModalVectorIndexRetriever)
# retrieve for the query using text to image retrieval
retrieval_results = retriever_engine.text_to_image_retrieve(query)

# Check the retrieved results from Experiment 1

retrieved_images = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_images.append(res_node.node.metadata["file_path"])
    else:
        #display_source_node(res_node, source_length=200)

plot_images(retrieved_images)

retrieved_images

# Now let's send the retrieved images to GPT4-V for image reasoning

image_documents = [
    ImageDocument(image_path=image_path) for image_path in retrieved_images
]

response = openai_mm_llm.complete(
    prompt="Compare llama2 with llama1?",
    image_documents=image_documents,
)

print(response)

# ### Observation:
# 
# As you can see even though there is answer in the images, it is unable to give correct answer.

# ## Experiment-2: Parse each pdf page as an image and get table date directly from GPT4-V. Index tables data and then do text retrieval
# 
# Steps:
# - Extract and separate each PDF page as an image document
# - Let GPT4V identify table and extract table information from each PDF page
# - Index GPT4V understandings for each page into `Image Reasoning Vector Store`
# - Retrieve answer from this `Image Reasoning Vector Store`

# ### Load each pdf page as a image document

from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index import SimpleDirectoryReader

# put your local directore here
documents_images_v2 = SimpleDirectoryReader("./llama2/").load_data()

# ### Select one Image for Showcase the GPT4-V Table Reasoning Results

image = Image.open(documents_images_v2[15].image_path).convert("RGB")

plt.figure(figsize=(16, 9))
plt.imshow(image)

# ### Using this one Image of PDF file for GPT4-V understanding as an Example

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=1500
)

image_prompt = """
    Please load the table data and output in the json format from the image.
    Please try your best to extract the table data from the image.
    If you can't extract the table data, please summarize image and return the summary.
"""
response = openai_mm_llm.complete(
    prompt=image_prompt,
    image_documents=[documents_images_v2[15]],
)

print(response)

# ### Reuse the same prompt for all the pages in the PDF file

image_results = {}
for img_doc in documents_images_v2:
    try:
        image_table_result = openai_mm_llm.complete(
            prompt=image_prompt,
            image_documents=[img_doc],
        )
    except Exception as e:
        print(
            f"Error understanding for image {img_doc.image_path} from GPT4V API"
        )
        continue
    # image_results.append((image_document.image_path, image_table_result))
    image_results[img_doc.image_path] = image_table_result

# ### Build Text-Only Vector Store by Indexing the Image Understandings from GPT4-V

from llama_index.schema import Document

text_docs = [
    Document(
        text=str(image_results[image_path]),
        metadata={"image_path": image_path},
    )
    for image_path in image_results
]

from llama_index.indices.multi_modal.base import VectorStoreIndex
from llama_index.vector_stores import QdrantVectorStore
from llama_index import SimpleDirectoryReader, StorageContext

import qdrant_client
from llama_index import (
    SimpleDirectoryReader,
)

# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_mm_db_llama_v3")

llama_text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)

storage_context = StorageContext.from_defaults(vector_store=llama_text_store)

# Create the Text Vector index
index = VectorStoreIndex.from_documents(
    text_docs,
    storage_context=storage_context,
)

# ### Build Top k retrieval for Vector Store Index

MAX_TOKENS = 50
retriever_engine = index.as_retriever(
    similarity_top_k=3,
)
# retrieve more information from the GPT4V response
retrieval_results = retriever_engine.retrieve("Compare llama2 with llama1?")

from llama_index.response.notebook_utils import #display_source_node

retrieved_image = []
for res_node in retrieval_results:
    #display_source_node(res_node, source_length=1000)

# ### Perform query engine on the index and answer the question

query_engine = index.as_query_engine()
query_engine.query("Compare llama2 with llama1?")

# Observation:
# 
# * GPT4V is not stable to identify table and extract table content from image espcially when the image is mixed with tables, texts, and images. It is common in `PDF` format.
# * By splitting PDF files into single images and let GPT4V understand/summarize each PDF page as an single image, then build RAG based on PDF image to text index. This method *is not performing well* for this task.

# ## Experiment-3: Let's use microsoft `Table Transformer` to crop tables from the images and see if it gives the correct answer.

# Thanks to [Neils](https://twitter.com/NielsRogge). We have modified the utils from the [repository](https://huggingface.co/spaces/nielsr/tatr-demo) for our task.

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image

detection_transform = transforms.Compose(
    [
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

structure_transform = transforms.Compose(
    [
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# load table detection model
# processor = TableTransformerImageProcessor(max_size=800)
model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection", revision="no_timm"
).to(device)

# load table structure recognition model
# structure_processor = TableTransformerImageProcessor(max_size=1000)
structure_model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all"
).to(device)

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor(
        [width, height, width, height], dtype=torch.float32
    )
    return boxes

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [
        elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)
    ]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )

    return objects

def detect_and_crop_save_table(
    file_path, cropped_table_directory="./table_images/"
):
    image = Image.open(file_path)

    filename, _ = os.path.splitext(file_path.split("/")[-1])

    if not os.path.exists(cropped_table_directory):
        os.makedirs(cropped_table_directory)

    # prepare image for the model
    # pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(pixel_values)

    # postprocess to get detected tables
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)

    print(f"number of tables detected {len(detected_tables)}")

    for idx in range(len(detected_tables)):
        #   # crop detected table out of image
        cropped_table = image.crop(detected_tables[idx]["bbox"])
        cropped_table.save(f"./{cropped_table_directory}/{filename}_{idx}.png")

def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break

# Crop the tables

for file_path in retrieved_images:
    detect_and_crop_save_table(file_path)

# Read the cropped tables
image_documents = SimpleDirectoryReader("./table_images/").load_data()

# Generate response for the query.

response = openai_mm_llm.complete(
    prompt="Compare llama2 with llama1?",
    image_documents=image_documents,
)

print(response)

import glob

table_images_paths = glob.glob("./table_images/*.png")

plot_images(table_images_paths)

# ### Observation:
#  As demonstrated, the model now provides accurate answers. This aligns with our findings from the Chain of Thought (COT) experiments, where supplying GPT-4-V with specific image information significantly enhances its ability to deliver correct responses.

# ## Experiment-4: Applying OCR on cropped table images and send the data to GPT4/ GPT-3.5 to answer the query.
# 
# The experiment depends highly on the OCR model used. Here we are using easyocr with few modifications from [repository](https://huggingface.co/spaces/nielsr/tatr-demo).

import easyocr

reader = easyocr.Reader(["en"])

def detect_and_crop_table(image):
    # prepare image for the model
    # pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(pixel_values)

    # postprocess to get detected tables
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)

    # visualize
    # fig = visualize_detected_tables(image, detected_tables)
    # image = fig2img(fig)

    # crop first detected table out of image
    cropped_table = image.crop(detected_tables[0]["bbox"])

    return cropped_table

def recognize_table(image):
    # prepare image for the model
    # pixel_values = structure_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = structure_transform(image).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        outputs = structure_model(pixel_values)

    # postprocess to get individual elements
    id2label = structure_model.config.id2label
    id2label[len(structure_model.config.id2label)] = "no object"
    cells = outputs_to_objects(outputs, image.size, id2label)

    # visualize cells on cropped table
    draw = ImageDraw.Draw(image)

    for cell in cells:
        draw.rectangle(cell["bbox"], outline="red")

    return image, cells

def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry["label"] == "table row"]
    columns = [
        entry for entry in table_data if entry["label"] == "table column"
    ]

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x["bbox"][1])
    columns.sort(key=lambda x: x["bbox"][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [
            column["bbox"][0],
            row["bbox"][1],
            column["bbox"][2],
            row["bbox"][3],
        ]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({"column": column["bbox"], "cell": cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x["column"][0])

        # Append row information to cell_coordinates
        cell_coordinates.append(
            {
                "row": row["bbox"],
                "cells": row_cells,
                "cell_count": len(row_cells),
            }
        )

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x["row"][1])

    return cell_coordinates

def apply_ocr(cell_coordinates, cropped_table):
    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(cell_coordinates):
        row_text = []
        for cell in row["cells"]:
            # crop cell out of image
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            # apply OCR
            result = reader.readtext(np.array(cell_image))
            if len(result) > 0:
                text = " ".join([x[1] for x in result])
                row_text.append(text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[str(idx)] = row_text

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for idx, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + [
                "" for _ in range(max_num_columns - len(row_data))
            ]
        data[str(idx)] = row_data

    text = ", ".join(f"{key}={value}" for key, value in data.items())

    return text

# Extract table information from the table images.

table_text = ""

for table_image in table_images_paths:
    try:
        cropped_table = Image.open(table_image)
        image, cells = recognize_table(cropped_table)

        cell_coordinates = get_cell_coordinates_by_row(cells)

        text = apply_ocr(cell_coordinates, image)

        table_text = table_text + text + "\n"
    except:
        continue

print(table_text)

# As you can see the tablex extracted is not very accurate. (Each row represents a table information)
# 
# Let's now send it LLM to answer our query.

from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-4", temperature=0)

query = f"""Based on the following table information extracted, answer the query: \n

            TABLE INFORMATION:

            {table_text}

            Query:

            Compare llama2 with llama1?
            """
response = llm.complete(query)

print(response)

# ### Observation
# 
# Because we could not extract the table information from image, the answer is wrong.

# ## Conclusion
# 

