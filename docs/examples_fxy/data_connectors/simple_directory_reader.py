#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/data_connectors/simple_directory_reader.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Simple Directory Reader

# The `SimpleDirectoryReader` is the most commonly used data connector that _just works_.  
# Simply pass in a input directory or a list of files.  
# It will select the best file reader based on the file extensions.  

# ### Get Started

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay1.txt'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay2.txt'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay3.txt'")

from llama_index import SimpleDirectoryReader

# Load specific files 

reader = SimpleDirectoryReader(
    input_files=["./data/paul_graham/paul_graham_essay1.txt"]
)

docs = reader.load_data()
print(f"Loaded {len(docs)} docs")

# Load all (top-level) files from directory

reader = SimpleDirectoryReader(input_dir="./data/paul_graham/")

docs = reader.load_data()
print(f"Loaded {len(docs)} docs")

# Load all (recursive) files from directory 

#("mkdir -p 'data/paul_graham/nested'")
#('echo "This is a nested file" > \'data/paul_graham/nested/nested_file.md\'')

# only load markdown files
required_exts = [".md"]

reader = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=required_exts,
    recursive=True,
)

docs = reader.load_data()
print(f"Loaded {len(docs)} docs")

# Create an iterator to load files and process them as they load

reader = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,
)

all_docs = []
for docs in reader.iter_data():
    for doc in docs:
        # do something with the doc
        doc.text = doc.text.upper()
        all_docs.append(doc)

print(len(all_docs))

# ## Full Configuration

# This is the full list of arguments that can be passed to the `SimpleDirectoryReader`:
# 
# ```python
# class SimpleDirectoryReader(BaseReader):
#     """Simple directory reader.
# 
#     Load files from file directory. 
#     Automatically select the best file reader given file extensions.
# 
# 
#     Args:
#         input_dir (str): Path to the directory.
#         input_files (List): List of file paths to read
#             (Optional; overrides input_dir, exclude)
#         exclude (List): glob of python file paths to exclude (Optional)
#         exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
#         encoding (str): Encoding of the files.
#             Default is utf-8.
#         errors (str): how encoding and decoding errors are to be handled,
#                 see https://docs.python.org/3/library/functions.html#open
#         recursive (bool): Whether to recursively search in subdirectories.
#             False by default.
#         filename_as_id (bool): Whether to use the filename as the document id.
#             False by default.
#         required_exts (Optional[List[str]]): List of required extensions.
#             Default is None.
#         file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file
#             extension to a BaseReader class that specifies how to convert that file
#             to text. If not specified, use default from DEFAULT_FILE_READER_CLS.
#         num_files_limit (Optional[int]): Maximum number of files to read.
#             Default is None.
#         file_metadata (Optional[Callable[str, Dict]]): A function that takes
#             in a filename and returns a Dict of metadata for the Document.
#             Default is None.
# """
# ```
# 
