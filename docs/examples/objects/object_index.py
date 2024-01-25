#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/objects/object_index.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # The `ObjectIndex` Class
# 
# The `ObjectIndex` class is one that allows for the indexing of arbitrary Python objects. As such, it is quite flexible and applicable to a wide-range of use cases. As examples:
# - [Use an `ObjectIndex` to index Tool objects to then be used by an agent.](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_retrieval.html#building-an-object-index)
# - [Use an `ObjectIndex` to index a SQLTableSchema objects](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo.html#part-2-query-time-retrieval-of-tables-for-text-to-sql)
# 
# To construct an `ObjectIndex`, we require an index as well as another abstraction, namely `ObjectNodeMapping`. This mapping, as its name suggests, provides the means to go between node and the associated object, and vice versa. Alternatively, there exists a `from_objects()` class method, that can conveniently construct an `ObjectIndex` from a set of objects.
# 

from llama_index import VectorStoreIndex
from llama_index.objects import ObjectIndex, SimpleObjectNodeMapping

# some really arbitrary objects
obj1 = {"input": "Hey, how's it going"}
obj2 = ["a", "b", "c", "d"]
obj3 = "llamaindex is an awesome library!"
arbitrary_objects = [obj1, obj2, obj3]

# object-node mapping
obj_node_mapping = SimpleObjectNodeMapping.from_objects(arbitrary_objects)
nodes = obj_node_mapping.to_nodes(arbitrary_objects)

# object index
object_index = ObjectIndex(
    index=VectorStoreIndex(nodes=nodes), object_node_mapping=obj_node_mapping
)

# ### As a retriever
# With the `object_index` in hand, we can use it as a retriever, to retrieve against the index objects.

object_retriever = object_index.as_retriever(similarity_top_k=1)
object_retriever.retrieve("llamaindex")

# ## Persisting `ObjectIndex`
# 
# When it comes to persisting the `ObjectIndex`, we have to handle both the index as well as the object-node mapping. Persisting the index is straightforward and can be handled by usual means (e.g., see this [guide](https://docs.llamaindex.ai/en/stable/module_guides/storing/save_load.html#persisting-loading-data)). However, it's a bit of a different story when it comes to persisting the `ObjectNodeMapping`. Since we're indexing aribtrary Python objects with the `ObjectIndex`, it may be the case (and perhaps more often than we'd like), that the arbitrary objects are not serializable. In those cases, you can persist the index, but the user would have to maintain a way to re-construct the `ObjectNodeMapping` to be able to re-construct the `ObjectIndex`. For convenience, there are the `persist` and `from_persist_dir` methods on the `ObjectIndex` that will attempt to persist and load a previously saved `ObjectIndex`, respectively.

# ### Happy example

# persist to disk (no path provided will persist to the default path ./storage)
object_index.persist()

# re-loading (no path provided will attempt to load from the default path ./storage)
reloaded_object_index = ObjectIndex.from_persist_dir()

reloaded_object_index._object_node_mapping.obj_node_mapping

object_index._object_node_mapping.obj_node_mapping

# ### Example of when it doesn't work

from llama_index.tools.function_tool import FunctionTool
from llama_index.indices.list.base import SummaryIndex
from llama_index.objects import SimpleToolNodeMapping

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

object_mapping = SimpleToolNodeMapping.from_objects([add_tool, multiply_tool])
object_index = ObjectIndex.from_objects(
    [add_tool, multiply_tool], object_mapping
)

# trying to persist the object_mapping directly will raise an error
object_mapping.persist()

# try to persist the object index here will throw a Warning to the user
object_index.persist()

# **In this case, only the index has been persisted.** In order to re-construct the `ObjectIndex` as mentioned above, we will need to manually re-construct `ObjectNodeMapping` and supply that to the `ObjectIndex.from_persist_dir` method.

reloaded_object_index = ObjectIndex.from_persist_dir(
    object_node_mapping=object_mapping  # without this, an error will be thrown
)

