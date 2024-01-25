#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/evaluation/semantic_similarity_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Embedding Similarity Evaluator

# This notebook shows the `SemanticSimilarityEvaluator`, which evaluates the quality of a question answering system via semantic similarity.
# 
# Concretely, it calculates the similarity score between embeddings of the generated answer and the reference answer.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.evaluation import SemanticSimilarityEvaluator

evaluator = SemanticSimilarityEvaluator()

# This evaluator only uses `response` and `reference`, passing in query does not influence the evaluation
# query = 'What is the color of the sky'

response = "The sky is typically blue"
reference = """The color of the sky can vary depending on several factors, including time of day, weather conditions, and location.

During the day, when the sun is in the sky, the sky often appears blue. 
This is because of a phenomenon called Rayleigh scattering, where molecules and particles in the Earth's atmosphere scatter sunlight in all directions, and blue light is scattered more than other colors because it travels as shorter, smaller waves. 
This is why we perceive the sky as blue on a clear day.
"""

result = await evaluator.aevaluate(
    response=response,
    reference=reference,
)

print("Score: ", result.score)
print("Passing: ", result.passing)  # default similarity threshold is 0.8

response = "Sorry, I do not have sufficient context to answer this question."
reference = """The color of the sky can vary depending on several factors, including time of day, weather conditions, and location.

During the day, when the sun is in the sky, the sky often appears blue. 
This is because of a phenomenon called Rayleigh scattering, where molecules and particles in the Earth's atmosphere scatter sunlight in all directions, and blue light is scattered more than other colors because it travels as shorter, smaller waves. 
This is why we perceive the sky as blue on a clear day.
"""

result = await evaluator.aevaluate(
    response=response,
    reference=reference,
)

print("Score: ", result.score)
print("Passing: ", result.passing)  # default similarity threshold is 0.8

# ### Customization

from llama_index.evaluation import SemanticSimilarityEvaluator
from llama_index import ServiceContext
from llama_index.embeddings import SimilarityMode

service_context = ServiceContext.from_defaults(embed_model="local")
evaluator = SemanticSimilarityEvaluator(
    service_context=service_context,
    similarity_mode=SimilarityMode.DEFAULT,
    similarity_threshold=0.6,
)

response = "The sky is yellow."
reference = "The sky is blue."

result = await evaluator.aevaluate(
    response=response,
    reference=reference,
)

print("Score: ", result.score)
print("Passing: ", result.passing)

# We note here that a high score does not imply the answer is always correct.  
# 
# Embedding similarity primarily captures the notion of "relevancy". Since both the response and reference discuss "the sky" and colors, they are semantically similar.
