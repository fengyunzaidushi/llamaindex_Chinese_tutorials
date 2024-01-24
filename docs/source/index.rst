Welcome to LlamaIndex's Chinese documentation!
===================================

LlamaIndex is a data framework for LLM-based applications to ingest, structure, and access private or domain-specific data. It’s available in Python (these docs) and Typescript.
LlamaIndex是一个数据框架，用于基于LLM的应用程序摄取、结构化和访问私有或领域特定的数据。它可在Python（本文档）和Typescript中使用。

🚀 Why LlamaIndex? 为什么选择LlamaIndex？
LLMs offer a natural language interface between humans and data. Widely available models come pre-trained on huge amounts of publicly available data like Wikipedia, mailing lists, textbooks, source code and more.
LLMs提供了人类和数据之间的自然语言接口。广泛可用的模型已经在大量公开可用的数据上进行了预训练，如维基百科、邮件列表、教科书、源代码等。

However, while LLMs are trained on a great deal of data, they are not trained on your data, which may be private or specific to the problem you’re trying to solve. It’s behind APIs, in SQL databases, or trapped in PDFs and slide decks.
然而，尽管LLMs是在大量数据上进行训练的，但它们并没有在您的数据上进行训练，这些数据可能是私有的或特定于您试图解决的问题。它们可能存在于API后面，在SQL数据库中，或者被困在PDF和幻灯片中。

You may choose to fine-tune a LLM with your data, but:
您可以选择使用您的数据来微调LLM，但是：

Training a LLM is expensive.
训练一个LLM是昂贵的。

Due to the cost to train, it’s hard to update a LLM with latest information.
由于培训成本高昂，更新LLM的最新信息很困难。

Observability is lacking. When you ask a LLM a question, it’s not obvious how the LLM arrived at its answer.
可观察性不足。当你向一个LLM提问时，很难看出LLM是如何得出答案的。

LlamaIndex takes a different approach called Retrieval-Augmented Generation (RAG). Instead of asking LLM to generate an answer immediately, LlamaIndex:
LlamaIndex采用一种不同的方法，称为检索增强生成（RAG）。而不是要求LLM立即生成答案，LlamaIndex：

retrieves information from your data sources first,
首先从您的数据源中检索信息

adds it to your question as context, and
将其作为背景添加到您的问题中，并返回译文

asks the LLM to answer based on the enriched prompt.
根据丰富的提示，要求LLM进行回答。

RAG overcomes all three weaknesses of the fine-tuning approach:
RAG克服了微调方法的三个弱点

There’s no training involved, so it’s cheap.
没有培训参与，所以很便宜。

Data is fetched only when you ask for them, so it’s always up to date.
数据只在您请求时获取，因此始终保持最新。

LlamaIndex can show you the retrieved documents, so it’s more trustworthy.
LlamaIndex可以显示您检索到的文档，因此更可靠。

LlamaIndex imposes no restriction on how you use LLMs. You can still use LLMs as auto-complete, chatbots, semi-autonomous agents, and more (see Use Cases on the left). It only makes LLMs more relevant to you.
LlamaIndex对于您如何使用LLMs没有任何限制。您仍然可以将LLMs用作自动完成、聊天机器人、半自主代理等（请参见左侧的使用案例）。它只会使LLMs对您更加相关。
