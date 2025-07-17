# [Meshagent](https://www.meshagent.com)

## MeshAgent OpenAI
The ``meshagent.openai`` package provides adapters to integrate OpenAI models with MeshAgent tools and agents. 

### Completions Adapter and Responses Adapter
MeshAgent supports both the OpenAI Chat Completions API and Responses API. It is recommended to use the Responses adapter given the newer OpenAI models and functionality use the Responses adapter.

- ``OpenAICompletionsAdapter``: wraps the OpenAI Chat Completions API. It turns Toolkit objects into OpenAI-style tool definitions and processes tool calls appropriately.
- ``OpenAIResponsesAdapter``: wraps the newer OpenAI Responses API. It collects tools, handles streaming events, and provides callbacks for advanced features like image generation or web search. 

```Python Python
from meshagent.openai import OpenAIResponsesAdapter
from openai import AsyncOpenAI

# Use an OpenAI client inside a MeshAgent LLMAdapter
adapter = OpenAIResponsesAdapter(client=AsyncOpenAI(api_key="sk-..."))
```

### Tool Response Adapter
The ``OpenAICompletionsToolResponseAdapter`` and ``OpenAIResponsesToolResponseAdapter``convert a tool's structured response into plain text or JSOn that can beinserted into an OpenAI chat context. 

---
### Learn more about MeshAgent on our website or check out the docs for additional examples!

**Website**: [www.meshagent.com](https://www.meshagent.com/)

**Documentation**: [docs.meshagent.com](https://docs.meshagent.com/)

---