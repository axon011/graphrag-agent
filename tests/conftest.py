"""Keep the test suite hermetic: never reach for a real LLM/CLI provider."""
import os

os.environ["GRAPHRAG_LLM"] = "off"
