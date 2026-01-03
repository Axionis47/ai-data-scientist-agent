"""
Vertex AI clients for production use.
Uses google-cloud-aiplatform or langchain-google-vertexai.
Configured via environment variables.
"""

import os


def get_vertex_config() -> dict:
    """Get Vertex AI configuration from environment variables."""
    return {
        "project": os.getenv("GCP_PROJECT", ""),
        "location": os.getenv("GCP_LOCATION", "us-central1"),
        "llm_model": os.getenv("VERTEX_LLM_MODEL", "gemini-1.5-flash"),
        "embed_model": os.getenv("VERTEX_EMBED_MODEL", "text-embedding-005"),
    }


class VertexEmbeddingsClient:
    """
    Vertex AI embeddings client using langchain-google-vertexai.
    """

    def __init__(self, project: str | None = None, location: str | None = None, model: str | None = None):
        config = get_vertex_config()
        self.project = project or config["project"]
        self.location = location or config["location"]
        self.model = model or config["embed_model"]
        self._client = None

    def _get_client(self):
        """Lazy initialization of the embeddings client."""
        if self._client is None:
            try:
                from langchain_google_vertexai import VertexAIEmbeddings
                self._client = VertexAIEmbeddings(
                    project=self.project,
                    location=self.location,
                    model_name=self.model,
                )
            except ImportError as e:
                raise ImportError(
                    "langchain-google-vertexai is required for Vertex AI. "
                    "Install with: pip install langchain-google-vertexai"
                ) from e
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using Vertex AI."""
        client = self._get_client()
        return client.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query using Vertex AI."""
        client = self._get_client()
        return client.embed_query(query)


class VertexLLMClient:
    """
    Vertex AI LLM client using langchain-google-vertexai.
    """

    def __init__(self, project: str | None = None, location: str | None = None, model: str | None = None):
        config = get_vertex_config()
        self.project = project or config["project"]
        self.location = location or config["location"]
        self.model = model or config["llm_model"]
        self._client = None

    def _get_client(self):
        """Lazy initialization of the LLM client."""
        if self._client is None:
            try:
                from langchain_google_vertexai import ChatVertexAI
                self._client = ChatVertexAI(
                    project=self.project,
                    location=self.location,
                    model_name=self.model,
                    temperature=0.1,
                    max_output_tokens=1024,
                )
            except ImportError as e:
                raise ImportError(
                    "langchain-google-vertexai is required for Vertex AI. "
                    "Install with: pip install langchain-google-vertexai"
                ) from e
        return self._client

    def generate(self, prompt: str) -> str:
        """Generate text using Vertex AI."""
        client = self._get_client()
        response = client.invoke(prompt)
        return response.content

