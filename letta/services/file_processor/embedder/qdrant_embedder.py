from typing import List, Optional

from letta.helpers.qdrant_client import LettaQdrantClient
from letta.llm_api.llm_client import LLMClient
from letta.log import get_logger
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import VectorDBProvider
from letta.schemas.passage import Passage
from letta.schemas.user import User
from letta.services.file_processor.embedder.base_embedder import BaseEmbedder

logger = get_logger(__name__)


class QdrantEmbedder(BaseEmbedder):
    def __init__(self, embedding_config: Optional[EmbeddingConfig] = None):
        super().__init__()
        self.vector_db_type = VectorDBProvider.QDRANT
        self.embedding_config = embedding_config or LettaQdrantClient.default_embedding_config
        self.qdrant_client = LettaQdrantClient()

    async def generate_embedded_passages(self, file_id: str, source_id: str, chunks: List[str], actor: User) -> List[Passage]:
        if not chunks:
            return []

        valid_chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
        if not valid_chunks:
            return []

        embedding_client = LLMClient.create(
            provider_type=self.embedding_config.embedding_endpoint_type,
            actor=actor,
        )
        embeddings = await embedding_client.request_embeddings(valid_chunks, self.embedding_config)

        passage_ids = [Passage.generate_id() for _ in valid_chunks]
        vector_size = len(embeddings[0]) if embeddings else self.embedding_config.embedding_dim

        import asyncio

        await asyncio.to_thread(
            self.qdrant_client.insert_archival_memories,
            archive_id=source_id,
            text_chunks=valid_chunks,
            passage_ids=passage_ids,
            organization_id=actor.organization_id,
            actor=actor,
            vectors=embeddings,
            vector_size=vector_size,
        )

        return [
            Passage(
                id=pid,
                text=text,
                file_id=file_id,
                source_id=source_id,
                embedding=embedding,
                embedding_config=self.embedding_config,
                organization_id=actor.organization_id,
            )
            for pid, text, embedding in zip(passage_ids, valid_chunks, embeddings)
        ]
