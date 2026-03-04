import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from qdrant_client import QdrantClient, models

from letta.log import get_logger
from letta.schemas.enums import TagMatchMode
from letta.schemas.passage import Passage
from letta.schemas.embedding_config import EmbeddingConfig
from letta.settings import settings
from letta.constants import DEFAULT_EMBEDDING_CHUNK_SIZE
from pydantic import ValidationError

if TYPE_CHECKING:
    from letta.schemas.user import User as PydanticUser

logger = get_logger(__name__)


def should_use_qdrant() -> bool:
    return bool(settings.enable_qdrant and settings.qdrant_url)


def get_qdrant_client() -> Optional[QdrantClient]:
    if not should_use_qdrant():
        return None

    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def create_collection_if_not_exists(client: QdrantClient, collection_name: str, vector_size: int = 1536):
    if not client.collection_exists(collection_name):
        logger.info(f"Creating Qdrant collection {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def _convert_tags_to_qdrant_filter(tags: Optional[List[str]], tag_match_mode: TagMatchMode) -> Optional[models.Filter]:
    if not tags:
        return None

    conditions = [models.FieldCondition(key="tags", match=models.MatchValue(value=tag)) for tag in tags]

    if tag_match_mode == TagMatchMode.ALL:
        return models.Filter(must=conditions)
    else:  # ANY mode
        return models.Filter(should=conditions)


class LettaQdrantClient:
    default_embedding_config = EmbeddingConfig(
        embedding_model="text-embedding-3-small",
        embedding_endpoint_type="openai",
        embedding_endpoint="https://api.openai.com/v1",
        embedding_dim=1536,
        embedding_chunk_size=DEFAULT_EMBEDDING_CHUNK_SIZE,
    )

    def __init__(self):
        self.client = get_qdrant_client()
        if not self.client:
            raise ValueError("Qdrant integration is not enabled or improperly configured.")

    def insert_archival_memories(
        self,
        archive_id: str,
        text_chunks: List[str],
        passage_ids: List[str],
        organization_id: str,
        actor: "PydanticUser",
        vectors: Optional[List[List[float]]] = None,
        tags: Optional[List[str]] = None,
        vector_size: int = 1536,
        **kwargs,
    ) -> List[str]:
        collection_name = archive_id

        if not text_chunks or not vectors:
            return []

        create_collection_if_not_exists(self.client, collection_name, vector_size=vector_size)

        points = []
        for i, (text, p_id) in enumerate(zip(text_chunks, passage_ids)):
            payload = {
                "text": text,
                "passage_id": p_id,
                "organization_id": organization_id,
            }
            if tags:
                payload["tags"] = tags
            payload.update(kwargs)

            vector = vectors[i] if vectors else [0.0] * vector_size

            try:
                point_id = str(uuid.UUID(p_id))
            except ValueError:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, p_id))

            points.append(
                models.PointStruct(
                    id=point_id,
                    payload=payload,
                    vector=vector,
                )
            )

        self.client.upload_points(collection_name=collection_name, points=points)

        return passage_ids

    def query_passages(
        self,
        archive_id: str,
        query_vector: List[float],
        top_k: int = 5,
        tags: Optional[List[str]] = None,
        tag_match_mode: TagMatchMode = TagMatchMode.ANY,
    ) -> List[Tuple[Passage, float, Dict[str, Any]]]:
        collection_name = archive_id

        if not self.client.collection_exists(collection_name):
            return []

        qdrant_filter = _convert_tags_to_qdrant_filter(tags, tag_match_mode)

        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results = []
        for scored_point in search_result.points:
            payload = scored_point.payload or {}

            passage_id = payload.get("passage_id", str(scored_point.id))
            text = payload.get("text", "")

            try:
                passage = Passage(
                    id=passage_id,
                    text=text,
                    organization_id=payload.get("organization_id", None),
                    embedding=[],
                    embedding_config=self.default_embedding_config,
                )
            except ValidationError as e:
                logger.error(f"Failed to reconstruct Passage from Qdrant payload: {e}")
                continue

            score = scored_point.score
            results.append((passage, score, payload))

        return results

    def delete_passages(self, archive_id: str, passage_ids: List[str]):
        collection_name = archive_id

        if not self.client.collection_exists(collection_name):
            return

        # Qdrant only allows UUIDs and +ve integers and point IDs.
        # Ref: https://qdrant.tech/documentation/concepts/points/#point-ids
        point_ids = []
        for p_id in passage_ids:
            try:
                point_ids.append(str(uuid.UUID(p_id)))
            except ValueError:
                point_ids.append(str(uuid.uuid5(uuid.NAMESPACE_OID, p_id)))

        self.client.delete(
            collection_name=collection_name,
            points_selector=point_ids,
        )

    def delete_all_passages(self, archive_id: str):
        if self.client.collection_exists(archive_id):
            self.client.delete_collection(archive_id)
