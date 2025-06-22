# services/vector_service.py

import os
import time
import hashlib
from datetime import datetime, timezone
from pinecone import Pinecone

from dotenv import load_dotenv
load_dotenv()

# 1) Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "discord-memory")

# 2) Ensure index exists, with integrated embedding
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",  # or "gcp"
        region="us-east-1",  # your chosen region
        embed={  # integrated-embedding config
            "model": "llama-text-embed-v2",
            "field_map": {"text": "chunk_text"}
        }
    )
    # wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

# 3) Grab the index handle
index = pc.Index(INDEX_NAME)


def _gen_id(text: str, user: str, ts: str) -> str:
    """Generate a stable unique ID for each record."""
    return hashlib.md5(f"{user}_{ts}_{text[:50]}".encode()).hexdigest()


class VectorService:
    async def store_conversation(self, user_id: str, channel_id: str,
                                 message: str, response: str,
                                 ai_model: str) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        combined = f"User: {message}\nAssistant: {response}"
        vid = _gen_id(combined, user_id, ts)

        index.upsert_records(records=[{
            "_id": vid,
            "chunk_text": combined,
            "user_id": user_id,
            "channel_id": channel_id,
            "ai_model": ai_model,
            "timestamp": ts,
            "type": "conversation",
            # Add these fields for search display
            "message": message,
            "response": response,
        }], namespace="")
        return vid

    async def store_memory(self, user_id: str, channel_id: str,
                           tag: str, content: str) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        text = f"Memory [{tag}]: {content}"
        vid = _gen_id(text, user_id, ts)

        index.upsert_records(records=[{
            "_id": vid,
            "chunk_text": text,
            "user_id": user_id,
            "channel_id": channel_id,
            "tag": tag,
            "timestamp": ts,
            "type": "memory"
        }], namespace="")
        return vid

    async def search_similar(self, query: str,
                             channel_id: str = None,
                             user_id: str = None,
                             content_type: list[str] = None,
                             top_k: int = 5) -> list[dict]:
        # Build filter
        filt = {}
        if channel_id:
            filt["channel_id"] = {"$eq": channel_id}
        if user_id:
            filt["user_id"] = {"$eq": user_id}
        if content_type:
            filt["type"] = {"$in": content_type}

        res = index.query(
            query=query,  # raw text → server does embedding
            top_k=top_k,
            include_metadata=True,
            filter=filt or None
        )

        return [
            {"id": m.id, "score": m.score, "metadata": m.metadata}
            for m in res.matches
        ]

    async def get_context_for_ai(self, query: str,
                                 channel_id: str,
                                 max_context_length: int = 3000) -> str:
        results = await self.search_similar(
            query=query,
            channel_id=channel_id,
            top_k=10
        )
        parts = []
        length = 0
        for m in results:
            if m["score"] < 0.7:
                continue
            md = m["metadata"]

            # Build context text based on type
            if md["type"] == "conversation":
                text = f"Previous conversation:\nUser: {md.get('message', 'N/A')}\nAssistant: {md.get('response', 'N/A')}"
            elif md["type"] == "memory":
                text = f"Memory [{md.get('tag', 'N/A')}]: {md.get('content', 'N/A')}"
            elif md["type"] == "document":
                text = f"Document [{md.get('filename', 'N/A')}]: {md.get('content', 'N/A')}"
            elif md["type"] == "github":
                text = f"GitHub [{md.get('repo_name', 'N/A')}]: {md.get('content', 'N/A')}"
            else:
                text = md.get('chunk_text', md.get('content', 'N/A'))

            if length + len(text) > max_context_length:
                break
            parts.append(text)
            length += len(text)

        return "\n\n".join(parts) if parts else ""

    async def delete_by_tag(self, channel_id: str, tag: str) -> bool:
        # Find all memory entries matching tag
        res = index.query(
            query="", top_k=1000, include_metadata=True,
            filter={
                "channel_id": {"$eq": channel_id},
                "tag": {"$eq": tag},
                "type": {"$eq": "memory"}
            }
        )
        ids = [m.id for m in res.matches]
        if ids:
            index.delete(ids=ids)
            return True
        return False

    async def get_stats(self) -> dict:
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
            "index_fullness": stats.get("index_fullness", 0)
        }
