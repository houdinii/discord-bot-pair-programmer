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
            "field_map": {"text": "text"}
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
            "text": combined,
            "user_id": user_id,
            "channel_id": channel_id,
            "ai_model": ai_model,
            "timestamp": ts,
            "type": "conversation",
            "message": message,
            "response": response
        }], namespace="default")
        return vid

    async def store_memory(self, user_id: str, channel_id: str,
                           tag: str, content: str) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        text_content = f"Memory [{tag}]: {content}"
        vid = _gen_id(text_content, user_id, ts)

        index.upsert_records(records=[{
            "_id": vid,
            "text": text_content,
            "user_id": user_id,
            "channel_id": channel_id,
            "tag": tag,
            "timestamp": ts,
            "type": "memory",
            "content": content
        }], namespace="default")
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
                text = md.get('text', md.get('content', 'N/A'))

            if length + len(text) > max_context_length:
                break
            parts.append(text)
            length += len(text)

        return "\n\n".join(parts) if parts else ""

    async def delete_by_tag(self, channel_id: str, tag: str) -> bool:
        # We need to search for memories with this tag
        # Since we can't do empty query, we'll search with the tag itself
        res = index.query(
            query=tag,  # Search with the tag name
            top_k=100,
            include_metadata=True,
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

    async def get_memory_by_tag(self, channel_id: str, tag: str) -> dict:
        """Get a specific memory by its exact tag"""
        # Search using the tag as query since we can't do empty queries
        res = index.query(
            query=tag,  # Use tag as query
            top_k=10,
            include_metadata=True,
            filter={
                "channel_id": {"$eq": channel_id},
                "tag": {"$eq": tag},
                "type": {"$eq": "memory"}
            }
        )

        if res.matches:
            m = res.matches[0]
            return {
                "tag": m.metadata.get("tag"),
                "content": m.metadata.get("content"),
                "timestamp": m.metadata.get("timestamp")
            }
        # noinspection PyTypeChecker
        return None

    async def list_memory_tags(self, channel_id: str) -> list[dict]:
        """List all memory tags in a channel"""
        # Since we can't do empty queries with integrated embeddings,
        # we'll search with a generic term and rely on filters
        res = index.query(
            query="memory",  # Generic search term
            top_k=100,
            include_metadata=True,
            filter={
                "channel_id": {"$eq": channel_id},
                "type": {"$eq": "memory"}
            }
        )

        memories = []
        seen_tags = set()

        for m in res.matches:
            tag = m.metadata.get("tag")
            if tag and tag not in seen_tags:
                seen_tags.add(tag)
                memories.append({
                    "tag": tag,
                    "content": m.metadata.get("content", ""),
                    "timestamp": m.metadata.get("timestamp", "")
                })

        return sorted(memories, key=lambda x: x["tag"])

    async def get_all_memories(self, channel_id: str) -> list[dict]:
        """Alternative method to get all memories using multiple queries"""
        # Try multiple generic queries to catch different memories
        all_memories = {}

        # Try different search terms to catch various memories
        search_terms = ["memory", "project", "api", "auth", "config", "setup", "note", "important"]

        for term in search_terms:
            try:
                res = index.query(
                    query=term,
                    top_k=50,
                    include_metadata=True,
                    filter={
                        "channel_id": {"$eq": channel_id},
                        "type": {"$eq": "memory"}
                    }
                )

                for m in res.matches:
                    tag = m.metadata.get("tag")
                    if tag and tag not in all_memories:
                        all_memories[tag] = {
                            "tag": tag,
                            "content": m.metadata.get("content", ""),
                            "timestamp": m.metadata.get("timestamp", "")
                        }
            except:
                continue

        return sorted(all_memories.values(), key=lambda x: x["tag"])

    async def get_stats(self) -> dict:
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
            "index_fullness": stats.get("index_fullness", 0)
        }