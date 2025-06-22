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

# 2) Get the index handle (assuming it already exists)
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

        index.upsert(
            vectors=[{
                "id": vid,
                "text": combined,
                "metadata": {
                    "user_id": user_id,
                    "channel_id": channel_id,
                    "ai_model": ai_model,
                    "timestamp": ts,
                    "type": "conversation",
                    "message": message,
                    "response": response
                }
            }],
            namespace="default"
        )
        return vid

    async def store_memory(self, user_id: str, channel_id: str,
                           tag: str, content: str) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        text_content = f"Memory [{tag}]: {content}"
        vid = _gen_id(text_content, user_id, ts)

        index.upsert(
            vectors=[{
                "id": vid,
                "text": text_content,
                "metadata": {
                    "user_id": user_id,
                    "channel_id": channel_id,
                    "tag": tag,
                    "timestamp": ts,
                    "type": "memory",
                    "content": content
                }
            }],
            namespace="default"
        )
        return vid

    async def search_similar(self, query: str,
                             channel_id: str = None,
                             user_id: str = None,
                             content_type: list[str] = None,
                             top_k: int = 5) -> list[dict]:
        # Build filter with correct syntax
        filter_conditions = []

        if channel_id:
            filter_conditions.append({"channel_id": {"$eq": channel_id}})
        if user_id:
            filter_conditions.append({"user_id": {"$eq": user_id}})
        if content_type:
            filter_conditions.append({"type": {"$in": content_type}})

        # Combine conditions
        if len(filter_conditions) == 0:
            filter_dict = None
        elif len(filter_conditions) == 1:
            filter_dict = filter_conditions[0]
        else:
            filter_dict = {"$and": filter_conditions}

        res = index.query(
            data=query,  # For integrated embeddings, use 'data' parameter
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
            namespace="default"
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
        # Search for memories with this tag
        filter_dict = {
            "$and": [
                {"channel_id": {"$eq": channel_id}},
                {"tag": {"$eq": tag}},
                {"type": {"$eq": "memory"}}
            ]
        }

        res = index.query(
            data=tag,  # Search with the tag name
            top_k=100,
            include_metadata=True,
            filter=filter_dict,
            namespace="default"
        )

        ids = [m.id for m in res.matches]
        if ids:
            index.delete(ids=ids, namespace="default")
            return True
        return False

    async def get_memory_by_tag(self, channel_id: str, tag: str) -> dict:
        """Get a specific memory by its exact tag"""
        filter_dict = {
            "$and": [
                {"channel_id": {"$eq": channel_id}},
                {"tag": {"$eq": tag}},
                {"type": {"$eq": "memory"}}
            ]
        }

        res = index.query(
            data=tag,  # Use tag as query
            top_k=10,
            include_metadata=True,
            filter=filter_dict,
            namespace="default"
        )

        if res.matches:
            m = res.matches[0]
            return {
                "tag": m.metadata.get("tag"),
                "content": m.metadata.get("content"),
                "timestamp": m.metadata.get("timestamp")
            }
        return None

    async def list_memory_tags(self, channel_id: str) -> list[dict]:
        """List all memory tags in a channel"""
        # Search with a generic term
        filter_dict = {
            "$and": [
                {"channel_id": {"$eq": channel_id}},
                {"type": {"$eq": "memory"}}
            ]
        }

        res = index.query(
            data="memory project api auth config setup note important",  # Generic search terms
            top_k=100,
            include_metadata=True,
            filter=filter_dict,
            namespace="default"
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
        all_memories = {}

        # Try different search terms to catch various memories
        search_terms = [
            "memory", "project", "api", "auth", "config", "setup",
            "note", "important", "stack", "endpoint", "method",
            "React", "Node", "PostgreSQL", "JWT", "token"
        ]

        filter_dict = {
            "$and": [
                {"channel_id": {"$eq": channel_id}},
                {"type": {"$eq": "memory"}}
            ]
        }

        for term in search_terms:
            try:
                res = index.query(
                    data=term,
                    top_k=50,
                    include_metadata=True,
                    filter=filter_dict,
                    namespace="default"
                )

                for m in res.matches:
                    tag = m.metadata.get("tag")
                    if tag and tag not in all_memories:
                        all_memories[tag] = {
                            "tag": tag,
                            "content": m.metadata.get("content", ""),
                            "timestamp": m.metadata.get("timestamp", "")
                        }
            except Exception as e:
                print(f"Error searching for term '{term}': {e}")
                continue

        return sorted(all_memories.values(), key=lambda x: x["tag"])

    async def get_stats(self) -> dict:
        stats = index.describe_index_stats()
        # Get namespace-specific stats
        namespace_stats = stats.get('namespaces', {}).get('default', {})
        return {
            "total_vectors": namespace_stats.get('vector_count', 0),
            "dimension": stats.get('dimension', 0),
            "index_fullness": stats.get('index_fullness', 0)
        }