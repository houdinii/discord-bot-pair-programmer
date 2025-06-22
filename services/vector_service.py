# services/vector_service.py

import hashlib
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "discord-memory")
NAMESPACE = "default"  # Use a consistent namespace

# Get the index handle (assuming it already exists)
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

        # Upsert with namespace
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
            namespace=NAMESPACE
        )
        return vid

    async def store_memory(self, user_id: str, channel_id: str,
                           tag: str, content: str) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        text_content = f"Memory [{tag}]: {content}"
        vid = _gen_id(text_content, user_id, ts)

        # Upsert with namespace
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
            namespace=NAMESPACE
        )
        return vid

    async def search_similar(self, query: str,
                             channel_id: Optional[str] = None,
                             user_id: Optional[str] = None,
                             content_type: Optional[List[str]] = None,
                             top_k: int = 5) -> List[Dict[str, Any]]:
        # Build filter with proper typing
        filter_dict: Dict[str, Any] = {}

        if channel_id:
            filter_dict["channel_id"] = channel_id
        if user_id:
            filter_dict["user_id"] = user_id
        if content_type:
            filter_dict["type"] = {"$in": content_type}

        # Query with namespace
        res = index.query(
            data=query,  # The text to embed and search
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None,
            namespace=NAMESPACE
        )

        return [
            {"id": m["id"], "score": m["score"], "metadata": m.get("metadata", {})}
            for m in res["matches"]
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
            if md.get("type") == "conversation":
                text = f"Previous conversation:\nUser: {md.get('message', 'N/A')}\nAssistant: {md.get('response', 'N/A')}"
            elif md.get("type") == "memory":
                text = f"Memory [{md.get('tag', 'N/A')}]: {md.get('content', 'N/A')}"
            elif md.get("type") == "document":
                text = f"Document [{md.get('filename', 'N/A')}]: {md.get('content', 'N/A')}"
            elif md.get("type") == "github":
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
            "channel_id": channel_id,
            "tag": tag,
            "type": "memory"
        }

        res = index.query(
            data=f"Memory tag: {tag}",  # Query text
            top_k=100,
            include_metadata=True,
            filter=filter_dict,
            namespace=NAMESPACE
        )

        ids = [m["id"] for m in res["matches"]]
        if ids:
            index.delete(ids=ids, namespace=NAMESPACE)
            return True
        return False

    async def get_memory_by_tag(self, channel_id: str, tag: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by its exact tag"""
        filter_dict = {
            "channel_id": channel_id,
            "tag": tag,
            "type": "memory"
        }

        res = index.query(
            data=f"Memory tag: {tag}",
            top_k=1,
            include_metadata=True,
            filter=filter_dict,
            namespace=NAMESPACE
        )

        if res["matches"]:
            m = res["matches"][0]
            metadata = m.get("metadata", {})
            return {
                "tag": metadata.get("tag"),
                "content": metadata.get("content"),
                "timestamp": metadata.get("timestamp")
            }
        return None

    async def list_memory_tags(self, channel_id: str) -> List[Dict[str, Any]]:
        """List all memory tags in a channel"""
        # Search with a generic memory-related query
        filter_dict = {
            "channel_id": channel_id,
            "type": "memory"
        }

        res = index.query(
            data="memory context information data",  # Generic memory-related terms
            top_k=100,
            include_metadata=True,
            filter=filter_dict,
            namespace=NAMESPACE
        )

        memories = []
        seen_tags = set()

        for m in res["matches"]:
            metadata = m.get("metadata", {})
            tag = metadata.get("tag")
            if tag and tag not in seen_tags:
                seen_tags.add(tag)
                memories.append({
                    "tag": tag,
                    "content": metadata.get("content", ""),
                    "timestamp": metadata.get("timestamp", "")
                })

        return sorted(memories, key=lambda x: x["tag"])

    async def get_all_memories(self, channel_id: str) -> List[Dict[str, Any]]:
        """Alternative method to get all memories using multiple queries"""
        all_memories = {}

        # Try different search terms to catch various memories
        search_terms = [
            "memory", "project", "api", "auth", "config",
            "setup", "note", "important", "remember", "context"
        ]

        filter_dict = {
            "channel_id": channel_id,
            "type": "memory"
        }

        for term in search_terms:
            try:
                res = index.query(
                    data=term,
                    top_k=50,
                    include_metadata=True,
                    filter=filter_dict,
                    namespace=NAMESPACE
                )

                for m in res["matches"]:
                    metadata = m.get("metadata", {})
                    tag = metadata.get("tag")
                    if tag and tag not in all_memories:
                        all_memories[tag] = {
                            "tag": tag,
                            "content": metadata.get("content", ""),
                            "timestamp": metadata.get("timestamp", ""),
                            "score": m["score"]
                        }
            except Exception as e:
                print(f"Error searching for term '{term}': {e}")
                continue

        return sorted(all_memories.values(), key=lambda x: x["tag"])

    async def fetch_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch vectors by their IDs"""
        try:
            res = index.fetch(ids=ids, namespace=NAMESPACE)
            vectors = []
            for vid, data in res["vectors"].items():
                metadata = data.get("metadata", {})
                vectors.append({
                    "id": vid,
                    "metadata": metadata
                })
            return vectors
        except Exception as e:
            print(f"Error fetching by IDs: {e}")
            return []

    async def list_all_in_namespace(self, prefix: str = "", limit: int = 100) -> List[str]:
        """List vector IDs in namespace"""
        try:
            res = index.list(prefix=prefix, limit=limit, namespace=NAMESPACE)
            return res.get("vectors", [])
        except Exception as e:
            print(f"Error listing vectors: {e}")
            return []

    async def get_stats(self) -> Dict[str, Any]:
        stats = index.describe_index_stats()
        namespace_stats = stats.get("namespaces", {}).get(NAMESPACE, {})
        return {
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
            "index_fullness": stats.get("index_fullness", 0),
            "namespace_vectors": namespace_stats.get("vector_count", 0)
        }
