# services/vector_service.py

import os
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Optional

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document

from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "discord-pair-programmer")

# Create index if it doesn't exist
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)


def _gen_id(text: str, user: str, ts: str) -> str:
    """Generate a stable unique ID for each record."""
    return hashlib.md5(f"{user}_{ts}_{text[:50]}".encode()).hexdigest()


class VectorService:
    def __init__(self):
        self.vector_store = vector_store
        self.embeddings = embeddings

    async def store_conversation(self, user_id: str, channel_id: str,
                                 message: str, response: str,
                                 ai_model: str) -> str:
        """Store a conversation in the vector database"""
        ts = datetime.now(timezone.utc).isoformat()
        combined = f"User: {message}\nAssistant: {response}"
        vid = _gen_id(combined, user_id, ts)

        doc = Document(
            page_content=combined,
            metadata={
                "id": vid,
                "user_id": user_id,
                "channel_id": channel_id,
                "ai_model": ai_model,
                "timestamp": ts,
                "type": "conversation",
                "message": message,
                "response": response
            }
        )

        self.vector_store.add_documents([doc], ids=[vid])
        return vid

    async def store_memory(self, user_id: str, channel_id: str,
                           tag: str, content: str) -> str:
        """Store a memory with a tag"""
        ts = datetime.now(timezone.utc).isoformat()
        text_content = f"Memory [{tag}]: {content}"
        vid = _gen_id(text_content, user_id, ts)

        doc = Document(
            page_content=text_content,
            metadata={
                "id": vid,
                "user_id": user_id,
                "channel_id": channel_id,
                "tag": tag,
                "timestamp": ts,
                "type": "memory",
                "content": content
            }
        )

        self.vector_store.add_documents([doc], ids=[vid])
        return vid

    async def search_similar(self, query: str,
                             channel_id: str = None,
                             user_id: str = None,
                             content_type: List[str] = None,
                             top_k: int = 5) -> List[Dict]:
        """Search for similar content in the vector database"""
        # Build filter
        filter_dict = {}

        if channel_id:
            filter_dict["channel_id"] = channel_id
        if user_id:
            filter_dict["user_id"] = user_id
        if content_type:
            filter_dict["type"] = {"$in": content_type}

        # Use similarity search with score
        results = self.vector_store.similarity_search_with_score(
            query,
            k=top_k,
            filter=filter_dict if filter_dict else None
        )

        return [
            {
                "id": doc.metadata.get("id"),
                "score": 1 - score,  # Convert distance to similarity score
                "metadata": doc.metadata,
                "content": doc.page_content
            }
            for doc, score in results
        ]

    async def get_context_for_ai(self, query: str,
                                 channel_id: str,
                                 max_context_length: int = 3000) -> str:
        """Get relevant context for AI response"""
        results = await self.search_similar(
            query=query,
            channel_id=channel_id,
            top_k=10
        )

        parts = []
        length = 0

        for result in results:
            if result["score"] < 0.7:
                continue

            md = result["metadata"]

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
                text = result.get('content', md.get('content', 'N/A'))

            if length + len(text) > max_context_length:
                break

            parts.append(text)
            length += len(text)

        return "\n\n".join(parts) if parts else ""

    async def delete_by_tag(self, channel_id: str, tag: str) -> bool:
        """Delete memories by tag"""
        # Search for memories with this tag
        results = await self.search_similar(
            query=tag,
            channel_id=channel_id,
            content_type=["memory"],
            top_k=100
        )

        # Filter for exact tag match
        ids_to_delete = []
        for result in results:
            if result["metadata"].get("tag") == tag:
                ids_to_delete.append(result["id"])

        if ids_to_delete:
            # Delete from Pinecone index directly
            index = pc.Index(INDEX_NAME)
            index.delete(ids=ids_to_delete)
            return True
        return False

    async def get_memory_by_tag(self, channel_id: str, tag: str) -> Optional[Dict]:
        """Get a specific memory by its exact tag"""
        results = await self.search_similar(
            query=tag,
            channel_id=channel_id,
            content_type=["memory"],
            top_k=10
        )

        # Find exact tag match
        for result in results:
            if result["metadata"].get("tag") == tag:
                return {
                    "tag": result["metadata"].get("tag"),
                    "content": result["metadata"].get("content"),
                    "timestamp": result["metadata"].get("timestamp")
                }
        return None

    async def get_all_memories(self, channel_id: str) -> List[Dict]:
        """Get all memories in a channel"""
        all_memories = {}

        # Try different search terms to catch various memories
        search_terms = [
            "memory", "project", "api", "auth", "config", "setup",
            "note", "important", "stack", "endpoint", "method",
            "React", "Node", "PostgreSQL", "JWT", "token",
            ""  # Empty string to potentially catch all
        ]

        for term in search_terms:
            try:
                results = await self.search_similar(
                    query=term,
                    channel_id=channel_id,
                    content_type=["memory"],
                    top_k=50
                )

                for result in results:
                    tag = result["metadata"].get("tag")
                    if tag and tag not in all_memories:
                        all_memories[tag] = {
                            "tag": tag,
                            "content": result["metadata"].get("content", ""),
                            "timestamp": result["metadata"].get("timestamp", "")
                        }
            except Exception as e:
                print(f"Error searching for term '{term}': {e}")
                continue

        return sorted(all_memories.values(), key=lambda x: x["tag"])

    async def get_stats(self) -> Dict:
        """Get vector database statistics"""
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()

        # Get namespace-specific stats (default namespace)
        namespace_stats = stats.get('namespaces', {}).get('', {})  # Empty string for default namespace

        return {
            "total_vectors": namespace_stats.get('vector_count', 0),
            "dimension": stats.get('dimension', 1536),
            "index_fullness": stats.get('index_fullness', 0.0)
        }

    async def store_document(self, filename: str, content: str,
                             user_id: str, channel_id: str,
                             file_type: str = None) -> str:
        """Store document content in vector database"""
        ts = datetime.now(timezone.utc).isoformat()
        text_content = f"Document [{filename}]: {content[:1000]}"  # Limit preview
        vid = _gen_id(text_content, user_id, ts)

        doc = Document(
            page_content=content,
            metadata={
                "id": vid,
                "user_id": user_id,
                "channel_id": channel_id,
                "filename": filename,
                "file_type": file_type,
                "timestamp": ts,
                "type": "document"
            }
        )

        self.vector_store.add_documents([doc], ids=[vid])
        return vid

    async def store_github_content(self, repo_name: str, content: str,
                                   content_type: str, channel_id: str) -> str:
        """Store GitHub-related content"""
        ts = datetime.now(timezone.utc).isoformat()
        text_content = f"GitHub [{repo_name}] {content_type}: {content[:500]}"
        vid = _gen_id(text_content, repo_name, ts)

        doc = Document(
            page_content=content,
            metadata={
                "id": vid,
                "repo_name": repo_name,
                "channel_id": channel_id,
                "content_type": content_type,
                "timestamp": ts,
                "type": "github"
            }
        )

        self.vector_store.add_documents([doc], ids=[vid])
        return vid
