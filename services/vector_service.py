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
from utils.logger import get_logger, log_method

logger = get_logger(__name__)

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
        logger.logger.info("VectorService initialized with Pinecone")

    @log_method()
    async def store_conversation(self, user_id: str, channel_id: str,
                                 message: str, response: str,
                                 ai_model: str) -> str:
        """Store a conversation in the vector database"""
        logger.log_data('IN', 'STORE_CONVERSATION', {
            'user_id': user_id,
            'channel_id': channel_id,
            'message_preview': message[:100] + '...' if len(message) > 100 else message,
            'response_preview': response[:100] + '...' if len(response) > 100 else response,
            'ai_model': ai_model
        })

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

        logger.log_data('OUT', 'CONVERSATION_STORED', {
            'vector_id': vid,
            'timestamp': ts
        })

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

    @log_method()
    async def search_similar(self, query: str,
                             channel_id: str = None,
                             user_id: str = None,
                             content_type: List[str] = None,
                             top_k: int = 5) -> List[Dict]:
        """Search for similar content in the vector database"""
        logger.log_data('IN', 'VECTOR_SEARCH', {
            'query': query[:100] + '...' if len(query) > 100 else query,
            'channel_id': channel_id,
            'user_id': user_id,
            'content_type': content_type,
            'top_k': top_k
        })

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

        formatted_results = [
            {
                "id": doc.metadata.get("id"),
                "score": 1 - score,  # Convert distance to similarity score
                "metadata": doc.metadata,
                "content": doc.page_content
            }
            for doc, score in results
        ]

        logger.log_data('OUT', 'SEARCH_RESULTS', {
            'results_count': len(formatted_results),
            'scores': [r['score'] for r in formatted_results],
            'types': [r['metadata'].get('type') for r in formatted_results]
        })

        return formatted_results

    @log_method()
    async def get_context_for_ai(self, query: str,
                                 channel_id: str,
                                 max_context_length: int = 3000) -> str:
        """Get relevant context for AI response"""
        logger.log_data('IN', 'GET_AI_CONTEXT', {
            'query': query[:100] + '...' if len(query) > 100 else query,
            'channel_id': channel_id,
            'max_context_length': max_context_length
        })

        # Search for relevant content - increase top_k for more context
        results = await self.search_similar(
            query=query,
            channel_id=channel_id,
            top_k=15  # Increased from 10
        )

        parts = []
        length = 0
        used_results = 0

        for result in results:
            # FIX: Score is similarity (higher is better) after the inversion in search_similar
            if result["score"] < 0.5:  # Changed from < 0.7 to < 0.5, and logic is now correct
                logger.logger.debug(f"Skipping result with low score: {result['score']}")
                continue

            md = result["metadata"]

            # Build context text based on type
            if md["type"] == "conversation":
                # Include timestamp for conversation context
                timestamp = md.get('timestamp', 'Unknown time')
                text = f"[{timestamp[:19]}] Previous conversation:\nUser: {md.get('message', 'N/A')}\nAssistant: {md.get('response', 'N/A')}"
            elif md["type"] == "memory":
                text = f"Saved Memory [{md.get('tag', 'N/A')}]: {md.get('content', 'N/A')}"
            elif md["type"] == "document":
                text = f"Document [{md.get('filename', 'N/A')}]: {md.get('content', 'N/A')[:500]}..."
            elif md["type"] == "github":
                text = f"GitHub [{md.get('repo_name', 'N/A')}]: {md.get('content', 'N/A')[:500]}..."
            else:
                text = result.get('content', md.get('content', 'N/A'))

            # Check if adding this would exceed max length
            if length + len(text) + 50 > max_context_length:  # 50 char buffer
                logger.logger.debug(f"Reached max context length at {length} chars")
                break

            parts.append(text)
            length += len(text) + 2  # Account for newlines
            used_results += 1

        # Sort parts by timestamp if available (newest first for conversations)
        context = "\n\n---\n\n".join(parts) if parts else ""

        logger.log_data('OUT', 'AI_CONTEXT_BUILT', {
            'total_length': len(context),
            'parts_used': used_results,
            'total_results': len(results),
            'context_preview': context[:200] + '...' if len(context) > 200 else context
        })

        return context

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
            # Delete it from Pinecone index directly
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
