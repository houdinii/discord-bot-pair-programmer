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

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def gen_id(text: str, user: str, ts: str) -> str:
    """Generate a stable unique ID for each record."""
    # noinspection PyTypeChecker
    return hashlib.md5(f"{user}_{ts}_{text[:50]}".encode()).hexdigest()


class VectorService:
    def __init__(self):
        self.embeddings = embeddings
        self.index_name = INDEX_NAME
        logger.logger.info("VectorService initialized with Pinecone")

    def _get_vector_store(self, namespace: str = None):
        """Get a vector store with optional namespace"""
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=namespace
        )

    @log_method()
    async def store_conversation(self, user_id: str, channel_id: str,
                                 message: str, response: str,
                                 ai_model: str) -> str:
        """Store a conversation in the vector database"""
        logger.log_data('IN', 'STORE_CONVERSATION', {
            'user_id': user_id,
            'channel_id': channel_id,
            'namespace': channel_id,
            'message_preview': message[:100] + '...' if len(message) > 100 else message,
            'response_preview': response[:100] + '...' if len(response) > 100 else response,
            'ai_model': ai_model
        })

        ts = datetime.now(timezone.utc).isoformat()
        combined = f"User: {message}\nAssistant: {response}"
        vid = gen_id(combined, user_id, ts)

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

        # Use channel_id as namespace
        vector_store = self._get_vector_store(namespace=channel_id)
        vector_store.add_documents([doc], ids=[vid])

        logger.log_data('OUT', 'CONVERSATION_STORED', {
            'vector_id': vid,
            'timestamp': ts,
            'namespace': channel_id
        })

        return vid

    async def delete_document_chunks(self, file_id: str, channel_id: str) -> bool:
        """Delete all chunks for a specific document"""
        # Search for all chunks of this document
        results = await self.search_similar(
            query=f"file_id:{file_id}",
            channel_id=channel_id,
            content_type=["document"],
            top_k=100
        )

        # Filter for exact file_id match
        ids_to_delete = []
        for result in results:
            if str(result["metadata"].get("file_id")) == str(file_id):
                ids_to_delete.append(result["id"])

        if ids_to_delete:
            index = pc.Index(INDEX_NAME)
            index.delete(ids=ids_to_delete, namespace=channel_id)
            return True
        return False

    async def get_document_stats(self, channel_id: str) -> Dict:
        """Get statistics about documents in a channel"""
        # Search for all documents
        results = await self.search_similar(
            query="",
            channel_id=channel_id,
            content_type=["document"],
            top_k=1000
        )

        # Aggregate stats
        unique_files = set()
        total_chunks = 0
        file_types = {}

        for result in results:
            metadata = result["metadata"]
            if metadata.get("type") == "document":
                file_id = metadata.get("file_id")
                if file_id:
                    unique_files.add(file_id)
                    total_chunks += 1

                    file_type = metadata.get("file_type", "unknown")
                    file_types[file_type] = file_types.get(file_type, 0) + 1

        return {
            "unique_files": len(unique_files),
            "total_chunks": total_chunks,
            "file_types": file_types
        }

    async def store_memory(self, user_id: str, channel_id: str,
                           tag: str, content: str) -> str:
        """Store a memory with a tag"""
        ts = datetime.now(timezone.utc).isoformat()
        text_content = f"Memory [{tag}]: {content}"
        vid = gen_id(text_content, user_id, ts)

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

        # Use channel_id as namespace
        vector_store = self._get_vector_store(namespace=channel_id)
        vector_store.add_documents([doc], ids=[vid])
        return vid

    @log_method()
    async def store_github_repo(self, repo_name: str, channel_id: str,
                                readme_content: str = None,
                                description: str = None,
                                language: str = None,
                                topics: List[str] = None) -> List[str]:
        """Store GitHub repository information in vector database"""
        logger.log_data('IN', 'STORE_GITHUB_REPO', {
            'repo_name': repo_name,
            'channel_id': channel_id,
            'has_readme': bool(readme_content),
            'language': language
        })

        ts = datetime.now(timezone.utc).isoformat()
        vector_ids = []

        # Store repository overview
        overview = f"GitHub Repository: {repo_name}"
        if description:
            overview += f"\nDescription: {description}"
        if language:
            overview += f"\nLanguage: {language}"
        if topics:
            overview += f"\nTopics: {', '.join(topics)}"

        overview_id = gen_id(overview, repo_name, ts + "_overview")

        doc_overview = Document(
            page_content=overview,
            metadata={
                "id": overview_id,
                "repo_name": repo_name,
                "channel_id": channel_id,
                "timestamp": ts,
                "type": "github",
                "github_type": "overview",
                "language": language
            }
        )

        # Store README content if available
        docs_to_add = [doc_overview]
        ids_to_add = [overview_id]
        vector_ids.append(overview_id)

        if readme_content:
            # Split README into chunks if it's too long
            chunks = self._chunk_text(readme_content, chunk_size=1500)

            for i, chunk in enumerate(chunks):
                chunk_id = gen_id(chunk, repo_name, ts + f"_readme_{i}")
                doc_readme = Document(
                    page_content=f"README for {repo_name} (part {i + 1}/{len(chunks)}):\n{chunk}",
                    metadata={
                        "id": chunk_id,
                        "repo_name": repo_name,
                        "channel_id": channel_id,
                        "timestamp": ts,
                        "type": "github",
                        "github_type": "readme",
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                docs_to_add.append(doc_readme)
                ids_to_add.append(chunk_id)
                vector_ids.append(chunk_id)

        # Use channel_id as namespace
        vector_store = self._get_vector_store(namespace=channel_id)
        vector_store.add_documents(docs_to_add, ids=ids_to_add)

        logger.log_data('OUT', 'GITHUB_REPO_STORED', {
            'repo_name': repo_name,
            'vector_ids_count': len(vector_ids),
            'namespace': channel_id
        })

        return vector_ids

    @log_method()
    async def store_github_file(self, repo_name: str, channel_id: str,
                                file_path: str, content: str,
                                file_type: str = None) -> List[str]:
        """Store a GitHub file in the vector database"""
        logger.log_data('IN', 'STORE_GITHUB_FILE', {
            'repo_name': repo_name,
            'file_path': file_path,
            'file_type': file_type,
            'content_length': len(content)
        })

        ts = datetime.now(timezone.utc).isoformat()
        vector_ids = []

        # Determine file type if not provided
        if not file_type:
            if file_path.endswith('.py'):
                file_type = 'python'
            elif file_path.endswith('.js') or file_path.endswith('.ts'):
                file_type = 'javascript'
            elif file_path.endswith('.md'):
                file_type = 'markdown'
            elif file_path.endswith('.yml') or file_path.endswith('.yaml'):
                file_type = 'yaml'
            elif file_path.endswith('.json'):
                file_type = 'json'
            else:
                file_type = 'text'

        # Chunk the file content if it's large
        chunks = self._chunk_code(content, chunk_size=1200)

        for i, chunk in enumerate(chunks):
            chunk_id = gen_id(chunk, repo_name, ts + f"_file_{i}")

            # Create descriptive content
            chunk_content = f"File: {file_path} from {repo_name}\n"
            if len(chunks) > 1:
                chunk_content += f"Part {i + 1}/{len(chunks)}\n"
            chunk_content += f"Type: {file_type}\n"
            chunk_content += f"```{file_type}\n{chunk}\n```"

            doc = Document(
                page_content=chunk_content,
                metadata={
                    "id": chunk_id,
                    "repo_name": repo_name,
                    "channel_id": channel_id,
                    "file_path": file_path,
                    "file_type": file_type,
                    "timestamp": ts,
                    "type": "github",
                    "github_type": "code",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )

            vector_ids.append(chunk_id)

            # Use channel_id as namespace
            vector_store = self._get_vector_store(namespace=channel_id)
            vector_store.add_documents([doc], ids=[chunk_id])

        logger.log_data('OUT', 'GITHUB_FILE_STORED', {
            'file_path': file_path,
            'chunks_stored': len(vector_ids)
        })

        return vector_ids

    async def store_github_structure(self, repo_name: str, channel_id: str,
                                     tree_structure: str) -> str:
        """Store repository structure/tree"""
        ts = datetime.now(timezone.utc).isoformat()
        tree_id = gen_id(tree_structure, repo_name, ts + "_tree")

        doc = Document(
            page_content=f"Repository structure for {repo_name}:\n{tree_structure}",
            metadata={
                "id": tree_id,
                "repo_name": repo_name,
                "channel_id": channel_id,
                "timestamp": ts,
                "type": "github",
                "github_type": "structure"
            }
        )

        vector_store = self._get_vector_store(namespace=channel_id)
        vector_store.add_documents([doc], ids=[tree_id])

        return tree_id

    def _chunk_code(self, code: str, chunk_size: int = 1200) -> List[str]:
        """Chunk code while trying to preserve logical boundaries"""
        if len(code) <= chunk_size:
            return [code]

        chunks = []
        lines = code.split('\n')
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            # If adding this line would exceed chunk size
            if current_size + line_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # Remember the last chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _chunk_text(self, text: str, chunk_size: int = 1500) -> List[str]:
        """Split text into chunks"""
        chunks = []
        lines = text.split('\n')
        current_chunk = ""

        for line in lines:
            if len(current_chunk) + len(line) + 1 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += ("\n" if current_chunk else "") + line

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

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
            'namespace': channel_id,
            'user_id': user_id,
            'content_type': content_type,
            'top_k': top_k
        })

        # Build filter
        filter_dict = {}

        # Note: namespace now handles channel_id filtering
        if user_id:
            filter_dict["user_id"] = user_id
        if content_type:
            filter_dict["type"] = {"$in": content_type}

        # Use namespace for channel isolation
        vector_store = self._get_vector_store(namespace=channel_id)

        # Use similarity search with score
        results = vector_store.similarity_search_with_score(
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
            'types': [r['metadata'].get('type') for r in formatted_results],
            'namespace': channel_id
        })

        return formatted_results

    @log_method()
    async def get_context_for_ai(self, query: str,
                                 channel_id: str,
                                 max_context_length: int = 3000) -> str:
        """Get relevant context for AI response including documents"""
        logger.log_data('IN', 'GET_AI_CONTEXT', {
            'query': query[:100] + '...' if len(query) > 100 else query,
            'channel_id': channel_id,
            'namespace': channel_id,
            'max_context_length': max_context_length
        })

        # Search for relevant content - use a higher top_k for documents
        results = await self.search_similar(
            query=query,
            channel_id=channel_id,
            top_k=25  # Increased to ensure we get documents
        )

        # Also do a specific document search
        doc_results = await self.search_documents(
            query=query,
            channel_id=channel_id,
            top_k=10
        )

        # Combine and deduplicate results
        all_results = results + doc_results
        seen_ids = set()
        unique_results = []

        for result in all_results:
            result_id = result.get('id') or result['metadata'].get('id')
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)

        # Sort by score
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        parts = []
        length = 0
        used_results = 0
        used_files = {}  # Track which files we've included

        for result in unique_results:
            # Score threshold
            if result.get("score", 1) < 0.5:
                continue

            md = result["metadata"]
            content = result.get('content', '')

            # Build context text based on type
            if md["type"] == "conversation":
                timestamp = md.get('timestamp', 'Unknown time')
                text = f"[Previous conversation - {timestamp[:19]}]\n"
                text += f"User: {md.get('message', 'N/A')}\n"
                text += f"Assistant: {md.get('response', 'N/A')}"

            elif md["type"] == "memory":
                text = f"[Saved Memory - {md.get('tag', 'N/A')}]\n{md.get('content', 'N/A')}"

            elif md["type"] == "document":
                file_id = md.get('file_id')
                filename = md.get('filename', 'Unknown')

                # Track how many chunks per file we've included
                if file_id:
                    if file_id not in used_files:
                        used_files[file_id] = 0

                    # Limit chunks per file to avoid one document dominating
                    if used_files[file_id] >= 3:
                        continue

                    used_files[file_id] += 1

                chunk_info = f"Part {md.get('chunk_index', 0) + 1}/{md.get('total_chunks', 1)}"

                # Extract actual content (remove search enhancement prefix)
                if "Content:\n" in content:
                    actual_content = content.split("Content:\n", 1)[1]
                else:
                    actual_content = content

                text = f"[Document: {filename} - {chunk_info}]\n{actual_content[:600]}..."

            elif md["type"] == "github":
                github_type = md.get('github_type', 'content')
                repo_name = md.get('repo_name', 'Unknown')

                if github_type == 'code':
                    file_path = md.get('file_path', 'Unknown')
                    text = f"[GitHub Code - {repo_name}/{file_path}]\n{content[:500]}..."
                else:
                    text = f"[GitHub {github_type} - {repo_name}]\n{content[:500]}..."

            else:
                continue  # Skip unknown types

            # Check length limit
            if length + len(text) + 50 > max_context_length:
                break

            parts.append(text)
            length += len(text) + 4  # Account for separators
            used_results += 1

        context = "\n\n---\n\n".join(parts) if parts else ""

        logger.log_data('OUT', 'AI_CONTEXT_BUILT', {
            'total_length': len(context),
            'parts_used': used_results,
            'total_results': len(unique_results),
            'unique_documents': len(used_files),
            'document_chunks_used': sum(used_files.values()),
            'context_preview': context[:200] + '...' if len(context) > 200 else context,
            'namespace': channel_id
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

    async def store_document_chunk(self, filename: str, content: str,
                                   user_id: str, channel_id: str,
                                   metadata: Dict) -> str:
        """Store a document chunk with additional metadata"""
        ts = datetime.now(timezone.utc).isoformat()

        # Create a unique ID that includes file_id for better tracking
        file_id = metadata.get('file_id', 'unknown')
        chunk_index = metadata.get('chunk_index', 0)
        vid = gen_id(f"{file_id}_{chunk_index}_{content[:50]}", user_id, ts)

        # Create comprehensive content for better search
        # Include filename and description in the searchable content
        enhanced_content = f"Document: {filename}\n"
        if metadata.get('description'):
            enhanced_content += f"Description: {metadata['description']}\n"
        enhanced_content += f"Content:\n{content}"

        doc = Document(
            page_content=enhanced_content,  # Use enhanced content for better search
            metadata={
                "id": vid,
                "user_id": user_id,
                "channel_id": channel_id,
                "filename": filename,
                "timestamp": ts,
                "type": "document",
                **metadata  # Include all additional metadata
            }
        )

        vector_store = self._get_vector_store(namespace=channel_id)
        vector_store.add_documents([doc], ids=[vid])

        logger.log_data('OUT', 'DOCUMENT_CHUNK_STORED', {
            'vector_id': vid,
            'file_id': file_id,
            'chunk_index': chunk_index,
            'namespace': channel_id
        })

        return vid

    async def search_documents(self, query: str, channel_id: str,
                               file_id: Optional[str] = None, top_k: int = 10) -> List[Dict]:
        """Search specifically for document content"""
        logger.log_data('IN', 'SEARCH_DOCUMENTS', {
            'query': query,
            'channel_id': channel_id,
            'file_id': file_id,
            'top_k': top_k
        })

        # Build filter
        filter_dict = {"type": "document"}
        if file_id:
            filter_dict["file_id"] = str(file_id)

        # Use namespace for channel isolation
        vector_store = self._get_vector_store(namespace=channel_id)

        # Search with the query
        results = vector_store.similarity_search_with_score(
            query,
            k=top_k,
            filter=filter_dict
        )

        formatted_results = [
            {
                "id": doc.metadata.get("id"),
                "score": 1 - score,  # Convert distance to similarity
                "metadata": doc.metadata,
                "content": doc.page_content,
                "file_id": doc.metadata.get("file_id"),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            }
            for doc, score in results
        ]

        logger.log_data('OUT', 'DOCUMENT_SEARCH_RESULTS', {
            'results_count': len(formatted_results),
            'file_ids': list(set(r['file_id'] for r in formatted_results if r['file_id']))
        })

        return formatted_results

    async def store_github_content(self, repo_name: str, content: str,
                                   content_type: str, channel_id: str) -> str:
        """Store GitHub-related content"""
        ts = datetime.now(timezone.utc).isoformat()
        text_content = f"GitHub [{repo_name}] {content_type}: {content[:500]}"
        vid = gen_id(text_content, repo_name, ts)

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
        vector_store = self._get_vector_store(namespace=channel_id)
        vector_store.add_documents([doc], ids=[vid])
        return vid
