import asyncio
import io
from typing import List, Dict, Optional, Tuple

import fitz  # PyMuPDF
import requests
from arxiv import Search, SortCriterion, SortOrder, Client
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.logger import get_logger

logger = get_logger(__name__)


class ArxivService:
    """Enhanced arXiv service with paper fetching, parsing, and search capabilities"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

    @staticmethod
    def clean_paper_id(paper_id: str) -> str:
        """Clean and normalize paper ID from various formats"""
        # Handle URLs like https://arxiv.org/abs/2304.03442v1
        if 'arxiv.org' in paper_id:
            paper_id = paper_id.split('/')[-1]

        # Remove version number (v1, v2, etc.) but be careful with old format
        if 'v' in paper_id:
            # Old format: math-ph/0003065v1 -> math-ph/0003065
            # New format: 2304.03442v1 -> 2304.03442
            parts = paper_id.split('v')
            if len(parts) == 2 and parts[1].isdigit():
                paper_id = parts[0]

        return paper_id.strip()

    # Also update the _short_id method:
    @staticmethod
    def _short_id(url: str):
        """Convert https://arxiv.org/abs/XXXX.YYYYYvZ to proper format"""
        paper_id = url.split("/")[-1]

        # Remove version number
        if 'v' in paper_id:
            parts = paper_id.split('v')
            if len(parts) == 2 and parts[1].isdigit():
                paper_id = parts[0]

        return paper_id

    @staticmethod
    def get_pdf_url(paper_id: str) -> str:
        """Get PDF URL from paper ID"""
        clean_id = ArxivService.clean_paper_id(paper_id)
        return f"https://arxiv.org/pdf/{clean_id}.pdf"

    @staticmethod
    def get_abs_url(paper_id: str) -> str:
        """Get abstract page URL from paper ID"""
        clean_id = ArxivService.clean_paper_id(paper_id)
        return f"https://arxiv.org/abs/{clean_id}"

    async def search_papers(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search arXiv papers with enhanced metadata"""
        logger.logger.info(f"Searching arXiv for: {query} (max: {max_results})")

        def _search_sync():
            search = Search(
                query=query,
                max_results=max_results,
                sort_by=SortCriterion.Relevance,
                sort_order=SortOrder.Descending
            )
            client = Client()

            results = []
            for result in client.results(search=search):
                # Extract ID properly from the entry_id
                # result.entry_id looks like: "http://arxiv.org/abs/2412.13419v1"
                raw_id = result.entry_id.split('/')[-1]  # Get "2412.13419v1"

                # Clean the ID properly
                clean_id = self.clean_paper_id(raw_id)

                paper_data = {
                    'id': clean_id,  # Use the cleaned ID
                    'title': result.title.strip(),
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary.strip(),
                    'published': result.published,
                    'updated': result.updated,
                    'categories': result.categories,
                    'primary_category': result.primary_category,
                    'pdf_url': self.get_pdf_url(clean_id),  # Use cleaned ID
                    'abs_url': self.get_abs_url(clean_id),  # Use cleaned ID
                    'journal_ref': getattr(result, 'journal_ref', None),
                    'doi': getattr(result, 'doi', None)
                }
                results.append(paper_data)

            return results

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _search_sync)

        logger.logger.info(f"Found {len(results)} papers")
        return results

    async def get_paper_metadata(self, paper_id: str) -> Optional[Dict]:
        """Get detailed metadata for a specific paper"""
        clean_id = self.clean_paper_id(paper_id)

        def _get_metadata_sync():
            search = Search(query=f"id:{clean_id}", max_results=1)
            client = Client()

            results = []
            for result in client.results(search=search):
                return {
                    'id': clean_id,
                    'title': result.title.strip(),
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary.strip(),
                    'published': result.published,
                    'updated': result.updated,
                    'categories': result.categories,
                    'primary_category': result.primary_category,
                    'pdf_url': self.get_pdf_url(result.entry_id),
                    'abs_url': self.get_abs_url(result.entry_id),
                    'journal_ref': getattr(result, 'journal_ref', None),
                    'doi': getattr(result, 'doi', None)
                }
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_metadata_sync)

    async def download_and_parse_pdf(self, paper_id: str, exclude_references: bool = True) -> Tuple[str, Dict]:
        """Download PDF and extract text content"""
        clean_id = self.clean_paper_id(paper_id)
        pdf_url = self.get_pdf_url(clean_id)

        logger.logger.info(f"Downloading PDF: {pdf_url}")

        def _download_and_parse():
            # Download PDF
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()

            # Parse PDF with PyMuPDF
            pdf_stream = io.BytesIO(response.content)

            page_texts = []
            with fitz.Document(stream=pdf_stream) as pdf:
                for page_num, page in enumerate(pdf.pages()):
                    text = page.get_text()
                    if text.strip():  # Only add non-empty pages
                        page_texts.append({
                            'page_num': page_num + 1,
                            'text': text
                        })

            # Combine all text
            full_text = "\n".join([page['text'] for page in page_texts])

            # Remove references section if requested
            if exclude_references:
                text_lower = full_text.lower()
                ref_positions = []

                # Look for common reference section headers
                ref_headers = ['references', 'bibliography', 'works cited']
                for header in ref_headers:
                    pos = text_lower.rfind(header)
                    if pos > len(full_text) * 0.5:  # Only if in latter half of paper
                        ref_positions.append(pos)

                if ref_positions:
                    cut_pos = min(ref_positions)
                    full_text = full_text[:cut_pos]

            metadata = {
                'total_pages': len(page_texts),
                'total_chars': len(full_text),
                'pdf_size': len(response.content)
            }

            return full_text, metadata

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _download_and_parse)

    def chunk_paper_content(self, content: str, paper_id: str, title: str) -> List[Document]:
        """Split paper content into chunks for vector storage"""
        chunks = self.text_splitter.split_text(content)

        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'paper_id': paper_id,
                    'title': title,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'source': f"arxiv:{paper_id}",
                    'type': 'arxiv_paper'
                }
            )
            documents.append(doc)

        return documents

    async def get_paper_suggestions(self, categories: List[str] = None,
                                    max_results: int = 5) -> List[Dict]:
        """Get paper suggestions based on categories or recent papers"""
        if not categories:
            categories = ['cs.AI', 'cs.LG', 'cs.CL']  # Default to AI/ML categories

        # Search for recent papers in specified categories
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])

        def _get_suggestions():
            search = Search(
                query=category_query,
                max_results=max_results,
                sort_by=SortCriterion.SubmittedDate,
                sort_order=SortOrder.Descending
            )
            client = Client()

            results = []
            for result in client.results(search=search):
                paper_data = {
                    'id': self.clean_paper_id(result.entry_id),
                    'title': result.title.strip(),
                    'authors': [author.name for author in result.authors][:3],  # Limit authors
                    'abstract': result.summary.strip()[:300] + "...",  # Truncate abstract
                    'published': result.published,
                    'primary_category': result.primary_category,
                    'abs_url': self.get_abs_url(result.entry_id)
                }
                results.append(paper_data)

            return results

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_suggestions)
