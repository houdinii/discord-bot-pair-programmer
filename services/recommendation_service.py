import re
from collections import Counter
from datetime import datetime, timezone
from typing import List, Dict, Set

from services.arxiv_service import ArxivService
from services.vector_service import VectorService
from utils.logger import get_logger

logger = get_logger(__name__)


class RecommendationService:
    """Intelligent paper recommendation service based on user context and interests"""

    def __init__(self, vector_service: VectorService, arxiv_service: ArxivService):
        self.vector_service = vector_service
        self.arxiv_service = arxiv_service

    async def analyze_user_interests(self, channel_id: str, days_back: int = 30) -> Dict:
        """Analyze user interests from recent conversations and loaded papers"""
        logger.logger.info(f"Analyzing interests for channel {channel_id}")

        # Get recent conversations and documents
        results = await self.vector_service.search_similar(
            query="research paper machine learning AI",  # Broad query to get diverse results
            channel_id=channel_id,
            top_k=100  # Get many results for analysis
        )

        # Extract key information
        topics = []
        categories = []
        paper_ids = set()
        keywords = []
        authors = []

        for result in results:
            metadata = result['metadata']
            content = result['content']

            # Extract from arxiv papers
            if metadata.get('type') == 'arxiv_paper':
                paper_ids.add(metadata.get('arxiv_id'))
                if metadata.get('categories'):
                    categories.extend(metadata['categories'].split(','))
                if metadata.get('authors'):
                    authors.extend(metadata['authors'].split(','))

            # Extract from documents
            elif metadata.get('type') == 'document':
                if 'arxiv' in metadata.get('filename', '').lower():
                    paper_id = metadata.get('arxiv_id')
                    if paper_id:
                        paper_ids.add(paper_id)

            # Extract from conversations
            elif metadata.get('type') == 'conversation':
                # Extract topics from Q&A
                message = metadata.get('message', '')
                response = metadata.get('response', '')

                # Look for technical terms and topics
                technical_terms = self._extract_technical_terms(message + ' ' + response)
                keywords.extend(technical_terms)

            # Extract from memories
            elif metadata.get('type') == 'memory':
                tag = metadata.get('tag', '')
                content_text = metadata.get('content', '')

                # Tags often contain important topics
                if tag:
                    topics.append(tag)

                # Extract technical terms from memory content
                technical_terms = self._extract_technical_terms(content_text)
                keywords.extend(technical_terms)

        # Analyze GitHub repos for additional context
        github_results = await self.vector_service.search_similar(
            query="github repository",
            channel_id=channel_id,
            content_type=['github'],
            top_k=50
        )

        for result in github_results:
            metadata = result['metadata']
            if metadata.get('language'):
                topics.append(metadata['language'].lower())
            if metadata.get('repo_name'):
                # Extract likely topics from repo names
                repo_topics = self._extract_topics_from_repo_name(metadata['repo_name'])
                topics.extend(repo_topics)

        # Process and rank interests
        topic_counts = Counter(topics)
        keyword_counts = Counter(keywords)
        category_counts = Counter(categories)
        author_counts = Counter([a.strip() for a in authors])

        return {
            'top_topics': topic_counts.most_common(10),
            'top_keywords': keyword_counts.most_common(20),
            'top_categories': category_counts.most_common(5),
            'top_authors': author_counts.most_common(5),
            'paper_ids': list(paper_ids),
            'total_papers': len(paper_ids),
            'total_interactions': len(results)
        }

    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract likely technical terms and topics from text"""
        terms = []

        # Common ML/AI terms to look for
        ml_terms = [
            'transformer', 'attention', 'neural', 'network', 'deep learning',
            'machine learning', 'reinforcement learning', 'supervised', 'unsupervised',
            'classification', 'regression', 'clustering', 'gan', 'vae', 'bert',
            'gpt', 'lstm', 'rnn', 'cnn', 'convolution', 'embedding', 'vector',
            'gradient', 'backpropagation', 'optimization', 'loss function',
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'roc',
            'dataset', 'training', 'validation', 'testing', 'cross-validation',
            'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'numpy', 'pandas',
            'nlp', 'computer vision', 'cv', 'object detection', 'segmentation',
            'natural language', 'language model', 'tokenization', 'bert', 'gpt',
            'diffusion', 'stable diffusion', 'midjourney', 'dalle', 'clip',
            'multimodal', 'vision transformer', 'vit', 'mae', 'sam',
            'prompt', 'fine-tuning', 'transfer learning', 'few-shot', 'zero-shot',
            'rag', 'retrieval', 'augmented', 'generation', 'llm', 'large language model'
        ]

        text_lower = text.lower()
        for term in ml_terms:
            if term in text_lower:
                terms.append(term)

        # Extract potential paper titles or technical phrases in quotes
        quoted = re.findall(r'"([^"]+)"', text)
        for quote in quoted:
            if 5 < len(quote) < 100:  # Reasonable length for technical term
                terms.append(quote.lower())

        # Extract acronyms (likely technical terms)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        terms.extend([a.lower() for a in acronyms])

        return terms

    def _extract_topics_from_repo_name(self, repo_name: str) -> List[str]:
        """Extract likely topics from repository name"""
        topics = []

        # Split on common separators
        parts = re.split(r'[-_/]', repo_name.lower())

        # Common repo name patterns
        tech_keywords = [
            'ml', 'ai', 'deep', 'learning', 'neural', 'network',
            'transformer', 'bert', 'gpt', 'llm', 'nlp', 'cv',
            'pytorch', 'tensorflow', 'keras', 'model', 'train',
            'classification', 'detection', 'segmentation', 'generation'
        ]

        for part in parts:
            if part in tech_keywords:
                topics.append(part)

        return topics

    async def get_personalized_recommendations(self, channel_id: str,
                                               max_results: int = 5) -> List[Dict]:
        """Get personalized paper recommendations based on context analysis"""
        logger.logger.info(f"Generating personalized recommendations for channel {channel_id}")

        # Analyze user interests
        interests = await self.analyze_user_interests(channel_id)

        # Build search queries based on interests
        search_queries = []

        # Use top keywords
        if interests['top_keywords']:
            top_keywords = [kw for kw, _ in interests['top_keywords'][:5]]
            search_queries.append(' '.join(top_keywords))

        # Use top categories
        if interests['top_categories']:
            for category, _ in interests['top_categories'][:2]:
                search_queries.append(f"cat:{category}")

        # Search for papers by top authors
        if interests['top_authors']:
            for author, _ in interests['top_authors'][:2]:
                search_queries.append(f"au:{author}")

        # Combine queries and search
        all_papers = []
        seen_ids = set()

        # Get papers that are ACTUALLY loaded by the user (not just recommended)
        actually_loaded_papers = await self._get_user_loaded_papers(channel_id)
        seen_ids.update(actually_loaded_papers)

        for query in search_queries[:3]:  # Limit queries to avoid too many API calls
            try:
                papers = await self.arxiv_service.search_papers(query, max_results=10)

                # Filter out already seen papers
                new_papers = [p for p in papers if p['id'] not in seen_ids]
                all_papers.extend(new_papers)

                # Track seen IDs
                for paper in new_papers:
                    seen_ids.add(paper['id'])

            except Exception as e:
                logger.logger.error(f"Error searching with query '{query}': {e}")
                continue

        # Score and rank papers based on relevance to interests
        scored_papers = []
        for paper in all_papers:
            score = self._score_paper_relevance(paper, interests)
            scored_papers.append((score, paper))

        # Sort by score and return top results
        scored_papers.sort(key=lambda x: x[0], reverse=True)

        recommendations = []
        for score, paper in scored_papers[:max_results]:
            paper['relevance_score'] = score
            paper['relevance_reason'] = self._get_relevance_reason(paper, interests)
            recommendations.append(paper)

        logger.logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations

    async def _get_user_loaded_papers(self, channel_id: str) -> Set[str]:
        """Get papers actually loaded by users (not just recommended)"""
        # Search for papers that were explicitly loaded via !arxiv_load
        results = await self.vector_service.search_similar(
            query="arxiv paper",
            channel_id=channel_id,
            content_type=['arxiv_paper'],
            top_k=100
        )

        loaded_papers = set()
        for result in results:
            metadata = result['metadata']
            # Only count papers that users have explicitly loaded
            # These should have the 'arxiv_id' metadata
            arxiv_id = metadata.get('arxiv_id')
            if arxiv_id:
                loaded_papers.add(arxiv_id)

        return loaded_papers

    def _score_paper_relevance(self, paper: Dict, interests: Dict) -> float:
        """Score a paper's relevance to user interests"""
        score = 0.0

        # Check title and abstract for keywords
        text = f"{paper['title']} {paper['abstract']}".lower()

        # Score based on keyword matches
        for keyword, count in interests['top_keywords']:
            if keyword in text:
                score += count * 2  # Weight by frequency in user's context

        # Score based on category matches
        paper_categories = paper.get('categories', [])
        for category, count in interests['top_categories']:
            if category in paper_categories:
                score += count * 3  # Categories are strong indicators

        # Score based on author matches
        paper_authors = [a.lower() for a in paper.get('authors', [])]
        for author, count in interests['top_authors']:
            if any(author.lower() in pa for pa in paper_authors):
                score += count * 4  # Author match is very relevant

        # Boost recent papers
        if paper.get('published'):
            days_old = (datetime.now(timezone.utc) - paper['published']).days
            if days_old < 7:
                score += 5
            elif days_old < 30:
                score += 2

        return score

    def _get_relevance_reason(self, paper: Dict, interests: Dict) -> str:
        """Generate a human-readable reason for why this paper is relevant"""
        reasons = []

        text = f"{paper['title']} {paper['abstract']}".lower()

        # Check for keyword matches
        matching_keywords = []
        for keyword, _ in interests['top_keywords'][:5]:
            if keyword in text:
                matching_keywords.append(keyword)

        if matching_keywords:
            reasons.append(f"Related to your interests in: {', '.join(matching_keywords[:3])}")

        # Check for author matches
        paper_authors = [a.lower() for a in paper.get('authors', [])]
        for author, _ in interests['top_authors']:
            if any(author.lower() in pa for pa in paper_authors):
                reasons.append(f"By author you follow: {author}")
                break

        # Check for category matches
        paper_categories = paper.get('categories', [])
        matching_categories = []
        for category, _ in interests['top_categories']:
            if category in paper_categories:
                matching_categories.append(category)

        if matching_categories:
            reasons.append(f"In your preferred categories: {', '.join(matching_categories)}")

        # Default reason
        if not reasons:
            reasons.append("Potentially relevant to your research interests")

        return " | ".join(reasons)

    async def generate_recommendation_summary(self, channel_id: str,
                                              recommendations: List[Dict]) -> str:
        """Generate a summary explanation of why these papers were recommended"""
        interests = await self.analyze_user_interests(channel_id)

        summary_parts = []

        # Summarize user's interests
        if interests['top_keywords']:
            top_3_keywords = [kw for kw, _ in interests['top_keywords'][:3]]
            summary_parts.append(f"Based on your interest in **{', '.join(top_3_keywords)}**")

        if interests['total_papers'] > 0:
            summary_parts.append(f"You've explored **{interests['total_papers']} papers** recently")

        if interests['top_categories']:
            top_cat = interests['top_categories'][0][0]
            summary_parts.append(f"Focusing on **{top_cat}** research")

        return " | ".join(summary_parts) if summary_parts else "Curated based on trending AI/ML research"
