#!/usr/bin/env python3
"""
RAG Retrievers - Comprehensive Collection
Various retriever implementations for RAG systems

Features:
- Basic retrievers (BM25, Dense, GraphRAG)
- Advanced retrievers (Hybrid, Ensemble, MultiQuery)
- OpenAI integration for embeddings and query expansion
- Flexible base class architecture
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import threading
import openai
import os
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryVariations(BaseModel):
    """Structured output for query variations"""
    variations: List[str]

# ============================================================================
# CORE CLASSES
# ============================================================================

@dataclass 
class QueryResult:
    """Result from a retriever query"""
    documents: List[str]
    scores: List[float]
    retriever_name: str
    query: str
    timestamp: datetime
    doc_ids: Optional[List[str]] = None  # For alignment with upstream DB

class BaseRetriever(ABC):
    """Abstract base class for all retrievers"""
    
    def __init__(self, name: str):
        self.name = name
        self.documents = []
        self._lock = threading.RLock()  # Thread-safe access to shared state
    
    @abstractmethod
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        """Add documents to the retriever"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = True) -> QueryResult:
        """Retrieve relevant documents"""
        pass

# ============================================================================
# BASIC RETRIEVERS
# ============================================================================

class BM25Retriever(BaseRetriever):
    """BM25 lexical retriever"""
    
    def __init__(self, name: str = "BM25"):
        super().__init__(name)
        self.bm25 = None
        self.doc_ids = []  # Store document IDs for alignment
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        with self._lock:  
            # Only add new documents, avoid duplicates
            new_docs = [doc for doc in documents if doc not in self.documents]
            if new_docs:
                self.documents.extend(new_docs)
                
                # Handle document IDs for upstream DB alignment
                if doc_ids:
                    # Map new_docs to their corresponding IDs from original list
                    for i, doc in enumerate(new_docs):
                        orig_idx = documents.index(doc)  # Find position in original list
                        self.doc_ids.append(doc_ids[orig_idx])
                else:
                    # Generate default IDs
                    start_id = len(self.doc_ids)
                    self.doc_ids.extend([f"doc_{start_id + i}" for i in range(len(new_docs))])
                
                try:
                    from rank_bm25 import BM25Okapi
                    tokenized_docs = [doc.lower().split() for doc in self.documents]
                    self.bm25 = BM25Okapi(tokenized_docs)
                except ImportError:
                    logger.warning("rank_bm25 not available, using simple matching")
                    self.bm25 = None
    
    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = True) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now(), doc_ids=[])
        
        if self.bm25:
            # Use BM25
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_docs = [self.documents[i] for i in top_indices]
            top_scores = [scores[i] for i in top_indices]
            # Safe doc_ids extraction - check if we have enough IDs
            if self.doc_ids and len(self.doc_ids) >= len(self.documents) and top_indices.size > 0:
                max_idx = max(top_indices) if len(top_indices) > 0 else 0
                top_ids = [self.doc_ids[i] for i in top_indices] if max_idx < len(self.doc_ids) else []
            else:
                top_ids = []
        else:
            # Fallback to simple matching
            query_words = set(query.lower().split())
            doc_scores = []
            for doc in self.documents:
                doc_words = set(doc.lower().split())
                score = len(query_words.intersection(doc_words)) / len(query_words.union(doc_words))
                doc_scores.append(score)
            
            top_indices = np.argsort(doc_scores)[::-1][:top_k]
            top_docs = [self.documents[i] for i in top_indices]
            # Safe doc_ids extraction - check if we have enough IDs
            if self.doc_ids and len(self.doc_ids) >= len(self.documents) and top_indices.size > 0:
                max_idx = max(top_indices) if len(top_indices) > 0 else 0
                top_ids = [self.doc_ids[i] for i in top_indices] if max_idx < len(self.doc_ids) else []
            else:
                top_ids = []
            if return_scores:
                top_scores = [doc_scores[i] for i in top_indices]
            else:
                top_scores = [1.0] * len(top_docs)
        
        return QueryResult(top_docs, top_scores, self.name, query, datetime.now(), doc_ids=top_ids)

class DenseRetriever(BaseRetriever):
    """Dense semantic retriever using OpenAI embeddings"""
    
    def __init__(self, name: str = "Dense", embedding_function: Optional[callable] = None):
        super().__init__(name)
        self.embeddings = None
        self.doc_ids = []  # Store document IDs for alignment
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.embedding_model = "text-embedding-3-large"
        self.embedding_function = embedding_function  # Allow external embedding function
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API or external function"""
        # Use external embedding function if provided (for testing/local models)
        if self.embedding_function:
            try:
                return self.embedding_function(texts)
            except Exception as e:
                logger.warning(f"External embedding function failed: {e}, falling back to OpenAI")
        
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings)
        except Exception as e:
            logger.warning(f"OpenAI embeddings failed: {e}, using random embeddings")
            # Dimension based on model
            dim = 3072 if "large" in self.embedding_model else 1536
            return np.random.randn(len(texts), dim)
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        with self._lock:  # Thread-safe document addition
            # Only add new documents, avoid duplicates
            new_docs = [doc for doc in documents if doc not in self.documents]
            if new_docs:
                self.documents.extend(new_docs)
                
                # Handle document IDs for upstream DB alignment  
                if doc_ids:
                    # Map new_docs to their corresponding IDs from original list
                    for doc in new_docs:
                        orig_idx = documents.index(doc)  # Find position in original list
                        self.doc_ids.append(doc_ids[orig_idx])
                else:
                    # Generate default IDs
                    start_id = len(self.doc_ids)
                    self.doc_ids.extend([f"doc_{start_id + i}" for i in range(len(new_docs))])
                
                # Incremental embedding: only compute embeddings for new docs
                new_embeddings = self._get_embeddings(new_docs)
                if self.embeddings is None:
                    self.embeddings = new_embeddings
                else:
                    # Stack new embeddings instead of recomputing all
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = True) -> QueryResult:
        if not self.documents or self.embeddings is None:
            return QueryResult([], [], self.name, query, datetime.now(), doc_ids=[])
        
        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]
        
        # Cosine similarity with epsilon to prevent division by zero
        eps = 1e-8
        doc_norms = np.linalg.norm(self.embeddings, axis=1) + eps
        query_norm = np.linalg.norm(query_embedding) + eps
        similarities = np.dot(self.embeddings, query_embedding) / (doc_norms * query_norm)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_docs = [self.documents[i] for i in top_indices]
        top_ids = [self.doc_ids[i] for i in top_indices] if len(self.doc_ids) > max(top_indices) else []
        
        # Optionally return scores
        if return_scores:
            top_scores = [similarities[i] for i in top_indices]
        else:
            top_scores = [1.0] * len(top_docs)  # Dummy scores when not requested
        
        return QueryResult(top_docs, top_scores, self.name, query, datetime.now(), doc_ids=top_ids)

class GraphRAGRetriever(BaseRetriever):
    """Graph-based retriever using PageRank"""
    
    def __init__(self, name: str = "GraphRAG"):
        super().__init__(name)
        self.graph = None
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        # Only add new documents, avoid duplicates
        new_docs = [doc for doc in documents if doc not in self.documents]
        if new_docs:
            self.documents.extend(new_docs)
            
            # Scale limit: disable graph for large corpora
            if len(self.documents) > 5_000:
                logger.warning("GraphRAG disabled: corpus too large (>5000 docs)")
                self.graph = None
                return
            
            try:
                import networkx as nx
                self.graph = nx.Graph()
                
                # Add nodes
                for i, doc in enumerate(self.documents):
                    self.graph.add_node(i, content=doc)
                
                # Add edges based on similarity (O(N²) - expensive!)
                for i in range(len(self.documents)):
                    for j in range(i + 1, len(self.documents)):
                        doc1_words = set(self.documents[i].lower().split())
                        doc2_words = set(self.documents[j].lower().split())
                        similarity = len(doc1_words.intersection(doc2_words)) / len(doc1_words.union(doc2_words))
                        if similarity > 0.1:
                            self.graph.add_edge(i, j, weight=similarity)
            except ImportError:
                logger.warning("networkx not available, using fallback")
                self.graph = None
    
    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = True) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now(), doc_ids=[])
        
        if self.graph:
            import networkx as nx
            pagerank_scores = nx.pagerank(self.graph)
            
            # Combine with query relevance
            query_words = set(query.lower().split())
            final_scores = []
            
            for i, doc in enumerate(self.documents):
                doc_words = set(doc.lower().split())
                relevance = len(query_words.intersection(doc_words)) / len(query_words.union(doc_words))
                final_score = 0.7 * relevance + 0.3 * pagerank_scores.get(i, 0)
                final_scores.append(final_score)
        else:
            # Fallback to simple matching
            query_words = set(query.lower().split())
            final_scores = []
            for doc in self.documents:
                doc_words = set(doc.lower().split())
                score = len(query_words.intersection(doc_words)) / len(query_words.union(doc_words))
                final_scores.append(score)
        
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        top_docs = [self.documents[i] for i in top_indices]
        if return_scores:
            top_scores = [final_scores[i] for i in top_indices]
        else:
            top_scores = [1.0] * len(top_docs)
        
        return QueryResult(top_docs, top_scores, self.name, query, datetime.now(), doc_ids=[])

# ============================================================================
# ADVANCED RETRIEVERS
# ============================================================================

class HybridRetriever(BaseRetriever):
    """Combines BM25 and Dense retrieval"""
    
    def __init__(self, name: str = "Hybrid", dense_weight: float = 0.7):
        super().__init__(name)
        self.bm25_retriever = BM25Retriever("BM25_Internal")
        self.dense_retriever = DenseRetriever("Dense_Internal")
        self.dense_weight = dense_weight
        self.bm25_weight = 1.0 - dense_weight
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        with self._lock:  # Thread-safe document addition
            # Only add new documents, avoid duplicates
            new_docs = [doc for doc in documents if doc not in self.documents]
            if new_docs:
                self.documents.extend(new_docs)
                self.bm25_retriever.add_documents(new_docs, metadata, doc_ids)
                self.dense_retriever.add_documents(new_docs, metadata, doc_ids)
    
    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = True) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
        # Get results from both retrievers
        bm25_result = self.bm25_retriever.retrieve(query, top_k * 2, return_scores=True)
        dense_result = self.dense_retriever.retrieve(query, top_k * 2, return_scores=True)
        
        # Normalize scores to [0,1] range for fair combination
        def normalize_scores(scores):
            if not scores or max(scores) == min(scores):
                return scores
            min_score, max_score = min(scores), max(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]
        
        norm_bm25_scores = normalize_scores(bm25_result.scores)
        norm_dense_scores = normalize_scores(dense_result.scores)
        
        # Combine normalized scores
        doc_scores = {}
        
        # Add normalized BM25 scores
        for doc, score in zip(bm25_result.documents, norm_bm25_scores):
            if doc not in doc_scores:
                doc_scores[doc] = 0
            doc_scores[doc] += self.bm25_weight * score
        
        # Add normalized Dense scores
        for doc, score in zip(dense_result.documents, norm_dense_scores):
            if doc not in doc_scores:
                doc_scores[doc] = 0
            doc_scores[doc] += self.dense_weight * score
        
        # Sort and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in sorted_docs[:top_k]]
        if return_scores:
            top_scores = [score for doc, score in sorted_docs[:top_k]]
        else:
            top_scores = [1.0] * len(top_docs)
        
        return QueryResult(top_docs, top_scores, self.name, query, datetime.now())

class EnsembleRetriever(BaseRetriever):
    """Combines multiple retrievers using Reciprocal Rank Fusion"""
    
    def __init__(self, name: str = "Ensemble", retrievers: List[BaseRetriever] = None):
        super().__init__(name)
        self.retrievers = retrievers or []
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        with self._lock: 
            # Only add new documents, avoid duplicates
            new_docs = [doc for doc in documents if doc not in self.documents]
            if new_docs:
                self.documents.extend(new_docs)
                for retriever in self.retrievers:
                    retriever.add_documents(new_docs, metadata, doc_ids)
    
    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = True) -> QueryResult:
        if not self.retrievers or not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
        # Get results from all retrievers
        all_results = []
        for retriever in self.retrievers:
            try:
                result = retriever.retrieve(query, top_k * 2, return_scores=True)
                all_results.append(result)
            except Exception as e:
                logger.warning(f"Error in retriever {retriever.name}: {e}")
        
        # Apply Reciprocal Rank Fusion
        doc_scores = {}
        k = 60  # RRF parameter
        
        for result in all_results:
            for rank, doc in enumerate(result.documents):
                if doc not in doc_scores:
                    doc_scores[doc] = 0
                doc_scores[doc] += 1 / (k + rank + 1)
        
        # Sort and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in sorted_docs[:top_k]]
        if return_scores:
            top_scores = [score for doc, score in sorted_docs[:top_k]]
        else:
            top_scores = [1.0] * len(top_docs)
        
        return QueryResult(top_docs, top_scores, self.name, query, datetime.now())

class MultiQueryRetriever(BaseRetriever):
    """Generates multiple query variations for improved recall using OpenAI"""
    
    def __init__(self, name: str = "MultiQuery", base_retriever: Optional[BaseRetriever] = None):
        super().__init__(name)
        self.base_retriever = base_retriever or DenseRetriever("Dense_Base")
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        with self._lock:  # Thread-safe document addition
            # Only add new documents, avoid duplicates
            new_docs = [doc for doc in documents if doc not in self.documents]
            if new_docs:
                self.documents.extend(new_docs)
                self.base_retriever.add_documents(new_docs, metadata, doc_ids)
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations using OpenAI structured outputs"""
        variations = [query]
        
        try:
            prompt = f"""Generate 3 different variations of this search query to improve information retrieval:

Original query: "{query}"

Generate variations that:
1. Use different wording but same meaning
2. Use synonyms or related terms  
3. Rephrase as questions or statements differently

Each variation should be distinct and useful for finding relevant information."""

            response = self.openai_client.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates query variations for information retrieval."},
                    {"role": "user", "content": prompt}
                ],
                response_format=QueryVariations,
                max_tokens=5000,
                temperature=0.7
            )
            
            if response.choices and response.choices[0].message.parsed:
                query_variations = response.choices[0].message.parsed
                for var in query_variations.variations:
                    if var and var.strip() not in variations:
                        variations.append(var.strip())
            
        except Exception as e:
            logger.warning(f"OpenAI query generation failed: {e}, falling back to simple expansion")
            # Fallback to simple expansion
            synonyms = {
                'machine learning': ['ML', 'AI'],
                'neural network': ['neural net', 'deep learning'],
                'algorithm': ['method', 'technique'],
            }
            
            for original, syns in synonyms.items():
                if original in query.lower():
                    for syn in syns[:1]:
                        variations.append(query.lower().replace(original, syn))
            
            # Question reformulation
            if query.lower().startswith('what is'):
                topic = query[7:].strip()
                variations.append(f"{topic} definition")
        
        return list(dict.fromkeys(variations))[:4]  # Remove duplicates, limit to 4
    
    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = True) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
        # Generate variations and retrieve
        variations = self._generate_query_variations(query)
        all_docs = []
        doc_counts = {}
        
        for var_query in variations:
            result = self.base_retriever.retrieve(var_query, top_k, return_scores=True)
            for doc in result.documents:
                if doc not in doc_counts:
                    doc_counts[doc] = 0
                    all_docs.append(doc)
                doc_counts[doc] += 1
        
        # Score by frequency of appearance
        doc_scores = [(doc, doc_counts[doc] / len(variations)) for doc in all_docs]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_docs = [doc for doc, score in doc_scores[:top_k]]
        if return_scores:
            top_scores = [score for doc, score in doc_scores[:top_k]]
        else:
            top_scores = [1.0] * len(top_docs)
        
        return QueryResult(top_docs, top_scores, self.name, query, datetime.now())

class ContextualCompressionRetriever(BaseRetriever):
    """Compresses long passages to only query-relevant sentences"""
    
    def __init__(self, name: str = "ContextualCompression", base_retriever: Optional[BaseRetriever] = None):
        super().__init__(name)
        self.base_retriever = base_retriever or DenseRetriever("Dense_Base")
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        with self._lock:  # Thread-safe document addition
            # Only add new documents, avoid duplicates
            new_docs = [doc for doc in documents if doc not in self.documents]
            if new_docs:
                self.documents.extend(new_docs)
                self.base_retriever.add_documents(new_docs, metadata, doc_ids)
    
    def _compress_document(self, query: str, document: str) -> str:
        """Compress document to query-relevant sentences"""
        try:
            prompt = f"""Extract only the sentences from this document that are directly relevant to the query.

Query: "{query}"

Document: {document}

Return only the relevant sentences, preserving their original wording. If no sentences are relevant, return the first 2 sentences."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model for compression
                messages=[
                    {"role": "system", "content": "You extract only query-relevant sentences from documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            if response.choices and response.choices[0].message.content:
                compressed = response.choices[0].message.content.strip()
                return compressed if compressed else document[:200]  # Fallback
            
        except Exception as e:
            logger.warning(f"Document compression failed: {e}")
        
        # Fallback: simple sentence extraction
        sentences = document.split('.')
        query_words = set(query.lower().split())
        relevant_sentences = []
        
        for sentence in sentences[:5]:  # Check first 5 sentences
            sentence_words = set(sentence.lower().split())
            if query_words.intersection(sentence_words):
                relevant_sentences.append(sentence.strip())
        
        return '. '.join(relevant_sentences[:3]) if relevant_sentences else document[:200]
    
    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = True) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
        # Get initial results
        result = self.base_retriever.retrieve(query, top_k, return_scores=True)
        
        # Compress each document
        compressed_docs = []
        for doc in result.documents:
            compressed = self._compress_document(query, doc)
            compressed_docs.append(compressed)
        
        # Honor return_scores flag
        scores = result.scores if return_scores else [1.0] * len(compressed_docs)
        
        return QueryResult(compressed_docs, scores, self.name, query, datetime.now())

class ReRankerRetriever(BaseRetriever):
    """Two-stage retrieval: base + cross-encoder rerank"""
    
    def __init__(self, name: str = "ReRanker", base_retriever: Optional[BaseRetriever] = None):
        super().__init__(name)
        self.base_retriever = base_retriever or DenseRetriever("Dense_Base")
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        with self._lock:  # Thread-safe document addition
            # Only add new documents, avoid duplicates
            new_docs = [doc for doc in documents if doc not in self.documents]
            if new_docs:
                self.documents.extend(new_docs)
                self.base_retriever.add_documents(new_docs, metadata, doc_ids)
    
    def _rerank_score(self, query: str, document: str) -> float:
        """Score query-document pair for reranking"""
        try:
            prompt = f"""Rate how relevant this document is to the query on a scale of 0.0 to 1.0.

Query: "{query}"
Document: "{document}"

Consider:
- Direct relevance to the query topic
- Quality and completeness of information
- How well it answers the query

Respond with just a number between 0.0 and 1.0."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise document relevance scorer. Respond only with a number."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            if response.choices and response.choices[0].message.content:
                score_text = response.choices[0].message.content.strip()
                return float(score_text)
                
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
        
        # Fallback: simple word overlap
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        return len(query_words.intersection(doc_words)) / len(query_words.union(doc_words))
    
    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = True) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
        # Get more candidates from base retriever
        first_stage = self.base_retriever.retrieve(query, min(top_k * 3, 20), return_scores=True)
        
        # Rerank with cross-encoder scoring
        rerank_scores = []
        for doc in first_stage.documents:
            score = self._rerank_score(query, doc)
            rerank_scores.append(score)
        
        # Sort by rerank scores
        sorted_indices = np.argsort(rerank_scores)[::-1][:top_k]
        final_docs = [first_stage.documents[i] for i in sorted_indices]
        if return_scores:
            final_scores = [rerank_scores[i] for i in sorted_indices]
        else:
            final_scores = [1.0] * len(final_docs)
        
        return QueryResult(final_docs, final_scores, self.name, query, datetime.now())

class HyDERetriever(BaseRetriever):
    """Hypothetical Document Embeddings - generates hypothetical answer first"""
    
    def __init__(self, name: str = "HyDE", base_retriever: Optional[BaseRetriever] = None):
        super().__init__(name)
        self.base_retriever = base_retriever or DenseRetriever("Dense_Base")
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        with self._lock:  # Thread-safe document addition
            # Only add new documents, avoid duplicates
            new_docs = [doc for doc in documents if doc not in self.documents]
            if new_docs:
                self.documents.extend(new_docs)
                self.base_retriever.add_documents(new_docs, metadata, doc_ids)
    
    def _generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query"""
        try:
            prompt = f"""Write a short, informative passage that would answer this query:

Query: "{query}"

Write as if you're providing a direct, factual answer. Keep it concise (2-3 sentences) and focused on the specific information requested."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You write concise, factual passages that answer queries directly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.warning(f"Hypothetical document generation failed: {e}")
        
        # Fallback: use original query
        return query
    
    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = True) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
        # Generate hypothetical document
        hyp_doc = self._generate_hypothetical_document(query)
        
        # Search using hypothetical document
        result = self.base_retriever.retrieve(hyp_doc, top_k, return_scores=True)
        
        # Honor return_scores flag
        scores = result.scores if return_scores else [1.0] * len(result.documents)
        
        # Return results with original query
        return QueryResult(result.documents, scores, self.name, query, datetime.now())

class TimeWeightedRetriever(BaseRetriever):
    """Biases toward fresher documents using timestamps"""
    
    def __init__(self, name: str = "TimeWeighted", base_retriever: Optional[BaseRetriever] = None, decay_factor: float = 0.1):
        super().__init__(name)
        self.base_retriever = base_retriever or DenseRetriever("Dense_Base")
        self.decay_factor = decay_factor  # How much to weight recency
        self.doc_timestamps = {}
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        with self._lock:  # Thread-safe document addition
            # Only add new documents, avoid duplicates
            new_docs = [doc for doc in documents if doc not in self.documents]
            if new_docs:
                # Capture previous length BEFORE extending
                prev_len = len(self.documents)
                self.documents.extend(new_docs)
                self.base_retriever.add_documents(new_docs, metadata, doc_ids)
                
                # Extract or assign timestamps
                current_time = datetime.now()
                
                for i, doc in enumerate(new_docs):  # Iterate over new_docs, not documents
                    doc_idx = prev_len + i  # Use captured previous length
                    if metadata and i < len(metadata) and 'timestamp' in metadata[i]:
                        # Use provided timestamp
                        if isinstance(metadata[i]['timestamp'], str):
                            self.doc_timestamps[doc_idx] = datetime.fromisoformat(metadata[i]['timestamp'])
                        else:
                            self.doc_timestamps[doc_idx] = metadata[i]['timestamp']
                    else:
                        # Assign current timestamp
                        self.doc_timestamps[doc_idx] = current_time
    
    def _calculate_time_weight(self, doc_timestamp: datetime) -> float:
        """Calculate time-based weight (more recent = higher weight)"""
        now = datetime.now()
        age_days = (now - doc_timestamp).days
        # Exponential decay: weight = e^(-decay_factor * age_days)
        return np.exp(-self.decay_factor * age_days)
    
    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = True) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
        # Get base retrieval results
        result = self.base_retriever.retrieve(query, min(top_k * 2, len(self.documents)), return_scores=True)
        
        # Apply time weighting
        time_weighted_scores = []
        for i, (doc, base_score) in enumerate(zip(result.documents, result.scores)):
            # Find document index
            doc_idx = None
            for idx, stored_doc in enumerate(self.documents):
                if stored_doc == doc:
                    doc_idx = idx
                    break
            
            if doc_idx is not None and doc_idx in self.doc_timestamps:
                time_weight = self._calculate_time_weight(self.doc_timestamps[doc_idx])
                # Combine base score with time weight
                final_score = 0.7 * base_score + 0.3 * time_weight
            else:
                final_score = base_score
            
            time_weighted_scores.append(final_score)
        
        # Sort by time-weighted scores
        sorted_indices = np.argsort(time_weighted_scores)[::-1][:top_k]
        final_docs = [result.documents[i] for i in sorted_indices]
        if return_scores:
            final_scores = [time_weighted_scores[i] for i in sorted_indices]
        else:
            final_scores = [1.0] * len(final_docs)
        
        return QueryResult(final_docs, final_scores, self.name, query, datetime.now())

class SelfQueryRetriever(BaseRetriever):
    """Parses natural language queries into semantic search + metadata filters"""
    
    def __init__(self, name: str = "SelfQuery", base_retriever: Optional[BaseRetriever] = None):
        super().__init__(name)
        self.base_retriever = base_retriever or DenseRetriever("Dense_Base")
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.doc_metadata = []
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        with self._lock:  # Thread-safe document addition
            # Only add new documents, avoid duplicates
            new_docs = [doc for doc in documents if doc not in self.documents]
            if new_docs:
                self.documents.extend(new_docs)
                self.base_retriever.add_documents(new_docs, metadata, doc_ids)
            
                # Store metadata for filtering
                if metadata:
                    # Only extend with metadata for new documents
                    new_metadata = metadata[:len(new_docs)]
                    self.doc_metadata.extend(new_metadata)
                else:
                    # Add empty metadata for each new document
                    self.doc_metadata.extend([{} for _ in new_docs])
    
    def _parse_query(self, query: str) -> Tuple[str, Dict]:
        """Parse natural language query into search terms + filters"""
        try:
            prompt = f"""Parse this natural language query into:
1. Core search terms (what to search for)
2. Metadata filters (constraints like date, author, type, etc.)

Query: "{query}"

Respond in this JSON format:
{{
    "search_terms": "core search terms here",
    "filters": {{
        "key": "value for any detected constraints"
    }}
}}

Common filter types: date, year, author, type, category, status, language"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You parse queries into search terms and metadata filters. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5000,
                temperature=0.1
            )
            
            if response.choices and response.choices[0].message.content:
                import json as json_lib
                parsed = json_lib.loads(response.choices[0].message.content.strip())
                return parsed.get('search_terms', query), parsed.get('filters', {})
                
        except Exception as e:
            logger.warning(f"Query parsing failed: {e}")
        
        # Fallback: no filters
        return query, {}
    
    def _apply_filters(self, documents: List[str], scores: List[float], filters: Dict) -> Tuple[List[str], List[float]]:
        """Apply metadata filters to results"""
        if not filters:
            return documents, scores
        
        filtered_docs = []
        filtered_scores = []
        
        for doc, score in zip(documents, scores):
            # Find document metadata
            doc_idx = None
            for i, stored_doc in enumerate(self.documents):
                if stored_doc == doc:
                    doc_idx = i
                    break
            
            if doc_idx is None or doc_idx >= len(self.doc_metadata):
                continue
            
            doc_meta = self.doc_metadata[doc_idx]
            
            # Check if document matches all filters
            matches = True
            for filter_key, filter_value in filters.items():
                if filter_key in doc_meta:
                    doc_value = doc_meta[filter_key]
                    # Simple string matching (could be enhanced)
                    if str(filter_value).lower() not in str(doc_value).lower():
                        matches = False
                        break
                else:
                    # Filter key not found in metadata
                    matches = False
                    break
            
            if matches:
                filtered_docs.append(doc)
                filtered_scores.append(score)
        
        return filtered_docs, filtered_scores
    
    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = True) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
        # Parse query into search terms and filters
        search_terms, filters = self._parse_query(query)
        
        # Search with parsed terms
        result = self.base_retriever.retrieve(search_terms, top_k * 2, return_scores=True)  # Get more candidates
        
        # Apply metadata filters
        filtered_docs, filtered_scores = self._apply_filters(result.documents, result.scores, filters)
        
        # Return top-k filtered results
        final_docs = filtered_docs[:top_k]
        if return_scores:
            final_scores = filtered_scores[:top_k]
        else:
            final_scores = [1.0] * len(final_docs)
        
        return QueryResult(final_docs, final_scores, self.name, query, datetime.now())

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_retrievers() -> List[BaseRetriever]:
    """Create a comprehensive set of sample retrievers (11 total)"""
    # Basic retrievers (3)
    basic_retrievers = [
        BM25Retriever("BM25"),
        DenseRetriever("Dense"),
        GraphRAGRetriever("GraphRAG")
    ]
    
    # Advanced retrievers (3) - Create fresh retrievers for Ensemble to avoid sharing
    advanced_retrievers = [
        HybridRetriever("Hybrid"),
        EnsembleRetriever("Ensemble", [BM25Retriever("BM25_Ensemble"), DenseRetriever("Dense_Ensemble")]),
        MultiQueryRetriever("MultiQuery")
    ]
    
    # Next-gen retrievers (5)
    nextgen_retrievers = [
        ReRankerRetriever("ReRanker"),
        HyDERetriever("HyDE"),
        SelfQueryRetriever("SelfQuery"),  
        TimeWeightedRetriever("TimeWeighted"),
        ContextualCompressionRetriever("ContextualCompression")
    ]
    
    return basic_retrievers + advanced_retrievers + nextgen_retrievers 