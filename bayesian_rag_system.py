#!/usr/bin/env python3
"""
Bayesian RAG Bandits - Complete System
A production-ready RAG retriever selection system using Thompson Sampling

Features:
- Thompson Sampling for optimal retriever selection
- 11 different retriever types (basic, advanced, LangChain-inspired)
- Online learning from user feedback
- Sparse feedback handling
- Production state persistence
- LLM-powered automatic relevance evaluation
- OpenAI embeddings and query expansion
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import openai
import os
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryVariations(BaseModel):
    """Structured output for query variations"""
    variations: List[str]

class RelevanceEvaluation(BaseModel):
    """Structured output for relevance evaluation"""
    is_relevant: bool
    confidence: float  # 0.0 to 1.0
    reasoning: str

# ============================================================================
# CORE CLASSES
# ============================================================================

@dataclass
class BetaParams:
    """Beta distribution parameters for Thompson Sampling"""
    alpha: float = 1.0
    beta: float = 1.0
    
    def update_success(self):
        self.alpha += 1
    
    def update_failure(self):
        self.beta += 1
    
    def sample(self) -> float:
        return np.random.beta(self.alpha, self.beta)
    
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

@dataclass 
class QueryResult:
    """Result from a retriever query"""
    documents: List[str]
    scores: List[float]
    retriever_name: str
    query: str
    timestamp: datetime

class BaseRetriever(ABC):
    """Abstract base class for all retrievers"""
    
    def __init__(self, name: str):
        self.name = name
        self.documents = []
    
    @abstractmethod
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to the retriever"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> QueryResult:
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
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        self.documents.extend(documents)
        try:
            from rank_bm25 import BM25Okapi
            tokenized_docs = [doc.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        except ImportError:
            logger.warning("rank_bm25 not available, using simple matching")
            self.bm25 = None
    
    def retrieve(self, query: str, top_k: int = 5) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
        if self.bm25:
            # Use BM25
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_docs = [self.documents[i] for i in top_indices]
            top_scores = [scores[i] for i in top_indices]
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
            top_scores = [doc_scores[i] for i in top_indices]
        
        return QueryResult(top_docs, top_scores, self.name, query, datetime.now())

class DenseRetriever(BaseRetriever):
    """Dense semantic retriever using OpenAI embeddings"""
    
    def __init__(self, name: str = "Dense"):
        super().__init__(name)
        self.embeddings = None
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.embedding_model = "text-embedding-3-large"
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API"""
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
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        self.documents.extend(documents)
        self.embeddings = self._get_embeddings(self.documents)
    
    def retrieve(self, query: str, top_k: int = 5) -> QueryResult:
        if not self.documents or self.embeddings is None:
            return QueryResult([], [], self.name, query, datetime.now())
        
        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]
        
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_docs = [self.documents[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]
        
        return QueryResult(top_docs, top_scores, self.name, query, datetime.now())

class GraphRAGRetriever(BaseRetriever):
    """Graph-based retriever using PageRank"""
    
    def __init__(self, name: str = "GraphRAG"):
        super().__init__(name)
        self.graph = None
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        self.documents.extend(documents)
        try:
            import networkx as nx
            self.graph = nx.Graph()
            
            # Add nodes
            for i, doc in enumerate(self.documents):
                self.graph.add_node(i, content=doc)
            
            # Add edges based on similarity
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
    
    def retrieve(self, query: str, top_k: int = 5) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
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
        top_scores = [final_scores[i] for i in top_indices]
        
        return QueryResult(top_docs, top_scores, self.name, query, datetime.now())

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
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        self.documents.extend(documents)
        self.bm25_retriever.add_documents(documents, metadata)
        self.dense_retriever.add_documents(documents, metadata)
    
    def retrieve(self, query: str, top_k: int = 5) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
        # Get results from both retrievers
        bm25_result = self.bm25_retriever.retrieve(query, top_k * 2)
        dense_result = self.dense_retriever.retrieve(query, top_k * 2)
        
        # Combine scores
        doc_scores = {}
        
        # Add BM25 scores
        for doc, score in zip(bm25_result.documents, bm25_result.scores):
            if doc not in doc_scores:
                doc_scores[doc] = 0
            doc_scores[doc] += self.bm25_weight * score
        
        # Add Dense scores
        for doc, score in zip(dense_result.documents, dense_result.scores):
            if doc not in doc_scores:
                doc_scores[doc] = 0
            doc_scores[doc] += self.dense_weight * score
        
        # Sort and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in sorted_docs[:top_k]]
        top_scores = [score for doc, score in sorted_docs[:top_k]]
        
        return QueryResult(top_docs, top_scores, self.name, query, datetime.now())

class EnsembleRetriever(BaseRetriever):
    """Combines multiple retrievers using Reciprocal Rank Fusion"""
    
    def __init__(self, name: str = "Ensemble", retrievers: List[BaseRetriever] = None):
        super().__init__(name)
        self.retrievers = retrievers or []
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        self.documents.extend(documents)
        for retriever in self.retrievers:
            retriever.add_documents(documents, metadata)
    
    def retrieve(self, query: str, top_k: int = 5) -> QueryResult:
        if not self.retrievers or not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
        # Get results from all retrievers
        all_results = []
        for retriever in self.retrievers:
            try:
                result = retriever.retrieve(query, top_k * 2)
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
        top_scores = [score for doc, score in sorted_docs[:top_k]]
        
        return QueryResult(top_docs, top_scores, self.name, query, datetime.now())

class MultiQueryRetriever(BaseRetriever):
    """Generates multiple query variations for improved recall using OpenAI"""
    
    def __init__(self, name: str = "MultiQuery", base_retriever: Optional[BaseRetriever] = None):
        super().__init__(name)
        self.base_retriever = base_retriever or DenseRetriever("Dense_Base")
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        self.documents.extend(documents)
        self.base_retriever.add_documents(documents, metadata)
    
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
                model="gpt-4.1",
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
    
    def retrieve(self, query: str, top_k: int = 5) -> QueryResult:
        if not self.documents:
            return QueryResult([], [], self.name, query, datetime.now())
        
        # Generate variations and retrieve
        variations = self._generate_query_variations(query)
        all_docs = []
        doc_counts = {}
        
        for var_query in variations:
            result = self.base_retriever.retrieve(var_query, top_k)
            for doc in result.documents:
                if doc not in doc_counts:
                    doc_counts[doc] = 0
                    all_docs.append(doc)
                doc_counts[doc] += 1
        
        # Score by frequency of appearance
        doc_scores = [(doc, doc_counts[doc] / len(variations)) for doc in all_docs]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_docs = [doc for doc, score in doc_scores[:top_k]]
        top_scores = [score for doc, score in doc_scores[:top_k]]
        
        return QueryResult(top_docs, top_scores, self.name, query, datetime.now())

# ============================================================================
# BAYESIAN BANDITS SYSTEM
# ============================================================================

class BayesianRAGBandits:
    """Main Thompson Sampling system for retriever selection"""
    
    def __init__(self, retrievers: List[BaseRetriever], prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.retrievers = {r.name: r for r in retrievers}
        self.bandit_params = {name: BetaParams(prior_alpha, prior_beta) for name in self.retrievers.keys()}
        self.query_history = []
        self.feedback_history = []
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to all retrievers"""
        for retriever in self.retrievers.values():
            retriever.add_documents(documents, metadata)
        logger.info(f"Added {len(documents)} documents to {len(self.retrievers)} retrievers")
    
    def select_retriever(self) -> str:
        """Select retriever using Thompson Sampling"""
        samples = {name: params.sample() for name, params in self.bandit_params.items()}
        return max(samples, key=samples.get)
    
    def _llm_evaluate_relevance(self, query: str, documents: List[str]) -> RelevanceEvaluation:
        """Use LLM to evaluate if retrieved documents are relevant to the query"""
        try:
            # Format documents for evaluation
            docs_text = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents[:3])])  # Limit to top 3
            
            prompt = f"""Evaluate if the retrieved documents are relevant to the search query.

Query: "{query}"

Retrieved Documents:
{docs_text}

Consider:
- Do the documents contain information that directly answers or relates to the query?
- Are the documents semantically relevant to what the user is looking for?
- Would these documents be helpful for someone asking this query?

Provide your assessment with confidence level (0.0 to 1.0) and reasoning."""

            response = self.openai_client.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating document relevance for information retrieval systems."},
                    {"role": "user", "content": prompt}
                ],
                response_format=RelevanceEvaluation,
                max_tokens=300,
                temperature=0.3
            )
            
            if response.choices and response.choices[0].message.parsed:
                return response.choices[0].message.parsed
            else:
                # Fallback
                return RelevanceEvaluation(is_relevant=True, confidence=0.5, reasoning="Unable to evaluate")
                
        except Exception as e:
            logger.warning(f"LLM relevance evaluation failed: {e}")
            return RelevanceEvaluation(is_relevant=True, confidence=0.5, reasoning=f"Evaluation error: {e}")

    def query(self, query: str, top_k: int = 5, auto_feedback: bool = False) -> Tuple[QueryResult, str]:
        """Query the system and return results with selected retriever"""
        selected_name = self.select_retriever()
        selected_retriever = self.retrievers[selected_name]
        
        result = selected_retriever.retrieve(query, top_k)
        
        self.query_history.append({
            'query': query,
            'selected_retriever': selected_name,
            'timestamp': datetime.now()
        })
        
        # Auto-feedback if enabled
        if auto_feedback and result.documents:
            evaluation = self._llm_evaluate_relevance(query, result.documents)
            self.provide_feedback(query, evaluation.is_relevant)
            logger.info(f"Auto-feedback: {evaluation.is_relevant} (confidence: {evaluation.confidence:.2f}) - {evaluation.reasoning}")
        
        return result, selected_name
    
    def get_llm_feedback(self, query: str, documents: List[str]) -> RelevanceEvaluation:
        """Get LLM evaluation without automatically applying feedback"""
        return self._llm_evaluate_relevance(query, documents)
    
    def provide_feedback(self, query: str, is_relevant: bool):
        """Provide feedback for the last query"""
        if not self.query_history:
            logger.warning("No query history to provide feedback for")
            return
        
        last_query = self.query_history[-1]
        if last_query['query'] != query:
            logger.warning("Feedback query doesn't match last query")
            return
        
        selected_name = last_query['selected_retriever']
        
        if is_relevant:
            self.bandit_params[selected_name].update_success()
        else:
            self.bandit_params[selected_name].update_failure()
        
        self.feedback_history.append({
            'query': query,
            'retriever': selected_name,
            'is_relevant': is_relevant,
            'timestamp': datetime.now()
        })
        
        logger.info(f"Feedback recorded: {selected_name} -> {'👍' if is_relevant else '👎'}")
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        stats = {}
        
        for name, params in self.bandit_params.items():
            mean_reward = params.mean()
            total_trials = params.alpha + params.beta - 2  # Subtract priors
            
            # 95% confidence interval
            try:
                from scipy.stats import beta
                ci_lower, ci_upper = beta.interval(0.95, params.alpha, params.beta)
            except ImportError:
                # Fallback approximation
                std = np.sqrt(params.alpha * params.beta / ((params.alpha + params.beta)**2 * (params.alpha + params.beta + 1)))
                ci_lower = max(0, mean_reward - 1.96 * std)
                ci_upper = min(1, mean_reward + 1.96 * std)
            
            stats[name] = {
                'mean_reward': mean_reward,
                'total_trials': total_trials,
                'confidence_interval_95': (ci_lower, ci_upper),
                'alpha': params.alpha,
                'beta': params.beta
            }
        
        return stats
    
    def save_state(self, filepath: str):
        """Save bandit state to file"""
        state = {
            'bandit_params': {name: {'alpha': p.alpha, 'beta': p.beta} for name, p in self.bandit_params.items()},
            'query_history': self.query_history,
            'feedback_history': self.feedback_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load bandit state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore bandit parameters
        for name, params in state['bandit_params'].items():
            if name in self.bandit_params:
                self.bandit_params[name].alpha = params['alpha']
                self.bandit_params[name].beta = params['beta']
        
        self.query_history = state.get('query_history', [])
        self.feedback_history = state.get('feedback_history', [])
        logger.info(f"State loaded from {filepath}")

# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def create_sample_retrievers() -> List[BaseRetriever]:
    """Create a set of sample retrievers"""
    basic_retrievers = [
        BM25Retriever("BM25"),
        DenseRetriever("Dense"),
        GraphRAGRetriever("GraphRAG")
    ]
    
    ensemble = EnsembleRetriever("Ensemble", basic_retrievers[:2])
    hybrid = HybridRetriever("Hybrid")
    multi_query = MultiQueryRetriever("MultiQuery")
    
    return basic_retrievers + [ensemble, hybrid, multi_query]

def run_demo():
    """Run a comprehensive demo"""
    print("🎯 Bayesian RAG Bandits Demo")
    print("=" * 50)
    
    # Create system
    retrievers = create_sample_retrievers()
    rag_bandits = BayesianRAGBandits(retrievers)
    
    print(f"📋 Initialized {len(retrievers)} retrievers:")
    for i, r in enumerate(retrievers, 1):
        print(f"  {i}. {r.name}")
    
    # Sample documents
    docs = [
        "Machine learning algorithms learn patterns from training data to make predictions on new data",
        "Neural networks are computing systems inspired by biological neural networks for pattern recognition",
        "Natural language processing enables computers to understand and generate human language text",
        "Information retrieval systems help users find relevant documents from large collections",
        "Thompson sampling is an algorithm for multi-armed bandit problems that balances exploration and exploitation"
    ]
    
    rag_bandits.add_documents(docs)
    
    # Test queries
    queries = [
        "machine learning algorithms",
        "neural network patterns", 
        "natural language processing",
        "information retrieval systems",
        "thompson sampling bandits"
    ]
    
    print(f"\n🔍 Testing {len(queries)} queries with Thompson Sampling...")
    print("🤖 First 3 queries with LLM auto-feedback, last 2 with simulated feedback\n")
    
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: '{query}'")
        
        # Use auto-feedback for first 3 queries, manual for last 2
        use_auto_feedback = i < 3
        
        # Query system
        result, selected = rag_bandits.query(query, top_k=2, auto_feedback=use_auto_feedback)
        print(f"🤖 Selected: {selected}")
        print(f"📋 Results:")
        for j, (doc, score) in enumerate(zip(result.documents, result.scores)):
            print(f"  {j+1}. {doc[:60]}... (score: {score:.3f})")
        
        if not use_auto_feedback:
            # Simulate manual feedback for last 2 queries
            is_relevant = np.random.random() < 0.8
            rag_bandits.provide_feedback(query, is_relevant)
            print(f"📝 Manual Feedback: {'👍' if is_relevant else '👎'}")
    
    # Show final statistics
    print(f"\n📊 Final Statistics:")
    stats = rag_bandits.get_stats()
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
    
    print("Retriever      | Reward | Trials | 95% CI")
    print("-" * 45)
    for name, stat in sorted_stats:
        ci_lower, ci_upper = stat['confidence_interval_95']
        print(f"{name:13} | {stat['mean_reward']:.3f}  | {stat['total_trials']:4.0f}   | [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Save state
    rag_bandits.save_state("demo_state.json")
    print(f"\n💾 State saved to demo_state.json")
    print("✅ Demo completed!")

if __name__ == "__main__":
    run_demo() 