"""
Bayesian RAG Bandits - Thompson Sampling for RAG Retriever Selection

A production-ready RAG retriever selection system using Thompson Sampling
for optimal retriever selection with online learning from user feedback.

Main Components:
- BayesianRAGBandits: Core Thompson Sampling system
- 11 retrievers: BM25, Dense, GraphRAG, Hybrid, Ensemble, MultiQuery, ReRanker, HyDE, SelfQuery, TimeWeighted, ContextualCompression
- Production utilities: State persistence, LLM evaluation, statistics

Example Usage:
    from src import BayesianRAGBandits, create_sample_retrievers
    
    retrievers = create_sample_retrievers()
    system = BayesianRAGBandits(retrievers)
    system.add_documents(["doc1", "doc2", "doc3"])
    
    result, selected = system.query("your query", auto_feedback=True)
"""

from .bayesian_rag_system import (
    BayesianRAGBandits,
    BetaParams,
    RelevanceEvaluation,
    run_demo
)

from .retrievers import (
    # Base classes
    BaseRetriever,
    QueryResult,
    
    # Basic retrievers
    BM25Retriever,
    DenseRetriever,
    GraphRAGRetriever,
    
    # Advanced retrievers
    HybridRetriever,
    EnsembleRetriever,
    MultiQueryRetriever,
    
    # Next-gen retrievers
    ReRankerRetriever,
    HyDERetriever,
    SelfQueryRetriever,
    TimeWeightedRetriever,
    ContextualCompressionRetriever,
    
    # Utility functions
    create_sample_retrievers,
    
    # Models
    QueryVariations
)

__version__ = "1.0.0"
__author__ = "Tyler Gibbs"

# Main exports
__all__ = [
    # Core system
    "BayesianRAGBandits",
    "BetaParams",
    "RelevanceEvaluation",
    
    # Base classes
    "BaseRetriever", 
    "QueryResult",
    
    # Basic retrievers
    "BM25Retriever",
    "DenseRetriever", 
    "GraphRAGRetriever",
    
    # Advanced retrievers
    "HybridRetriever",
    "EnsembleRetriever",
    "MultiQueryRetriever",
    
    # Next-gen retrievers
    "ReRankerRetriever",
    "HyDERetriever",
    "SelfQueryRetriever",
    "TimeWeightedRetriever",
    "ContextualCompressionRetriever",
    
    # Utilities
    "create_sample_retrievers",
    "run_demo",
    
    # Models
    "QueryVariations"
]
