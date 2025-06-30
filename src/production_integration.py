#!/usr/bin/env python3
"""
Production Integration Example

Shows how to integrate Bayesian RAG Bandits into production systems:
- API endpoints
- Database integration  
- Async feedback collection
- Performance monitoring
"""

from src import BayesianRAGBandits, BM25Retriever, DenseRetriever, HybridRetriever
from typing import List, Dict, Any, Optional
import json
import asyncio
from datetime import datetime

class ProductionRAGService:
    """Production-ready RAG service with bandit optimization"""
    
    def __init__(self, config_path: str = "rag_config.json"):
        self.rag_bandits = None
        self.config = self._load_config(config_path)
        self._initialize_system()
        
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration (creates default if missing)"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default config
            config = {
                "retrievers": ["BM25", "Dense", "Hybrid"],
                "prior_alpha": 1.0,
                "prior_beta": 1.0,
                "top_k_default": 5,
                "state_file": "production_rag_state.json"
            }
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            return config
    
    def _initialize_system(self):
        """Initialize retrievers and bandit system"""
        # Create retrievers based on config
        retrievers = []
        for retriever_name in self.config["retrievers"]:
            if retriever_name == "BM25":
                retrievers.append(BM25Retriever("BM25"))
            elif retriever_name == "Dense":
                retrievers.append(DenseRetriever("Dense"))
            elif retriever_name == "Hybrid":
                retrievers.append(HybridRetriever("Hybrid"))
        
        # Initialize bandit system
        self.rag_bandits = BayesianRAGBandits(
            retrievers,
            prior_alpha=self.config["prior_alpha"],
            prior_beta=self.config["prior_beta"]
        )
        
        # Load existing state if available
        try:
            self.rag_bandits.load_state(self.config["state_file"])
            print(f"✅ Loaded existing state from {self.config['state_file']}")
        except FileNotFoundError:
            print("🆕 Starting with fresh state")
    
    def add_document_collection(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to the system"""
        self.rag_bandits.add_documents(documents, metadata)
        print(f"📚 Added {len(documents)} documents to retrieval system")
    
    def search(self, query: str, top_k: Optional[int] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Main search function - returns results with metadata"""
        if top_k is None:
            top_k = self.config["top_k_default"]
        
        # Get results from bandit system
        result, selected_retriever = self.rag_bandits.query(query, top_k)
        
        # Format response
        response = {
            "query": query,
            "results": [
                {
                    "document": doc,
                    "score": score,
                    "rank": i + 1
                }
                for i, (doc, score) in enumerate(zip(result.documents, result.scores))
            ],
            "retriever_used": selected_retriever,
            "timestamp": result.timestamp.isoformat(),
            "user_id": user_id,
            "query_id": f"{user_id}_{int(result.timestamp.timestamp())}" if user_id else None
        }
        
        return response
    
    def provide_feedback(self, query: str, is_helpful: bool, user_id: Optional[str] = None):
        """Record user feedback for learning"""
        self.rag_bandits.provide_feedback(query, is_helpful)
        
        # Log feedback for analytics
        feedback_log = {
            "query": query,
            "is_helpful": is_helpful,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # In production, you'd save this to a database
        print(f"📝 Feedback recorded: {query} -> {'👍' if is_helpful else '👎'}")
        
        # Auto-save state periodically
        self._save_state()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance analytics"""
        stats = self.rag_bandits.get_stats()
        
        # Calculate additional metrics
        total_queries = len(self.rag_bandits.query_history)
        total_feedback = len(self.rag_bandits.feedback_history)
        feedback_rate = total_feedback / total_queries if total_queries > 0 else 0
        
        metrics = {
            "total_queries": total_queries,
            "total_feedback": total_feedback,
            "feedback_rate": feedback_rate,
            "retriever_performance": stats,
            "last_updated": datetime.now().isoformat()
        }
        
        return metrics
    
    def _save_state(self):
        """Save current bandit state"""
        self.rag_bandits.save_state(self.config["state_file"])

# Example usage in different production scenarios

def api_endpoint_example():
    """Example: Using in a REST API"""
    print("🌐 API Endpoint Example")
    print("-" * 30)
    
    # Initialize service
    rag_service = ProductionRAGService()
    
    # Add documents (in production, this might come from a database)
    documents = [
        "FastAPI is a modern Python web framework for building APIs",
        "Docker containers provide consistent deployment environments",
        "PostgreSQL is a powerful open-source relational database",
        "Redis provides in-memory data structure storage and caching"
    ]
    rag_service.add_document_collection(documents)
    
    # Simulate API calls
    api_calls = [
        {"query": "python web framework", "user_id": "user123"},
        {"query": "container deployment", "user_id": "user456"},
        {"query": "database storage", "user_id": "user123"}
    ]
    
    for call in api_calls:
        # Search
        response = rag_service.search(call["query"], user_id=call["user_id"])
        print(f"🔍 Query: {call['query']}")
        print(f"   Retriever: {response['retriever_used']}")
        print(f"   Top result: {response['results'][0]['document'][:50]}...")
        
        # Simulate user feedback (in production, this comes later via separate endpoint)
        is_helpful = len(response['results']) > 0  # Simple simulation
        rag_service.provide_feedback(call["query"], is_helpful, call["user_id"])
        print()

def batch_processing_example():
    """Example: Batch processing with feedback collection"""
    print("📊 Batch Processing Example")
    print("-" * 30)
    
    service = ProductionRAGService()
    
    # Add documents
    docs = [
        "Machine learning models require training data and validation",
        "Data preprocessing is crucial for model performance",
        "Feature engineering improves prediction accuracy"
    ]
    service.add_document_collection(docs)
    
    # Batch queries
    queries = [
        "training machine learning models",
        "data preprocessing steps", 
        "feature engineering techniques"
    ]
    
    results = []
    for query in queries:
        result = service.search(query)
        results.append(result)
        
        # Simulate automated feedback based on result quality
        has_results = len(result['results']) > 0
        service.provide_feedback(query, has_results)
    
    # Get performance metrics
    metrics = service.get_performance_metrics()
    print(f"Processed {metrics['total_queries']} queries")
    print(f"Feedback rate: {metrics['feedback_rate']:.1%}")

async def async_feedback_example():
    """Example: Asynchronous feedback collection"""
    print("⚡ Async Feedback Example")
    print("-" * 30)
    
    service = ProductionRAGService()
    
    # Add documents
    docs = ["Async programming enables concurrent execution of tasks"]
    service.add_document_collection(docs)
    
    # Simulate delayed feedback (common in production)
    query = "async programming tasks"
    result = service.search(query, user_id="async_user")
    
    print(f"🔍 Immediate search: {query}")
    print(f"   Results returned: {len(result['results'])}")
    
    # Simulate user taking time to evaluate
    await asyncio.sleep(1)  # User thinks about it
    
    # Feedback comes later
    service.provide_feedback(query, True, "async_user")
    print("📝 Delayed feedback received")

def main():
    """Run all production examples"""
    print("🏭 Production Integration Examples")
    print("=" * 50)
    print()
    
    # Run examples
    api_endpoint_example()
    print()
    batch_processing_example()
    print()
    
    # Run async example
    asyncio.run(async_feedback_example())
    
    print("\n✅ All production examples completed!")
    print("\nKey takeaways:")
    print("• No user interaction required for core functionality")
    print("• Feedback can be collected asynchronously") 
    print("• System learns continuously from any feedback source")
    print("• State persists between application restarts")
    print("• Performance metrics available for monitoring")

if __name__ == "__main__":
    main() 