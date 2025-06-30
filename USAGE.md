# Bayesian RAG Bandits - Usage Guide

## Quick Start

```python
from bayesian_rag_system import BayesianRAGBandits, BM25Retriever, DenseRetriever

# 1. Create retrievers
retrievers = [
    BM25Retriever("BM25"),      # Keyword matching
    DenseRetriever("Dense")     # Semantic search
]

# 2. Initialize system
rag_system = BayesianRAGBandits(retrievers)

# 3. Add documents
docs = ["Your documents here..."]
rag_system.add_documents(docs)

# 4. Query (Thompson Sampling selects best retriever)
result, selected_retriever = rag_system.query("your query")

# 5. Provide real feedback when available
rag_system.provide_feedback("your query", is_helpful=True)
```

## Core Concepts

### Thompson Sampling
- Automatically selects the best retriever for each query
- Balances exploration vs exploitation
- Learns from minimal feedback (even 1-2 samples help)

### Real Feedback Sources
- **User clicks**: Which results do users click?
- **Ratings**: Explicit thumbs up/down or star ratings  
- **Session success**: Did user find what they needed?
- **Business metrics**: Conversions, purchases, signups
- **Time spent**: How long users engage with results

### No Simulation Needed
The system works with actual user behavior:
```python
# Real click tracking
if user_clicked_result:
    rank = user_clicked_rank
    is_good = rank <= 2  # Top 2 clicks = positive
    rag_system.provide_feedback(query, is_good)

# Real ratings
if user_rating >= 3:  # 3+ stars out of 5
    rag_system.provide_feedback(query, True)

# Real business outcomes
if user_made_purchase:
    rag_system.provide_feedback(query, True)
```

## Available Retrievers

1. **BM25Retriever**: Keyword/lexical matching
2. **DenseRetriever**: Semantic embeddings  
3. **GraphRAGRetriever**: Graph-based with PageRank
4. **HybridRetriever**: Combines BM25 + Dense
5. **EnsembleRetriever**: Combines multiple retrievers
6. **MultiQueryRetriever**: Generates query variations

## Production Integration

See `production_integration.py` for complete examples:
- REST API integration
- Async feedback collection
- Performance monitoring
- State persistence

## Files

- `bayesian_rag_system.py` - Main system (all retrievers included)
- `real_feedback_example.py` - Real feedback integration examples
- `production_integration.py` - Production-ready service examples
- `requirements.txt` - Dependencies

## Key Benefits

✅ **No retraining required** - learns online from feedback  
✅ **Works with sparse feedback** - even 1-2 samples help  
✅ **Production ready** - state persistence, error handling  
✅ **Multiple retrievers** - automatically finds the best one  
✅ **Real metrics** - learns from actual user behavior  