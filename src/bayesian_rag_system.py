#!/usr/bin/env python3
"""
Bayesian RAG Bandits - Complete System
A production-ready RAG retriever selection system using Thompson Sampling

Features:
- Thompson Sampling for optimal retriever selection
- 11 different retriever types (basic, advanced, next-gen with LLM enhancement)
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
import logging
import openai
import os
from pydantic import BaseModel

# Import only what we actually use
from .retrievers import (
    BaseRetriever,     # For type hints
    QueryResult,       # For return types
    create_sample_retrievers  # Factory function that creates all retrievers
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

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
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, doc_ids: Optional[List[str]] = None):
        """Add documents to all retrievers"""
        for retriever in self.retrievers.values():
            try:
                retriever.add_documents(documents, metadata, doc_ids)
            except TypeError:
                # Fallback for retrievers that don't support doc_ids yet
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