#!/usr/bin/env python3
"""
Main entry point for Bayesian RAG Bandits System

Enhanced with Rich terminal UI for beautiful output.
Run this file to execute the interactive demo.
"""

import time
import logging
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule
from rich.status import Status
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich.layout import Layout
from rich.align import Align

from src import BayesianRAGBandits, create_sample_retrievers

# Configure logging AFTER import to suppress logs for clean Rich output
# (bayesian_rag_system.py sets basicConfig, so we override after import)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("src.retrievers").setLevel(logging.WARNING)
logging.getLogger("__main__").setLevel(logging.WARNING)
# Keep bayesian_rag_system at INFO to see auto-feedback messages
logging.getLogger("src.bayesian_rag_system").setLevel(logging.INFO)
# Set root logger to INFO to see auto-feedback
logging.getLogger().setLevel(logging.INFO)

# Initialize Rich console
console = Console()

def create_title_panel():
    """Create the main title panel"""
    title = Text.assemble(
        ("🎯 ", "bold red"),
        ("Bayesian RAG Bandits", "bold blue"),
        (" - Thompson Sampling Demo", "bold white")
    )
    return Panel(
        Align.center(title),
        border_style="bright_blue",
        padding=(1, 2)
    )

def create_retrievers_table(retrievers):
    """Create a table showing all retrievers"""
    table = Table(title="🤖 Available Retrievers", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="cyan", width=20)
    table.add_column("Type", style="green", width=15)
    table.add_column("Description", style="white")

    retriever_info = {
        "BM25": ("Basic", "Lexical matching with BM25 scoring"),
        "Dense": ("Basic", "Semantic embeddings with OpenAI"),
        "GraphRAG": ("Basic", "PageRank-based graph retrieval"),
        "Hybrid": ("Advanced", "BM25 + Dense combination"),
        "Ensemble": ("Advanced", "Reciprocal Rank Fusion"),
        "MultiQuery": ("Advanced", "Query variations with LLM"),
        "ReRanker": ("Next-gen", "Cross-encoder LLM reranking"),
        "HyDE": ("Next-gen", "Hypothetical document generation"),
        "SelfQuery": ("Next-gen", "Natural language query parsing"),
        "TimeWeighted": ("Next-gen", "Recency-biased retrieval"),
        "ContextualCompression": ("Next-gen", "Query-relevant compression")
    }
    
    for i, retriever in enumerate(retrievers, 1):
        name = retriever.name
        ret_type, desc = retriever_info.get(name, ("Unknown", "Custom retriever"))
        
        # Color code by type
        type_style = {
            "Basic": "green",
            "Advanced": "yellow", 
            "Next-gen": "magenta"
        }.get(ret_type, "white")
        
        table.add_row(
            str(i),
            f"[bold]{name}[/bold]",
            f"[{type_style}]{ret_type}[/{type_style}]",
            desc
        )
    
    return table

def create_stats_table(stats):
    """Create a table showing retriever statistics"""
    table = Table(title="📊 Performance Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Retriever", style="white", width=18)
    table.add_column("Mean Reward", style="green", width=12)
    table.add_column("Trials", style="yellow", width=8)
    table.add_column("95% CI", style="blue", width=20)
    table.add_column("Status", style="magenta", width=10)

    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
    
    for i, (name, stat) in enumerate(sorted_stats):
        ci_lower, ci_upper = stat['confidence_interval_95']
        
        # Determine status emoji
        if i == 0:
            status = "🏆 Best"
        elif stat['mean_reward'] > 0.7:
            status = "✅ Good"
        elif stat['mean_reward'] > 0.5:
            status = "⚠️ Fair"
        else:
            status = "❌ Poor"
        
        # Color code the reward
        reward = stat['mean_reward']
        if reward > 0.7:
            reward_str = f"[bold green]{reward:.3f}[/bold green]"
        elif reward > 0.5:
            reward_str = f"[yellow]{reward:.3f}[/yellow]"
        else:
            reward_str = f"[red]{reward:.3f}[/red]"
        
        table.add_row(
            f"[bold]{name}[/bold]",
            reward_str,
            str(int(stat['total_trials'])),
            f"[{ci_lower:.3f}, {ci_upper:.3f}]",
            status
        )
    
    return table

def run_rich_demo():
    """Run the enhanced demo with Rich formatting"""
    # Title
    console.print(create_title_panel())
    console.print()
    
    # Initialize system with progress
    with Status("[bold green]Initializing retrieval system...", console=console) as status:
        retrievers = create_sample_retrievers()
        rag_bandits = BayesianRAGBandits(retrievers)
        time.sleep(1)  # Dramatic pause
    
    console.print("✅ [bold green]System initialized successfully![/bold green]\n")
    
    # Show retrievers table
    console.print(create_retrievers_table(retrievers))
    console.print()
    
    # Sample documents
    docs = [
        "Machine learning algorithms learn patterns from training data to make predictions on new data",
        "Neural networks are computing systems inspired by biological neural networks for pattern recognition", 
        "Natural language processing enables computers to understand and generate human language text",
        "Information retrieval systems help users find relevant documents from large collections",
        "Thompson sampling is an algorithm for multi-armed bandit problems that balances exploration and exploitation"
    ]
    
    # Add documents with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Adding documents..."),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Adding documents", total=len(retrievers))
        
        for retriever in retrievers:
            retriever.add_documents(docs)
            progress.advance(task)
            time.sleep(0.1)  # Visual effect
    
    console.print(f"✅ [bold green]Added {len(docs)} documents to all {len(retrievers)} retrievers[/bold green]\n")
    
    # Test queries
    queries = [
        "machine learning algorithms",
        "neural network patterns", 
        "natural language processing", 
        "information retrieval systems",
        "thompson sampling bandits"
    ]
    
    # Query execution
    console.print(Panel(
        f"[bold yellow]🔍 Testing {len(queries)} queries with Thompson Sampling[/bold yellow]\n"
        f"[dim]First 3 queries use LLM auto-feedback, last 2 use simulated feedback[/dim]",
        border_style="yellow"
    ))
    console.print()
    
    for i, query in enumerate(queries):
        # Query header
        query_panel = Panel(
            f"[bold cyan]Query {i+1}:[/bold cyan] '{query}'",
            border_style="cyan",
            padding=(0, 2)
        )
        console.print(query_panel)
        
        # Execute query with status
        with Status(f"[bold blue]Processing query {i+1}...", console=console):
            use_auto_feedback = i < 3
            result, selected = rag_bandits.query(query, top_k=2, auto_feedback=use_auto_feedback)
            time.sleep(0.5)  # Dramatic pause
        
        # Results table
        results_table = Table(show_header=True, header_style="bold white", border_style="blue")
        results_table.add_column("Rank", width=6)
        results_table.add_column("Document", width=60)
        results_table.add_column("Score", width=10)
        
        for j, (doc, score) in enumerate(zip(result.documents, result.scores)):
            rank_style = "bold green" if j == 0 else "yellow" if j == 1 else "white"
            results_table.add_row(
                f"[{rank_style}]#{j+1}[/{rank_style}]",
                doc[:55] + "..." if len(doc) > 55 else doc,
                f"[bold]{score:.3f}[/bold]"
            )
        
        console.print(f"🤖 [bold magenta]Selected Retriever:[/bold magenta] [bold white]{selected}[/bold white]")
        console.print(results_table)
        
        # Feedback
        if not use_auto_feedback:
            is_relevant = np.random.random() < 0.8
            rag_bandits.provide_feedback(query, is_relevant)
            feedback_color = "green" if is_relevant else "red"
            feedback_emoji = "👍" if is_relevant else "👎"
            console.print(f"📝 [bold {feedback_color}]Manual Feedback: {feedback_emoji}[/bold {feedback_color}]")
        
        console.print()
    
    # Final statistics
    console.print(Rule(title="[bold blue]Final Results[/bold blue]", style="blue"))
    console.print()
    
    stats = rag_bandits.get_stats()
    console.print(create_stats_table(stats))
    console.print()
    
    # Summary panels
    summary_panels = [
        Panel(
            "[bold green]✅ Demo Completed![/bold green]\n"
            f"• {len(queries)} queries processed\n"
            f"• {len(retrievers)} retrievers tested\n"
            "• Thompson Sampling active",
            title="Summary",
            border_style="green"
        ),
        Panel(
            "State saved to:\n"
            "[bold]demo_state.json[/bold]\n\n"
            "Ready for production!",
            title="💾 Persistence",
            border_style="blue"
        )
    ]
    
    console.print(Columns(summary_panels))
    
    # Save state
    with Status("[bold blue]Saving state...", console=console):
        rag_bandits.save_state("demo_state.json")
        time.sleep(0.5)
    
    console.print("\n🎉 [bold green]All done! Your Bayesian RAG system is ready to deploy.[/bold green]")

def main():
    """Main entry point with Rich UI"""
    try:
        run_rich_demo()
    except KeyboardInterrupt:
        console.print("\n[bold red]Demo interrupted by user.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        console.print_exception()

if __name__ == "__main__":
    main() 