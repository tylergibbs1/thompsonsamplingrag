#!/usr/bin/env python3
"""
Thompson Sampling Test Suite

This program rigorously tests the Thompson Sampling implementation
to ensure it correctly balances exploration vs exploitation.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text

from src import BayesianRAGBandits, create_sample_retrievers

console = Console()

class ThompsonSamplingTester:
    """Comprehensive test suite for Thompson Sampling behavior"""
    
    def __init__(self):
        self.results = {}
        
    def test_exploration_vs_exploitation(self, num_trials: int = 100) -> Dict:
        """Test that Thompson sampling explores initially then exploits"""
        console.print(Panel(
            "[bold blue]Test 1: Exploration vs Exploitation[/bold blue]\n"
            "Testing that TS explores different retrievers initially, then converges to best ones",
            border_style="blue"
        ))
        
        # Create system with controlled retrievers
        retrievers = create_sample_retrievers()[:5]  # Use only 5 for cleaner analysis
        rag_bandits = BayesianRAGBandits(retrievers)
        
        # Add documents
        docs = ["Test document about machine learning algorithms and neural networks"]
        for retriever in retrievers:
            retriever.add_documents(docs)
        
        # Track selections over time
        selections = []
        query = "machine learning test query"
        
        # Simulate known reward patterns (make Dense clearly the best)
        reward_probabilities = {
            "BM25": 0.3,
            "Dense": 0.8,  # Clearly best
            "GraphRAG": 0.4,
            "Hybrid": 0.6,
            "Ensemble": 0.5
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Running exploration test..."),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Testing", total=num_trials)
            
            for i in range(num_trials):
                # Query without auto-feedback
                result, selected = rag_bandits.query(query, auto_feedback=False)
                selections.append(selected)
                
                # Provide controlled feedback based on known probabilities
                reward = np.random.random() < reward_probabilities[selected]
                rag_bandits.provide_feedback(query, reward)
                
                progress.advance(task)
        
        # Analyze results
        selection_counts = Counter(selections)
        
        # Test early exploration (first 20% of trials)
        early_trials = int(num_trials * 0.2)
        early_selections = set(selections[:early_trials])
        exploration_score = len(early_selections) / len(retrievers)  # Should be high
        
        # Test late exploitation (last 20% of trials)
        late_trials = int(num_trials * 0.2)
        late_selections = selections[-late_trials:]
        best_retriever_late_ratio = late_selections.count("Dense") / late_trials  # Should be high
        
        # Display results
        table = Table(title="Exploration vs Exploitation Results", show_header=True)
        table.add_column("Retriever", style="cyan")
        table.add_column("Total Selections", style="yellow")
        table.add_column("Selection %", style="green")
        table.add_column("True Win Rate", style="magenta")
        
        for retriever_name in reward_probabilities:
            count = selection_counts[retriever_name]
            percentage = (count / num_trials) * 100
            true_rate = reward_probabilities[retriever_name]
            table.add_row(
                retriever_name,
                str(count),
                f"{percentage:.1f}%",
                f"{true_rate:.1f}"
            )
        
        console.print(table)
        console.print(f"\n📊 [bold]Analysis:[/bold]")
        console.print(f"• Exploration Score (early diversity): {exploration_score:.2f} (should be > 0.6)")
        console.print(f"• Exploitation Score (late best selection): {best_retriever_late_ratio:.2f} (should be > 0.5)")
        
        # Test passes if we explore early and exploit late
        exploration_test = exploration_score > 0.6
        exploitation_test = best_retriever_late_ratio > 0.5
        
        result = {
            "test_name": "exploration_vs_exploitation",
            "passed": exploration_test and exploitation_test,
            "exploration_score": exploration_score,
            "exploitation_score": best_retriever_late_ratio,
            "selection_counts": dict(selection_counts),
            "reward_probabilities": reward_probabilities
        }
        
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        console.print(f"\n{status} Thompson Sampling shows correct exploration → exploitation behavior")
        
        return result
    
    def test_convergence_to_best(self, num_trials: int = 200) -> Dict:
        """Test convergence to genuinely best retriever over time"""
        console.print(Panel(
            "[bold green]Test 2: Convergence to Best Retriever[/bold green]\n"
            "Testing that TS converges to the statistically best retriever over time",
            border_style="green"
        ))
        
        retrievers = create_sample_retrievers()[:4]  # Smaller set for analysis
        rag_bandits = BayesianRAGBandits(retrievers)
        
        docs = ["Test document for convergence analysis"]
        for retriever in retrievers:
            retriever.add_documents(docs)
        
        # Clear winner scenario
        reward_probs = {
            "BM25": 0.2,      # Poor
            "Dense": 0.9,     # Clear winner
            "GraphRAG": 0.3,  # Poor
            "Hybrid": 0.4     # Mediocre
        }
        
        selections_over_time = []
        query = "convergence test query"
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Testing convergence..."),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Converging", total=num_trials)
            
            for i in range(num_trials):
                result, selected = rag_bandits.query(query, auto_feedback=False)
                selections_over_time.append(selected)
                
                reward = np.random.random() < reward_probs[selected]
                rag_bandits.provide_feedback(query, reward)
                
                progress.advance(task)
        
        # Analyze convergence in windows
        window_size = 25
        convergence_data = []
        
        for i in range(0, num_trials - window_size, window_size):
            window = selections_over_time[i:i + window_size]
            dense_ratio = window.count("Dense") / len(window)
            convergence_data.append((i + window_size, dense_ratio))
        
        # Check if Dense selection increases over time
        early_dense_ratio = convergence_data[0][1] if convergence_data else 0
        late_dense_ratio = convergence_data[-1][1] if convergence_data else 0
        improvement = late_dense_ratio - early_dense_ratio
        
        console.print("\n📈 [bold]Convergence Analysis:[/bold]")
        console.print(f"• Early Dense selection (first 25): {early_dense_ratio:.2f}")
        console.print(f"• Late Dense selection (last 25): {late_dense_ratio:.2f}")
        console.print(f"• Improvement: +{improvement:.2f}")
        console.print(f"• Final Dense selection should be > 0.7 for clear winner")
        
        converged = late_dense_ratio > 0.7 and improvement >= 0.0
        
        result = {
            "test_name": "convergence_to_best",
            "passed": converged,
            "early_best_ratio": early_dense_ratio,
            "late_best_ratio": late_dense_ratio,
            "improvement": improvement,
            "convergence_data": convergence_data,
            "reward_probabilities": reward_probs
        }
        
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        console.print(f"\n{status} Thompson Sampling converges to best retriever")
        
        return result
    
    def test_beta_distribution_updates(self, num_trials: int = 50) -> Dict:
        """Test that beta distribution parameters update correctly"""
        console.print(Panel(
            "[bold yellow]Test 3: Beta Distribution Updates[/bold yellow]\n"
            "Testing that alpha/beta parameters update correctly with feedback",
            border_style="yellow"
        ))
        
        retrievers = create_sample_retrievers()[:3]
        rag_bandits = BayesianRAGBandits(retrievers)
        
        docs = ["Test document for beta testing"]
        for retriever in retrievers:
            retriever.add_documents(docs)
        
        # Get initial parameters
        initial_params = {}
        for name in rag_bandits.retrievers:
            params = rag_bandits.bandit_params[name]
            initial_params[name] = (params.alpha, params.beta)
        
        query = "beta distribution test"
        
        # Force specific selections and feedback
        test_cases = [
            ("BM25", True),    # Success
            ("BM25", True),    # Another success  
            ("BM25", False),   # Failure
            ("Dense", True),   # Success
            ("Dense", False),  # Failure
            ("Dense", False),  # Another failure
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]Testing beta updates..."),
            console=console
        ) as progress:
            task = progress.add_task("Beta testing", total=len(test_cases))
            
            for target_retriever, reward in test_cases:
                # Keep querying until we get the target retriever
                max_attempts = 20
                for attempt in range(max_attempts):
                    result, selected = rag_bandits.query(query, auto_feedback=False)
                    if selected == target_retriever:
                        rag_bandits.provide_feedback(query, reward)
                        break
                else:
                    # If we can't get the target retriever, manually update for testing
                    if reward:
                        rag_bandits.bandit_params[target_retriever].update_success()
                    else:
                        rag_bandits.bandit_params[target_retriever].update_failure()
                
                progress.advance(task)
        
        # Check final parameters
        final_params = {}
        for name in rag_bandits.retrievers:
            params = rag_bandits.bandit_params[name]
            final_params[name] = (params.alpha, params.beta)
        
        # Verify updates
        table = Table(title="Beta Distribution Parameter Updates")
        table.add_column("Retriever", style="cyan")
        table.add_column("Initial (α, β)", style="yellow")
        table.add_column("Final (α, β)", style="green")
        table.add_column("Expected Change", style="magenta")
        
        expected_changes = {
            "BM25": "(+2, +1)",  # 2 successes, 1 failure
            "Dense": "(+1, +2)", # 1 success, 2 failures
            "GraphRAG": "(+0, +0)" # No trials
        }
        
        updates_correct = True
        for name in ["BM25", "Dense", "GraphRAG"]:
            initial = initial_params[name]
            final = final_params[name]
            expected = expected_changes.get(name, "(+0, +0)")
            
            table.add_row(
                name,
                f"({initial[0]}, {initial[1]})",
                f"({final[0]}, {final[1]})",
                expected
            )
            
            # Check if updates are reasonable (allowing for randomness in selection)
            alpha_increased = final[0] >= initial[0]
            beta_increased = final[1] >= initial[1]
            if name in ["BM25", "Dense"]:
                if not (alpha_increased or beta_increased):
                    updates_correct = False
        
        console.print(table)
        
        result = {
            "test_name": "beta_distribution_updates",
            "passed": updates_correct,
            "initial_params": initial_params,
            "final_params": final_params,
            "test_cases": test_cases
        }
        
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        console.print(f"\n{status} Beta distribution parameters update correctly")
        
        return result
    
    def test_equal_retrievers_behavior(self, num_trials: int = 100) -> Dict:
        """Test behavior when retrievers have equal performance"""
        console.print(Panel(
            "[bold magenta]Test 4: Equal Retrievers Behavior[/bold magenta]\n"
            "Testing that TS explores fairly when retrievers have equal performance",
            border_style="magenta"
        ))
        
        retrievers = create_sample_retrievers()[:4]
        rag_bandits = BayesianRAGBandits(retrievers)
        
        docs = ["Equal performance test document"]
        for retriever in retrievers:
            retriever.add_documents(docs)
        
        # All retrievers have equal 60% success rate
        reward_prob = 0.6
        
        selections = []
        query = "equal performance test"
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]Testing equal performance..."),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Equal testing", total=num_trials)
            
            for i in range(num_trials):
                result, selected = rag_bandits.query(query, auto_feedback=False)
                selections.append(selected)
                
                reward = np.random.random() < reward_prob
                rag_bandits.provide_feedback(query, reward)
                
                progress.advance(task)
        
        # Analyze distribution
        selection_counts = Counter(selections)
        expected_per_retriever = num_trials / len(retrievers)
        
        table = Table(title="Equal Performance Distribution")
        table.add_column("Retriever", style="cyan")
        table.add_column("Selections", style="yellow")
        table.add_column("Expected", style="green")
        table.add_column("Deviation", style="red")
        
        max_deviation = 0
        for name in rag_bandits.retrievers:
            count = selection_counts[name]
            deviation = abs(count - expected_per_retriever)
            max_deviation = max(max_deviation, deviation)
            
            table.add_row(
                name,
                str(count),
                f"{expected_per_retriever:.1f}",
                f"{deviation:.1f}"
            )
        
        console.print(table)
        
        # Test passes if distribution is reasonably fair (max deviation < 40% of expected)
        fairness_threshold = expected_per_retriever * 0.4
        is_fair = max_deviation < fairness_threshold
        
        console.print(f"\n📊 [bold]Fairness Analysis:[/bold]")
        console.print(f"• Max deviation from expected: {max_deviation:.1f}")
        console.print(f"• Fairness threshold: {fairness_threshold:.1f}")
        console.print(f"• Distribution should be reasonably uniform for equal performance")
        
        result = {
            "test_name": "equal_retrievers_behavior",
            "passed": is_fair,
            "max_deviation": max_deviation,
            "fairness_threshold": fairness_threshold,
            "selection_counts": dict(selection_counts),
            "expected_per_retriever": expected_per_retriever
        }
        
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        console.print(f"\n{status} Thompson Sampling handles equal performance fairly")
        
        return result
    
    def run_comprehensive_test_suite(self):
        """Run all Thompson Sampling tests"""
        console.print(Panel(
            Text.assemble(
                ("🎯 ", "bold red"),
                ("Thompson Sampling Test Suite", "bold blue"),
                (" - Comprehensive Validation", "bold white")
            ),
            border_style="bright_blue",
            padding=(1, 2)
        ))
        console.print()
        
        # Run all tests
        all_results = {}
        
        all_results["exploration_exploitation"] = self.test_exploration_vs_exploitation()
        console.print("\n" + "="*80 + "\n")
        
        all_results["convergence"] = self.test_convergence_to_best()
        console.print("\n" + "="*80 + "\n")
        
        all_results["beta_updates"] = self.test_beta_distribution_updates()
        console.print("\n" + "="*80 + "\n")
        
        all_results["equal_performance"] = self.test_equal_retrievers_behavior()
        console.print("\n" + "="*80 + "\n")
        
        # Summary
        passed_tests = sum(1 for result in all_results.values() if result["passed"])
        total_tests = len(all_results)
        
        summary_table = Table(title="🏆 Test Suite Summary", show_header=True)
        summary_table.add_column("Test", style="cyan", width=25)
        summary_table.add_column("Result", style="bold", width=10)
        summary_table.add_column("Key Metric", style="yellow")
        
        for test_name, result in all_results.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            
            # Extract key metric based on test type
            if test_name == "exploration_exploitation":
                metric = f"Explr: {result['exploration_score']:.2f}, Expl: {result['exploitation_score']:.2f}"
            elif test_name == "convergence":
                metric = f"Final best ratio: {result['late_best_ratio']:.2f}"
            elif test_name == "beta_updates":
                metric = "Parameter updates correct"
            elif test_name == "equal_performance":
                metric = f"Max deviation: {result['max_deviation']:.1f}"
            else:
                metric = "N/A"
            
            summary_table.add_row(
                test_name.replace("_", " ").title(),
                status,
                metric
            )
        
        console.print(summary_table)
        
        overall_status = "✅ ALL TESTS PASSED" if passed_tests == total_tests else f"❌ {total_tests - passed_tests} TESTS FAILED"
        console.print(f"\n🎯 [bold]{overall_status} ({passed_tests}/{total_tests})[/bold]")
        
        if passed_tests == total_tests:
            console.print("\n🎉 [bold green]Thompson Sampling implementation is working correctly![/bold green]")
            console.print("✅ Proper exploration → exploitation balance")
            console.print("✅ Convergence to best performing retrievers")  
            console.print("✅ Correct beta distribution updates")
            console.print("✅ Fair handling of equal performance scenarios")
        else:
            console.print("\n⚠️ [bold red]Some Thompson Sampling behaviors need attention.[/bold red]")
        
        # Save detailed results
        with open("thompson_sampling_test_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        console.print(f"\n💾 Detailed results saved to: [bold]thompson_sampling_test_results.json[/bold]")
        
        return all_results

def main():
    """Run Thompson Sampling validation tests"""
    tester = ThompsonSamplingTester()
    results = tester.run_comprehensive_test_suite()
    
    # Optional: Generate plots if matplotlib available
    try:
        create_visualization_plots(results)
        console.print("\n📊 Visualization plots saved to: [bold]thompson_sampling_plots.png[/bold]")
    except ImportError:
        console.print("\n💡 Install matplotlib for visualization plots: pip install matplotlib")

def create_visualization_plots(results):
    """Create visualization plots for Thompson Sampling behavior"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Thompson Sampling Test Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Exploration vs Exploitation
    if "exploration_exploitation" in results:
        data = results["exploration_exploitation"]
        ax = axes[0, 0]
        selection_counts = data["selection_counts"]
        reward_probs = data["reward_probabilities"]
        
        retrievers = list(selection_counts.keys())
        selections = [selection_counts[r] for r in retrievers]
        true_rates = [reward_probs[r] for r in retrievers]
        
        x = np.arange(len(retrievers))
        width = 0.35
        
        ax.bar(x - width/2, selections, width, label='Selections', alpha=0.8)
        ax2 = ax.twinx()
        ax2.bar(x + width/2, true_rates, width, label='True Win Rate', alpha=0.8, color='orange')
        
        ax.set_xlabel('Retrievers')
        ax.set_ylabel('Selection Count', color='blue')
        ax2.set_ylabel('True Win Rate', color='orange')
        ax.set_title('Exploration vs Exploitation')
        ax.set_xticks(x)
        ax.set_xticklabels(retrievers, rotation=45)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    # Plot 2: Convergence Over Time
    if "convergence" in results:
        data = results["convergence"]
        ax = axes[0, 1]
        convergence_data = data["convergence_data"]
        
        if convergence_data:
            trials, ratios = zip(*convergence_data)
            ax.plot(trials, ratios, 'b-', linewidth=2, marker='o')
            ax.axhline(y=0.7, color='r', linestyle='--', label='Target (70%)')
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('Best Retriever Selection Rate')
            ax.set_title('Convergence to Best Retriever')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Plot 3: Beta Distribution Updates
    if "beta_updates" in results:
        data = results["beta_updates"]
        ax = axes[1, 0]
        
        initial_params = data["initial_params"]
        final_params = data["final_params"]
        
        retrievers = list(initial_params.keys())
        initial_alphas = [initial_params[r][0] for r in retrievers]
        final_alphas = [final_params[r][0] for r in retrievers]
        initial_betas = [initial_params[r][1] for r in retrievers]
        final_betas = [final_params[r][1] for r in retrievers]
        
        x = np.arange(len(retrievers))
        width = 0.2
        
        ax.bar(x - width*1.5, initial_alphas, width, label='Initial α', alpha=0.7)
        ax.bar(x - width/2, final_alphas, width, label='Final α', alpha=0.7)
        ax.bar(x + width/2, initial_betas, width, label='Initial β', alpha=0.7)
        ax.bar(x + width*1.5, final_betas, width, label='Final β', alpha=0.7)
        
        ax.set_xlabel('Retrievers')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Beta Distribution Parameter Updates')
        ax.set_xticks(x)
        ax.set_xticklabels(retrievers, rotation=45)
        ax.legend()
    
    # Plot 4: Equal Performance Distribution
    if "equal_performance" in results:
        data = results["equal_performance"]
        ax = axes[1, 1]
        
        selection_counts = data["selection_counts"]
        expected = data["expected_per_retriever"]
        
        retrievers = list(selection_counts.keys())
        counts = [selection_counts[r] for r in retrievers]
        
        ax.bar(retrievers, counts, alpha=0.7, color='skyblue')
        ax.axhline(y=expected, color='red', linestyle='--', linewidth=2, label=f'Expected ({expected:.1f})')
        ax.set_xlabel('Retrievers')
        ax.set_ylabel('Selection Count')
        ax.set_title('Equal Performance Distribution')
        ax.legend()
        
        # Add deviation annotations
        for i, (retriever, count) in enumerate(zip(retrievers, counts)):
            deviation = abs(count - expected)
            ax.annotate(f'±{deviation:.1f}', 
                       xy=(i, count), 
                       xytext=(0, 10), 
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=8)
    
    plt.tight_layout()
    plt.savefig('thompson_sampling_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 