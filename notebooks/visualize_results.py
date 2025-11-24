"""
Visualize Existing Deep Research Evaluation Results
====================================================

This script loads previously generated results and creates visualizations
without re-running the evaluations.

Usage:
    python visualize_results.py
    python visualize_results.py --results-dir path/to/results
"""

import os
import sys
import json
import argparse
from typing import Dict, List
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set plotting style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12


def load_results_from_directory(results_dir: str) -> Dict:
    """Load all evaluation results from a directory."""
    print(f"Loading results from: {results_dir}")
    
    all_results = {}
    result_files = glob(os.path.join(results_dir, "*_results.json"))
    
    if not result_files:
        print(f"⚠️  No result files found in {results_dir}")
        return {}
    
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)
            approach = data['approach']
            benchmark = data['benchmark']
            
            if approach not in all_results:
                all_results[approach] = {}
            
            all_results[approach][benchmark] = data
    
    print(f"✓ Loaded {len(result_files)} result files")
    return all_results


def compile_results_dataframes(all_results: Dict) -> tuple:
    """Compile results into dataframes."""
    accuracy_data = []
    performance_data = []
    
    for approach, benchmarks in all_results.items():
        for benchmark_name, summary in benchmarks.items():
            accuracy_data.append({
                "Approach": approach.replace('_', ' ').title(),
                "Benchmark": benchmark_name,
                "Accuracy (%)": summary["accuracy"],
                "Correct": summary["correct"],
                "Total": summary["total_problems"]
            })
            
            performance_data.append({
                "Approach": approach.replace('_', ' ').title(),
                "Benchmark": benchmark_name,
                "Avg Tokens": summary["avg_tokens"],
                "Avg Time (s)": summary["avg_time_seconds"]
            })
    
    return pd.DataFrame(accuracy_data), pd.DataFrame(performance_data)


def create_all_visualizations(accuracy_df: pd.DataFrame, performance_df: pd.DataFrame, 
                              output_dir: str):
    """Create all visualizations."""
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80 + "\n")
    
    # 1. Accuracy comparison
    fig = px.bar(
        accuracy_df,
        x="Benchmark",
        y="Accuracy (%)",
        color="Approach",
        barmode="group",
        title="Accuracy Across Benchmarks by Approach",
        text="Accuracy (%)"
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=500, xaxis_tickangle=-45)
    
    output_file = os.path.join(output_dir, "accuracy_comparison.html")
    fig.write_html(output_file)
    print(f"✓ Saved: {output_file}")
    
    # 2. Token usage comparison
    fig = px.bar(
        performance_df,
        x="Benchmark",
        y="Avg Tokens",
        color="Approach",
        barmode="group",
        title="Average Token Usage by Approach",
        text="Avg Tokens"
    )
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig.update_layout(height=500, xaxis_tickangle=-45)
    
    output_file = os.path.join(output_dir, "token_usage_comparison.html")
    fig.write_html(output_file)
    print(f"✓ Saved: {output_file}")
    
    # 3. Time comparison
    fig = px.bar(
        performance_df,
        x="Benchmark",
        y="Avg Time (s)",
        color="Approach",
        barmode="group",
        title="Average Time per Problem by Approach",
        text="Avg Time (s)"
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(height=500, xaxis_tickangle=-45)
    
    output_file = os.path.join(output_dir, "time_comparison.html")
    fig.write_html(output_file)
    print(f"✓ Saved: {output_file}")
    
    # 4. Scatter plot: Accuracy vs Efficiency
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Accuracy vs Tokens', 'Accuracy vs Time')
    )
    
    for approach in accuracy_df["Approach"].unique():
        approach_acc = accuracy_df[accuracy_df["Approach"] == approach]
        approach_perf = performance_df[performance_df["Approach"] == approach]
        
        avg_accuracy = approach_acc["Accuracy (%)"].mean()
        avg_tokens = approach_perf["Avg Tokens"].mean()
        avg_time = approach_perf["Avg Time (s)"].mean()
        
        fig.add_trace(
            go.Scatter(
                x=[avg_tokens],
                y=[avg_accuracy],
                mode='markers+text',
                name=approach,
                text=[approach],
                textposition="top center",
                marker=dict(size=15)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[avg_time],
                y=[avg_accuracy],
                mode='markers+text',
                name=approach,
                text=[approach],
                textposition="top center",
                marker=dict(size=15),
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Average Tokens", row=1, col=1)
    fig.update_xaxes(title_text="Average Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Average Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Average Accuracy (%)", row=1, col=2)
    
    fig.update_layout(title="Efficiency Analysis", height=500)
    
    output_file = os.path.join(output_dir, "efficiency_scatter.html")
    fig.write_html(output_file)
    print(f"✓ Saved: {output_file}")
    
    # 5. Static matplotlib summary
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy by benchmark
    accuracy_pivot = accuracy_df.pivot(index="Benchmark", columns="Approach", values="Accuracy (%)")
    accuracy_pivot.plot(kind='bar', ax=axes[0, 0], rot=45)
    axes[0, 0].set_title("Accuracy by Benchmark", fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend(title="Approach")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Token usage by benchmark
    tokens_pivot = performance_df.pivot(index="Benchmark", columns="Approach", values="Avg Tokens")
    tokens_pivot.plot(kind='bar', ax=axes[0, 1], rot=45)
    axes[0, 1].set_title("Token Usage by Benchmark", fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel("Average Tokens")
    axes[0, 1].legend(title="Approach")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Overall accuracy by approach
    overall_acc = accuracy_df.groupby("Approach")["Accuracy (%)"].mean().sort_values(ascending=False)
    overall_acc.plot(kind='barh', ax=axes[1, 0], color=['#2ecc71', '#e74c3c'])
    axes[1, 0].set_title("Overall Accuracy by Approach", fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel("Mean Accuracy (%)")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overall time by approach
    overall_time = performance_df.groupby("Approach")["Avg Time (s)"].mean().sort_values(ascending=False)
    overall_time.plot(kind='barh', ax=axes[1, 1], color=['#3498db', '#f39c12'])
    axes[1, 1].set_title("Overall Time by Approach", fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel("Mean Time (s)")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "summary_dashboard.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    print("\n✓ All visualizations generated successfully!")


def print_summary_statistics(accuracy_df: pd.DataFrame, performance_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\n📊 Accuracy Summary:")
    print("-" * 80)
    print(accuracy_df.groupby("Approach").agg({
        "Accuracy (%)": ["mean", "std", "min", "max"],
        "Correct": "sum",
        "Total": "sum"
    }).round(2).to_string())
    
    print("\n\n⚡ Performance Summary:")
    print("-" * 80)
    print(performance_df.groupby("Approach").agg({
        "Avg Tokens": ["mean", "std"],
        "Avg Time (s)": ["mean", "std"]
    }).round(2).to_string())
    
    print("\n\n🏆 Best Performance:")
    print("-" * 80)
    best_overall = accuracy_df.groupby("Approach")["Accuracy (%)"].mean().idxmax()
    best_accuracy = accuracy_df.groupby("Approach")["Accuracy (%)"].mean().max()
    print(f"  Best Overall Accuracy: {best_overall} ({best_accuracy:.2f}%)")
    
    for benchmark in accuracy_df["Benchmark"].unique():
        bench_data = accuracy_df[accuracy_df["Benchmark"] == benchmark]
        best_approach = bench_data.loc[bench_data["Accuracy (%)"].idxmax(), "Approach"]
        best_acc = bench_data["Accuracy (%)"].max()
        print(f"  Best on {benchmark}: {best_approach} ({best_acc:.1f}%)")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize Deep Research evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="deep_research_benchmark_results",
        help="Directory containing result files (default: deep_research_benchmark_results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: same as results-dir)"
    )
    
    args = parser.parse_args()
    
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    
    if not os.path.exists(results_dir):
        print(f"❌ Error: Results directory not found: {results_dir}")
        print("\nPlease run the evaluation first or specify a valid results directory.")
        sys.exit(1)
    
    print("="*80)
    print("Deep Research Results Visualization")
    print("="*80)
    print(f"Results Directory: {results_dir}")
    print(f"Output Directory: {output_dir}")
    print("="*80 + "\n")
    
    # Load results
    all_results = load_results_from_directory(results_dir)
    
    if not all_results:
        print("\n❌ No results found. Please run the evaluation first.")
        sys.exit(1)
    
    # Compile dataframes
    accuracy_df, performance_df = compile_results_dataframes(all_results)
    
    # Print data
    print("\n" + "="*80)
    print("Accuracy Results:")
    print("="*80)
    print(accuracy_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("Performance Metrics:")
    print("="*80)
    print(performance_df.to_string(index=False))
    
    # Create visualizations
    create_all_visualizations(accuracy_df, performance_df, output_dir)
    
    # Print summary statistics
    print_summary_statistics(accuracy_df, performance_df)
    
    print("\n✅ Visualization complete!")
    print(f"📁 Output saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


