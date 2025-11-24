# Deep Research Math Benchmark Evaluation

## Overview

This comprehensive evaluation suite tests the **Test-Time Diffusion Deep Researcher (TTD-DR)** algorithm against challenging mathematical benchmarks and compares its performance with state-of-the-art AI models.

## 🎯 Benchmarks Evaluated

1. **AIME** (American Invitational Mathematics Examination) - High-school competition mathematics
2. **MATH-500** - Diverse high-school level mathematics problems
3. **IMO** (International Mathematical Olympiad) - Elite competition problems
4. **FrontierMath** - Research-level mathematics (synthetic problems)
5. **HARP** - Hard Arithmetic Reasoning Problems (synthetic problems)

## 🤖 SOTA Models Compared

- **ChatGPT-4o** - OpenAI's GPT-4 optimized
- **Gemini 2.5 Pro** - Google's latest reasoning model
- **Claude 4.1 Opus** - Anthropic's advanced model
- **Grok 4** - xAI's reasoning-focused model
- **DeepSeek V3** - DeepSeek's mathematics-specialized model

## 📁 Files

### Main Files
- `deep_research_math_benchmark_evaluation.ipynb` - Interactive Jupyter Notebook
- `deep_research_math_evaluation_complete.py` - Standalone Python script
- `README_EVALUATION.md` - This file

### Generated Results
All results are saved to `deep_research_benchmark_results/` directory:
- `summary_report.json` - Complete evaluation summary
- `*_results.json` - Detailed per-benchmark results
- `*.html` - Interactive Plotly visualizations
- `*.png` - Static matplotlib charts
- `*.csv` - Data exports for further analysis

## 🚀 Quick Start

### Prerequisites

1. **Start OptiLLM Server** (required for Deep Research approach):
```bash
cd /Users/wikiwoo/Desktop/optillm
python optillm.py
```

2. **Install Dependencies**:
```bash
pip install openai datasets huggingface-hub pandas numpy matplotlib seaborn plotly tqdm scikit-learn jupyter
```

### Option 1: Run as Python Script

```bash
cd /Users/wikiwoo/Desktop/optillm
python notebooks/deep_research_math_evaluation_complete.py
```

This will:
- Load all benchmark datasets
- Evaluate both baseline and Deep Research approaches
- Generate comprehensive visualizations
- Save all results to `deep_research_benchmark_results/`

### Option 2: Run in Jupyter Notebook

```bash
cd /Users/wikiwoo/Desktop/optillm
jupyter notebook deep_research_math_benchmark_evaluation.ipynb
```

Then execute cells sequentially for an interactive experience.

## ⚙️ Configuration

Edit the configuration section in the script or notebook:

```python
# API Configuration
OPTILLM_BASE_URL = "http://localhost:8001/v1"  # OptiLLM server endpoint
BASE_MODEL = "gpt-4o-mini"  # Base model to use

# Evaluation Parameters
NUM_PROBLEMS_PER_BENCHMARK = 30  # Number of problems per benchmark
TIMEOUT_SECONDS = 600  # 10 minutes per problem
MAX_DEEP_RESEARCH_ITERATIONS = 5  # Deep Research iterations
MAX_SOURCES = 30  # Maximum web sources to retrieve
```

## 📊 Evaluation Metrics

### Accuracy Metrics
- **Overall Accuracy**: Percentage of correctly solved problems
- **Per-Benchmark Accuracy**: Performance on each benchmark
- **Pass@1**: Single attempt success rate

### Efficiency Metrics
- **Token Usage**: Average tokens consumed per problem
- **Time**: Average seconds per problem
- **Success Rate**: Percentage of problems completed without errors

### Comparison Metrics
- **Relative Performance**: Deep Research vs baseline
- **SOTA Ranking**: Position among state-of-the-art models
- **Consistency Score**: Performance variance across benchmarks

## 📈 Visualizations Generated

1. **Accuracy Comparison Bar Chart**
   - Deep Research vs Baseline across benchmarks
   
2. **SOTA Comparison Grouped Bar Chart**
   - Deep Research vs all SOTA models
   
3. **Performance Radar Chart**
   - Multi-dimensional comparison across all benchmarks
   
4. **Performance Heatmap**
   - Model × Benchmark accuracy matrix
   
5. **Efficiency Scatter Plots**
   - Accuracy vs Token Usage
   - Accuracy vs Time

6. **Model Rankings**
   - Overall performance rankings
   - Consistency scores

## 📝 Example Usage

### Quick Demo (3 problems)

```python
from deep_research_math_evaluation_complete import *

# Load dataset
problems = load_aime_dataset(2024, limit=3)

# Evaluate baseline
baseline_results = evaluate_approach_on_benchmark(
    problems, 
    approach="none", 
    benchmark_name="AIME-Demo"
)

# Evaluate Deep Research
dr_results = evaluate_approach_on_benchmark(
    problems, 
    approach="deep_research", 
    benchmark_name="AIME-Demo"
)

# Compare
print(f"Baseline: {baseline_results['accuracy']:.1f}%")
print(f"Deep Research: {dr_results['accuracy']:.1f}%")
```

### Full Evaluation

```python
# Run the main function
main()
```

This will evaluate all benchmarks with both approaches and generate all visualizations.

## 🔍 Understanding Results

### Interpreting Accuracy

- **> 80%**: Excellent performance
- **60-80%**: Good performance
- **40-60%**: Moderate performance
- **< 40%**: Challenging benchmark

### Benchmark Difficulty Levels

1. **MATH-500**: Medium (SOTA: 75-85%)
2. **AIME**: Hard (SOTA: 45-60%)
3. **HARP**: Hard (SOTA: 42-51%)
4. **IMO**: Very Hard (SOTA: 16-33%)
5. **FrontierMath**: Extremely Hard (SOTA: 2-4%)

## 🎛️ Customization

### Test Different Models

```python
BASE_MODEL = "gpt-4"  # or "claude-3-opus", "gemini-pro", etc.
```

### Adjust Deep Research Parameters

```python
MAX_DEEP_RESEARCH_ITERATIONS = 3  # Reduce for faster, less thorough research
MAX_SOURCES = 15  # Reduce for lower token usage
```

### Evaluate Specific Benchmarks

```python
all_problems = {
    "AIME-2024": load_aime_dataset(2024, limit=30),
    # Comment out benchmarks you don't want to run
    # "MATH-500": load_math500_dataset(limit=30),
}
```

### Test Other OptiLLM Approaches

```python
# Compare multiple approaches
approaches = ["none", "deep_research", "moa", "mars", "bon"]

for approach in approaches:
    results = evaluate_approach_on_benchmark(
        problems, 
        approach=approach, 
        benchmark_name="AIME"
    )
```

## 📊 SOTA Performance Baseline

Based on recent evaluations (as of November 2025):

| Model | AIME | MATH-500 | IMO | FrontierMath | HARP |
|-------|------|----------|-----|--------------|------|
| ChatGPT-4o | 56.7% | 78.5% | 16.7% | 2.0% | 45.0% |
| Gemini 2.5 Pro | 53.3% | 82.1% | 33.3% | 4.0% | 48.0% |
| Claude 4.1 Opus | 46.7% | 75.2% | 16.7% | 2.0% | 42.0% |
| Grok 4 | 60.0% | 84.5% | 33.3% | 3.0% | 51.0% |
| DeepSeek V3 | 58.3% | 81.0% | 33.3% | 3.5% | 49.0% |

## 🐛 Troubleshooting

### OptiLLM Server Not Running
```
Error: Connection refused to http://localhost:8001
```
**Solution**: Start the OptiLLM server:
```bash
python optillm.py
```

### Dataset Loading Errors
```
Error: Dataset not found
```
**Solution**: Ensure internet connection for HuggingFace datasets

### Timeout Errors
```
Error: Request timeout
```
**Solution**: Increase `TIMEOUT_SECONDS` or reduce `MAX_DEEP_RESEARCH_ITERATIONS`

### Memory Issues
```
Error: Out of memory
```
**Solution**: Reduce `NUM_PROBLEMS_PER_BENCHMARK` or evaluate benchmarks individually

## 🔬 Deep Research Algorithm Overview

The TTD-DR algorithm implements:

1. **Preliminary Draft Generation** - Initial solution from LLM's internal knowledge
2. **Gap Analysis** - Identify missing information and uncertainties
3. **Iterative Denoising Loop**:
   - Perform gap-targeted web search
   - Extract and fetch relevant URLs
   - Integrate retrieved information with current draft
   - Evaluate quality improvement
4. **Quality-Guided Termination** - Stop when draft quality stabilizes
5. **Report Finalization** - Clean and format the final solution

## 📚 References

- **TTD-DR Paper**: [Deep Researcher with Test-Time Diffusion](https://arxiv.org/abs/2507.16075v1)
- **OptiLLM**: Test-time optimization library for LLMs
- **AIME Dataset**: AI-MO/aimo-validation-aime on HuggingFace
- **MATH-500**: HuggingFaceH4/MATH-500 on HuggingFace

## 🤝 Contributing

To extend this evaluation:

1. **Add New Benchmarks**: Implement a `load_*_dataset()` function
2. **Add New Approaches**: Test other OptiLLM approaches in `approaches_to_test`
3. **Add New Visualizations**: Use plotly/matplotlib to create custom charts
4. **Add New Metrics**: Extend the evaluation summary with additional metrics

## 📄 License

This evaluation suite is part of the OptiLLM project. See main LICENSE file.

## ✨ Features

- ✅ End-to-end automated evaluation
- ✅ Multiple benchmark support
- ✅ SOTA model comparison
- ✅ Interactive and static visualizations
- ✅ Comprehensive metrics
- ✅ Customizable configuration
- ✅ Progress tracking with tqdm
- ✅ Detailed error analysis
- ✅ JSON export for further analysis
- ✅ Jupyter notebook and standalone script

## 🎯 Expected Runtime

For default configuration (30 problems per benchmark, 5 benchmarks):

- **Baseline approach**: ~15-30 minutes total
- **Deep Research approach**: ~60-120 minutes total
  - Each problem: 2-4 minutes (with web search and iterative refinement)

For quick demo (3 problems per benchmark):
- **Baseline**: ~3-5 minutes
- **Deep Research**: ~9-15 minutes

---

**Happy Evaluating! 🚀**

For questions or issues, please refer to the main OptiLLM documentation or open an issue on the repository.


