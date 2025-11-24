# Quick Start Guide: Deep Research Math Benchmark Evaluation

## 🚀 5-Minute Quick Start

### Step 1: Start OptiLLM Server (Terminal 1)

```bash
cd /Users/wikiwoo/Desktop/optillm
python optillm.py
```

Wait for: `✓ Server running on http://localhost:8001`

### Step 2: Run Evaluation (Terminal 2)

```bash
cd /Users/wikiwoo/Desktop/optillm
python notebooks/deep_research_math_evaluation_complete.py
```

That's it! Results will be saved to `deep_research_benchmark_results/`

---

## 📓 Alternative: Jupyter Notebook

### Option A: Full Automated Run

```bash
jupyter notebook deep_research_math_benchmark_evaluation.ipynb
```

Then execute Cell 7 (Quick Start cell) to run everything automatically.

### Option B: Interactive Step-by-Step

Execute cells 9-12 for a customizable, interactive evaluation of 3 problems.

---

## 📊 Visualize Existing Results

If you've already run the evaluation:

```bash
python notebooks/visualize_results.py
```

Or specify a custom results directory:

```bash
python notebooks/visualize_results.py --results-dir path/to/results
```

---

## ⚙️ Quick Configuration Changes

Edit the configuration section at the top of `deep_research_math_evaluation_complete.py`:

```python
# Quick demo (fast)
NUM_PROBLEMS_PER_BENCHMARK = 3
TIMEOUT_SECONDS = 300

# Medium evaluation
NUM_PROBLEMS_PER_BENCHMARK = 10
TIMEOUT_SECONDS = 600

# Full evaluation (slow)
NUM_PROBLEMS_PER_BENCHMARK = 30
TIMEOUT_SECONDS = 600
```

---

## 📁 What Gets Generated

```
deep_research_benchmark_results/
├── summary_report.json          # Complete evaluation data
├── none_AIME-2024_results.json  # Baseline results per benchmark
├── deep_research_AIME-2024_results.json  # Deep research results
├── accuracy_comparison.html     # Interactive accuracy chart
├── sota_comparison.html         # SOTA models comparison
├── radar_comparison.html        # Multi-dimensional radar
├── performance_heatmap.html     # Performance matrix
└── summary_dashboard.png        # Static summary chart
```

---

## 🎯 Expected Runtime

| Configuration | Baseline | Deep Research | Total |
|---------------|----------|---------------|-------|
| Quick (3 problems) | 3-5 min | 9-15 min | 12-20 min |
| Medium (10 problems) | 10-15 min | 30-45 min | 40-60 min |
| Full (30 problems) | 30-45 min | 90-120 min | 2-3 hours |

---

## ❓ Troubleshooting

### "Connection refused to http://localhost:8001"
**Problem**: OptiLLM server not running  
**Solution**: Start it in a separate terminal:
```bash
python optillm.py
```

### "No module named 'openai'"
**Problem**: Missing dependencies  
**Solution**: Install them:
```bash
pip install openai datasets pandas matplotlib seaborn plotly tqdm
```

### "Request timeout"
**Problem**: Problem too complex or iterations too many  
**Solution**: Increase `TIMEOUT_SECONDS` or reduce `MAX_DEEP_RESEARCH_ITERATIONS`

### "Out of memory"
**Problem**: Too many problems loaded  
**Solution**: Reduce `NUM_PROBLEMS_PER_BENCHMARK`

---

## 📚 Next Steps

1. ✅ Run quick demo (3 problems)
2. ✅ Review generated visualizations in `deep_research_benchmark_results/`
3. ✅ Read detailed analysis in `summary_report.json`
4. ✅ Run full evaluation (30 problems per benchmark)
5. ✅ Experiment with different base models
6. ✅ Compare with other OptiLLM approaches (moa, mars, bon, etc.)

---

## 🎓 Understanding the Results

### Accuracy Metrics
- **> 80%**: Excellent - Model solves most problems correctly
- **60-80%**: Good - Model handles most common problem types
- **40-60%**: Moderate - Model struggles with harder problems
- **< 40%**: Challenging - Very difficult benchmark

### Benchmark Difficulty
1. **MATH-500** (Easiest) - High-school mathematics
2. **AIME** (Hard) - Competition-level high school
3. **HARP** (Hard) - Complex arithmetic reasoning
4. **IMO** (Very Hard) - Olympiad-level proofs
5. **FrontierMath** (Extremely Hard) - Research-level mathematics

### When to Use Deep Research
✅ Complex multi-step problems  
✅ Problems requiring external knowledge  
✅ Research-style questions  
✅ When accuracy > speed

### When to Use Baseline
✅ Simple problems  
✅ Time-sensitive applications  
✅ Cost-conscious scenarios  
✅ When speed > accuracy

---

## 🔗 Useful Links

- **Full Documentation**: `notebooks/README_EVALUATION.md`
- **OptiLLM Docs**: Main repository README
- **Deep Research Paper**: https://arxiv.org/abs/2507.16075v1

---

**Happy Evaluating! 🎉**

For questions or issues, check the full README or open an issue on GitHub.


