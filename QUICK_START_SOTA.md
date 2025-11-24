# Quick Start: Deep Research with SOTA Models

Get started with Deep Research and evaluate it on hard math benchmarks using state-of-the-art models.

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
cd /Users/wikiwoo/Desktop/optillm
pip install -r deep_research_requirements.txt
pip install -e .  # Install optillm package
```

### 2. Set Up API Keys

Choose your preferred provider and set the API key:

```bash
# Option A: OpenRouter (recommended - access to all models)
export OPENROUTER_API_KEY="sk-or-v1-..."

# Option B: Direct OpenAI
export OPENAI_API_KEY="sk-..."

# Option C: Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Option D: Google
export GOOGLE_API_KEY="..."

# Option E: DeepSeek
export DEEPSEEK_API_KEY="sk-..."
```

### 3. Start Deep Research Server

**With GPT-5.1 (OpenRouter):**
```bash
python deep_research_app.py \
    --port 8000 \
    --provider openrouter \
    --model gpt-5.1 \
    --max-iterations 5 \
    --max-sources 30
```

**With Claude Sonnet 4.5 (OpenRouter):**
```bash
python deep_research_app.py \
    --port 8000 \
    --provider openrouter \
    --model claude-sonnet-4.5
```

**With Gemini 3.0 Deep Think (OpenRouter):**
```bash
python deep_research_app.py \
    --port 8000 \
    --provider openrouter \
    --model gemini-3.0-deep-think
```

**With o1 (OpenAI Direct):**
```bash
python deep_research_app.py \
    --port 8000 \
    --provider openai \
    --model o1
```

### 4. Test the Server

In a new terminal:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="gpt-5.1",
    messages=[{
        "role": "user",
        "content": "Find the sum of all positive integers n ≤ 1000 for which n^2 + 15n + 225 is a perfect square."
    }]
)

print(response.choices[0].message.content)
```

## 📊 Math Benchmark Evaluation

### Single Model Evaluation

1. **Start Deep Research server** (see step 3 above)

2. **Start baseline server** (for comparison):
```bash
# In another terminal
cd /Users/wikiwoo/Desktop/optillm
python optillm.py --port 8001
```

3. **Run the evaluation notebook**:
```bash
jupyter notebook deep_research_math_benchmarks.ipynb
```

4. **Update notebook configuration**:
```python
# In cell 5 of the notebook
DEEP_RESEARCH_BASE_URL = "http://localhost:8000/v1"
BASELINE_BASE_URL = "http://localhost:8001/v1"
MODEL = "gpt-5.1"  # Or your chosen model
```

5. **Run all cells** and wait for results!

### Multi-Model Comparison

Compare multiple SOTA models at once:

**Terminal 1 - GPT-5.1:**
```bash
export OPENROUTER_API_KEY="..."
python deep_research_app.py --port 8000 --provider openrouter --model gpt-5.1
```

**Terminal 2 - Claude Sonnet 4.5:**
```bash
python deep_research_app.py --port 8001 --provider openrouter --model claude-sonnet-4.5
```

**Terminal 3 - Gemini 3.0:**
```bash
python deep_research_app.py --port 8002 --provider openrouter --model gemini-3.0-deep-think
```

**Terminal 4 - o1:**
```bash
export OPENAI_API_KEY="..."
python deep_research_app.py --port 8003 --provider openai --model o1
```

**Terminal 5 - Run Notebook:**
```bash
jupyter notebook deep_research_math_benchmarks.ipynb
```

Update the notebook to evaluate all models:

```python
# In the notebook
MODELS_TO_EVALUATE = {
    "GPT-5.1": {"url": "http://localhost:8000/v1", "name": "gpt-5.1"},
    "Claude-4.5": {"url": "http://localhost:8001/v1", "name": "claude-sonnet-4.5"},
    "Gemini-3.0": {"url": "http://localhost:8002/v1", "name": "gemini-3.0-deep-think"},
    "o1": {"url": "http://localhost:8003/v1", "name": "o1"},
}
```

## 📈 Expected Results

After running the evaluation, you'll get:

### Files Generated
- `deep_research_math_results/comprehensive_summary.json` - Complete data
- `deep_research_math_results/*_results.json` - Per-benchmark results
- `deep_research_math_results/*.html` - Interactive visualizations
- `deep_research_math_results/*.csv` - Comparison tables

### Visualizations
- Accuracy comparison across benchmarks
- Token usage analysis
- Time efficiency comparison
- Improvement over baseline
- SOTA model comparison

### Example Output
```
📊 COMPREHENSIVE RESULTS
======================================================================
Benchmark      Approach         Accuracy (%)  Correct  Avg Tokens  Avg Time (s)
AIME           Baseline         15.0          3/20     2,500       12.3
AIME           Deep Research    65.0          13/20    15,000      156.8
MATH-500       Baseline         42.0          8/20     2,800       14.2
MATH-500       Deep Research    87.0          17/20    16,500      168.4
FrontierMath   Baseline         2.0           0/20     3,200       15.7
FrontierMath   Deep Research    12.0          2/20     18,000      185.3
======================================================================
```

## 🎯 Recommended Configurations by Model

### GPT-5.1
```bash
--max-iterations 5 --max-sources 30
```
- Best for: General math problems
- Strength: Broad knowledge, good reasoning
- Speed: Medium (2-4 min/problem)

### Claude Sonnet 4.5
```bash
--max-iterations 4 --max-sources 25
```
- Best for: Proof-based problems
- Strength: Step-by-step reasoning, structured output
- Speed: Medium-Fast (1.5-3 min/problem)

### Gemini 3.0 Deep Think
```bash
--max-iterations 3 --max-sources 20
```
- Best for: Computational problems
- Strength: Built-in thinking, fast inference
- Speed: Fast (1-2 min/problem)

### o1
```bash
--max-iterations 2 --max-sources 15
```
- Best for: Complex reasoning
- Strength: Extended internal reasoning
- Speed: Slow (3-5 min/problem)
- Note: Already has reasoning, fewer iterations needed

### DeepSeek-Reasoner
```bash
--max-iterations 3 --max-sources 20
```
- Best for: Cost-effective evaluation
- Strength: Good math performance, low cost
- Speed: Fast (1-2 min/problem)

## 💡 Tips for Best Results

1. **Start Small**: Begin with 5-10 problems to test your setup
   ```python
   NUM_PROBLEMS_PER_BENCHMARK = 5
   ```

2. **Monitor Progress**: Watch server logs for debugging
   ```bash
   python deep_research_app.py --log-level DEBUG
   ```

3. **Save Costs**: Use smaller models for testing
   ```bash
   python deep_research_app.py --model gpt-4o-mini
   ```

4. **Optimize Configs**: Adjust based on problem type
   - Simple arithmetic: 2 iterations, 10 sources
   - Competition math: 3-4 iterations, 20 sources
   - Research-level: 5-7 iterations, 30+ sources

5. **Handle Rate Limits**: Add delays between requests in notebook
   ```python
   import time
   time.sleep(2)  # Between problems
   ```

## 🐛 Troubleshooting

### Server won't start
```bash
# Check port availability
lsof -i :8000

# Try different port
python deep_research_app.py --port 8080
```

### API key errors
```bash
# Verify key is set
echo $OPENROUTER_API_KEY

# Test with minimal example
python -c "from openai import OpenAI; c=OpenAI(base_url='https://openrouter.ai/api/v1', api_key='$OPENROUTER_API_KEY'); print(c.models.list())"
```

### Chrome/Selenium issues
```bash
# Install Chrome
brew install --cask google-chrome  # macOS
sudo apt-get install google-chrome-stable  # Linux

# Test browser
python -c "from selenium import webdriver; driver = webdriver.Chrome(); driver.quit()"
```

### Out of memory
```bash
# Reduce sources and iterations
python deep_research_app.py --max-sources 10 --max-iterations 2
```

## 📚 Next Steps

1. **Read the full documentation**: `DEEP_RESEARCH_README.md`
2. **Explore SOTA model setup**: `SOTA_MODELS_SETUP.md`
3. **Check example reports**: `optillm/plugins/deep_research/sample_reports/`
4. **Customize for your domain**: Modify prompts in `research_engine.py`
5. **Join the community**: https://github.com/codelion/optillm

## 🎉 You're Ready!

Start researching with cutting-edge AI models and see how Deep Research enhances their mathematical reasoning capabilities!

```bash
# One-liner to get started
python deep_research_app.py --provider openrouter --model gpt-5.1 && \
jupyter notebook deep_research_math_benchmarks.ipynb
```

Good luck with your research! 🚀


