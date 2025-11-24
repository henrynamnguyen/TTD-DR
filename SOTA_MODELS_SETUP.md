# Using SOTA Models with Deep Research

This guide shows how to use state-of-the-art models (GPT-5.1, Claude Sonnet 4.5, Gemini 3.0 Deep Think) with the Deep Research app.

## Setup Instructions

### Option 1: OpenRouter (Recommended - Unified Access)

OpenRouter provides access to all major models through a single API:

```bash
# Sign up at https://openrouter.ai and get your API key
export OPENROUTER_API_KEY="your-openrouter-key"

# Start Deep Research with GPT-5.1
python deep_research_app.py \
    --provider openrouter \
    --model gpt-5.1 \
    --api-key $OPENROUTER_API_KEY

# Or with Claude Sonnet 4.5
python deep_research_app.py \
    --provider openrouter \
    --model claude-sonnet-4.5 \
    --api-key $OPENROUTER_API_KEY

# Or with Gemini 3.0 Deep Think
python deep_research_app.py \
    --provider openrouter \
    --model gemini-3.0-deep-think \
    --api-key $OPENROUTER_API_KEY
```

### Option 2: Direct Provider Access

#### OpenAI (GPT-5.1, o1)

```bash
export OPENAI_API_KEY="your-openai-key"

# GPT-5.1
python deep_research_app.py \
    --provider openai \
    --model gpt-5.1

# o1 (extended reasoning)
python deep_research_app.py \
    --provider openai \
    --model o1
```

#### Anthropic (Claude Sonnet 4.5)

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"

python deep_research_app.py \
    --provider anthropic \
    --model claude-sonnet-4.5
```

#### Google (Gemini 3.0 Deep Think)

```bash
export GOOGLE_API_KEY="your-google-key"

python deep_research_app.py \
    --provider google \
    --model gemini-3.0-deep-think
```

#### DeepSeek (DeepSeek-V3, DeepSeek-Reasoner)

```bash
export DEEPSEEK_API_KEY="your-deepseek-key"

python deep_research_app.py \
    --provider deepseek \
    --model deepseek-reasoner
```

## Using in Python

### With OpenRouter (All Models)

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# Research with GPT-5.1
response = client.chat.completions.create(
    model="gpt-5.1",  # Server was started with this model
    messages=[
        {"role": "user", "content": "Research quantum error correction techniques"}
    ],
    extra_body={
        "request_config": {
            "max_iterations": 5,
            "max_sources": 30
        }
    }
)

print(response.choices[0].message.content)
```

### Math Benchmark Evaluation with SOTA Models

Update the notebook configuration:

```python
# Configuration for SOTA model evaluation
MODELS_TO_EVALUATE = {
    "GPT-5.1": {
        "base_url": "http://localhost:8000/v1",  # Deep Research server
        "model": "gpt-5.1",
        "provider": "openrouter"
    },
    "Claude-4.5-Sonnet": {
        "base_url": "http://localhost:8001/v1",
        "model": "claude-sonnet-4.5",
        "provider": "openrouter"
    },
    "Gemini-3.0-DeepThink": {
        "base_url": "http://localhost:8002/v1",
        "model": "gemini-3.0-deep-think",
        "provider": "openrouter"
    },
    "o1": {
        "base_url": "http://localhost:8003/v1",
        "model": "o1",
        "provider": "openai"
    }
}
```

## Running Multiple Models Simultaneously

Start multiple Deep Research servers on different ports:

```bash
# Terminal 1: GPT-5.1
export OPENROUTER_API_KEY="your-key"
python deep_research_app.py --port 8000 --provider openrouter --model gpt-5.1

# Terminal 2: Claude Sonnet 4.5
python deep_research_app.py --port 8001 --provider openrouter --model claude-sonnet-4.5

# Terminal 3: Gemini 3.0 Deep Think
python deep_research_app.py --port 8002 --provider openrouter --model gemini-3.0-deep-think

# Terminal 4: o1 (reasoning model)
export OPENAI_API_KEY="your-openai-key"
python deep_research_app.py --port 8003 --provider openai --model o1
```

Then run the evaluation notebook to compare all models!

## Model Recommendations for Math Benchmarks

### Best Overall: o1
- Excellent at mathematical reasoning
- Built-in extended thinking
- Config: `max_iterations: 2-3` (already has reasoning)

### Best for Research: GPT-5.1
- Strong general knowledge
- Good at integrating external sources
- Config: `max_iterations: 4-5, max_sources: 30`

### Best for Structured Reasoning: Claude Sonnet 4.5
- Excellent at step-by-step reasoning
- Very good citations
- Config: `max_iterations: 3-4, max_sources: 25`

### Best for Speed: Gemini 3.0 Deep Think
- Fast inference with reasoning
- Good math capabilities
- Config: `max_iterations: 3, max_sources: 20`

### Best Value: DeepSeek-Reasoner
- Open source reasoning model
- Strong math performance
- Config: `max_iterations: 2-3, max_sources: 20`

## Cost Comparison (Approximate)

| Model | Cost per 1M tokens | Math Benchmark Cost (20 problems) |
|-------|-------------------|-----------------------------------|
| GPT-5.1 | $15.00 | ~$3.00 |
| o1 | $60.00 | ~$12.00 |
| Claude Sonnet 4.5 | $15.00 | ~$3.00 |
| Gemini 3.0 Deep Think | $7.50 | ~$1.50 |
| DeepSeek-Reasoner | $0.60 | ~$0.12 |

*Note: Costs are estimates and vary based on usage. Deep Research uses more tokens due to iteration.*

## Expected Performance (Estimated)

Based on model capabilities and Deep Research enhancement:

### AIME Benchmark
- o1: 80-85% (best)
- GPT-5.1 + Deep Research: 60-70%
- Claude Sonnet 4.5 + Deep Research: 55-65%
- Gemini 3.0 Deep Think: 50-60%
- DeepSeek-Reasoner: 45-55%

### FrontierMath
- o1: 15-20% (best)
- GPT-5.1 + Deep Research: 10-15%
- Claude Sonnet 4.5 + Deep Research: 8-12%
- Gemini 3.0 Deep Think: 7-10%
- DeepSeek-Reasoner: 5-8%

### MATH-500
- o1: 94-96%
- GPT-5.1 + Deep Research: 85-90%
- Claude Sonnet 4.5 + Deep Research: 83-88%
- Gemini 3.0 Deep Think: 80-85%
- DeepSeek-Reasoner: 75-80%

## Troubleshooting

### Rate Limits

If you hit rate limits:

```bash
# Reduce iterations and sources
python deep_research_app.py \
    --max-iterations 2 \
    --max-sources 10 \
    --model gpt-5.1
```

### API Key Issues

Check your environment variables:

```bash
echo $OPENROUTER_API_KEY
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
echo $DEEPSEEK_API_KEY
```

### Model Not Available

Some models may not be available yet. Check:
- OpenRouter dashboard: https://openrouter.ai/models
- Direct provider documentation

Use fallback models:
- GPT-5.1 → gpt-4o or o1-preview
- Claude Sonnet 4.5 → claude-3-5-sonnet-20241022
- Gemini 3.0 Deep Think → gemini-2.0-flash-thinking-exp

## Full Example: Complete Evaluation

```bash
# 1. Start Deep Research servers for each model (separate terminals)

# Terminal 1: GPT-5.1
export OPENROUTER_API_KEY="sk-or-..."
python deep_research_app.py --port 8000 --provider openrouter --model gpt-5.1

# Terminal 2: Claude
python deep_research_app.py --port 8001 --provider openrouter --model claude-sonnet-4.5

# Terminal 3: Gemini
python deep_research_app.py --port 8002 --provider openrouter --model gemini-3.0-deep-think

# Terminal 4: o1
export OPENAI_API_KEY="sk-..."
python deep_research_app.py --port 8003 --provider openai --model o1

# 2. Update notebook configuration
# Edit deep_research_math_benchmarks.ipynb:
# - Set up multiple clients (one per port)
# - Run evaluations for each model
# - Compare results

# 3. Run notebook
jupyter notebook deep_research_math_benchmarks.ipynb
```

Results will be saved to `deep_research_math_results/` with separate files for each model!


