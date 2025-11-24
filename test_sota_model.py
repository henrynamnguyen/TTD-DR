#!/usr/bin/env python3
"""
Quick test script for SOTA models with Deep Research

Usage:
    python test_sota_model.py --model gpt-5.1
    python test_sota_model.py --model claude-sonnet-4.5 --provider openrouter
"""

import argparse
import os
from openai import OpenAI

# Test math problems
TEST_PROBLEMS = [
    {
        "name": "AIME-style",
        "problem": "Find the number of positive integers n ≤ 100 for which n^2 + 15n + 56 is a perfect square.",
        "expected_approach": "Complete the square or systematic checking"
    },
    {
        "name": "Competition Math",
        "problem": "What is the sum of all positive divisors of 360?",
        "expected_approach": "Prime factorization and divisor formula"
    },
    {
        "name": "Geometry",
        "problem": "A circle is inscribed in a square with side length 10. What is the area of the region inside the square but outside the circle?",
        "expected_approach": "Square area minus circle area"
    }
]


def test_model(base_url: str, model: str, problem: dict, use_deep_research: bool = True):
    """Test a single problem with the model"""
    client = OpenAI(base_url=base_url, api_key="dummy")
    
    print(f"\n{'='*70}")
    print(f"Testing: {problem['name']}")
    print(f"{'='*70}")
    print(f"Problem: {problem['problem']}")
    print(f"Expected approach: {problem['expected_approach']}")
    print()
    
    extra_body = {}
    if use_deep_research:
        extra_body = {
            "request_config": {
                "max_iterations": 3,
                "max_sources": 15
            }
        }
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a mathematical expert. Solve the problem step by step."
                },
                {
                    "role": "user",
                    "content": problem['problem']
                }
            ],
            extra_body=extra_body if extra_body else None
        )
        
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
        
        print(f"✅ Success!")
        print(f"Tokens used: {tokens}")
        print(f"\nAnswer preview (first 500 chars):")
        print("-" * 70)
        print(answer[:500] + ("..." if len(answer) > 500 else ""))
        print("-" * 70)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test SOTA models with Deep Research")
    parser.add_argument("--model", default="gpt-5.1", help="Model to test")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="Deep Research server URL")
    parser.add_argument("--no-deep-research", action="store_true", help="Test without deep research")
    parser.add_argument("--problem-index", type=int, help="Test specific problem (0-2)")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🧪 SOTA Model Testing with Deep Research")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Server: {args.base_url}")
    print(f"Deep Research: {'Enabled' if not args.no_deep_research else 'Disabled'}")
    print("="*70)
    
    # Test connection
    print("\n📡 Testing connection...")
    try:
        client = OpenAI(base_url=args.base_url, api_key="dummy")
        # Try a simple request
        client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("✅ Connection successful!")
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print("\nMake sure the Deep Research server is running:")
        print(f"  python deep_research_app.py --provider openrouter --model {args.model}")
        return
    
    # Test problems
    problems = [TEST_PROBLEMS[args.problem_index]] if args.problem_index is not None else TEST_PROBLEMS
    
    results = []
    for i, problem in enumerate(problems):
        success = test_model(
            args.base_url,
            args.model,
            problem,
            use_deep_research=not args.no_deep_research
        )
        results.append(success)
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Total problems: {len(results)}")
    print(f"Successful: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    print("="*70)
    
    if all(results):
        print("\n✅ All tests passed! Your setup is working correctly.")
        print("\nNext steps:")
        print("  1. Run the full evaluation notebook:")
        print("     jupyter notebook deep_research_math_benchmarks.ipynb")
        print("  2. Try other SOTA models:")
        print(f"     python test_sota_model.py --model claude-sonnet-4.5")
        print(f"     python test_sota_model.py --model gemini-3.0-deep-think")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()


