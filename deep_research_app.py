#!/usr/bin/env python3
"""
Deep Research Standalone Application

This is a standalone application that implements the Test-Time Diffusion Deep Researcher (TTD-DR)
algorithm for comprehensive research report generation.

Based on: "Deep Researcher with Test-Time Diffusion" (https://arxiv.org/abs/2507.16075v1)

The app provides a simple API server that exposes the deep research functionality
without the complexity of the full OptILLM proxy.
"""

import argparse
import logging
import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, Response
from openai import OpenAI, AzureOpenAI
from cerebras.cloud.sdk import Cerebras
from typing import Dict, Any, Optional
import time

# Import deep research components
from optillm.plugins.deep_research_plugin import run as deep_research_run

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global configuration
config = {
    'model': 'gpt-4o-mini',  # Default model, can be overridden
    'base_url': None,
    'api_key': None,
    'max_iterations': 5,
    'max_sources': 30
}

# Supported SOTA models mapping
SOTA_MODELS = {
    # OpenAI models
    'gpt-5.1': 'gpt-5.1',
    'gpt-4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
    'o1': 'o1',
    'o1-mini': 'o1-mini',
    'o1-preview': 'o1-preview',
    'gpt-4-turbo': 'gpt-4-turbo',
    'gpt-4': 'gpt-4',
    
    # Claude models (via OpenRouter or direct Anthropic)
    'claude-sonnet-4.5': 'claude-sonnet-4.5',
    'claude-3-5-sonnet-20241022': 'claude-3-5-sonnet-20241022',
    'claude-3-opus': 'claude-3-opus-20240229',
    'claude-3-sonnet': 'claude-3-sonnet-20240229',
    
    # Gemini models (via OpenRouter or direct Google)
    'gemini-3.0-deep-think': 'gemini-3.0-deep-think',
    'gemini-2.0-flash-thinking-exp': 'gemini-2.0-flash-thinking-exp',
    'gemini-2.0-flash-exp': 'gemini-2.0-flash-exp',
    'gemini-1.5-pro': 'gemini-1.5-pro',
    
    # DeepSeek models
    'deepseek-v3': 'deepseek-chat',
    'deepseek-reasoner': 'deepseek-reasoner',
    
    # Other SOTA models
    'qwen-2.5-72b': 'qwen/qwen-2.5-72b-instruct',
    'llama-3.3-70b': 'meta-llama/llama-3.3-70b-instruct'
}

# Global client
llm_client = None


def create_llm_client(provider: str = 'openai', **kwargs):
    """
    Create an LLM client based on provider type
    
    Args:
        provider: One of 'openai', 'azure', 'cerebras', 'anthropic', 'google', 'openrouter'
        **kwargs: Provider-specific configuration
    
    Returns:
        Configured LLM client
    """
    if provider == 'openai':
        return OpenAI(
            api_key=kwargs.get('api_key', os.environ.get('OPENAI_API_KEY')),
            base_url=kwargs.get('base_url')
        )
    elif provider == 'azure':
        return AzureOpenAI(
            api_key=kwargs.get('api_key', os.environ.get('AZURE_OPENAI_API_KEY')),
            api_version=kwargs.get('api_version', '2024-02-15-preview'),
            azure_endpoint=kwargs.get('azure_endpoint', os.environ.get('AZURE_OPENAI_ENDPOINT'))
        )
    elif provider == 'cerebras':
        return Cerebras(
            api_key=kwargs.get('api_key', os.environ.get('CEREBRAS_API_KEY'))
        )
    elif provider == 'anthropic':
        # Use OpenAI client with Anthropic-compatible base URL
        return OpenAI(
            api_key=kwargs.get('api_key', os.environ.get('ANTHROPIC_API_KEY')),
            base_url='https://api.anthropic.com/v1'
        )
    elif provider == 'google':
        # Use OpenAI client with Google AI-compatible base URL
        return OpenAI(
            api_key=kwargs.get('api_key', os.environ.get('GOOGLE_API_KEY')),
            base_url='https://generativelanguage.googleapis.com/v1beta'
        )
    elif provider == 'openrouter':
        # OpenRouter provides unified access to many models
        return OpenAI(
            api_key=kwargs.get('api_key', os.environ.get('OPENROUTER_API_KEY')),
            base_url='https://openrouter.ai/api/v1'
        )
    elif provider == 'deepseek':
        # DeepSeek API (OpenAI-compatible)
        return OpenAI(
            api_key=kwargs.get('api_key', os.environ.get('DEEPSEEK_API_KEY')),
            base_url='https://api.deepseek.com/v1'
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, azure, cerebras, anthropic, google, openrouter, deepseek")


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    OpenAI-compatible chat completions endpoint
    
    Performs deep research on the user's query and returns a comprehensive report.
    """
    try:
        data = request.json
        
        # Extract messages
        messages = data.get('messages', [])
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        # Get the latest user message
        user_message = None
        system_prompt = "You are a helpful research assistant."
        
        for msg in messages:
            if msg.get('role') == 'system':
                system_prompt = msg.get('content', system_prompt)
            elif msg.get('role') == 'user':
                user_message = msg.get('content', '')
        
        if not user_message:
            return jsonify({'error': 'No user message found'}), 400
        
        # Extract configuration
        model = data.get('model', config['model'])
        extra_body = data.get('extra_body', {})
        request_config = extra_body.get('request_config', {})
        
        # Merge with default config
        research_config = {
            'max_iterations': request_config.get('max_iterations', config['max_iterations']),
            'max_sources': request_config.get('max_sources', config['max_sources'])
        }
        
        # Check if streaming is requested
        stream = data.get('stream', False)
        
        logger.info(f"🔬 Starting deep research for query: {user_message[:100]}...")
        logger.info(f"   Config: max_iterations={research_config['max_iterations']}, max_sources={research_config['max_sources']}")
        
        start_time = time.time()
        
        # Perform deep research
        result, total_tokens = deep_research_run(
            system_prompt=system_prompt,
            initial_query=user_message,
            client=llm_client,
            model=model,
            request_config=research_config
        )
        
        duration = time.time() - start_time
        logger.info(f"✅ Deep research completed in {duration:.1f}s, used {total_tokens} tokens")
        
        # Prepare response
        if stream:
            # Streaming response
            def generate():
                response_data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": result
                        },
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(response_data)}\n\n"
                
                # Final chunk
                final_data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            # Non-streaming response
            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,  # Not tracked separately
                    "completion_tokens": total_tokens,
                    "total_tokens": total_tokens
                }
            }
            return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in chat_completions: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": config['model'],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deep-research"
            }
        ]
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'deep-research'})


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with service information"""
    return jsonify({
        'service': 'Deep Research Standalone Application',
        'algorithm': 'Test-Time Diffusion Deep Researcher (TTD-DR)',
        'version': '1.0.0',
        'endpoints': {
            'chat': '/v1/chat/completions',
            'models': '/v1/models',
            'health': '/health'
        },
        'config': {
            'model': config['model'],
            'max_iterations': config['max_iterations'],
            'max_sources': config['max_sources']
        }
    })


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Deep Research Standalone Application - TTD-DR Algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Server configuration
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to run the server on (default: 8000)'
    )
    
    # LLM provider configuration
    parser.add_argument(
        '--provider',
        type=str,
        default='openai',
        choices=['openai', 'azure', 'cerebras', 'anthropic', 'google', 'openrouter', 'deepseek'],
        help='LLM provider to use (default: openai)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help=f'Model to use for research. Examples: {", ".join(list(SOTA_MODELS.keys())[:6])}... (default: gpt-4o-mini)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for the LLM provider (can also use environment variables)'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        help='Base URL for OpenAI-compatible API (optional)'
    )
    
    parser.add_argument(
        '--azure-endpoint',
        type=str,
        help='Azure OpenAI endpoint (for Azure provider)'
    )
    
    parser.add_argument(
        '--azure-api-version',
        type=str,
        default='2024-02-15-preview',
        help='Azure API version (default: 2024-02-15-preview)'
    )
    
    # Research configuration
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum research iterations (default: 5)'
    )
    
    parser.add_argument(
        '--max-sources',
        type=int,
        default=30,
        help='Maximum sources to retrieve per search (default: 30)'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    global config, llm_client
    
    args = parse_args()
    
    # Update logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Update configuration
    config.update({
        'model': args.model,
        'base_url': args.base_url,
        'api_key': args.api_key,
        'max_iterations': args.max_iterations,
        'max_sources': args.max_sources
    })
    
    # Create LLM client
    logger.info(f"🚀 Initializing Deep Research Application")
    logger.info(f"   Provider: {args.provider}")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Max iterations: {args.max_iterations}")
    logger.info(f"   Max sources: {args.max_sources}")
    
    try:
        if args.provider == 'azure':
            llm_client = create_llm_client(
                provider='azure',
                api_key=args.api_key,
                azure_endpoint=args.azure_endpoint,
                api_version=args.azure_api_version
            )
        elif args.provider == 'cerebras':
            llm_client = create_llm_client(
                provider='cerebras',
                api_key=args.api_key
            )
        elif args.provider in ['anthropic', 'google', 'openrouter', 'deepseek']:
            llm_client = create_llm_client(
                provider=args.provider,
                api_key=args.api_key
            )
        else:  # openai or custom
            llm_client = create_llm_client(
                provider='openai',
                api_key=args.api_key,
                base_url=args.base_url
            )
        
        logger.info("✅ LLM client initialized successfully")
        
        # Display model info
        model_display = SOTA_MODELS.get(args.model, args.model)
        if model_display != args.model:
            logger.info(f"   Using model alias: {args.model} → {model_display}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM client: {str(e)}")
        return
    
    # Check for required environment
    logger.info("\n📋 Checking environment...")
    
    # Check for Chrome (required for web search)
    import shutil
    if not shutil.which('google-chrome') and not shutil.which('chrome'):
        logger.warning("⚠️  Chrome browser not found. Web search may not work properly.")
        logger.warning("   Please install Chrome for full functionality.")
    else:
        logger.info("✅ Chrome browser found")
    
    # Start server
    logger.info(f"\n🌐 Starting Deep Research server on http://0.0.0.0:{args.port}")
    logger.info(f"   API endpoint: http://localhost:{args.port}/v1/chat/completions")
    logger.info(f"   Health check: http://localhost:{args.port}/health")
    logger.info("\n" + "="*60)
    logger.info("Example usage with OpenAI Python client:")
    logger.info("="*60)
    logger.info("""
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not used, but required by client
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Research the latest advances in quantum computing"}
    ],
    extra_body={
        "request_config": {
            "max_iterations": 3,
            "max_sources": 20
        }
    }
)

print(response.choices[0].message.content)
    """)
    logger.info("="*60 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=args.port)


if __name__ == "__main__":
    main()

