"""OpenAI API provider."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import (
    ModelInfo, ModelType, TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    AudioTranscriptionModelInfo, EmbeddingModelInfo, Provider, ProviderInfo,
)

_STATIC_MODELS = [
    TextModelInfo(
        model_id='gpt-4',
        display_name='gpt-4',
        description='Original GPT-4 model with strong reasoning capabilities',
        context_window=8_192,
        max_output_tokens=8_192,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=30.0,
        output_cost_per_mtok=60.0,
    ),
    TextModelInfo(
        model_id='gpt-4-0613',
        display_name='gpt-4-0613',
        description='GPT-4 snapshot from June 2023 with function calling',
        context_window=8_192,
        max_output_tokens=8_192,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=30.0,
        output_cost_per_mtok=60.0,
    ),
    TextModelInfo(
        model_id='gpt-4-turbo',
        display_name='gpt-4-turbo',
        description='GPT-4 Turbo with vision, 128K context',
        context_window=128_000,
        max_output_tokens=4_096,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=10.0,
        output_cost_per_mtok=30.0,
    ),
    TextModelInfo(
        model_id='gpt-4-turbo-2024-04-09',
        display_name='gpt-4-turbo-2024-04-09',
        description='GPT-4 Turbo with vision snapshot from April 2024',
        context_window=128_000,
        max_output_tokens=4_096,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=10.0,
        output_cost_per_mtok=30.0,
    ),
    TextModelInfo(
        model_id='gpt-4.1',
        display_name='gpt-4.1',
        description='Previous generation GPT model (retired from ChatGPT, still in API)',
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=8.0,
    ),
    TextModelInfo(
        model_id='gpt-4.1-2025-04-14',
        display_name='gpt-4.1-2025-04-14',
        description='GPT-4.1 snapshot from April 2025',
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=8.0,
    ),
    TextModelInfo(
        model_id='gpt-4.1-mini',
        display_name='gpt-4.1-mini',
        description='Previous generation balanced model (retired from ChatGPT, still in API)',
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.4,
        output_cost_per_mtok=1.6,
    ),
    TextModelInfo(
        model_id='gpt-4.1-mini-2025-04-14',
        display_name='gpt-4.1-mini-2025-04-14',
        description='GPT-4.1 Mini snapshot from April 2025',
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.4,
        output_cost_per_mtok=1.6,
    ),
    TextModelInfo(
        model_id='gpt-4.1-nano',
        display_name='gpt-4.1-nano',
        description='Previous generation fastest model (retired from ChatGPT, still in API)',
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.1,
        output_cost_per_mtok=0.4,
    ),
    TextModelInfo(
        model_id='gpt-4.1-nano-2025-04-14',
        display_name='gpt-4.1-nano-2025-04-14',
        description='GPT-4.1 Nano snapshot from April 2025',
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.1,
        output_cost_per_mtok=0.4,
    ),
    TextModelInfo(
        model_id='gpt-4o',
        display_name='gpt-4o',
        description='Multimodal flagship model with vision and tool use',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.5,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-4o-2024-05-13',
        display_name='gpt-4o-2024-05-13',
        description='GPT-4o snapshot from May 2024',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.5,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-4o-2024-08-06',
        display_name='gpt-4o-2024-08-06',
        description='GPT-4o snapshot from August 2024',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.5,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-4o-2024-11-20',
        display_name='gpt-4o-2024-11-20',
        description='GPT-4o snapshot from November 2024',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.5,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-4o-audio-preview',
        display_name='gpt-4o-audio-preview',
        description='GPT-4o with audio input/output capabilities',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.5,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-4o-audio-preview-2024-12-17',
        display_name='gpt-4o-audio-preview-2024-12-17',
        description='GPT-4o audio preview snapshot from December 2024',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.5,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-4o-audio-preview-2025-06-03',
        display_name='gpt-4o-audio-preview-2025-06-03',
        description='GPT-4o audio preview snapshot from June 2025',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.5,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-4o-mini',
        display_name='gpt-4o-mini',
        description='Small, fast, and affordable multimodal model',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.15,
        output_cost_per_mtok=0.6,
    ),
    TextModelInfo(
        model_id='gpt-4o-mini-2024-07-18',
        display_name='gpt-4o-mini-2024-07-18',
        description='GPT-4o Mini snapshot from July 2024',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.15,
        output_cost_per_mtok=0.6,
    ),
    TextModelInfo(
        model_id='gpt-4o-mini-audio-preview',
        display_name='gpt-4o-mini-audio-preview',
        description='GPT-4o Mini with audio input/output capabilities',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.15,
        output_cost_per_mtok=0.6,
    ),
    TextModelInfo(
        model_id='gpt-4o-mini-audio-preview-2024-12-17',
        display_name='gpt-4o-mini-audio-preview-2024-12-17',
        description='GPT-4o Mini audio preview snapshot from December 2024',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.15,
        output_cost_per_mtok=0.6,
    ),
    TextModelInfo(
        model_id='gpt-4o-mini-realtime-preview',
        display_name='gpt-4o-mini-realtime-preview',
        description='GPT-4o Mini for low-latency realtime voice and text',
        context_window=128_000,
        max_output_tokens=4_096,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.6,
        output_cost_per_mtok=2.4,
    ),
    TextModelInfo(
        model_id='gpt-4o-mini-realtime-preview-2024-12-17',
        display_name='gpt-4o-mini-realtime-preview-2024-12-17',
        description='GPT-4o Mini realtime preview snapshot from December 2024',
        context_window=128_000,
        max_output_tokens=4_096,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.6,
        output_cost_per_mtok=2.4,
    ),
    TextModelInfo(
        model_id='gpt-4o-mini-search-preview',
        display_name='gpt-4o-mini-search-preview',
        description='GPT-4o Mini with built-in web search capabilities',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.15,
        output_cost_per_mtok=0.6,
    ),
    TextModelInfo(
        model_id='gpt-4o-mini-search-preview-2025-03-11',
        display_name='gpt-4o-mini-search-preview-2025-03-11',
        description='GPT-4o Mini search preview snapshot from March 2025',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.15,
        output_cost_per_mtok=0.6,
    ),
    AudioTranscriptionModelInfo(
        model_id='gpt-4o-mini-transcribe',
        display_name='gpt-4o-mini-transcribe',
        description='Cost-effective speech-to-text model',
        supported_input_formats=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'],
        max_file_size_mb=25,
        cost_per_minute=0.003,
    ),
    AudioTranscriptionModelInfo(
        model_id='gpt-4o-mini-transcribe-2025-03-20',
        display_name='gpt-4o-mini-transcribe-2025-03-20',
        description='Cost-effective speech-to-text model',
        supported_input_formats=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'],
        max_file_size_mb=25,
        cost_per_minute=0.003,
    ),
    AudioTranscriptionModelInfo(
        model_id='gpt-4o-mini-transcribe-2025-12-15',
        display_name='gpt-4o-mini-transcribe-2025-12-15',
        description='Cost-effective speech-to-text model snapshot from December 2025',
        supported_input_formats=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'],
        max_file_size_mb=25,
        cost_per_minute=0.003,
    ),
    AudioTTSModelInfo(
        model_id='gpt-4o-mini-tts',
        display_name='gpt-4o-mini-tts',
        description='Instruction-steerable text-to-speech model',
        supported_voices=['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer', 'verse', 'marin', 'cedar'],
        supported_output_formats=['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'],
        cost_per_mchars=None,  # TODO: add pricing
        input_cost_per_mtok=0.6,
        output_cost_per_mtok=12.0,
    ),
    AudioTTSModelInfo(
        model_id='gpt-4o-mini-tts-2025-03-20',
        display_name='gpt-4o-mini-tts-2025-03-20',
        description='Instruction-steerable text-to-speech model',
        supported_voices=['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer', 'verse', 'marin', 'cedar'],
        supported_output_formats=['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'],
        cost_per_mchars=None,  # TODO: add pricing
        input_cost_per_mtok=0.6,
        output_cost_per_mtok=12.0,
    ),
    AudioTTSModelInfo(
        model_id='gpt-4o-mini-tts-2025-12-15',
        display_name='gpt-4o-mini-tts-2025-12-15',
        description='Instruction-steerable text-to-speech model snapshot from December 2025',
        supported_voices=['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer', 'verse', 'marin', 'cedar'],
        supported_output_formats=['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'],
        cost_per_mchars=None,  # TODO: add pricing
        input_cost_per_mtok=0.6,
        output_cost_per_mtok=12.0,
    ),
    TextModelInfo(
        model_id='gpt-4o-realtime-preview',
        display_name='gpt-4o-realtime-preview',
        description='GPT-4o for low-latency realtime voice and text',
        context_window=128_000,
        max_output_tokens=4_096,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=5.0,
        output_cost_per_mtok=20.0,
    ),
    TextModelInfo(
        model_id='gpt-4o-realtime-preview-2024-12-17',
        display_name='gpt-4o-realtime-preview-2024-12-17',
        description='GPT-4o realtime preview snapshot from December 2024',
        context_window=128_000,
        max_output_tokens=4_096,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=5.0,
        output_cost_per_mtok=20.0,
    ),
    TextModelInfo(
        model_id='gpt-4o-realtime-preview-2025-06-03',
        display_name='gpt-4o-realtime-preview-2025-06-03',
        description='GPT-4o realtime preview snapshot from June 2025',
        context_window=128_000,
        max_output_tokens=4_096,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=5.0,
        output_cost_per_mtok=20.0,
    ),
    TextModelInfo(
        model_id='gpt-4o-search-preview',
        display_name='gpt-4o-search-preview',
        description='GPT-4o with built-in web search capabilities',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.5,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-4o-search-preview-2025-03-11',
        display_name='gpt-4o-search-preview-2025-03-11',
        description='GPT-4o search preview snapshot from March 2025',
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.5,
        output_cost_per_mtok=10.0,
    ),
    AudioTranscriptionModelInfo(
        model_id='gpt-4o-transcribe',
        display_name='gpt-4o-transcribe',
        description='High-accuracy speech-to-text model',
        supported_input_formats=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'],
        max_file_size_mb=25,
        cost_per_minute=0.006,
    ),
    AudioTranscriptionModelInfo(
        model_id='gpt-4o-transcribe-diarize',
        display_name='gpt-4o-transcribe-diarize',
        description='Speech-to-text model with speaker diarization',
        supported_input_formats=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'],
        max_file_size_mb=25,
        cost_per_minute=0.006,
    ),
    TextModelInfo(
        model_id='gpt-5',
        display_name='gpt-5',
        description='Reasoning model with configurable reasoning effort',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-5-2025-08-07',
        display_name='gpt-5-2025-08-07',
        description='GPT-5 snapshot from August 2025',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-5-chat-latest',
        display_name='gpt-5-chat-latest',
        description='GPT-5 latest chat alias',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-5-codex',
        display_name='gpt-5-codex',
        description='GPT-5 optimized for Codex coding environments',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-5-mini',
        display_name='gpt-5-mini',
        description='Near-frontier intelligence optimized for cost-sensitive workloads',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.25,
        output_cost_per_mtok=2.0,
    ),
    TextModelInfo(
        model_id='gpt-5-mini-2025-08-07',
        display_name='gpt-5-mini-2025-08-07',
        description='GPT-5 Mini snapshot from August 2025',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.25,
        output_cost_per_mtok=2.0,
    ),
    TextModelInfo(
        model_id='gpt-5-nano',
        display_name='gpt-5-nano',
        description='Fastest, most cost-efficient GPT-5 variant',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.05,
        output_cost_per_mtok=0.4,
    ),
    TextModelInfo(
        model_id='gpt-5-nano-2025-08-07',
        display_name='gpt-5-nano-2025-08-07',
        description='GPT-5 Nano snapshot from August 2025',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.05,
        output_cost_per_mtok=0.4,
    ),
    TextModelInfo(
        model_id='gpt-5-pro',
        display_name='gpt-5-pro',
        description='GPT-5 with increased compute for harder problems',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=15.0,
        output_cost_per_mtok=120.0,
    ),
    TextModelInfo(
        model_id='gpt-5-pro-2025-10-06',
        display_name='gpt-5-pro-2025-10-06',
        description='GPT-5 Pro snapshot from October 2025',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=15.0,
        output_cost_per_mtok=120.0,
    ),
    TextModelInfo(
        model_id='gpt-5-search-api',
        display_name='gpt-5-search-api',
        description='GPT-5 with built-in web search capabilities',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-5-search-api-2025-10-14',
        display_name='gpt-5-search-api-2025-10-14',
        description='GPT-5 search API snapshot from October 2025',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-5.1',
        display_name='gpt-5.1',
        description='Earlier frontier iteration',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-5.1-2025-11-13',
        display_name='gpt-5.1-2025-11-13',
        description='GPT-5.1 snapshot from November 2025',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-5.1-chat-latest',
        display_name='gpt-5.1-chat-latest',
        description='GPT-5.1 latest chat alias',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-5.1-codex',
        display_name='gpt-5.1-codex',
        description='GPT-5.1 optimized for Codex coding environments',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-5.1-codex-max',
        display_name='gpt-5.1-codex-max',
        description='GPT-5.1 Codex with maximum compute for complex tasks',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gpt-5.1-codex-mini',
        display_name='gpt-5.1-codex-mini',
        description='GPT-5.1 Codex Mini for cost-effective coding tasks',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.25,
        output_cost_per_mtok=2.0,
    ),
    TextModelInfo(
        model_id='gpt-5.2',
        display_name='gpt-5.2',
        description='Previous frontier model for complex professional work',
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.75,
        output_cost_per_mtok=14.0,
    ),
    TextModelInfo(
        model_id='gpt-5.2-2025-12-11',
        display_name='gpt-5.2-2025-12-11',
        description='GPT-5.2 snapshot from December 2025',
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.75,
        output_cost_per_mtok=14.0,
    ),
    TextModelInfo(
        model_id='gpt-5.2-chat-latest',
        display_name='gpt-5.2-chat-latest',
        description='GPT-5.2 latest chat alias',
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.75,
        output_cost_per_mtok=14.0,
    ),
    TextModelInfo(
        model_id='gpt-5.2-codex',
        display_name='gpt-5.2-codex',
        description='GPT-5.2 optimized for Codex coding environments',
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.75,
        output_cost_per_mtok=14.0,
    ),
    TextModelInfo(
        model_id='gpt-5.2-pro',
        display_name='gpt-5.2-pro',
        description='GPT-5.2 with increased compute for harder problems',
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=21.0,
        output_cost_per_mtok=168.0,
    ),
    TextModelInfo(
        model_id='gpt-5.2-pro-2025-12-11',
        display_name='gpt-5.2-pro-2025-12-11',
        description='GPT-5.2 Pro snapshot from December 2025',
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=21.0,
        output_cost_per_mtok=168.0,
    ),
    TextModelInfo(
        model_id='gpt-5.3-chat-latest',
        display_name='gpt-5.3-chat-latest',
        description='GPT-5.3 latest chat alias',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.75,
        output_cost_per_mtok=14.0,
    ),
    TextModelInfo(
        model_id='gpt-5.3-codex',
        display_name='gpt-5.3-codex',
        description='Most capable agentic coding model, optimized for Codex environments',
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.75,
        output_cost_per_mtok=14.0,
    ),
    TextModelInfo(
        model_id='gpt-5.4',
        display_name='gpt-5.4',
        description='Best intelligence at scale for agentic, coding, and professional workflows',
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.5,
        output_cost_per_mtok=15.0,
    ),
    TextModelInfo(
        model_id='gpt-5.4-2026-03-05',
        display_name='gpt-5.4-2026-03-05',
        description='GPT-5.4 snapshot from March 2026',
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.5,
        output_cost_per_mtok=15.0,
    ),
    TextModelInfo(
        model_id='gpt-5.4-mini',
        display_name='gpt-5.4-mini',
        description='',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.75,
        output_cost_per_mtok=4.5,
    ),
    TextModelInfo(
        model_id='gpt-5.4-mini-2026-03-17',
        display_name='gpt-5.4-mini-2026-03-17',
        description='',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.75,
        output_cost_per_mtok=4.5,
    ),
    TextModelInfo(
        model_id='gpt-5.4-nano',
        display_name='gpt-5.4-nano',
        description='',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.2,
        output_cost_per_mtok=1.25,
    ),
    TextModelInfo(
        model_id='gpt-5.4-nano-2026-03-17',
        display_name='gpt-5.4-nano-2026-03-17',
        description='',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.2,
        output_cost_per_mtok=1.25,
    ),
    TextModelInfo(
        model_id='gpt-5.4-pro',
        display_name='gpt-5.4-pro',
        description='More compute for harder problems, supports reasoning effort levels',
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=30.0,
        output_cost_per_mtok=180.0,
    ),
    TextModelInfo(
        model_id='gpt-5.4-pro-2026-03-05',
        display_name='gpt-5.4-pro-2026-03-05',
        description='GPT-5.4 Pro snapshot from March 2026',
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=30.0,
        output_cost_per_mtok=180.0,
    ),
    TextModelInfo(
        model_id='gpt-5.5',
        display_name='gpt-5.5',
        description='Frontier model for agentic coding, computer use, professional knowledge work, tool use, and early scientific research',
        context_window=1_000_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=True,
        input_cost_per_mtok=5.0,
        output_cost_per_mtok=30.0,
    ),
    TextModelInfo(
        model_id='gpt-5.5-2026-04-23',
        display_name='gpt-5.5-2026-04-23',
        description='Dated snapshot of gpt-5.5 (Apr 23, 2026)',
        context_window=1_000_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=True,
        input_cost_per_mtok=5.0,
        output_cost_per_mtok=30.0,
    ),
    TextModelInfo(
        model_id='gpt-5.5-pro',
        display_name='gpt-5.5-pro',
        description="Higher-accuracy GPT-5.5 variant for the hardest questions and demanding professional, legal, education, data science, and research work",
        context_window=1_000_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=30.0,
        output_cost_per_mtok=180.0,
    ),
    TextModelInfo(
        model_id='gpt-5.5-pro-2026-04-23',
        display_name='gpt-5.5-pro-2026-04-23',
        description='Dated snapshot of gpt-5.5-pro (Apr 23, 2026)',
        context_window=1_000_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=30.0,
        output_cost_per_mtok=180.0,
    ),
    ImageModelInfo(
        model_id='gpt-image-1',
        display_name='gpt-image-1',
        description='Standard image generation model',
        supported_sizes=['1024x1024', '1024x1536', '1536x1024', 'auto'],
        supported_qualities=['low', 'medium', 'high', 'auto'],
        max_images_per_request=1,
        cost_per_image=0.042,
    ),
    ImageModelInfo(
        model_id='gpt-image-1-mini',
        display_name='gpt-image-1-mini',
        description='Cost-effective image generation model',
        supported_sizes=['1024x1024', '1024x1536', '1536x1024', 'auto'],
        supported_qualities=['low', 'medium', 'high', 'auto'],
        max_images_per_request=1,
        cost_per_image=0.011,
    ),
    ImageModelInfo(
        model_id='gpt-image-1.5',
        display_name='gpt-image-1.5',
        description='Flagship image generation model with transparent backgrounds and streaming',
        supported_sizes=['1024x1024', '1024x1536', '1536x1024', 'auto'],
        supported_qualities=['low', 'medium', 'high', 'auto'],
        max_images_per_request=1,
        cost_per_image=0.034,
    ),
    ImageModelInfo(
        model_id='gpt-image-2',
        display_name='gpt-image-2',
        description='State-of-the-art image generation model. Token-priced ($8/1M image input, $30/1M image output); cost_per_image is a representative medium-quality estimate.',
        supported_sizes=['1024x1024', '1024x1536', '1536x1024', 'auto'],
        supported_qualities=['low', 'medium', 'high', 'auto'],
        max_images_per_request=1,
        cost_per_image=0.04,
    ),
    ImageModelInfo(
        model_id='gpt-image-2-2026-04-21',
        display_name='gpt-image-2-2026-04-21',
        description='Dated snapshot of gpt-image-2 (Apr 21, 2026)',
        supported_sizes=['1024x1024', '1024x1536', '1536x1024', 'auto'],
        supported_qualities=['low', 'medium', 'high', 'auto'],
        max_images_per_request=1,
        cost_per_image=0.04,
    ),
    TextModelInfo(
        model_id='o3',
        display_name='o3',
        description='Reasoning model for complex tasks',
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=8.0,
    ),
    TextModelInfo(
        model_id='o3-2025-04-16',
        display_name='o3-2025-04-16',
        description='o3 reasoning model snapshot from April 2025',
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=10.0,
        output_cost_per_mtok=40.0,
    ),
    TextModelInfo(
        model_id='o3-mini',
        display_name='o3-mini',
        description='Fast, affordable reasoning model',
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.1,
        output_cost_per_mtok=4.4,
    ),
    TextModelInfo(
        model_id='o3-mini-2025-01-31',
        display_name='o3-mini-2025-01-31',
        description='o3-mini reasoning model snapshot from January 2025',
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.1,
        output_cost_per_mtok=4.4,
    ),
    TextModelInfo(
        model_id='o4-mini',
        display_name='o4-mini',
        description='Fast, affordable reasoning model',
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.1,
        output_cost_per_mtok=4.4,
    ),
    TextModelInfo(
        model_id='o4-mini-2025-04-16',
        display_name='o4-mini-2025-04-16',
        description='o4-mini reasoning model snapshot from April 2025',
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.1,
        output_cost_per_mtok=4.4,
    ),
    EmbeddingModelInfo(
        model_id='text-embedding-3-large',
        display_name='text-embedding-3-large',
        description='Most capable embedding model',
        dimensions=3_072,
        max_input_tokens=8_192,
        supports_multimodal=False,
        input_cost_per_mtok=0.13,
    ),
    EmbeddingModelInfo(
        model_id='text-embedding-3-small',
        display_name='text-embedding-3-small',
        description='Cost-effective embedding model',
        dimensions=1_536,
        max_input_tokens=8_192,
        supports_multimodal=False,
        input_cost_per_mtok=0.02,
    ),
    EmbeddingModelInfo(
        model_id='text-embedding-ada-002',
        display_name='text-embedding-ada-002',
        description='Legacy embedding model',
        dimensions=1_536,
        max_input_tokens=8_191,
        supports_multimodal=False,
        input_cost_per_mtok=0.1,
    ),
    AudioTTSModelInfo(
        model_id='tts-1',
        display_name='tts-1',
        description='Low-latency text-to-speech model',
        supported_voices=['alloy', 'ash', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer'],
        supported_output_formats=['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'],
        cost_per_mchars=15.0,
        input_cost_per_mtok=None,  # TODO: add pricing
        output_cost_per_mtok=None,  # TODO: add pricing
    ),
    AudioTTSModelInfo(
        model_id='tts-1-1106',
        display_name='tts-1-1106',
        description='Low-latency text-to-speech model snapshot from November 2023',
        supported_voices=['alloy', 'ash', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer'],
        supported_output_formats=['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'],
        cost_per_mchars=15.0,
        input_cost_per_mtok=None,  # TODO: add pricing
        output_cost_per_mtok=None,  # TODO: add pricing
    ),
    AudioTTSModelInfo(
        model_id='tts-1-hd',
        display_name='tts-1-hd',
        description='High-quality text-to-speech model',
        supported_voices=['alloy', 'ash', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer'],
        supported_output_formats=['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'],
        cost_per_mchars=30.0,
        input_cost_per_mtok=None,  # TODO: add pricing
        output_cost_per_mtok=None,  # TODO: add pricing
    ),
    AudioTTSModelInfo(
        model_id='tts-1-hd-1106',
        display_name='tts-1-hd-1106',
        description='High-quality text-to-speech model snapshot from November 2023',
        supported_voices=['alloy', 'ash', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer'],
        supported_output_formats=['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'],
        cost_per_mchars=30.0,
        input_cost_per_mtok=None,  # TODO: add pricing
        output_cost_per_mtok=None,  # TODO: add pricing
    ),
    AudioTranscriptionModelInfo(
        model_id='whisper-1',
        display_name='whisper-1',
        description='Open-source speech-to-text model',
        supported_input_formats=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'],
        max_file_size_mb=25,
        cost_per_minute=0.006,
    ),
    TextModelInfo(
        model_id='gpt-4-0125-preview',
        display_name='gpt-4-0125-preview',
        description='GPT-4 Turbo preview snapshot from January 2025',
        context_window=128_000,
        max_output_tokens=4_096,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=10.0,
        output_cost_per_mtok=30.0,
    ),
    TextModelInfo(
        model_id='gpt-4-1106-preview',
        display_name='gpt-4-1106-preview',
        description='GPT-4 Turbo preview snapshot from November 2023',
        context_window=128_000,
        max_output_tokens=4_096,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=10.0,
        output_cost_per_mtok=30.0,
    ),
    TextModelInfo(
        model_id='gpt-4-turbo-preview',
        display_name='gpt-4-turbo-preview',
        description='GPT-4 Turbo preview alias (text-only)',
        context_window=128_000,
        max_output_tokens=4_096,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=10.0,
        output_cost_per_mtok=30.0,
    ),
    TextModelInfo(
        model_id='o3-pro',
        display_name='o3 Pro',
        description='o3 with increased compute for harder reasoning problems',
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=20.0,
        output_cost_per_mtok=80.0,
    ),
]


class OpenAIProvider(Provider):
    """OpenAI provider."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="OpenAI",
            api_base_url="https://api.openai.com",
            api_version=None,
            auth_env_var="OPENAI_API_KEY",
            auth_header="Authorization",
            sdk_packages={
                "python": "openai",
                "typescript": "openai",
                "javascript": "openai",
                "java": "com.openai:openai-java",
            },
            sdk_installs={
                "python": "pip install openai",
                "typescript": "npm install openai",
                "javascript": "npm install openai",
                "java": "Maven/Gradle: com.openai:openai-java",
                "cpp": "No official SDK — use REST API via libcurl",
            },
            models=list(_STATIC_MODELS),
            documentation_url="https://platform.openai.com/docs/api-reference",
            rate_limits_url="https://platform.openai.com/docs/guides/rate-limits",
        )

    @staticmethod
    def _model_class_for_id(model_id: str):
        """Return the appropriate ModelInfo subclass for a model ID."""
        # Order matters: more specific prefixes before less specific ones.
        if model_id.startswith("gpt-image-"):
            return ImageModelInfo
        elif model_id.startswith("gpt-4o-mini-tts") or model_id.startswith("tts-"):
            return AudioTTSModelInfo
        elif model_id.startswith("whisper-") or model_id.startswith("gpt-4o-transcribe") or model_id.startswith("gpt-4o-mini-transcribe"):
            return AudioTranscriptionModelInfo
        elif model_id.startswith("text-embedding-"):
            return EmbeddingModelInfo
        return TextModelInfo

    def fetch_live_models(self) -> ProviderInfo:
        """Fetch models from the OpenAI API, fall back to static."""
        info = self.get_static_info()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return info

        # Well-known model prefixes we care about.
        _PREFIXES = ("gpt-5", "gpt-4", "o3", "o4", "gpt-image-", "tts-", "whisper-", "text-embedding-")

        try:
            req = urllib.request.Request(
                f"{info.api_base_url}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            live_models: list[ModelInfo] = []
            for m in data.get("data", []):
                mid = m.get("id", "")
                if any(mid.startswith(p) for p in _PREFIXES):
                    model_cls = self._model_class_for_id(mid)
                    live_models.append(model_cls(model_id=mid, display_name=mid))
            if live_models:
                live_models.sort(key=lambda x: x.model_id)
                info.models = live_models
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
            pass

        return info

    def _get_model_type(self, model_id: str) -> ModelType:
        """Look up the model type from _STATIC_MODELS."""
        for m in _STATIC_MODELS:
            if m.model_id == model_id:
                return m.model_type
        return ModelType.TEXT

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or "gpt-5.4"
        mtype = self._get_model_type(model)
        if mtype == ModelType.IMAGE:
            return self._image_snippet(model, language)
        elif mtype == ModelType.AUDIO_TTS:
            return self._tts_snippet(model, language)
        elif mtype == ModelType.AUDIO_TRANSCRIPTION:
            return self._transcription_snippet(model, language)
        elif mtype == ModelType.EMBEDDING:
            return self._embedding_snippet(model, language)
        return self._text_snippet(model, language)

    def _text_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from openai import OpenAI\n\n'
                'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
                'response = client.chat.completions.create(\n'
                f'    model="{model}",\n'
                '    messages=[{"role": "user", "content": "Hello!"}],\n'
                ')\n'
                'print(response.choices[0].message.content)\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.chat.completions.create({\n'
                f'  model: "{model}",\n'
                '  messages: [{ role: "user", content: "Hello!" }],\n'
                '});\n'
                'console.log(response.choices[0].message.content);\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.chat.completions.create({\n'
                f'  model: "{model}",\n'
                '  messages: [{ role: "user", content: "Hello!" }],\n'
                '});\n'
                'console.log(response.choices[0].message.content);\n'
            ),
            "java": (
                'import com.openai.client.OpenAIClient;\n'
                'import com.openai.client.okhttp.OpenAIOkHttpClient;\n'
                'import com.openai.models.chat.*;\n\n'
                '// Uses OPENAI_API_KEY env var\n'
                'OpenAIClient client = OpenAIOkHttpClient.builder().build();\n\n'
                'ChatCompletion response = client.chat().completions().create(\n'
                '    ChatCompletionCreateParams.builder()\n'
                f'        .model("{model}")\n'
                '        .addUserMessage("Hello!")\n'
                '        .build()\n'
                ');\n'
                'System.out.println(response.choices().get(0).message().content());\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <curl/curl.h>\n'
                '#include <nlohmann/json.hpp>\n\n'
                'using json = nlohmann::json;\n\n'
                'static size_t WriteCallback(void* contents, size_t size,\n'
                '                            size_t nmemb, std::string* out) {\n'
                '    out->append((char*)contents, size * nmemb);\n'
                '    return size * nmemb;\n'
                '}\n\n'
                'int main() {\n'
                '    const char* api_key = std::getenv("OPENAI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                f'        {{"model", "{model}"}},\n'
                '        {"messages", {{{"role", "user"}, {"content", "Hello!"}}}}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL,\n'
                '        "https://api.openai.com/v1/chat/completions");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["choices"][0]["message"]["content"]\n'
                '              .get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _image_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from openai import OpenAI\n\n'
                'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
                'response = client.images.generate(\n'
                f'    model="{model}",\n'
                '    prompt="A white siamese cat",\n'
                '    size="1024x1024",\n'
                ')\n'
                'print(response.data[0].url)\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.images.generate({\n'
                f'  model: "{model}",\n'
                '  prompt: "A white siamese cat",\n'
                '  size: "1024x1024",\n'
                '});\n'
                'console.log(response.data[0].url);\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.images.generate({\n'
                f'  model: "{model}",\n'
                '  prompt: "A white siamese cat",\n'
                '  size: "1024x1024",\n'
                '});\n'
                'console.log(response.data[0].url);\n'
            ),
            "java": (
                'import com.openai.client.OpenAIClient;\n'
                'import com.openai.client.okhttp.OpenAIOkHttpClient;\n'
                'import com.openai.models.images.*;\n\n'
                '// Uses OPENAI_API_KEY env var\n'
                'OpenAIClient client = OpenAIOkHttpClient.builder().build();\n\n'
                'ImagesResponse response = client.images().generate(\n'
                '    ImageGenerateParams.builder()\n'
                f'        .model("{model}")\n'
                '        .prompt("A white siamese cat")\n'
                '        .size("1024x1024")\n'
                '        .build()\n'
                ');\n'
                'System.out.println(response.data().get(0).url());\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <curl/curl.h>\n'
                '#include <nlohmann/json.hpp>\n\n'
                'using json = nlohmann::json;\n\n'
                'static size_t WriteCallback(void* contents, size_t size,\n'
                '                            size_t nmemb, std::string* out) {\n'
                '    out->append((char*)contents, size * nmemb);\n'
                '    return size * nmemb;\n'
                '}\n\n'
                'int main() {\n'
                '    const char* api_key = std::getenv("OPENAI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                f'        {{"model", "{model}"}},\n'
                '        {"prompt", "A white siamese cat"},\n'
                '        {"size", "1024x1024"}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL,\n'
                '        "https://api.openai.com/v1/images/generations");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["data"][0]["url"]\n'
                '              .get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _tts_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from openai import OpenAI\n\n'
                'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
                'response = client.audio.speech.create(\n'
                f'    model="{model}",\n'
                '    voice="alloy",\n'
                '    input="Hello!",\n'
                ')\n'
                'response.stream_to_file("output.mp3")\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n'
                'import fs from "fs";\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.audio.speech.create({\n'
                f'  model: "{model}",\n'
                '  voice: "alloy",\n'
                '  input: "Hello!",\n'
                '});\n'
                'const buffer = Buffer.from(await response.arrayBuffer());\n'
                'fs.writeFileSync("output.mp3", buffer);\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n'
                'const fs = require("fs");\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.audio.speech.create({\n'
                f'  model: "{model}",\n'
                '  voice: "alloy",\n'
                '  input: "Hello!",\n'
                '});\n'
                'const buffer = Buffer.from(await response.arrayBuffer());\n'
                'fs.writeFileSync("output.mp3", buffer);\n'
            ),
            "java": (
                'import com.openai.client.OpenAIClient;\n'
                'import com.openai.client.okhttp.OpenAIOkHttpClient;\n'
                'import com.openai.models.audio.*;\n'
                'import java.nio.file.Files;\n'
                'import java.nio.file.Path;\n\n'
                '// Uses OPENAI_API_KEY env var\n'
                'OpenAIClient client = OpenAIOkHttpClient.builder().build();\n\n'
                'byte[] response = client.audio().speech().create(\n'
                '    SpeechCreateParams.builder()\n'
                f'        .model("{model}")\n'
                '        .voice("alloy")\n'
                '        .input("Hello!")\n'
                '        .build()\n'
                ');\n'
                'Files.write(Path.of("output.mp3"), response);\n'
                'System.out.println("Audio saved to output.mp3");\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <fstream>\n'
                '#include <string>\n'
                '#include <curl/curl.h>\n'
                '#include <nlohmann/json.hpp>\n\n'
                'using json = nlohmann::json;\n\n'
                'static size_t WriteCallback(void* contents, size_t size,\n'
                '                            size_t nmemb, std::string* out) {\n'
                '    out->append((char*)contents, size * nmemb);\n'
                '    return size * nmemb;\n'
                '}\n\n'
                'int main() {\n'
                '    const char* api_key = std::getenv("OPENAI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                f'        {{"model", "{model}"}},\n'
                '        {"voice", "alloy"},\n'
                '        {"input", "Hello!"}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL,\n'
                '        "https://api.openai.com/v1/audio/speech");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    std::ofstream out("output.mp3", std::ios::binary);\n'
                '    out.write(response.data(), response.size());\n'
                '    std::cout << "Audio saved to output.mp3" << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _transcription_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from openai import OpenAI\n\n'
                'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
                'transcript = client.audio.transcriptions.create(\n'
                f'    model="{model}",\n'
                '    file=open("audio.mp3", "rb"),\n'
                ')\n'
                'print(transcript.text)\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n'
                'import fs from "fs";\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const transcript = await client.audio.transcriptions.create({\n'
                f'  model: "{model}",\n'
                '  file: fs.createReadStream("audio.mp3"),\n'
                '});\n'
                'console.log(transcript.text);\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n'
                'const fs = require("fs");\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const transcript = await client.audio.transcriptions.create({\n'
                f'  model: "{model}",\n'
                '  file: fs.createReadStream("audio.mp3"),\n'
                '});\n'
                'console.log(transcript.text);\n'
            ),
            "java": (
                'import com.openai.client.OpenAIClient;\n'
                'import com.openai.client.okhttp.OpenAIOkHttpClient;\n'
                'import com.openai.models.audio.*;\n'
                'import java.nio.file.Path;\n\n'
                '// Uses OPENAI_API_KEY env var\n'
                'OpenAIClient client = OpenAIOkHttpClient.builder().build();\n\n'
                'Transcription transcript = client.audio().transcriptions().create(\n'
                '    TranscriptionCreateParams.builder()\n'
                f'        .model("{model}")\n'
                '        .file(Path.of("audio.mp3"))\n'
                '        .build()\n'
                ');\n'
                'System.out.println(transcript.text());\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <curl/curl.h>\n'
                '#include <nlohmann/json.hpp>\n\n'
                'using json = nlohmann::json;\n\n'
                'static size_t WriteCallback(void* contents, size_t size,\n'
                '                            size_t nmemb, std::string* out) {\n'
                '    out->append((char*)contents, size * nmemb);\n'
                '    return size * nmemb;\n'
                '}\n\n'
                'int main() {\n'
                '    const char* api_key = std::getenv("OPENAI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    curl_mime* mime = curl_mime_init(curl);\n'
                '    curl_mimepart* part = curl_mime_addpart(mime);\n'
                '    curl_mime_name(part, "file");\n'
                '    curl_mime_filedata(part, "audio.mp3");\n'
                '    part = curl_mime_addpart(mime);\n'
                '    curl_mime_name(part, "model");\n'
                f'    curl_mime_data(part, "{model}", CURL_ZERO_TERMINATED);\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n\n'
                '    std::string response;\n'
                '    curl_easy_setopt(curl, CURLOPT_URL,\n'
                '        "https://api.openai.com/v1/audio/transcriptions");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_mime_free(mime);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["text"].get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _embedding_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from openai import OpenAI\n\n'
                'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
                'response = client.embeddings.create(\n'
                f'    model="{model}",\n'
                '    input="Hello!",\n'
                ')\n'
                'print(response.data[0].embedding[:5])\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.embeddings.create({\n'
                f'  model: "{model}",\n'
                '  input: "Hello!",\n'
                '});\n'
                'console.log(response.data[0].embedding.slice(0, 5));\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.embeddings.create({\n'
                f'  model: "{model}",\n'
                '  input: "Hello!",\n'
                '});\n'
                'console.log(response.data[0].embedding.slice(0, 5));\n'
            ),
            "java": (
                'import com.openai.client.OpenAIClient;\n'
                'import com.openai.client.okhttp.OpenAIOkHttpClient;\n'
                'import com.openai.models.embeddings.*;\n\n'
                '// Uses OPENAI_API_KEY env var\n'
                'OpenAIClient client = OpenAIOkHttpClient.builder().build();\n\n'
                'EmbeddingResponse response = client.embeddings().create(\n'
                '    EmbeddingCreateParams.builder()\n'
                f'        .model("{model}")\n'
                '        .input("Hello!")\n'
                '        .build()\n'
                ');\n'
                'System.out.println(response.data().get(0).embedding());\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <curl/curl.h>\n'
                '#include <nlohmann/json.hpp>\n\n'
                'using json = nlohmann::json;\n\n'
                'static size_t WriteCallback(void* contents, size_t size,\n'
                '                            size_t nmemb, std::string* out) {\n'
                '    out->append((char*)contents, size * nmemb);\n'
                '    return size * nmemb;\n'
                '}\n\n'
                'int main() {\n'
                '    const char* api_key = std::getenv("OPENAI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                f'        {{"model", "{model}"}},\n'
                '        {"input", "Hello!"}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL,\n'
                '        "https://api.openai.com/v1/embeddings");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["data"][0]["embedding"] << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])
