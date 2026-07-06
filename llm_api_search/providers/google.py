"""Google Gemini API provider."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import (
    ModelInfo, ModelType, TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    EmbeddingModelInfo, MusicModelInfo, VideoModelInfo, Provider, ProviderInfo,
)

_STATIC_MODELS = [
    TextModelInfo(
        model_id='gemini-2.5-flash',
        display_name='Gemini 2.5 Flash',
        description='Stable version of Gemini 2.5 Flash, our mid-size multimodal model that supports up to 1 million tokens, released in June of 2025.',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.3,
        output_cost_per_mtok=2.5,
    ),
    TextModelInfo(
        model_id='gemini-2.5-pro',
        display_name='Gemini 2.5 Pro',
        description='Stable release (June 17th, 2025) of Gemini 2.5 Pro',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gemini-2.0-flash',
        display_name='Gemini 2.0 Flash',
        description='Gemini 2.0 Flash',
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.1,
        output_cost_per_mtok=0.4,
    ),
    TextModelInfo(
        model_id='gemini-2.0-flash-001',
        display_name='Gemini 2.0 Flash 001',
        description='Stable version of Gemini 2.0 Flash, our fast and versatile multimodal model for scaling across diverse tasks, released in January of 2025.',
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.1,
        output_cost_per_mtok=0.4,
    ),
    TextModelInfo(
        model_id='gemini-2.0-flash-lite-001',
        display_name='Gemini 2.0 Flash-Lite 001',
        description='Stable version of Gemini 2.0 Flash-Lite',
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.075,
        output_cost_per_mtok=0.3,
    ),
    TextModelInfo(
        model_id='gemini-2.0-flash-lite',
        display_name='Gemini 2.0 Flash-Lite',
        description='Gemini 2.0 Flash-Lite',
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.075,
        output_cost_per_mtok=0.3,
    ),
    AudioTTSModelInfo(
        model_id='gemini-2.5-flash-preview-tts',
        display_name='Gemini 2.5 Flash Preview TTS',
        description='Gemini 2.5 Flash Preview TTS',
        supported_voices=['Zephyr', 'Puck', 'Charon', 'Kore', 'Fenrir', 'Leda', 'Orus', 'Aoede', 'Callirrhoe', 'Autonoe', 'Enceladus', 'Iapetus', 'Umbriel', 'Algieba', 'Despina', 'Erinome', 'Algenib', 'Rasalgethi', 'Laomedeia', 'Achernar', 'Alnilam', 'Schedar', 'Gacrux', 'Pulcherrima', 'Achird', 'Zubenelgenubi', 'Vindemiatrix', 'Sadachbia', 'Sadaltager', 'Sulafat'],
        supported_output_formats=['wav'],
        cost_per_mchars=None,
        input_cost_per_mtok=0.5,
        output_cost_per_mtok=10.0,
    ),
    AudioTTSModelInfo(
        model_id='gemini-2.5-pro-preview-tts',
        display_name='Gemini 2.5 Pro Preview TTS',
        description='Gemini 2.5 Pro Preview TTS',
        supported_voices=['Zephyr', 'Puck', 'Charon', 'Kore', 'Fenrir', 'Leda', 'Orus', 'Aoede', 'Callirrhoe', 'Autonoe', 'Enceladus', 'Iapetus', 'Umbriel', 'Algieba', 'Despina', 'Erinome', 'Algenib', 'Rasalgethi', 'Laomedeia', 'Achernar', 'Alnilam', 'Schedar', 'Gacrux', 'Pulcherrima', 'Achird', 'Zubenelgenubi', 'Vindemiatrix', 'Sadachbia', 'Sadaltager', 'Sulafat'],
        supported_output_formats=['wav'],
        cost_per_mchars=None,
        input_cost_per_mtok=1.0,
        output_cost_per_mtok=20.0,
    ),
    TextModelInfo(
        model_id='gemma-4-26b-a4b-it',
        display_name='Gemma 4 26B A4B IT',
        description='Gemma 4 26B A4B IT',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    ),
    TextModelInfo(
        model_id='gemma-4-31b-it',
        display_name='Gemma 4 31B IT',
        description='Gemma 4 31B IT',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    ),
    TextModelInfo(
        model_id='gemini-flash-latest',
        display_name='Gemini Flash Latest',
        description='Latest release of Gemini Flash',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.3,
        output_cost_per_mtok=2.5,
    ),
    TextModelInfo(
        model_id='gemini-flash-lite-latest',
        display_name='Gemini Flash-Lite Latest',
        description='Latest release of Gemini Flash-Lite',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.1,
        output_cost_per_mtok=0.4,
    ),
    TextModelInfo(
        model_id='gemini-pro-latest',
        display_name='Gemini Pro Latest',
        description='Latest release of Gemini Pro',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gemini-2.5-flash-lite',
        display_name='Gemini 2.5 Flash-Lite',
        description='Stable version of Gemini 2.5 Flash-Lite, released in July of 2025',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.1,
        output_cost_per_mtok=0.4,
    ),
    ImageModelInfo(
        model_id='gemini-2.5-flash-image',
        display_name='Nano Banana',
        description='Gemini 2.5 Flash Preview Image',
        supported_sizes=['1024x1024'],
        supported_qualities=['standard'],
        max_images_per_request=1,
        cost_per_image=0.039,
    ),
    TextModelInfo(
        model_id='gemini-3-pro-preview',
        display_name='Gemini 3 Pro Preview',
        description='Gemini 3 Pro Preview',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=True,
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=12.0,
    ),
    TextModelInfo(
        model_id='gemini-3-flash-preview',
        display_name='Gemini 3 Flash Preview',
        description='Gemini 3 Flash Preview',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=True,
        input_cost_per_mtok=0.5,
        output_cost_per_mtok=3.0,
    ),
    TextModelInfo(
        model_id='gemini-3.1-pro-preview',
        display_name='Gemini 3.1 Pro Preview',
        description='Gemini 3.1 Pro Preview',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=True,
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=12.0,
    ),
    TextModelInfo(
        model_id='gemini-3.1-pro-preview-customtools',
        display_name='Gemini 3.1 Pro Preview Custom Tools',
        description='Gemini 3.1 Pro Preview optimized for custom tool usage',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=True,
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=12.0,
    ),
    TextModelInfo(
        model_id='gemini-3.1-flash-lite-preview',
        display_name='Gemini 3.1 Flash Lite Preview',
        description='Gemini 3.1 Flash Lite Preview',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.25,
        output_cost_per_mtok=1.5,
    ),
    TextModelInfo(
        model_id='gemini-3.1-flash-lite',
        display_name='Gemini 3.1 Flash Lite',
        description='Most cost-efficient Gemini, optimized for high-volume agentic tasks, translation, and simple data processing.',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.25,
        output_cost_per_mtok=1.5,
    ),
    ImageModelInfo(
        model_id='gemini-3-pro-image-preview',
        display_name='Nano Banana Pro',
        description='Gemini 3 Pro Image Preview',
        supported_sizes=['1024x1024'],
        supported_qualities=['standard'],
        max_images_per_request=1,
        cost_per_image=0.134,  # 1K standard tier; 2K $0.134, 4K $0.24
    ),
    ImageModelInfo(
        model_id='gemini-3-pro-image',
        display_name='Nano Banana Pro',
        description='Gemini 3 Pro Image',
        supported_sizes=['1024x1024'],
        supported_qualities=['standard'],
        max_images_per_request=1,
        cost_per_image=0.134,  # 1K standard tier; 2K $0.134, 4K $0.24
    ),
    TextModelInfo(
        model_id='nano-banana-pro-preview',
        display_name='Nano Banana Pro',
        description='Gemini 3 Pro Image Preview',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=True,
        supports_computer_use=False,
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=12.0,
    ),
    ImageModelInfo(
        model_id='gemini-3.1-flash-image-preview',
        display_name='Nano Banana 2',
        description='Gemini 3.1 Flash Image Preview.',
        supported_sizes=['1024x1024'],
        supported_qualities=['standard'],
        max_images_per_request=1,
        cost_per_image=0.067,  # 1K standard tier; 0.5K $0.045, 2K $0.101, 4K $0.151
    ),
    ImageModelInfo(
        model_id='gemini-3.1-flash-image',
        display_name='Nano Banana 2',
        description='Gemini 3.1 Flash Image.',
        supported_sizes=['1024x1024'],
        supported_qualities=['standard'],
        max_images_per_request=1,
        cost_per_image=0.067,  # 1K standard tier; 0.5K $0.045, 2K $0.101, 4K $0.151
    ),
    TextModelInfo(
        model_id='gemini-3.5-flash',
        display_name='Gemini 3.5 Flash',
        description='Most intelligent Flash model built for speed, combining frontier intelligence with superior search and grounding.',
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.5,
        output_cost_per_mtok=9.0,
    ),
    MusicModelInfo(
        model_id='lyria-3-clip-preview',
        display_name='Lyria 3 Clip Preview',
        description='Lyria 3 30s model Preview',
        cost_per_second=None,  # TODO: add pricing
    ),
    MusicModelInfo(
        model_id='lyria-3-pro-preview',
        display_name='Lyria 3 Pro Preview',
        description='Lyria 3 Pro Preview',
        cost_per_second=None,  # TODO: add pricing
    ),
    AudioTTSModelInfo(
        model_id='gemini-3.1-flash-tts-preview',
        display_name='Gemini 3.1 Flash TTS Preview',
        description='Gemini 3.1 Flash TTS Preview',
        supported_voices=[],
        supported_output_formats=[],
        cost_per_mchars=None,  # TODO: add pricing
        input_cost_per_mtok=None,  # TODO: add pricing
        output_cost_per_mtok=None,  # TODO: add pricing
    ),
    TextModelInfo(
        model_id='gemini-robotics-er-1.5-preview',
        display_name='Gemini Robotics-ER 1.5 Preview',
        description='Gemini Robotics-ER 1.5 Preview',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='gemini-robotics-er-1.6-preview',
        display_name='Gemini Robotics-ER 1.6 Preview',
        description='Gemini Robotics-ER 1.6 Preview',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=None,  # TODO: add pricing
        output_cost_per_mtok=None,  # TODO: add pricing
    ),
    TextModelInfo(
        model_id='gemini-2.5-computer-use-preview-10-2025',
        display_name='Gemini 2.5 Computer Use Preview 10-2025',
        description='Gemini 2.5 Computer Use Preview 10-2025',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=True,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='antigravity-preview-05-2026',
        display_name='Antigravity Agent Preview',
        description='Preview release of Antigravity Agent (05-2026)',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.5,
        output_cost_per_mtok=9.0,
    ),
    TextModelInfo(
        model_id='deep-research-max-preview-04-2026',
        display_name='Deep Research Max Preview (Apr-21-2026)',
        description='Preview release (April 21st, 2026) of Deep Research Max',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=None,  # TODO: add pricing
        output_cost_per_mtok=None,  # TODO: add pricing
    ),
    TextModelInfo(
        model_id='deep-research-preview-04-2026',
        display_name='Deep Research Preview (Apr-21-2026)',
        description='Preview release (April 21th, 2026) of Deep Research',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=None,  # TODO: add pricing
        output_cost_per_mtok=None,  # TODO: add pricing
    ),
    TextModelInfo(
        model_id='deep-research-pro-preview-12-2025',
        display_name='Deep Research Pro Preview (Dec-12-2025)',
        description='Preview release (December 12th, 2025) of Deep Research Pro',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.5,
        output_cost_per_mtok=15.0,
    ),
    EmbeddingModelInfo(
        model_id='gemini-embedding-001',
        display_name='Gemini Embedding 001',
        description='Obtain a distributed representation of a text.',
        dimensions=3_072,
        max_input_tokens=2_048,
        supports_multimodal=False,
        input_cost_per_mtok=0.15,
    ),
    EmbeddingModelInfo(
        model_id='gemini-embedding-2-preview',
        display_name='Gemini Embedding 2 Preview',
        description='Obtain a distributed representation of multimodal content.',
        dimensions=3_072,
        max_input_tokens=8_192,
        supports_multimodal=True,
        input_cost_per_mtok=0.2,
    ),
    EmbeddingModelInfo(
        model_id='gemini-embedding-2',
        display_name='Gemini Embedding 2',
        description='Multimodal embedding model mapping text, images, video, audio, and PDFs into a unified embedding space.',
        dimensions=3_072,
        max_input_tokens=8_192,
        supports_multimodal=True,
        input_cost_per_mtok=0.2,
    ),
    TextModelInfo(
        model_id='aqa',
        display_name='Model that performs Attributed Question Answering.',
        description='Model trained to return answers to questions that are grounded in provided sources, along with estimating answerable probability.',
        context_window=7_168,
        max_output_tokens=1_024,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    ),
    ImageModelInfo(
        model_id='imagen-4.0-generate-001',
        display_name='Imagen 4',
        description='Vertex served Imagen 4.0 model',
        supported_sizes=['1024x1024', '2048x2048'],
        supported_qualities=['standard'],
        max_images_per_request=4,
        cost_per_image=0.04,
    ),
    ImageModelInfo(
        model_id='imagen-4.0-ultra-generate-001',
        display_name='Imagen 4 Ultra',
        description='Vertex served Imagen 4.0 ultra model',
        supported_sizes=['1024x1024', '2048x2048'],
        supported_qualities=['standard'],
        max_images_per_request=4,
        cost_per_image=0.06,
    ),
    ImageModelInfo(
        model_id='imagen-4.0-fast-generate-001',
        display_name='Imagen 4 Fast',
        description='Vertex served Imagen 4.0 Fast model',
        supported_sizes=['1024x1024'],
        supported_qualities=['standard'],
        max_images_per_request=4,
        cost_per_image=0.02,
    ),
    VideoModelInfo(
        model_id='veo-2.0-generate-001',
        display_name='Veo 2',
        description='Vertex served Veo 2 model. Access to this model requires billing to be enabled on the associated Google Cloud Platform account. Please visit https://console.cloud.google.com/billing to enable it.',
        supported_resolutions=['720p', '1080p'],
        supports_audio=False,
        cost_per_second=0.35,
        cost_per_video=None,
    ),
    VideoModelInfo(
        model_id='veo-3.0-generate-001',
        display_name='Veo 3',
        description='Veo 3',
        supported_resolutions=['720p', '1080p'],
        supports_audio=True,
        cost_per_second=0.5,
        cost_per_video=None,
    ),
    VideoModelInfo(
        model_id='veo-3.0-fast-generate-001',
        display_name='Veo 3 fast',
        description='Veo 3 fast',
        supported_resolutions=['720p', '1080p'],
        supports_audio=True,
        cost_per_second=0.25,
        cost_per_video=None,
    ),
    VideoModelInfo(
        model_id='veo-3.1-generate-preview',
        display_name='Veo 3.1',
        description='Veo 3.1',
        supported_resolutions=['720p', '1080p'],
        supports_audio=True,
        cost_per_second=0.5,
        cost_per_video=None,
    ),
    VideoModelInfo(
        model_id='veo-3.1-fast-generate-preview',
        display_name='Veo 3.1 fast',
        description='Veo 3.1 fast',
        supported_resolutions=['720p', '1080p'],
        supports_audio=True,
        cost_per_second=0.25,
        cost_per_video=None,
    ),
    VideoModelInfo(
        model_id='veo-3.1-lite-generate-preview',
        display_name='Veo 3.1 lite',
        description='Veo 3.1 lite',
        supported_resolutions=[],
        supports_audio=False,
        cost_per_second=None,  # TODO: add pricing
        cost_per_video=None,  # TODO: add pricing
    ),
    TextModelInfo(
        model_id='gemini-2.5-flash-native-audio-latest',
        display_name='Gemini 2.5 Flash Native Audio Latest',
        description='Latest release of Gemini 2.5 Flash Native Audio',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.3,
        output_cost_per_mtok=2.5,
    ),
    TextModelInfo(
        model_id='gemini-2.5-flash-native-audio-preview-09-2025',
        display_name='Gemini 2.5 Flash Native Audio Preview 09-2025',
        description='Gemini 2.5 Flash Native Audio Preview 09-2025',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.3,
        output_cost_per_mtok=2.5,
    ),
    TextModelInfo(
        model_id='gemini-2.5-flash-native-audio-preview-12-2025',
        display_name='Gemini 2.5 Flash Native Audio Preview 12-2025',
        description='Gemini 2.5 Flash Native Audio Preview 12-2025',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.3,
        output_cost_per_mtok=2.5,
    ),
    TextModelInfo(
        model_id='gemini-3.1-flash-live-preview',
        display_name='Gemini 3.1 Flash Live Preview',
        description='Gemini 3.1 Flash Live Preview',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=None,  # TODO: add pricing
        output_cost_per_mtok=None,  # TODO: add pricing
    ),
    TextModelInfo(
        model_id='gemma-3-1b-it',
        display_name='Gemma 3 1B',
        description='Gemma 3 1B instruction-tuned open model',
        context_window=32_768,
        max_output_tokens=8_192,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    ),
    TextModelInfo(
        model_id='gemma-3-4b-it',
        display_name='Gemma 3 4B',
        description='Gemma 3 4B instruction-tuned open model with vision support',
        context_window=131_072,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    ),
    TextModelInfo(
        model_id='gemma-3-12b-it',
        display_name='Gemma 3 12B',
        description='Gemma 3 12B instruction-tuned open model with vision support',
        context_window=131_072,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    ),
    TextModelInfo(
        model_id='gemma-3-27b-it',
        display_name='Gemma 3 27B',
        description='Gemma 3 27B instruction-tuned open model with vision and tool use support',
        context_window=131_072,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    ),
    TextModelInfo(
        model_id='gemma-3n-e4b-it',
        display_name='Gemma 3n E4B',
        description='Gemma 3n E4B instruction-tuned open model optimized for on-device efficiency',
        context_window=131_072,
        max_output_tokens=8_192,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    ),
    TextModelInfo(
        model_id='gemma-3n-e2b-it',
        display_name='Gemma 3n E2B',
        description='Gemma 3n E2B instruction-tuned open model optimized for on-device efficiency',
        context_window=131_072,
        max_output_tokens=8_192,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    ),
    TextModelInfo(
        model_id='gemini-2.5-flash-lite-preview-09-2025',
        display_name='Gemini 2.5 Flash-Lite Preview Sep 2025',
        description='Preview release (September 25th, 2025) of Gemini 2.5 Flash-Lite',
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.1,
        output_cost_per_mtok=0.4,
    ),
    EmbeddingModelInfo(
        model_id='multimodalembedding',
        display_name='Multimodal Embedding',
        description='Embeddings for text ($0.0002/1K chars), image ($0.0001/image), and video ($0.0005-$0.002/sec)',
        dimensions=1_408,
        max_input_tokens=None,
        supports_multimodal=True,
        input_cost_per_mtok=0.8,
    ),
    EmbeddingModelInfo(
        model_id='multilingual-e5-small',
        display_name='Multilingual E5 Small',
        description='Open-source multilingual text embedding model (small)',
        dimensions=384,
        max_input_tokens=512,
        supports_multimodal=False,
        input_cost_per_mtok=0.015,
    ),
    EmbeddingModelInfo(
        model_id='multilingual-e5-large',
        display_name='Multilingual E5 Large',
        description='Open-source multilingual text embedding model (large)',
        dimensions=1_024,
        max_input_tokens=512,
        supports_multimodal=False,
        input_cost_per_mtok=0.025,
    ),
    MusicModelInfo(
        model_id='lyria-2',
        display_name='Lyria 2',
        description='High-quality instrumental music generation from text prompts',
        cost_per_second=0.002,
    ),
    VideoModelInfo(
        model_id='veo-3.1',
        display_name='Veo 3.1',
        description='High-quality video generation with optional synchronized audio',
        supported_resolutions=['720p', '1080p', '4k'],
        supports_audio=True,
        cost_per_second=0.4,
        cost_per_video=None,
    ),
    VideoModelInfo(
        model_id='veo-3.1-fast',
        display_name='Veo 3.1 Fast',
        description='Faster video generation with optional synchronized audio',
        supported_resolutions=['720p', '1080p', '4k'],
        supports_audio=True,
        cost_per_second=0.15,
        cost_per_video=None,
    ),
    VideoModelInfo(
        model_id='veo-3',
        display_name='Veo 3',
        description='Video generation with synchronized speech and sound effects',
        supported_resolutions=['720p', '1080p'],
        supports_audio=True,
        cost_per_second=0.4,
        cost_per_video=None,
    ),
    VideoModelInfo(
        model_id='veo-3-fast',
        display_name='Veo 3 Fast',
        description='Faster video generation with synchronized audio',
        supported_resolutions=['720p', '1080p'],
        supports_audio=True,
        cost_per_second=0.15,
        cost_per_video=None,
    ),
    VideoModelInfo(
        model_id='veo-2',
        display_name='Veo 2',
        description='Video generation with advanced controls and frame interpolation',
        supported_resolutions=['720p'],
        supports_audio=False,
        cost_per_second=0.5,
        cost_per_video=None,
    ),
    ImageModelInfo(
        model_id='gemini-3.1-flash-lite-image',
        display_name='Nano Banana 2 Lite',
        description='Gemini 3.1 Flash Lite Image.',
        supported_sizes=[],
        supported_qualities=[],
        max_images_per_request=None,
        cost_per_image=0.0336,
    ),
    TextModelInfo(
        model_id='gemini-omni-flash-preview',
        display_name='Gemini Omni Flash Preview',
        description='Gemini Omni Flash Preview',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=None,  # TODO: add pricing
        output_cost_per_mtok=None,  # TODO: add pricing
    ),
]


class GeminiProvider(Provider):
    """Google Gemini provider."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Google (Gemini)",
            api_base_url="https://generativelanguage.googleapis.com",
            api_version="v1beta",
            auth_env_var="GEMINI_API_KEY",
            auth_header="x-goog-api-key",
            sdk_packages={
                "python": "google-genai",
                "typescript": "@google/genai",
                "javascript": "@google/genai",
                "java": "com.google.genai:google-genai",
            },
            sdk_installs={
                "python": "pip install google-genai",
                "typescript": "npm install @google/genai",
                "javascript": "npm install @google/genai",
                "java": "Maven/Gradle: com.google.genai:google-genai",
                "cpp": "No official SDK — use REST API via libcurl",
            },
            models=list(_STATIC_MODELS),
            documentation_url="https://ai.google.dev/gemini-api/docs",
            rate_limits_url="https://ai.google.dev/gemini-api/docs/rate-limits",
        )

    @staticmethod
    def _model_class_for_id(model_id: str):
        if model_id.startswith("imagen-"):
            return ImageModelInfo
        elif "image" in model_id and model_id.startswith("gemini-"):
            return ImageModelInfo
        elif model_id.startswith("gemini-embedding") or model_id.startswith("text-embedding"):
            return EmbeddingModelInfo
        elif "tts" in model_id:
            return AudioTTSModelInfo
        elif model_id.startswith("lyria"):
            return MusicModelInfo
        elif model_id.startswith("veo-"):
            return VideoModelInfo
        return TextModelInfo

    def fetch_live_models(self) -> ProviderInfo:
        """Fetch models from the Gemini API, fall back to static."""
        info = self.get_static_info()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return info

        try:
            url = (
                f"{info.api_base_url}/{info.api_version}/models"
                f"?key={api_key}"
            )
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            live_models: list[ModelInfo] = []
            for m in data.get("models", []):
                model_id = m.get("name", "").removeprefix("models/")
                if not model_id:
                    continue
                model_cls = self._model_class_for_id(model_id)
                live_models.append(
                    model_cls(
                        model_id=model_id,
                        display_name=m.get("displayName", model_id),
                        description=m.get("description", ""),
                    )
                )
            if live_models:
                info.models = live_models
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
            pass

        return info

    def _get_model_type(self, model_id: str) -> ModelType:
        for m in _STATIC_MODELS:
            if m.model_id == model_id:
                return m.model_type
        return ModelType.TEXT

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or "gemini-2.5-flash"
        mtype = self._get_model_type(model)
        if mtype == ModelType.IMAGE:
            if model.startswith("gemini-"):
                return self._gemini_image_snippet(model, language)
            return self._image_snippet(model, language)
        elif mtype == ModelType.AUDIO_TTS:
            return self._tts_snippet(model, language)
        elif mtype == ModelType.EMBEDDING:
            return self._embedding_snippet(model, language)
        elif mtype == ModelType.MUSIC:
            return self._music_snippet(model, language)
        elif mtype == ModelType.VIDEO:
            return self._video_snippet(model, language)
        if "native-audio" in model:
            return self._live_audio_snippet(model, language)
        if "deep-research" in model:
            return self._deep_research_snippet(model, language)
        if "robotics" in model:
            return self._robotics_snippet(model, language)
        return self._text_snippet(model, language)

    def _text_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from google import genai\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'response = client.models.generate_content(\n'
                f'    model="{model}",\n'
                '    contents="Hello!",\n'
                ')\n'
                'print(response.text)\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Hello!",\n'
                '});\n'
                'console.log(response.text);\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Hello!",\n'
                '});\n'
                'console.log(response.text);\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.GenerateContentResponse;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'GenerateContentResponse response = client.models.generateContent(\n'
                f'    "{model}",\n'
                '    "Hello!"\n'
                ');\n'
                'System.out.println(response.text());\n'
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
                '    const char* api_key = std::getenv("GEMINI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                '        {"contents", {{{"parts", {{{"text", "Hello!"}}}}}}}}\n'
                '    };\n\n'
                '    std::string url =\n'
                '        "https://generativelanguage.googleapis.com/v1beta/models/"\n'
                f'        "{model}:generateContent?key=" + std::string(api_key);\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["candidates"][0]["content"]["parts"][0]["text"]\n'
                '              .get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _live_audio_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'import asyncio\n'
                'from google import genai\n'
                'from google.genai import types\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'async def main():\n'
                '    async with client.aio.live.connect(\n'
                f'        model="{model}",\n'
                '        config=types.LiveConnectConfig(\n'
                '            response_modalities=["AUDIO"],\n'
                '        ),\n'
                '    ) as session:\n'
                '        await session.send("Hello, how are you?", end_of_turn=True)\n\n'
                '        async for response in session.receive():\n'
                '            if response.data:\n'
                '                # response.data contains raw audio bytes\n'
                '                print(f"Received {len(response.data)} audio bytes")\n\n'
                'asyncio.run(main())\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'const session = await ai.live.connect({\n'
                f'  model: "{model}",\n'
                '  config: { responseModalities: ["AUDIO"] },\n'
                '});\n\n'
                'session.send("Hello, how are you?", { endOfTurn: true });\n\n'
                'for await (const response of session) {\n'
                '  if (response.data) {\n'
                '    console.log(`Received ${response.data.length} audio bytes`);\n'
                '  }\n'
                '}\n'
                'session.close();\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'const session = await ai.live.connect({\n'
                f'  model: "{model}",\n'
                '  config: { responseModalities: ["AUDIO"] },\n'
                '});\n\n'
                'session.send("Hello, how are you?", { endOfTurn: true });\n\n'
                'for await (const response of session) {\n'
                '  if (response.data) {\n'
                '    console.log(`Received ${response.data.length} audio bytes`);\n'
                '  }\n'
                '}\n'
                'session.close();\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.LiveConnectConfig;\n'
                'import com.google.genai.types.LiveSession;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'LiveSession session = client.live.connect(\n'
                f'    "{model}",\n'
                '    LiveConnectConfig.builder()\n'
                '        .responseModalities(java.util.List.of("AUDIO"))\n'
                '        .build()\n'
                ');\n\n'
                'session.send("Hello, how are you?");\n'
                'System.out.println("Audio session established");\n'
                'session.close();\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <curl/curl.h>\n'
                '#include <nlohmann/json.hpp>\n\n'
                'using json = nlohmann::json;\n\n'
                '// The Live API uses WebSockets for real-time streaming.\n'
                '// For C++ you need a WebSocket library (e.g., libwebsockets, Boost.Beast).\n'
                '// Below is a simplified outline.\n\n'
                'int main() {\n'
                '    const char* api_key = std::getenv("GEMINI_API_KEY");\n\n'
                '    // WebSocket URL for the Live API\n'
                '    std::string ws_url =\n'
                '        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage"\n'
                f'        ".v1beta.GenerativeService.BidiGenerateContent?key=" +\n'
                '        std::string(api_key);\n\n'
                '    // Setup message to send after WebSocket connection\n'
                '    json setup = {\n'
                '        {"setup", {\n'
                f'            {{"model", "models/{model}"}},\n'
                '            {"generationConfig", {{"responseModalities", {"AUDIO"}}}}\n'
                '        }}\n'
                '    };\n\n'
                '    std::cout << "Connect to: " << ws_url << std::endl;\n'
                '    std::cout << "Send setup: " << setup.dump(2) << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _deep_research_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'import time\n'
                'from google import genai\n'
                'from google.genai import types\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'response = client.models.generate_content(\n'
                f'    model="{model}",\n'
                '    contents="Research the impact of quantum computing on cryptography",\n'
                '    config=types.GenerateContentConfig(\n'
                '        tools=[types.Tool(google_search=types.GoogleSearch())],\n'
                '    ),\n'
                ')\n\n'
                '# Deep research may return a long-running operation\n'
                'if hasattr(response, "operation") and response.operation:\n'
                '    while not response.operation.done:\n'
                '        time.sleep(10)\n'
                '        response = client.operations.get(response.operation)\n\n'
                'print(response.text)\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'let response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Research the impact of quantum computing on cryptography",\n'
                '  config: {\n'
                '    tools: [{ googleSearch: {} }],\n'
                '  },\n'
                '});\n\n'
                '// Deep research may return a long-running operation\n'
                'if (response.operation && !response.operation.done) {\n'
                '  while (!response.operation.done) {\n'
                '    await new Promise(r => setTimeout(r, 10000));\n'
                '    response = await ai.operations.get(response.operation);\n'
                '  }\n'
                '}\n\n'
                'console.log(response.text);\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'let response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Research the impact of quantum computing on cryptography",\n'
                '  config: {\n'
                '    tools: [{ googleSearch: {} }],\n'
                '  },\n'
                '});\n\n'
                '// Deep research may return a long-running operation\n'
                'if (response.operation && !response.operation.done) {\n'
                '  while (!response.operation.done) {\n'
                '    await new Promise(r => setTimeout(r, 10000));\n'
                '    response = await ai.operations.get(response.operation);\n'
                '  }\n'
                '}\n\n'
                'console.log(response.text);\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.GenerateContentConfig;\n'
                'import com.google.genai.types.GenerateContentResponse;\n'
                'import com.google.genai.types.Tool;\n'
                'import com.google.genai.types.GoogleSearch;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'GenerateContentResponse response = client.models.generateContent(\n'
                f'    "{model}",\n'
                '    "Research the impact of quantum computing on cryptography",\n'
                '    GenerateContentConfig.builder()\n'
                '        .tools(java.util.List.of(\n'
                '            Tool.builder().googleSearch(GoogleSearch.builder().build()).build()))\n'
                '        .build()\n'
                ');\n\n'
                'System.out.println(response.text());\n'
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
                '    const char* api_key = std::getenv("GEMINI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                '        {"contents", {{{"parts", {{{"text",\n'
                '            "Research the impact of quantum computing on cryptography"}}}}}}},\n'
                '        {"tools", {{{"googleSearch", json::object()}}}}\n'
                '    };\n\n'
                '    std::string url =\n'
                '        "https://generativelanguage.googleapis.com/v1beta/models/"\n'
                f'        "{model}:generateContent?key=" + std::string(api_key);\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["candidates"][0]["content"]["parts"][0]["text"]\n'
                '              .get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _robotics_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from google import genai\n'
                'from google.genai import types\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                '# Load a camera frame\n'
                'with open("camera_frame.jpg", "rb") as f:\n'
                '    image_data = f.read()\n\n'
                'response = client.models.generate_content(\n'
                f'    model="{model}",\n'
                '    contents=[\n'
                '        types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),\n'
                '        "Pick up the red block and place it on the blue plate.",\n'
                '    ],\n'
                ')\n'
                'print(response.text)  # action plan / spatial reasoning output\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n'
                'import { readFileSync } from "fs";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'const imageData = readFileSync("camera_frame.jpg");\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: [\n'
                '    { inlineData: { data: imageData.toString("base64"), mimeType: "image/jpeg" } },\n'
                '    { text: "Pick up the red block and place it on the blue plate." },\n'
                '  ],\n'
                '});\n'
                'console.log(response.text);\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n'
                'const { readFileSync } = require("fs");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'const imageData = readFileSync("camera_frame.jpg");\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: [\n'
                '    { inlineData: { data: imageData.toString("base64"), mimeType: "image/jpeg" } },\n'
                '    { text: "Pick up the red block and place it on the blue plate." },\n'
                '  ],\n'
                '});\n'
                'console.log(response.text);\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.GenerateContentResponse;\n'
                'import com.google.genai.types.Content;\n'
                'import com.google.genai.types.Part;\n'
                'import java.nio.file.Files;\n'
                'import java.nio.file.Path;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'byte[] imageBytes = Files.readAllBytes(Path.of("camera_frame.jpg"));\n\n'
                'GenerateContentResponse response = client.models.generateContent(\n'
                f'    "{model}",\n'
                '    Content.builder().parts(java.util.List.of(\n'
                '        Part.builder().inlineData(imageBytes, "image/jpeg").build(),\n'
                '        Part.builder().text(\n'
                '            "Pick up the red block and place it on the blue plate.").build()\n'
                '    )).build()\n'
                ');\n'
                'System.out.println(response.text());\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <fstream>\n'
                '#include <sstream>\n'
                '#include <curl/curl.h>\n'
                '#include <nlohmann/json.hpp>\n\n'
                'using json = nlohmann::json;\n\n'
                'static size_t WriteCallback(void* contents, size_t size,\n'
                '                            size_t nmemb, std::string* out) {\n'
                '    out->append((char*)contents, size * nmemb);\n'
                '    return size * nmemb;\n'
                '}\n\n'
                'std::string base64_encode(const std::string& data);\n'
                '// (use a base64 library such as boost::beast::detail::base64)\n\n'
                'int main() {\n'
                '    const char* api_key = std::getenv("GEMINI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    // Read camera frame\n'
                '    std::ifstream file("camera_frame.jpg", std::ios::binary);\n'
                '    std::ostringstream oss;\n'
                '    oss << file.rdbuf();\n'
                '    std::string image_b64 = base64_encode(oss.str());\n\n'
                '    json body = {\n'
                '        {"contents", {{{"parts", {\n'
                '            {{\"inlineData\", {{\"mimeType\", \"image/jpeg\"},\n'
                '                              {\"data\", image_b64}}}},\n'
                '            {{\"text\", \"Pick up the red block and place it on the blue plate.\"}}\n'
                '        }}}}}\n'
                '    };\n\n'
                '    std::string url =\n'
                '        "https://generativelanguage.googleapis.com/v1beta/models/"\n'
                f'        "{model}:generateContent?key=" + std::string(api_key);\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["candidates"][0]["content"]["parts"][0]["text"]\n'
                '              .get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _image_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from google import genai\n'
                'from google.genai import types\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'response = client.models.generate_images(\n'
                f'    model="{model}",\n'
                '    prompt="A white siamese cat",\n'
                '    config=types.GenerateImagesConfig(number_of_images=1),\n'
                ')\n'
                'response.generated_images[0].image.save("output.png")\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'const response = await ai.models.generateImages({\n'
                f'  model: "{model}",\n'
                '  prompt: "A white siamese cat",\n'
                '  config: { numberOfImages: 1 },\n'
                '});\n'
                'console.log(response.generatedImages[0]);\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'const response = await ai.models.generateImages({\n'
                f'  model: "{model}",\n'
                '  prompt: "A white siamese cat",\n'
                '  config: { numberOfImages: 1 },\n'
                '});\n'
                'console.log(response.generatedImages[0]);\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.GenerateImagesConfig;\n'
                'import com.google.genai.types.GenerateImagesResponse;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'GenerateImagesResponse response = client.models.generateImages(\n'
                f'    "{model}",\n'
                '    "A white siamese cat",\n'
                '    GenerateImagesConfig.builder().numberOfImages(1).build()\n'
                ');\n'
                'System.out.println(response.generatedImages().get(0));\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <fstream>\n'
                '#include <curl/curl.h>\n'
                '#include <nlohmann/json.hpp>\n\n'
                'using json = nlohmann::json;\n\n'
                'static size_t WriteCallback(void* contents, size_t size,\n'
                '                            size_t nmemb, std::string* out) {\n'
                '    out->append((char*)contents, size * nmemb);\n'
                '    return size * nmemb;\n'
                '}\n\n'
                'int main() {\n'
                '    const char* api_key = std::getenv("GEMINI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                '        {"instances", {{{"prompt", "A white siamese cat"}}}},\n'
                '        {"parameters", {{"sampleCount", 1}}}\n'
                '    };\n\n'
                '    std::string url =\n'
                '        "https://generativelanguage.googleapis.com/v1beta/models/"\n'
                f'        "{model}:predict?key=" + std::string(api_key);\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << "Image generated successfully" << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _gemini_image_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from google import genai\n'
                'from google.genai import types\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'response = client.models.generate_content(\n'
                f'    model="{model}",\n'
                '    contents="A white siamese cat",\n'
                '    config=types.GenerateContentConfig(\n'
                '        response_modalities=["IMAGE"],\n'
                '    ),\n'
                ')\n\n'
                'image = response.candidates[0].content.parts[0].inline_data\n'
                'with open("output.png", "wb") as f:\n'
                '    f.write(image.data)\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "A white siamese cat",\n'
                '  config: { responseModalities: ["IMAGE"] },\n'
                '});\n\n'
                'const imageData = response.candidates![0].content!.parts![0].inlineData!.data;\n'
                'console.log("Image generated, length:", imageData.length);\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "A white siamese cat",\n'
                '  config: { responseModalities: ["IMAGE"] },\n'
                '});\n\n'
                'const imageData = response.candidates[0].content.parts[0].inlineData.data;\n'
                'console.log("Image generated, length:", imageData.length);\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.GenerateContentConfig;\n'
                'import com.google.genai.types.GenerateContentResponse;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'GenerateContentResponse response = client.models.generateContent(\n'
                f'    "{model}",\n'
                '    "A white siamese cat",\n'
                '    GenerateContentConfig.builder()\n'
                '        .responseModalities(java.util.List.of("IMAGE"))\n'
                '        .build()\n'
                ');\n'
                'System.out.println("Image generated successfully");\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <fstream>\n'
                '#include <curl/curl.h>\n'
                '#include <nlohmann/json.hpp>\n\n'
                'using json = nlohmann::json;\n\n'
                'static size_t WriteCallback(void* contents, size_t size,\n'
                '                            size_t nmemb, std::string* out) {\n'
                '    out->append((char*)contents, size * nmemb);\n'
                '    return size * nmemb;\n'
                '}\n\n'
                'int main() {\n'
                '    const char* api_key = std::getenv("GEMINI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                '        {"contents", {{{"parts", {{{"text", "A white siamese cat"}}}}}}},\n'
                '        {"generationConfig", {{"responseModalities", {"IMAGE"}}}}\n'
                '    };\n\n'
                '    std::string url =\n'
                '        "https://generativelanguage.googleapis.com/v1beta/models/"\n'
                f'        "{model}:generateContent?key=" + std::string(api_key);\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << "Image generated successfully" << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _tts_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from google import genai\n'
                'from google.genai import types\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'response = client.models.generate_content(\n'
                f'    model="{model}",\n'
                '    contents="Hello, world!",\n'
                '    config=types.GenerateContentConfig(\n'
                '        response_modalities=["AUDIO"],\n'
                '        speech_config=types.SpeechConfig(\n'
                '            voice_config=types.VoiceConfig(\n'
                '                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")\n'
                '            )\n'
                '        ),\n'
                '    ),\n'
                ')\n'
                'with open("output.wav", "wb") as f:\n'
                '    f.write(response.candidates[0].content.parts[0].inline_data.data)\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Hello, world!",\n'
                '  config: {\n'
                '    responseModalities: ["AUDIO"],\n'
                '    speechConfig: {\n'
                '      voiceConfig: {\n'
                '        prebuiltVoiceConfig: { voiceName: "Kore" },\n'
                '      },\n'
                '    },\n'
                '  },\n'
                '});\n'
                'const audioData = response.candidates[0].content.parts[0].inlineData.data;\n'
                'console.log("Audio generated, length:", audioData.length);\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Hello, world!",\n'
                '  config: {\n'
                '    responseModalities: ["AUDIO"],\n'
                '    speechConfig: {\n'
                '      voiceConfig: {\n'
                '        prebuiltVoiceConfig: { voiceName: "Kore" },\n'
                '      },\n'
                '    },\n'
                '  },\n'
                '});\n'
                'const audioData = response.candidates[0].content.parts[0].inlineData.data;\n'
                'console.log("Audio generated, length:", audioData.length);\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.GenerateContentConfig;\n'
                'import com.google.genai.types.GenerateContentResponse;\n'
                'import com.google.genai.types.SpeechConfig;\n'
                'import com.google.genai.types.VoiceConfig;\n'
                'import com.google.genai.types.PrebuiltVoiceConfig;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'GenerateContentResponse response = client.models.generateContent(\n'
                f'    "{model}",\n'
                '    "Hello, world!",\n'
                '    GenerateContentConfig.builder()\n'
                '        .responseModalities(java.util.List.of("AUDIO"))\n'
                '        .speechConfig(SpeechConfig.builder()\n'
                '            .voiceConfig(VoiceConfig.builder()\n'
                '                .prebuiltVoiceConfig(PrebuiltVoiceConfig.builder()\n'
                '                    .voiceName("Kore").build()).build()).build())\n'
                '        .build()\n'
                ');\n'
                'System.out.println("Audio generated successfully");\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <fstream>\n'
                '#include <curl/curl.h>\n'
                '#include <nlohmann/json.hpp>\n\n'
                'using json = nlohmann::json;\n\n'
                'static size_t WriteCallback(void* contents, size_t size,\n'
                '                            size_t nmemb, std::string* out) {\n'
                '    out->append((char*)contents, size * nmemb);\n'
                '    return size * nmemb;\n'
                '}\n\n'
                'int main() {\n'
                '    const char* api_key = std::getenv("GEMINI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                '        {"contents", {{{"parts", {{{"text", "Hello, world!"}}}}}}}},\n'
                '        {"generationConfig", {\n'
                '            {"responseModalities", {"AUDIO"}},\n'
                '            {"speechConfig", {{"voiceConfig", {{"prebuiltVoiceConfig",\n'
                '                {{"voiceName", "Kore"}}}}}}}\n'
                '        }}\n'
                '    };\n\n'
                '    std::string url =\n'
                '        "https://generativelanguage.googleapis.com/v1beta/models/"\n'
                f'        "{model}:generateContent?key=" + std::string(api_key);\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << "Audio speech generated successfully" << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _embedding_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from google import genai\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'response = client.models.embed_content(\n'
                f'    model="{model}",\n'
                '    contents="Hello, world!",\n'
                ')\n'
                'print(response.embeddings[0].values[:5])\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'const response = await ai.models.embedContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Hello, world!",\n'
                '});\n'
                'console.log(response.embeddings[0].values.slice(0, 5));\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'const response = await ai.models.embedContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Hello, world!",\n'
                '});\n'
                'console.log(response.embeddings[0].values.slice(0, 5));\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.EmbedContentResponse;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'EmbedContentResponse response = client.models.embedContent(\n'
                f'    "{model}",\n'
                '    "Hello, world!"\n'
                ');\n'
                'System.out.println(response.embeddings().get(0).values());\n'
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
                '    const char* api_key = std::getenv("GEMINI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                '        {"content", {{"parts", {{{"text", "Hello, world!"}}}}}}\n'
                '    };\n\n'
                '    std::string url =\n'
                '        "https://generativelanguage.googleapis.com/v1beta/models/"\n'
                f'        "{model}:embedContent?key=" + std::string(api_key);\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << "Embedding: " << result["embedding"]["values"][0]\n'
                '              << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _music_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from google import genai\n'
                'from google.genai import types\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'response = client.models.generate_content(\n'
                f'    model="{model}",\n'
                '    contents="A mellow jazz piano piece with soft drums",\n'
                '    config=types.GenerateContentConfig(\n'
                '        response_modalities=["AUDIO"],\n'
                '    ),\n'
                ')\n\n'
                'with open("output.wav", "wb") as f:\n'
                '    f.write(response.candidates[0].content.parts[0].inline_data.data)\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "A mellow jazz piano piece with soft drums",\n'
                '  config: { responseModalities: ["AUDIO"] },\n'
                '});\n\n'
                'const audioData = response.candidates![0].content!.parts![0].inlineData!.data;\n'
                'console.log("Audio generated, length:", audioData.length);\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "A mellow jazz piano piece with soft drums",\n'
                '  config: { responseModalities: ["AUDIO"] },\n'
                '});\n\n'
                'const audioData = response.candidates[0].content.parts[0].inlineData.data;\n'
                'console.log("Audio generated, length:", audioData.length);\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.GenerateContentConfig;\n'
                'import com.google.genai.types.GenerateContentResponse;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'GenerateContentResponse response = client.models.generateContent(\n'
                f'    "{model}",\n'
                '    "A mellow jazz piano piece with soft drums",\n'
                '    GenerateContentConfig.builder()\n'
                '        .responseModalities(java.util.List.of("AUDIO"))\n'
                '        .build()\n'
                ');\n'
                'System.out.println("Audio generated");\n'
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
                '    const char* api_key = std::getenv("GEMINI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                '        {"contents", {{{"parts", {{{"text",\n'
                '            "A mellow jazz piano piece with soft drums"}}}}}}}},\n'
                '        {"generationConfig", {{"responseModalities", {"AUDIO"}}}}\n'
                '    };\n\n'
                '    std::string url =\n'
                '        "https://generativelanguage.googleapis.com/v1beta/models/"\n'
                f'        "{model}:generateContent?key=" + std::string(api_key);\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << "Generated audio response" << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _video_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from google import genai\n'
                'from google.genai import types\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'operation = client.models.generate_videos(\n'
                f'    model="{model}",\n'
                '    prompt="A cinematic drone shot over a mountain lake at sunrise",\n'
                '    config=types.GenerateVideosConfig(aspect_ratio="16:9"),\n'
                ')\n\n'
                '# Poll until complete\n'
                'import time\n'
                'while not operation.done:\n'
                '    time.sleep(10)\n'
                '    operation = client.operations.get(operation)\n\n'
                'video = operation.result.generated_videos[0]\n'
                'client.files.download(file=video.video, download_path="output.mp4")\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'let operation = await ai.models.generateVideos({\n'
                f'  model: "{model}",\n'
                '  prompt: "A cinematic drone shot over a mountain lake at sunrise",\n'
                '  config: { aspectRatio: "16:9" },\n'
                '});\n\n'
                '// Poll until complete\n'
                'while (!operation.done) {\n'
                '  await new Promise(r => setTimeout(r, 10000));\n'
                '  operation = await ai.operations.get(operation);\n'
                '}\n\n'
                'const video = operation.result!.generatedVideos![0];\n'
                'console.log("Video generated:", video.video);\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'let operation = await ai.models.generateVideos({\n'
                f'  model: "{model}",\n'
                '  prompt: "A cinematic drone shot over a mountain lake at sunrise",\n'
                '  config: { aspectRatio: "16:9" },\n'
                '});\n\n'
                '// Poll until complete\n'
                'while (!operation.done) {\n'
                '  await new Promise(r => setTimeout(r, 10000));\n'
                '  operation = await ai.operations.get(operation);\n'
                '}\n\n'
                'const video = operation.result.generatedVideos[0];\n'
                'console.log("Video generated:", video.video);\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.GenerateVideosConfig;\n'
                'import com.google.genai.types.Operation;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'Operation operation = client.models.generateVideos(\n'
                f'    "{model}",\n'
                '    "A cinematic drone shot over a mountain lake at sunrise",\n'
                '    GenerateVideosConfig.builder().aspectRatio("16:9").build()\n'
                ');\n\n'
                '// Poll until complete\n'
                'while (!operation.done()) {\n'
                '    Thread.sleep(10000);\n'
                '    operation = client.operations.get(operation);\n'
                '}\n'
                'System.out.println("Video: " + operation.result().generatedVideos().get(0));\n'
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
                '    const char* api_key = std::getenv("GEMINI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                f'        {{"model", "{model}"}},\n'
                '        {"prompt", "A cinematic drone shot over a mountain lake"},\n'
                '        {"config", {{"aspectRatio", "16:9"}}}\n'
                '    };\n\n'
                '    std::string url =\n'
                '        "https://generativelanguage.googleapis.com/v1beta/models/"\n'
                f'        "{model}:generateVideos?key=" + std::string(api_key);\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << "Operation: " << result.dump(2) << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])
