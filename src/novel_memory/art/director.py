"""
AI-powered Art Director Module
Generates visual descriptions and image prompts for story chapters.
"""
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from novel_memory.models.abstraction_layer import (
    ModelManager, 
    ModelCapability, 
    GenerationRequest
)

logger = logging.getLogger(__name__)

@dataclass
class ArtRequest:
    """Request for image prompt generation"""
    content: str
    chapter_title: str
    story_context: Dict[str, Any] = None
    style_guide: str = "cinematic, detailed, masterpiece"

@dataclass
class ArtResponse:
    """Response from art director"""
    image_prompt: str
    success: bool
    error_message: Optional[str] = None
    model_used: str = ""
    tokens_used: int = 0

class ArtDirector:
    """Handles visual direction and prompt generation"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
    async def generate_chapter_prompt(self, request: ArtRequest) -> ArtResponse:
        """Generate a compelling image prompt for the chapter"""
        
        try:
            logger.info(f"Generating art prompt for: {request.chapter_title}")
            
            prompt = f"""You are an expert concept artist and art director.

TASK:
Create a single, vivid, highly detailed image generation prompt (optimized for Midjourney/DALL-E 3) that captures the most visually striking scene or theme from this chapter.

INPUT CONTEXT:
Title: {request.chapter_title}
Style: {request.style_guide}
Story Info: {request.story_context}

CHAPTER CONTENT (Excerpt/Summary):
{request.content[:10000]} 

INSTRUCTIONS:
- focus on visual details: lighting, composition, color palette, mood, and texture.
- Do not describe abstract concepts; describe what is seen.
- Output ONLY the prompt text. Do not include "Here is the prompt:" or quotes unless necessary for the prompt itself.
- Keep it under 200 words.

OUTPUT:
"""

            generation_request = GenerationRequest(
                prompt=prompt,
                max_tokens=500, 
                temperature=0.8,
                context={"task": "art_direction"},
                capability_needed=ModelCapability.TEXT_GENERATION,
                stop_sequences=[]
            )
            
            response = await self.model_manager.generate_text(
                prompt=prompt,
                capability=ModelCapability.TEXT_GENERATION,
                max_tokens=500,
                temperature=0.8,
                context=generation_request.context
            )
            
            if not response.success:
                return ArtResponse(
                    image_prompt="",
                    success=False,
                    error_message=response.error_message
                )
             
            return ArtResponse(
                image_prompt=response.generated_text.strip().strip('"'),
                success=True,
                model_used="azure-gpt",
                tokens_used=response.tokens_used
            )

        except Exception as e:
            logger.error(f"Art direction failed: {str(e)}")
            return ArtResponse(
                image_prompt="",
                success=False,
                error_message=str(e)
            )
