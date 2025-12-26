"""
AI-powered Proofreading Module
Provides proofreading, renaming, and image prompt generation for chapters.
"""
import logging
import json
import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from novel_memory.models.abstraction_layer import (
    ModelManager, 
    ModelCapability, 
    GenerationRequest
)

logger = logging.getLogger(__name__)

@dataclass
class ProofreadingRequest:
    """Request for chapter proofreading"""
    content: str
    chapter_title: str
    story_context: Dict[str, Any] = None

@dataclass
class ProofreadingResponse:
    """Response from proofreading"""
    proofread_content: str
    new_title: str
    image_prompt: str
    success: bool
    error_message: Optional[str] = None
    model_used: str = ""
    tokens_used: int = 0

class StoryProofreader:
    """Handles proofreading of story content"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
    async def proofread_chapter(self, request: ProofreadingRequest) -> ProofreadingResponse:
        """Proofread chapter, suggest title, and generate image prompt"""
        
        try:
            logger.info(f"Proofreading chapter: {request.chapter_title}")
            
            prompt = f"""You are an expert editor and creative director for a publishing house.

TASK:
1. Proofread and polish the provided chapter content. Improve flow, grammar, and style without changing the core plot or characters.
2. Create a compelling, short title for this chapter based on the content.
3. Write a vivid, detailed image generation prompt (Midjourney/DALL-E style) for a key scene in this chapter.

INPUT CONTEXT:
Original Title: {request.chapter_title}
Story Context: {request.story_context}

CHAPTER CONTENT:
{request.content}

OUTPUT FORMAT:
Return a JSON object with exactly these fields:
{{
    "proofread_text": "(the full proofread chapter text)",
    "new_title": "(the suggested title)",
    "image_prompt": "(the image generation prompt)"
}}
"""

            # We'll use a larger context window if possible, or just standard generation
            generation_request = GenerationRequest(
                prompt=prompt,
                max_tokens=6000, 
                temperature=0.7,
                context={"task": "proofreading"},
                capability_needed=ModelCapability.TEXT_GENERATION,
                stop_sequences=[]
            )
            
            response = await self.model_manager.generate_text(
                prompt=prompt,
                capability=ModelCapability.TEXT_GENERATION,
                max_tokens=6000,
                temperature=0.7,
                context=generation_request.context
            )
            
            if not response.success:
                return ProofreadingResponse(
                    proofread_content=request.content,
                    new_title=request.chapter_title,
                    image_prompt="",
                    success=False,
                    error_message=response.error_message
                )
                
            # Parse JSON output
            try:
                # Find JSON block if wrapped in code fences
                text = response.generated_text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(text)
                
                return ProofreadingResponse(
                    proofread_content=result.get("proofread_text", request.content),
                    new_title=result.get("new_title", request.chapter_title),
                    image_prompt=result.get("image_prompt", ""),
                    success=True,
                    model_used="azure-gpt",
                    tokens_used=response.tokens_used
                )
                
            except json.JSONDecodeError as e:
                # Fallback: if JSON parsing fails, try to salvage or error out safely
                # In a robust system, we might retry. For now, we return failure but keep original content.
                logger.error(f"JSON parsing failed: {e}. Raw response start: {response.generated_text[:100]}")
                return ProofreadingResponse(
                    proofread_content=request.content,
                    new_title=request.chapter_title,
                    image_prompt="",
                    success=False,
                    error_message=f"Failed to parse model response: {str(e)}"
                )

        except Exception as e:
            logger.error(f"Proofreading failed: {str(e)}")
            return ProofreadingResponse(
                proofread_content=request.content,
                new_title=request.chapter_title,
                image_prompt="",
                success=False,
                error_message=str(e)
            )
