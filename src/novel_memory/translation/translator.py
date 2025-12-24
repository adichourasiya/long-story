"""
AI-powered Translation Module
Provides post-generation translation capabilities while preserving formatting
"""
import logging
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass

from novel_memory.models.abstraction_layer import (
    ModelManager, 
    ModelCapability, 
    GenerationRequest
)

logger = logging.getLogger(__name__)

@dataclass
class TranslationRequest:
    """Request for text translation"""
    content: str
    target_language: str
    source_language: str = "auto"
    preserve_formatting: bool = True
    context: Dict[str, Any] = None

@dataclass
class TranslationResponse:
    """Response from translation"""
    translated_content: str
    source_language: str
    target_language: str
    success: bool
    error_message: Optional[str] = None
    model_used: str = ""
    tokens_used: int = 0

class StoryTranslator:
    """Handles translation of story content while preserving formatting"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
        # Language code mappings for common requests
        self.language_codes = {
            'fr': 'French',
            'french': 'French', 
            'es': 'Spanish',
            'spanish': 'Spanish',
            'de': 'German', 
            'german': 'German',
            'it': 'Italian',
            'italian': 'Italian',
            'pt': 'Portuguese',
            'portuguese': 'Portuguese',
            'ru': 'Russian',
            'russian': 'Russian',
            'ja': 'Japanese',
            'japanese': 'Japanese',
            'ko': 'Korean',
            'korean': 'Korean',
            'zh': 'Chinese',
            'chinese': 'Chinese',
            'ar': 'Arabic',
            'arabic': 'Arabic',
            'hi': 'Hindi',
            'hindi': 'Hindi',
            'ur': 'Urdu',
            'urdu': 'Urdu',
            'ta': 'Tamil',
            'tamil': 'Tamil',
            'te': 'Telugu',
            'telugu': 'Telugu',
            'bn': 'Bengali',
            'bengali': 'Bengali',
            'mr': 'Marathi',
            'marathi': 'Marathi',
            'gu': 'Gujarati',
            'gujarati': 'Gujarati',
            'kn': 'Kannada',
            'kannada': 'Kannada',
            'ml': 'Malayalam',
            'malayalam': 'Malayalam',
            'pa': 'Punjabi',
            'punjabi': 'Punjabi'
        }
    
    def normalize_language_code(self, language: str) -> str:
        """Normalize language code to full name"""
        language_lower = language.lower().strip()
        return self.language_codes.get(language_lower, language.title())
    
    async def translate_story(self, request: TranslationRequest) -> TranslationResponse:
        """Translate story content while preserving structure"""
        
        try:
            # Normalize target language
            target_language = self.normalize_language_code(request.target_language)
            
            logger.info(f"Translating story to {target_language}")
            
            # Extract metadata and content structure
            metadata_section, main_content = self._extract_metadata_and_content(request.content)
            
            # If content is too short, translate as one piece
            if len(main_content.split()) < 200:
                translated_content = await self._translate_text(main_content, target_language)
                
                if translated_content is None:
                    return TranslationResponse(
                        translated_content=request.content,
                        source_language="unknown",
                        target_language=target_language,
                        success=False,
                        error_message="Translation failed"
                    )
                
                # Reconstruct with original metadata
                final_content = metadata_section + translated_content if metadata_section else translated_content
                
                return TranslationResponse(
                    translated_content=final_content,
                    source_language="auto-detected",
                    target_language=target_language,
                    success=True,
                    model_used="azure-gpt",
                    tokens_used=len(translated_content.split())
                )
            
            # For longer content, translate in chunks to maintain quality
            chunks = self._split_content_into_chunks(main_content)
            translated_chunks = []
            total_tokens = 0
            
            for chunk in chunks:
                translated_chunk = await self._translate_text(chunk, target_language)
                if translated_chunk is None:
                    return TranslationResponse(
                        translated_content=request.content,
                        source_language="unknown",
                        target_language=target_language,
                        success=False,
                        error_message="Translation failed for content chunk"
                    )
                
                translated_chunks.append(translated_chunk)
                total_tokens += len(translated_chunk.split())
            
            # Reconstruct the translated content
            translated_main_content = "\n\n".join(translated_chunks)
            
            # Combine with original metadata
            final_content = metadata_section + translated_main_content if metadata_section else translated_main_content
            
            return TranslationResponse(
                translated_content=final_content,
                source_language="auto-detected",
                target_language=target_language,
                success=True,
                model_used="azure-gpt",
                tokens_used=total_tokens
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return TranslationResponse(
                translated_content=request.content,
                source_language="unknown", 
                target_language=request.target_language,
                success=False,
                error_message=str(e)
            )
    
    def _extract_metadata_and_content(self, content: str) -> tuple[str, str]:
        """Extract YAML metadata and main content"""
        
        # Check for YAML frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                yaml_section = f"---{parts[1]}---\n\n"
                main_content = parts[2].strip()
                return yaml_section, main_content
        
        # Check for markdown-style header
        lines = content.split('\n')
        header_end = 0
        
        for i, line in enumerate(lines):
            if line.startswith('#') or line.strip() == '':
                header_end = i + 1
            else:
                break
        
        if header_end > 0:
            header = '\n'.join(lines[:header_end]) + '\n\n'
            main_content = '\n'.join(lines[header_end:]).strip()
            return header, main_content
        
        return "", content
    
    def _split_content_into_chunks(self, content: str, max_words: int = 500) -> list[str]:
        """Split content into smaller chunks for translation"""
        
        # First try to split by paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())
            
            # If adding this paragraph exceeds limit, finalize current chunk
            if current_word_count + paragraph_words > max_words and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_word_count = paragraph_words
            else:
                current_chunk.append(paragraph)
                current_word_count += paragraph_words
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    async def _translate_text(self, text: str, target_language: str) -> Optional[str]:
        """Translate a piece of text using the model manager"""
        
        prompt = f"""You are a professional literary translator specializing in high-quality story translation. 

TASK: Translate the following text to {target_language}.

CRITICAL REQUIREMENTS:
1. Maintain the literary style, tone, and narrative voice
2. Preserve all formatting including paragraph breaks, dialogue formatting, and emphasis
3. Keep proper names, titles, and cultural references appropriate to the story context
4. Ensure the translation flows naturally in {target_language} while staying faithful to the original meaning
5. Do not add any explanations, comments, or translator notes
6. Output ONLY the translated text

TEXT TO TRANSLATE:

{text}

TRANSLATION:"""

        request = GenerationRequest(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.3,  # Lower temperature for more consistent translation
            stop_sequences=[],
            context={"task": "translation", "target_language": target_language},
            capability_needed=ModelCapability.TEXT_GENERATION
        )
        
        try:
            response = await self.model_manager.generate_text(
                prompt=prompt,
                capability=ModelCapability.TEXT_GENERATION,
                max_tokens=2000,
                temperature=0.3,
                context=request.context
            )
            
            if response.success:
                return response.generated_text.strip()
            else:
                logger.error(f"Translation request failed: {response.error_message}")
                return None
                
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return None