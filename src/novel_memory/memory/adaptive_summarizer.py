"""
Adaptive Summarization Strategy
Implements importance-aware, context-sensitive compression of narrative content
"""
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SummaryType(Enum):
    """Types of narrative summaries"""
    CHAPTER_SUMMARY = "chapter_summary"
    CHARACTER_DEVELOPMENT = "character_development"
    PLOT_PROGRESSION = "plot_progression"
    WORLD_BUILDING = "world_building"
    DIALOGUE_ESSENCE = "dialogue_essence"
    ACTION_SEQUENCE = "action_sequence"
    EMOTIONAL_BEATS = "emotional_beats"

class CompressionLevel(Enum):
    """Levels of content compression"""
    MINIMAL = 1      # 90% retention
    LIGHT = 2        # 75% retention  
    MODERATE = 3     # 50% retention
    AGGRESSIVE = 4   # 25% retention
    EXTREME = 5      # 10% retention

@dataclass
class SummaryContext:
    """Context for generating summaries"""
    content: str
    content_type: str
    chapter_id: str
    word_count: int
    protected_elements: List[str]
    key_characters: List[str]
    important_events: List[str]
    emotional_intensity: float
    plot_importance: float
    target_compression: CompressionLevel

@dataclass
class SummaryResult:
    """Result of summarization process"""
    original_length: int
    summary_length: int
    compression_ratio: float
    summary_type: SummaryType
    preserved_elements: List[str]
    lost_elements: List[str]
    summary_text: str
    metadata: Dict[str, Any]

class SummarizationStrategy(ABC):
    """Abstract base class for summarization strategies"""
    
    @abstractmethod
    def can_summarize(self, context: SummaryContext) -> bool:
        """Check if this strategy can handle the given content"""
        pass
    
    @abstractmethod
    def generate_summary(self, context: SummaryContext) -> SummaryResult:
        """Generate summary for the given content"""
        pass
    
    @abstractmethod
    def estimate_compression(self, context: SummaryContext) -> float:
        """Estimate compression ratio for the content"""
        pass

class CharacterDevelopmentStrategy(SummarizationStrategy):
    """Summarization strategy focused on character development"""
    
    def can_summarize(self, context: SummaryContext) -> bool:
        return any(char in context.content.lower() for char in context.key_characters)
    
    def generate_summary(self, context: SummaryContext) -> SummaryResult:
        # Extract character-focused content
        char_mentions = {}
        content_lower = context.content.lower()
        
        for char in context.key_characters:
            char_lower = char.lower()
            if char_lower in content_lower:
                # Find sentences mentioning the character
                sentences = re.split(r'[.!?]+', context.content)
                char_sentences = [s.strip() for s in sentences if char_lower in s.lower()]
                char_mentions[char] = char_sentences
        
        # Build character-focused summary
        summary_parts = []
        preserved_elements = []
        
        for char, sentences in char_mentions.items():
            if sentences:
                # Prioritize emotional and developmental content
                important_sentences = []
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(emotion in sentence_lower for emotion in 
                          ['felt', 'realized', 'understood', 'decided', 'changed', 'learned']):
                        important_sentences.append(sentence)
                        preserved_elements.append(f"character_development_{char}")
                
                if important_sentences:
                    summary_parts.append(f"{char}: {' '.join(important_sentences[:2])}")
        
        summary_text = " | ".join(summary_parts) if summary_parts else "No significant character development."
        
        return SummaryResult(
            original_length=len(context.content),
            summary_length=len(summary_text),
            compression_ratio=len(summary_text) / len(context.content),
            summary_type=SummaryType.CHARACTER_DEVELOPMENT,
            preserved_elements=preserved_elements,
            lost_elements=[],  # Would be computed by comparing with original
            summary_text=summary_text,
            metadata={"character_focus": list(char_mentions.keys())}
        )
    
    def estimate_compression(self, context: SummaryContext) -> float:
        char_content_ratio = sum(1 for char in context.key_characters 
                               if char.lower() in context.content.lower()) / max(len(context.key_characters), 1)
        base_compression = 0.3  # 30% retention
        return base_compression * (1 + char_content_ratio)

class PlotProgressionStrategy(SummarizationStrategy):
    """Summarization strategy focused on plot advancement"""
    
    def can_summarize(self, context: SummaryContext) -> bool:
        plot_keywords = ['discovered', 'revealed', 'decided', 'attacked', 'traveled', 
                        'met', 'found', 'lost', 'won', 'failed', 'began', 'ended']
        content_lower = context.content.lower()
        return any(keyword in content_lower for keyword in plot_keywords)
    
    def generate_summary(self, context: SummaryContext) -> SummaryResult:
        # Extract plot-critical events
        plot_keywords = ['discovered', 'revealed', 'decided', 'attacked', 'traveled', 
                        'met', 'found', 'lost', 'won', 'failed', 'began', 'ended']
        
        sentences = re.split(r'[.!?]+', context.content)
        plot_sentences = []
        preserved_elements = []
        
        for sentence in sentences:
            sentence_lower = sentence.strip().lower()
            if any(keyword in sentence_lower for keyword in plot_keywords):
                plot_sentences.append(sentence.strip())
                preserved_elements.append("plot_event")
        
        # Prioritize based on protected elements
        prioritized_sentences = []
        for sentence in plot_sentences:
            sentence_lower = sentence.lower()
            for protected_elem in context.protected_elements:
                if protected_elem.lower() in sentence_lower:
                    prioritized_sentences.insert(0, sentence)
                    break
            else:
                prioritized_sentences.append(sentence)
        
        # Build plot summary
        max_sentences = max(1, len(prioritized_sentences) // 3)  # Keep top 1/3
        summary_sentences = prioritized_sentences[:max_sentences]
        summary_text = " ".join(summary_sentences) if summary_sentences else "No significant plot advancement."
        
        return SummaryResult(
            original_length=len(context.content),
            summary_length=len(summary_text),
            compression_ratio=len(summary_text) / len(context.content),
            summary_type=SummaryType.PLOT_PROGRESSION,
            preserved_elements=preserved_elements,
            lost_elements=[],
            summary_text=summary_text,
            metadata={"plot_events": len(plot_sentences)}
        )
    
    def estimate_compression(self, context: SummaryContext) -> float:
        return 0.4 * context.plot_importance  # Higher plot importance = higher retention

class DialogueEssenceStrategy(SummarizationStrategy):
    """Summarization strategy for dialogue content"""
    
    def can_summarize(self, context: SummaryContext) -> bool:
        # Simple dialogue detection
        quote_count = context.content.count('"')
        return quote_count >= 4  # At least 2 dialogue exchanges
    
    def generate_summary(self, context: SummaryContext) -> SummaryResult:
        # Extract dialogue
        dialogue_pattern = r'"([^"]*)"'
        dialogues = re.findall(dialogue_pattern, context.content)
        
        # Find speaker attributions
        attribution_pattern = r'"[^"]*"\s*([^.!?]*(?:said|asked|replied|whispered|shouted)[^.!?]*)'
        attributions = re.findall(attribution_pattern, context.content)
        
        # Summarize key dialogue points
        important_dialogues = []
        preserved_elements = []
        
        for i, dialogue in enumerate(dialogues):
            dialogue_lower = dialogue.lower()
            # Check for important content
            if any(important_word in dialogue_lower for important_word in 
                  ['secret', 'truth', 'plan', 'danger', 'love', 'betrayal', 'prophecy']):
                speaker = attributions[i] if i < len(attributions) else "someone"
                important_dialogues.append(f'"{dialogue}" ({speaker.strip()})')
                preserved_elements.append("important_dialogue")
        
        # If no important dialogue found, keep a few representative pieces
        if not important_dialogues and dialogues:
            important_dialogues = [f'"{dialogues[0]}"', f'"{dialogues[-1]}"']
        
        summary_text = " | ".join(important_dialogues) if important_dialogues else "Dialogue occurred."
        
        return SummaryResult(
            original_length=len(context.content),
            summary_length=len(summary_text),
            compression_ratio=len(summary_text) / len(context.content),
            summary_type=SummaryType.DIALOGUE_ESSENCE,
            preserved_elements=preserved_elements,
            lost_elements=[],
            summary_text=summary_text,
            metadata={"dialogue_count": len(dialogues), "important_count": len(important_dialogues)}
        )
    
    def estimate_compression(self, context: SummaryContext) -> float:
        quote_count = context.content.count('"')
        dialogue_ratio = min(quote_count / 20, 1.0)  # Normalize dialogue density
        return 0.2 + (0.3 * dialogue_ratio)  # 20-50% retention based on dialogue density

class EmotionalBeatsStrategy(SummarizationStrategy):
    """Summarization strategy focused on emotional content"""
    
    def can_summarize(self, context: SummaryContext) -> bool:
        return context.emotional_intensity > 0.5
    
    def generate_summary(self, context: SummaryContext) -> SummaryResult:
        emotion_words = [
            'angry', 'sad', 'happy', 'fear', 'love', 'hate', 'joy', 'grief',
            'excitement', 'terror', 'relief', 'anxiety', 'hope', 'despair',
            'surprised', 'shocked', 'amazed', 'disgusted', 'proud', 'ashamed'
        ]
        
        sentences = re.split(r'[.!?]+', context.content)
        emotional_sentences = []
        preserved_elements = []
        
        for sentence in sentences:
            sentence_lower = sentence.strip().lower()
            if any(emotion in sentence_lower for emotion in emotion_words):
                emotional_sentences.append(sentence.strip())
                preserved_elements.append("emotional_content")
        
        # Prioritize by emotional intensity and character involvement
        prioritized = []
        for sentence in emotional_sentences:
            sentence_lower = sentence.lower()
            score = 0
            
            # Score by emotion word count
            score += sum(1 for emotion in emotion_words if emotion in sentence_lower)
            
            # Score by character involvement
            score += sum(1 for char in context.key_characters if char.lower() in sentence_lower)
            
            prioritized.append((sentence, score))
        
        prioritized.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top emotional beats
        max_beats = max(1, len(prioritized) // 2)
        top_beats = [sentence for sentence, score in prioritized[:max_beats]]
        
        summary_text = " | ".join(top_beats) if top_beats else "Emotional content present."
        
        return SummaryResult(
            original_length=len(context.content),
            summary_length=len(summary_text),
            compression_ratio=len(summary_text) / len(context.content),
            summary_type=SummaryType.EMOTIONAL_BEATS,
            preserved_elements=preserved_elements,
            lost_elements=[],
            summary_text=summary_text,
            metadata={"emotional_intensity": context.emotional_intensity, "beats_count": len(top_beats)}
        )
    
    def estimate_compression(self, context: SummaryContext) -> float:
        return 0.3 + (0.4 * context.emotional_intensity)  # 30-70% retention based on intensity

class AdaptiveSummarizer:
    """
    Adaptive summarization system that selects appropriate strategies
    """
    
    def __init__(self):
        self.strategies = [
            CharacterDevelopmentStrategy(),
            PlotProgressionStrategy(),
            DialogueEssenceStrategy(),
            EmotionalBeatsStrategy()
        ]
        
        self.summarization_history: List[Dict[str, Any]] = []
    
    def analyze_content(self, content: str, chapter_id: str, 
                       protected_elements: List[str] = None,
                       key_characters: List[str] = None) -> SummaryContext:
        """Analyze content to determine summarization context"""
        if protected_elements is None:
            protected_elements = []
        if key_characters is None:
            key_characters = []
        
        word_count = len(content.split())
        
        # Estimate emotional intensity
        emotion_words = ['angry', 'sad', 'happy', 'fear', 'love', 'hate', 'joy', 'grief']
        emotion_count = sum(1 for word in emotion_words if word in content.lower())
        emotional_intensity = min(emotion_count / 10, 1.0)
        
        # Estimate plot importance
        plot_keywords = ['discovered', 'revealed', 'decided', 'attacked', 'traveled']
        plot_count = sum(1 for keyword in plot_keywords if keyword in content.lower())
        plot_importance = min(plot_count / 5, 1.0)
        
        # Determine compression level based on content length and importance
        if word_count < 500:
            target_compression = CompressionLevel.MINIMAL
        elif plot_importance > 0.7 or emotional_intensity > 0.8:
            target_compression = CompressionLevel.LIGHT
        elif word_count < 2000:
            target_compression = CompressionLevel.MODERATE
        else:
            target_compression = CompressionLevel.AGGRESSIVE
        
        return SummaryContext(
            content=content,
            content_type="chapter_content",
            chapter_id=chapter_id,
            word_count=word_count,
            protected_elements=protected_elements,
            key_characters=key_characters,
            important_events=[],
            emotional_intensity=emotional_intensity,
            plot_importance=plot_importance,
            target_compression=target_compression
        )
    
    def generate_multi_strategy_summary(self, context: SummaryContext) -> Dict[str, SummaryResult]:
        """Generate summaries using multiple applicable strategies"""
        results = {}
        
        for strategy in self.strategies:
            if strategy.can_summarize(context):
                try:
                    result = strategy.generate_summary(context)
                    results[result.summary_type.value] = result
                    logger.debug(f"Generated {result.summary_type.value} summary with {result.compression_ratio:.2f} compression")
                except Exception as e:
                    logger.error(f"Error in {strategy.__class__.__name__}: {e}")
        
        return results
    
    def create_layered_summary(self, context: SummaryContext) -> str:
        """Create a layered summary combining multiple strategies"""
        strategy_results = self.generate_multi_strategy_summary(context)
        
        if not strategy_results:
            # Fallback: simple truncation
            words = context.content.split()
            target_words = int(len(words) * 0.3)  # 30% retention
            return " ".join(words[:target_words]) + "..."
        
        # Combine summaries by importance
        summary_parts = []
        
        # Priority order for summary types
        priority_order = [
            SummaryType.PLOT_PROGRESSION,
            SummaryType.CHARACTER_DEVELOPMENT,
            SummaryType.EMOTIONAL_BEATS,
            SummaryType.DIALOGUE_ESSENCE
        ]
        
        for summary_type in priority_order:
            type_key = summary_type.value
            if type_key in strategy_results:
                result = strategy_results[type_key]
                if result.summary_text and result.summary_text != "No significant plot advancement." and result.summary_text != "Emotional content present.":
                    summary_parts.append(f"[{summary_type.value.replace('_', ' ').title()}] {result.summary_text}")
        
        layered_summary = " | ".join(summary_parts) if summary_parts else context.content[:200] + "..."
        
        # Record summarization history
        self.summarization_history.append({
            "chapter_id": context.chapter_id,
            "original_length": context.word_count,
            "summary_length": len(layered_summary.split()),
            "strategies_used": list(strategy_results.keys()),
            "timestamp": datetime.now().isoformat(),
            "compression_level": context.target_compression.value
        })
        
        return layered_summary
    
    def get_optimal_compression_level(self, content: str, target_length: int) -> CompressionLevel:
        """Determine optimal compression level to reach target length"""
        current_length = len(content.split())
        
        if current_length <= target_length:
            return CompressionLevel.MINIMAL
        
        compression_needed = target_length / current_length
        
        if compression_needed >= 0.9:
            return CompressionLevel.MINIMAL
        elif compression_needed >= 0.75:
            return CompressionLevel.LIGHT
        elif compression_needed >= 0.5:
            return CompressionLevel.MODERATE
        elif compression_needed >= 0.25:
            return CompressionLevel.AGGRESSIVE
        else:
            return CompressionLevel.EXTREME
    
    def get_summarization_metrics(self) -> Dict[str, Any]:
        """Get metrics on summarization performance"""
        if not self.summarization_history:
            return {"total_summaries": 0}
        
        total_summaries = len(self.summarization_history)
        avg_compression = sum(h["summary_length"] / h["original_length"] 
                            for h in self.summarization_history) / total_summaries
        
        strategy_usage = {}
        for history in self.summarization_history:
            for strategy in history["strategies_used"]:
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        compression_levels = {}
        for history in self.summarization_history:
            level = history["compression_level"]
            compression_levels[level] = compression_levels.get(level, 0) + 1
        
        return {
            "total_summaries": total_summaries,
            "average_compression_ratio": avg_compression,
            "strategy_usage": strategy_usage,
            "compression_levels": compression_levels,
            "most_used_strategy": max(strategy_usage.items(), key=lambda x: x[1])[0] if strategy_usage else None
        }