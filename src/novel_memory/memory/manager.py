"""
Narrative Memory Manager - Central coordination of memory systems.

Coordinates between ChromaDB semantic memory and markdown file persistence,
providing unified interface for narrative memory management with intelligent
context retrieval and consistency enforcement.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

from .chromadb_manager import ChromaDBManager
from .markdown_persistence import MarkdownStoryPersistence
from .enhanced_manager import EnhancedMemoryManager
from .hierarchical_memory import HierarchicalMemorySystem, MemoryTier, ProtectionLevel

logger = logging.getLogger(__name__)


@dataclass
class MemoryElement:
    """Unified representation of narrative memory element."""
    
    element_id: str
    element_type: str  # character, plot_thread, world_building, chapter, theme
    content: str
    metadata: Dict[str, Any]
    source: str  # chromadb, markdown, both
    last_updated: Optional[str] = None


@dataclass
class NarrativeContext:
    """Complete narrative context for generation."""
    
    current_chapter: int
    active_characters: List[str]
    active_plot_threads: List[str]
    current_setting: str
    narrative_tone: str
    recent_events: List[str]
    context_elements: List[MemoryElement]
    total_context_tokens: int


class NarrativeMemoryManager:
    """
    Central manager for narrative memory across ChromaDB and markdown persistence.
    
    Provides unified interface for storing, retrieving, and managing narrative
    elements with intelligent context assembly and consistency checking.
    """
    
    def __init__(
        self,
        db_path: str = "./chroma_db",
        story_path: str = "./story",
        embedding_model: str = "all-MiniLM-L6-v2",
        sync_mode: str = "bidirectional"  # chromadb_only, markdown_only, bidirectional
    ):
        """
        Initialize narrative memory manager.
        
        Args:
            db_path: Path to ChromaDB database
            story_path: Path to markdown story files
            embedding_model: SentenceTransformer model for embeddings
            sync_mode: How to sync between ChromaDB and markdown files
        """
        self.sync_mode = sync_mode
        
        # Initialize storage systems
        self.chromadb = ChromaDBManager(db_path, embedding_model)
        self.markdown = MarkdownStoryPersistence(story_path)
        
        # Create templates on first run
        self.markdown.create_story_templates()
        
        logger.info(f"NarrativeMemoryManager initialized (sync_mode: {sync_mode})")
    
    def store_character(
        self,
        character_name: str,
        character_data: Dict[str, Any],
        content: str,
        sync_to_chromadb: bool = True
    ) -> MemoryElement:
        """Store character profile in memory systems."""
        element_id = f"character_{self._sanitize_id(character_name)}"
        
        # Store in markdown
        self.markdown.save_character(character_name, character_data, content)
        
        # Store in ChromaDB if enabled
        if sync_to_chromadb and self.sync_mode in ["chromadb_only", "bidirectional"]:
            self.chromadb.store_narrative_element(
                "characters", element_id, content, character_data
            )
        
        return MemoryElement(
            element_id=element_id,
            element_type="character",
            content=content,
            metadata=character_data,
            source="both" if sync_to_chromadb else "markdown",
            last_updated=datetime.now().isoformat()
        )
    
    def store_plot_thread(
        self,
        thread_name: str,
        thread_data: Dict[str, Any],
        content: str,
        sync_to_chromadb: bool = True
    ) -> MemoryElement:
        """Store plot thread in memory systems."""
        element_id = f"plot_{self._sanitize_id(thread_name)}"
        
        # Store in markdown
        self.markdown.save_plot_thread(thread_name, thread_data, content)
        
        # Store in ChromaDB if enabled
        if sync_to_chromadb and self.sync_mode in ["chromadb_only", "bidirectional"]:
            self.chromadb.store_narrative_element(
                "plot_threads", element_id, content, thread_data
            )
        
        return MemoryElement(
            element_id=element_id,
            element_type="plot_thread",
            content=content,
            metadata=thread_data,
            source="both" if sync_to_chromadb else "markdown",
            last_updated=datetime.now().isoformat()
        )
    
    def store_world_element(
        self,
        element_name: str,
        element_data: Dict[str, Any],
        content: str,
        sync_to_chromadb: bool = True
    ) -> MemoryElement:
        """Store world building element in memory systems."""
        element_id = f"world_{self._sanitize_id(element_name)}"
        
        # Store in markdown
        self.markdown.save_world_element(element_name, element_data, content)
        
        # Store in ChromaDB if enabled
        if sync_to_chromadb and self.sync_mode in ["chromadb_only", "bidirectional"]:
            self.chromadb.store_narrative_element(
                "world_building", element_id, content, element_data
            )
        
        return MemoryElement(
            element_id=element_id,
            element_type="world_building",
            content=content,
            metadata=element_data,
            source="both" if sync_to_chromadb else "markdown",
            last_updated=datetime.now().isoformat()
        )
    
    def store_chapter(
        self,
        chapter_number: int,
        chapter_data: Dict[str, Any],
        content: str,
        sync_to_chromadb: bool = True
    ) -> MemoryElement:
        """Store chapter in memory systems."""
        element_id = f"chapter_{chapter_number:02d}"
        
        # Store in markdown
        self.markdown.save_chapter(chapter_number, chapter_data, content)
        
        # Store in ChromaDB if enabled
        if sync_to_chromadb and self.sync_mode in ["chromadb_only", "bidirectional"]:
            # Store chapter summary in ChromaDB, not full content
            summary = self._generate_chapter_summary(content, chapter_data)
            summary_data = chapter_data.copy()
            summary_data["type"] = "summary"
            
            self.chromadb.store_narrative_element(
                "chapters", element_id, summary, summary_data
            )
        
        return MemoryElement(
            element_id=element_id,
            element_type="chapter",
            content=content,
            metadata=chapter_data,
            source="both" if sync_to_chromadb else "markdown",
            last_updated=datetime.now().isoformat()
        )
    
    def store_theme(
        self,
        theme_name: str,
        theme_data: Dict[str, Any],
        content: str,
        sync_to_chromadb: bool = True
    ) -> MemoryElement:
        """Store theme development in memory systems."""
        element_id = f"theme_{self._sanitize_id(theme_name)}"
        
        # Store in markdown
        self.markdown.save_theme(theme_name, theme_data, content)
        
        # Store in ChromaDB if enabled
        if sync_to_chromadb and self.sync_mode in ["chromadb_only", "bidirectional"]:
            self.chromadb.store_narrative_element(
                "themes", element_id, content, theme_data
            )
        
        return MemoryElement(
            element_id=element_id,
            element_type="theme",
            content=content,
            metadata=theme_data,
            source="both" if sync_to_chromadb else "markdown",
            last_updated=datetime.now().isoformat()
        )
    
    def get_character(self, character_name: str) -> Optional[MemoryElement]:
        """Retrieve character profile."""
        # Try markdown first for complete profile
        markdown_data = self.markdown.load_character(character_name)
        if markdown_data:
            metadata, content = markdown_data
            return MemoryElement(
                element_id=f"character_{self._sanitize_id(character_name)}",
                element_type="character",
                content=content,
                metadata=metadata,
                source="markdown",
                last_updated=metadata.get("last_updated")
            )
        
        # Fallback to ChromaDB
        element_id = f"character_{self._sanitize_id(character_name)}"
        chromadb_data = self.chromadb.get_element_by_id("characters", element_id)
        if chromadb_data:
            return MemoryElement(
                element_id=element_id,
                element_type="character",
                content=chromadb_data["content"],
                metadata=chromadb_data["metadata"],
                source="chromadb"
            )
        
        return None
    
    def get_narrative_context(
        self,
        query: str,
        current_chapter: int,
        context_type: str = "comprehensive",  # focused, comprehensive, minimal
        max_tokens: int = 4000
    ) -> NarrativeContext:
        """
        Assemble comprehensive narrative context for generation.
        
        Args:
            query: Current writing query/prompt
            current_chapter: Current chapter number
            context_type: Type of context to retrieve
            max_tokens: Maximum tokens for context
            
        Returns:
            Assembled narrative context
        """
        context_elements = []
        total_tokens = 0
        
        # Get story state from most recent chapter
        story_state = self._get_current_story_state(current_chapter)
        
        if context_type == "comprehensive":
            # Get broad context across all narrative elements
            semantic_context = self.chromadb.search_narrative_context(
                query, max_results_per_collection=3
            )
            
            # Process results from each collection
            for collection_name, results in semantic_context.items():
                for result in results[:2]:  # Limit per collection
                    if total_tokens >= max_tokens:
                        break
                    
                    element = MemoryElement(
                        element_id=result["id"],
                        element_type=collection_name[:-1],  # Remove 's' from collection name
                        content=result["content"],
                        metadata=result["metadata"],
                        source="chromadb"
                    )
                    
                    # Rough token estimation
                    element_tokens = len(element.content) // 4
                    if total_tokens + element_tokens <= max_tokens:
                        context_elements.append(element)
                        total_tokens += element_tokens
        
        elif context_type == "focused":
            # Get focused context based on active story elements
            active_characters = story_state.get("active_characters", [])
            active_plots = story_state.get("active_plot_threads", [])
            
            # Get character context
            for character_name in active_characters[:3]:  # Limit active characters
                char_context = self.chromadb.get_character_context(character_name)
                if char_context["character_profile"]:
                    element = MemoryElement(
                        element_id=char_context["character_profile"]["id"],
                        element_type="character",
                        content=char_context["character_profile"]["content"],
                        metadata=char_context["character_profile"]["metadata"],
                        source="chromadb"
                    )
                    context_elements.append(element)
        
        elif context_type == "minimal":
            # Get only essential context for current scene
            recent_results = self.chromadb.search_similar_elements(
                "chapters", query, n_results=2
            )
            
            for result in recent_results:
                element = MemoryElement(
                    element_id=result["id"],
                    element_type="chapter",
                    content=result["content"],
                    metadata=result["metadata"],
                    source="chromadb"
                )
                context_elements.append(element)
        
        return NarrativeContext(
            current_chapter=current_chapter,
            active_characters=story_state.get("active_characters", []),
            active_plot_threads=story_state.get("active_plot_threads", []),
            current_setting=story_state.get("current_setting", ""),
            narrative_tone=story_state.get("narrative_tone", ""),
            recent_events=story_state.get("recent_events", []),
            context_elements=context_elements,
            total_context_tokens=total_tokens
        )
    
    def check_consistency(
        self,
        element_type: str,
        element_name: str,
        new_content: str,
        new_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check consistency of new content against existing narrative elements.
        
        Returns:
            Consistency report with warnings and suggestions
        """
        consistency_report = {
            "consistent": True,
            "warnings": [],
            "suggestions": [],
            "conflicts": []
        }
        
        # Check against existing element if it exists
        if element_type == "character":
            existing = self.get_character(element_name)
            if existing:
                consistency_report.update(
                    self._check_character_consistency(existing, new_content, new_metadata)
                )
        
        # Check for narrative contradictions
        similar_elements = self.chromadb.search_similar_elements(
            f"{element_type}s", new_content, n_results=5
        )
        
        for element in similar_elements:
            if element["similarity_score"] > 0.8:  # High similarity threshold
                conflicts = self._detect_content_conflicts(
                    new_content, element["content"], element["metadata"]
                )
                if conflicts:
                    consistency_report["conflicts"].extend(conflicts)
                    consistency_report["consistent"] = False
        
        return consistency_report
    
    def get_story_statistics(self) -> Dict[str, Any]:
        """Get comprehensive story statistics."""
        overview = self.markdown.get_story_overview()
        chromadb_stats = self.chromadb.get_collection_stats()
        
        return {
            "story_overview": overview,
            "chromadb_stats": chromadb_stats,
            "sync_mode": self.sync_mode,
            "total_elements": sum(
                stats.get("document_count", 0) 
                for stats in chromadb_stats.values()
                if isinstance(stats, dict)
            ),
            "generated_at": datetime.now().isoformat()
        }
    
    def synchronize_storage(self, direction: str = "both") -> Dict[str, int]:
        """
        Synchronize between ChromaDB and markdown storage.
        
        Args:
            direction: "to_chromadb", "to_markdown", or "both"
            
        Returns:
            Sync statistics
        """
        sync_stats = {"chromadb_updates": 0, "markdown_updates": 0, "errors": 0}
        
        if direction in ["to_chromadb", "both"]:
            # Sync markdown to ChromaDB
            try:
                # Sync characters
                for character_name in self.markdown.list_characters():
                    markdown_data = self.markdown.load_character(character_name)
                    if markdown_data:
                        metadata, content = markdown_data
                        element_id = f"character_{self._sanitize_id(character_name)}"
                        self.chromadb.store_narrative_element(
                            "characters", element_id, content, metadata
                        )
                        sync_stats["chromadb_updates"] += 1
                
                # Sync plot threads
                for thread_name in self.markdown.list_plot_threads():
                    markdown_data = self.markdown.load_plot_thread(thread_name)
                    if markdown_data:
                        metadata, content = markdown_data
                        element_id = f"plot_{self._sanitize_id(thread_name)}"
                        self.chromadb.store_narrative_element(
                            "plot_threads", element_id, content, metadata
                        )
                        sync_stats["chromadb_updates"] += 1
                        
            except Exception as e:
                logger.error(f"Error syncing to ChromaDB: {e}")
                sync_stats["errors"] += 1
        
        if direction in ["to_markdown", "both"]:
            # Sync ChromaDB to markdown would require more complex logic
            # For now, markdown is treated as authoritative
            logger.info("Markdown is treated as authoritative for sync operations")
        
        logger.info(f"Sync completed: {sync_stats}")
        return sync_stats
    
    def _get_current_story_state(self, current_chapter: int) -> Dict[str, Any]:
        """Extract current story state from recent chapters and elements."""
        story_state = {
            "active_characters": [],
            "active_plot_threads": [],
            "current_setting": "",
            "narrative_tone": "",
            "recent_events": []
        }
        
        # Get recent chapters for context
        recent_chapters = []
        for chapter_num in range(max(1, current_chapter - 2), current_chapter):
            chapter_data = self.markdown.load_chapter(chapter_num)
            if chapter_data:
                metadata, content = chapter_data
                recent_chapters.append(metadata)
        
        # Extract active elements from recent chapters
        for chapter_meta in recent_chapters:
            characters = chapter_meta.get("character_development", {}).keys()
            story_state["active_characters"].extend(characters)
            
            plot_threads = chapter_meta.get("plot_threads_advanced", [])
            story_state["active_plot_threads"].extend(plot_threads)
            
            if chapter_meta.get("setting"):
                story_state["current_setting"] = chapter_meta["setting"]
            
            if chapter_meta.get("emotional_tone"):
                story_state["narrative_tone"] = chapter_meta["emotional_tone"]
        
        # Deduplicate lists
        story_state["active_characters"] = list(set(story_state["active_characters"]))
        story_state["active_plot_threads"] = list(set(story_state["active_plot_threads"]))
        
        return story_state
    
    def _generate_chapter_summary(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> str:
        """Generate chapter summary for ChromaDB storage."""
        # Extract key information for summary
        summary_parts = []
        
        if metadata.get("title"):
            summary_parts.append(f"Chapter: {metadata['title']}")
        
        if metadata.get("pov_character"):
            summary_parts.append(f"POV: {metadata['pov_character']}")
        
        if metadata.get("setting"):
            summary_parts.append(f"Setting: {metadata['setting']}")
        
        if metadata.get("key_events"):
            events = ", ".join(metadata["key_events"][:3])  # Top 3 events
            summary_parts.append(f"Key events: {events}")
        
        # Add first paragraph as content sample
        first_paragraph = content.split('\n\n')[0] if content else ""
        if len(first_paragraph) > 200:
            first_paragraph = first_paragraph[:200] + "..."
        
        summary = ". ".join(summary_parts)
        if first_paragraph:
            summary += f"\n\nOpening: {first_paragraph}"
        
        return summary
    
    def _check_character_consistency(
        self,
        existing: MemoryElement,
        new_content: str,
        new_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check consistency between existing and new character information."""
        report = {"consistent": True, "warnings": [], "suggestions": []}
        
        # Check for trait changes
        existing_traits = set(existing.metadata.get("traits", []))
        new_traits = set(new_metadata.get("traits", []))
        
        removed_traits = existing_traits - new_traits
        if removed_traits:
            report["warnings"].append(
                f"Traits removed: {', '.join(removed_traits)}"
            )
            report["consistent"] = False
        
        # Check for relationship changes
        existing_relationships = existing.metadata.get("relationships", {})
        new_relationships = new_metadata.get("relationships", {})
        
        for person, old_relation in existing_relationships.items():
            new_relation = new_relationships.get(person)
            if new_relation and new_relation != old_relation:
                report["warnings"].append(
                    f"Relationship with {person} changed: {old_relation} -> {new_relation}"
                )
        
        return report
    
    def _detect_content_conflicts(
        self,
        new_content: str,
        existing_content: str,
        existing_metadata: Dict[str, Any]
    ) -> List[str]:
        """Detect potential conflicts between content pieces."""
        conflicts = []
        
        # This is a simplified conflict detection
        # In a full implementation, this would use NLP to detect semantic conflicts
        
        new_words = set(new_content.lower().split())
        existing_words = set(existing_content.lower().split())
        
        # Check for contradictory words
        contradictions = [
            ("alive", "dead"), ("young", "old"), ("happy", "sad"),
            ("married", "single"), ("rich", "poor")
        ]
        
        for word1, word2 in contradictions:
            if word1 in new_words and word2 in existing_words:
                conflicts.append(f"Potential contradiction: '{word1}' vs '{word2}'")
            elif word2 in new_words and word1 in existing_words:
                conflicts.append(f"Potential contradiction: '{word2}' vs '{word1}'")
        
        return conflicts
    
    def _sanitize_id(self, name: str) -> str:
        """Sanitize name for use as element ID."""
        return name.lower().replace(' ', '_').replace('-', '_')[:50]