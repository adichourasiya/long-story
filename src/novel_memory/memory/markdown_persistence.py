"""
Markdown Story Persistence Manager.

Handles markdown-based story storage with YAML frontmatter for structured
metadata and human-readable narrative content. Provides templates and
persistence for characters, plot threads, world building, and chapters.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MarkdownStoryPersistence:
    """Manage markdown-based story persistence with YAML frontmatter."""
    
    def __init__(self, story_base_path: str = "./story"):
        """
        Initialize markdown story persistence.
        
        Args:
            story_base_path: Base directory for story files
        """
        self.story_base_path = Path(story_base_path)
        self.ensure_directory_structure()
        
        logger.info(f"Markdown persistence initialized at {story_base_path}")
    
    def ensure_directory_structure(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.story_base_path,
            self.story_base_path / "characters",
            self.story_base_path / "plot" / "threads",
            self.story_base_path / "world" / "settings",
            self.story_base_path / "chapters",
            self.story_base_path / "chapters" / "summaries",
            self.story_base_path / "templates",
            self.story_base_path / "themes"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_character(
        self,
        character_name: str,
        metadata: Dict[str, Any],
        content: str,
        overwrite: bool = True
    ) -> Path:
        """
        Save character profile to markdown file.
        
        Args:
            character_name: Name of the character
            metadata: Character metadata for frontmatter
            content: Markdown content for character description
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path to saved file
        """
        filename = self._sanitize_filename(character_name) + ".md"
        file_path = self.story_base_path / "characters" / filename
        
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"Character file already exists: {file_path}")
        
        # Add timestamp to metadata
        metadata["last_updated"] = datetime.now().isoformat()
        
        self._write_markdown_file(file_path, metadata, content)
        logger.info(f"Saved character '{character_name}' to {file_path}")
        
        return file_path
    
    def load_character(self, character_name: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """Load character profile from markdown file."""
        filename = self._sanitize_filename(character_name) + ".md"
        file_path = self.story_base_path / "characters" / filename
        
        if not file_path.exists():
            return None
        
        return self._read_markdown_file(file_path)
    
    def list_characters(self) -> List[str]:
        """List all character names."""
        characters_dir = self.story_base_path / "characters"
        character_files = list(characters_dir.glob("*.md"))
        
        return [self._unsanitize_filename(f.stem) for f in character_files]
    
    def save_plot_thread(
        self,
        thread_name: str,
        metadata: Dict[str, Any],
        content: str,
        overwrite: bool = True
    ) -> Path:
        """Save plot thread to markdown file."""
        filename = self._sanitize_filename(thread_name) + ".md"
        file_path = self.story_base_path / "plot" / "threads" / filename
        
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"Plot thread file already exists: {file_path}")
        
        metadata["last_updated"] = datetime.now().isoformat()
        
        self._write_markdown_file(file_path, metadata, content)
        logger.info(f"Saved plot thread '{thread_name}' to {file_path}")
        
        return file_path
    
    def load_plot_thread(self, thread_name: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """Load plot thread from markdown file."""
        filename = self._sanitize_filename(thread_name) + ".md"
        file_path = self.story_base_path / "plot" / "threads" / filename
        
        if not file_path.exists():
            return None
        
        return self._read_markdown_file(file_path)
    
    def list_plot_threads(self) -> List[str]:
        """List all plot thread names."""
        threads_dir = self.story_base_path / "plot" / "threads"
        thread_files = list(threads_dir.glob("*.md"))
        
        return [self._unsanitize_filename(f.stem) for f in thread_files]
    
    def save_world_element(
        self,
        element_name: str,
        metadata: Dict[str, Any],
        content: str,
        overwrite: bool = True
    ) -> Path:
        """Save world building element to markdown file."""
        element_type = metadata.get('element_type', 'general')
        subdir = "settings" if element_type == "location" else element_type
        
        # Ensure subdirectory exists
        target_dir = self.story_base_path / "world" / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filename = self._sanitize_filename(element_name) + ".md"
        file_path = target_dir / filename
        
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"World element file already exists: {file_path}")
        
        metadata["last_updated"] = datetime.now().isoformat()
        
        self._write_markdown_file(file_path, metadata, content)
        logger.info(f"Saved world element '{element_name}' to {file_path}")
        
        return file_path
    
    def load_world_element(self, element_name: str, element_type: str = "settings") -> Optional[Tuple[Dict[str, Any], str]]:
        """Load world building element from markdown file."""
        subdir = "settings" if element_type == "location" else element_type
        filename = self._sanitize_filename(element_name) + ".md"
        file_path = self.story_base_path / "world" / subdir / filename
        
        if not file_path.exists():
            return None
        
        return self._read_markdown_file(file_path)
    
    def save_chapter(
        self,
        chapter_number: int,
        metadata: Dict[str, Any],
        content: str,
        overwrite: bool = True
    ) -> Path:
        """Save chapter to markdown file."""
        filename = f"chapter_{chapter_number:02d}.md"
        file_path = self.story_base_path / "chapters" / filename
        
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"Chapter file already exists: {file_path}")
        
        metadata["last_updated"] = datetime.now().isoformat()
        
        self._write_markdown_file(file_path, metadata, content)
        logger.info(f"Saved chapter {chapter_number} to {file_path}")
        
        return file_path
    
    def load_chapter(self, chapter_number: int) -> Optional[Tuple[Dict[str, Any], str]]:
        """Load chapter from markdown file."""
        filename = f"chapter_{chapter_number:02d}.md"
        file_path = self.story_base_path / "chapters" / filename
        
        if not file_path.exists():
            return None
        
        return self._read_markdown_file(file_path)
    
    def save_chapter_summary(
        self,
        chapter_number: int,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save chapter summary."""
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "chapter_number": chapter_number,
            "summary_created": datetime.now().isoformat(),
            "type": "summary"
        })
        
        filename = f"chapter_{chapter_number:02d}_summary.md"
        file_path = self.story_base_path / "chapters" / "summaries" / filename
        
        self._write_markdown_file(file_path, metadata, summary)
        logger.info(f"Saved chapter {chapter_number} summary to {file_path}")
        
        return file_path
    
    def save_theme(
        self,
        theme_name: str,
        metadata: Dict[str, Any],
        content: str,
        overwrite: bool = True
    ) -> Path:
        """Save theme development to markdown file."""
        filename = self._sanitize_filename(theme_name) + ".md"
        file_path = self.story_base_path / "themes" / filename
        
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"Theme file already exists: {file_path}")
        
        metadata["last_updated"] = datetime.now().isoformat()
        
        self._write_markdown_file(file_path, metadata, content)
        logger.info(f"Saved theme '{theme_name}' to {file_path}")
        
        return file_path
    
    def get_story_overview(self) -> Dict[str, Any]:
        """Get overview of all story elements."""
        overview = {
            "characters": self.list_characters(),
            "plot_threads": self.list_plot_threads(),
            "chapters": self._list_chapters(),
            "themes": self._list_themes(),
            "world_elements": self._list_world_elements(),
            "last_updated": datetime.now().isoformat()
        }
        
        return overview
    
    def create_story_templates(self) -> None:
        """Create template files for story elements."""
        templates = {
            "character_template.md": self._get_character_template(),
            "plot_thread_template.md": self._get_plot_thread_template(),
            "world_element_template.md": self._get_world_element_template(),
            "chapter_template.md": self._get_chapter_template(),
            "theme_template.md": self._get_theme_template()
        }
        
        templates_dir = self.story_base_path / "templates"
        
        for filename, content in templates.items():
            file_path = templates_dir / filename
            if not file_path.exists():
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Created template: {file_path}")
    
    def backup_story(self, backup_path: Optional[str] = None) -> Path:
        """Create backup of entire story directory."""
        import shutil
        import zipfile
        
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"story_backup_{timestamp}.zip"
        
        backup_path = Path(backup_path)
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.story_base_path):
                for file in files:
                    if file.endswith('.md'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.story_base_path)
                        zipf.write(file_path, arcname)
        
        logger.info(f"Created story backup: {backup_path}")
        return backup_path
    
    def _write_markdown_file(
        self, 
        file_path: Path, 
        metadata: Dict[str, Any], 
        content: str
    ) -> None:
        """Write markdown file with YAML frontmatter."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("---\n")
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
            f.write("---\n\n")
            f.write(content)
    
    def _read_markdown_file(self, file_path: Path) -> Tuple[Dict[str, Any], str]:
        """Read markdown file with YAML frontmatter."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if content.startswith('---\n'):
            try:
                _, frontmatter, markdown_content = content.split('---\n', 2)
                metadata = yaml.safe_load(frontmatter)
                return metadata, markdown_content.strip()
            except ValueError:
                # Invalid frontmatter format
                return {}, content
        else:
            # No frontmatter
            return {}, content
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize name for use as filename."""
        # Replace spaces with underscores and remove invalid characters
        sanitized = name.replace(' ', '_').lower()
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized
    
    def _unsanitize_filename(self, filename: str) -> str:
        """Convert filename back to readable name."""
        return filename.replace('_', ' ').title()
    
    def _list_chapters(self) -> List[int]:
        """List all chapter numbers."""
        chapters_dir = self.story_base_path / "chapters"
        chapter_files = list(chapters_dir.glob("chapter_*.md"))
        
        chapter_numbers = []
        for file in chapter_files:
            try:
                # Extract number from filename like "chapter_01.md"
                num_str = file.stem.split('_')[1]
                chapter_numbers.append(int(num_str))
            except (IndexError, ValueError):
                continue
        
        return sorted(chapter_numbers)
    
    def _list_themes(self) -> List[str]:
        """List all theme names."""
        themes_dir = self.story_base_path / "themes"
        theme_files = list(themes_dir.glob("*.md"))
        
        return [self._unsanitize_filename(f.stem) for f in theme_files]
    
    def _list_world_elements(self) -> Dict[str, List[str]]:
        """List all world building elements by type."""
        world_dir = self.story_base_path / "world"
        elements = {}
        
        for subdir in world_dir.iterdir():
            if subdir.is_dir():
                element_files = list(subdir.glob("*.md"))
                elements[subdir.name] = [
                    self._unsanitize_filename(f.stem) for f in element_files
                ]
        
        return elements
    
    def _get_character_template(self) -> str:
        """Get character profile template."""
        return '''---
# Character Metadata
character_name: ""
character_type: "protagonist"  # protagonist, antagonist, supporting, minor
importance: 0.5  # 0.0-1.0 scale
status: "active"  # active, resolved, deceased, absent

# Story Integration
chapter_introduced: 1
last_appearance: 1
pov_chapters: []
story_arc: "main"  # main, subplot, background

# Character Traits
personality_traits: []
physical_description:
  age: null
  height: ""
  hair: ""
  eyes: ""
  distinguishing_features: ""

# Relationships
relationships: {}

# Character Development
character_arc:
  starting_state: ""
  current_state: ""
  target_state: ""
  development_challenges: []

# Voice and Dialogue
speech_patterns: []
dialogue_examples: []

# Skills and Abilities
skills: []
limitations: []

# Narrative Function
purpose_in_story: ""
symbolic_representation: ""
reader_connection: ""

# Consistency Rules
voice_consistency_rules: []
trait_locked: false
last_updated: ""
---

# Character Name - Character Profile

## Background

[Character background and history]

## Personal History

[Personal history and formative events]

## Current Situation

[Current circumstances and role in story]

## Character Voice Notes

[Notes on how character speaks and acts]

## Development Arc

[Character growth and change throughout story]

## Relationships Dynamic

[Key relationships and how they evolve]

## Consistency Checkpoints

[Important consistency notes and rules]
'''
    
    def _get_plot_thread_template(self) -> str:
        """Get plot thread template."""
        return '''---
# Plot Thread Metadata
thread_name: ""
thread_type: "main"  # main, subplot, character_arc, world_building
priority: 1.0  # 0.0-1.0, higher = more important
status: "planned"  # planned, active, paused, resolved, abandoned

# Story Structure
chapters_planned: []
chapters_active: []
resolution_target_chapter: null

# Characters Involved
primary_characters: []
secondary_characters: []

# Conflict Structure
conflict_type: "external"  # internal, external, philosophical, supernatural
stakes: "personal"  # personal, local, regional, global
urgency: "medium"  # low, medium, high, critical

# Plot Progression
inciting_incident: ""
rising_action: []
climax: ""
falling_action: ""
resolution: ""

# Foreshadowing Elements
planted_clues: []

# Dependencies
prerequisite_threads: []
enables_threads: []
blocks_threads: []

# Tension Management
tension_curve: {}

# Consistency Rules
plot_rules: []
last_updated: ""
---

# Plot Thread Name - Plot Thread

## Thread Overview

[Overview of the plot thread and its significance]

## Plot Development Strategy

[How the plot will develop across chapters]

## Subplot Integration

[How this thread relates to other plot lines]

## Foreshadowing Strategy

[Planned clues and setup elements]

## Tension Pacing

[How tension builds and releases]

## Resolution Strategy

[How the thread will be resolved]

## Consistency Checkpoints

[Important plot consistency notes]
'''
    
    def _get_world_element_template(self) -> str:
        """Get world building element template."""
        return '''---
# World Building Metadata
element_name: ""
element_type: "location"  # location, culture, rule, history, technology
scope: "local"  # local, regional, global
importance: 0.5  # 0.0-1.0
consistency_level: "strict"  # strict, flexible, suggestion

# Story Integration
established_chapter: 1
relevant_chapters: []
character_associations: []

# Relationships
connected_elements: []

# Verification Requirements
fact_checking_required: false
research_sources: []
last_updated: ""
---

# World Element Name - World Building

## Element Overview

[Description of the world building element]

## Physical/Conceptual Description

[Detailed description of the element]

## Rules and Constraints

[Important rules that govern this element]

## Story Integration

[How this element fits into the narrative]

## Character Interactions

[How characters interact with this element]

## Consistency Rules

[Important consistency guidelines]
'''
    
    def _get_chapter_template(self) -> str:
        """Get chapter template."""
        return '''---
# Chapter Metadata
chapter_number: 1
title: ""
word_count: 0
target_word_count: 3000

# Narrative Structure
pov_character: ""
setting: ""
time_period: ""
season: ""

# Story Elements
plot_threads_advanced: []
plot_threads_introduced: []
plot_threads_resolved: []

character_development: {}
new_characters_introduced: []

# Scene Structure
scenes: []

# Emotional Arc
opening_mood: ""
climax_mood: ""
closing_mood: ""
emotional_beats: []

# Foreshadowing
clues_planted: []
setup_elements: []

# Dialogue and Voice
key_dialogue: []
voice_notes: []

# Technical Elements
research_verified: []

# Consistency Checks
character_consistency: []
plot_consistency: []

# Quality Metrics
pacing_score: 0.0  # 0.0-1.0
tension_management: 0.0
character_voice: 0.0
plot_advancement: 0.0

# Notes and Revisions
author_notes: ""
revision_history: []
last_updated: ""
---

# Chapter Title

[Chapter content goes here]

## Chapter Analysis

### Strengths
[What works well in this chapter]

### Areas for Enhancement
[What could be improved]

### Plot Function
[How this chapter serves the overall story]

### Character Development Notes
[Character growth and changes in this chapter]
'''
    
    def _get_theme_template(self) -> str:
        """Get theme development template."""
        return '''---
# Theme Metadata
theme_name: ""
theme_type: "major"  # major, minor, symbolic, motif
importance: 0.5  # 0.0-1.0

# Story Integration
first_introduction: 1
development_chapters: []
resolution_chapter: null
resolution_status: "developing"  # developing, resolved, abandoned

# Symbolic Elements
symbolic_objects: []
symbolic_locations: []
symbolic_actions: []

# Character Associations
primary_characters: []
character_relationships_to_theme: {}

# Thematic Development
exploration_methods: []
key_questions: []

# Literary Techniques
narrative_techniques: []
dialogue_patterns: []

# Thematic Arc
development_progression: {}

# Cultural Context
relevant_real_world_issues: []
historical_parallels: []

# Consistency Rules
thematic_guidelines: []
avoid_patterns: []
last_updated: ""
---

# Theme Name - Thematic Development

## Theme Overview

[Overview of the theme and its significance]

## Thematic Questions

[Key questions the theme explores]

## Character Embodiment

[How characters embody or explore this theme]

## Symbolic Development

[How symbols and motifs develop the theme]

## Narrative Techniques

[Literary techniques used to explore the theme]

## Cultural Resonance

[How the theme connects to broader cultural issues]

## Resolution Strategy

[How the theme will be resolved or developed]

## Consistency Guidelines

[Important thematic consistency notes]
'''