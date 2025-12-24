#!/usr/bin/env python3
"""
Novel Writer CLI - Simplified interface for generating novels
"""

import os
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

console = Console()

class NovelWriter:
    """Simple novel writing interface"""
    
    def __init__(self):
        self.base_path = Path.cwd()
        self.stories_path = self.base_path / "stories"
        self.stories_path.mkdir(exist_ok=True)
        self.current_story = None
        
    def list_stories(self) -> list:
        """List all existing stories"""
        if not self.stories_path.exists():
            return []
        
        stories = []
        for story_dir in self.stories_path.iterdir():
            if story_dir.is_dir():
                metadata_path = story_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        stories.append({
                            'id': story_dir.name,
                            'path': str(story_dir),
                            'metadata': metadata
                        })
                    except:
                        continue
        return stories
    
    # `create_story` removed — story creation is no longer available via CLI
    
    async def generate_chapter(self, story_id: str, chapter_title: str, chapter_prompt: str = None, translate_to: str = None) -> str:
        """Generate a chapter for the story using the full architecture"""
        
        story_path = self.stories_path / story_id
        if not story_path.exists():
            raise ValueError(f"Story '{story_id}' not found")
        
        # Load story metadata
        metadata_path = story_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            story_metadata = json.load(f)
        
        console.print(f"[bold cyan]Generating chapter: {chapter_title}[/bold cyan]")
        
        try:
            # Import the enhanced memory system
            from novel_memory.memory.enhanced_manager import EnhancedMemoryManager
            from novel_memory.models.abstraction_layer import ModelManager, ModelCapability, ModelConfig, ModelProvider, AzureOpenAIModel
            
            # Initialize systems
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                task = progress.add_task("Initializing AI systems...", total=None)
                
                # Setup model manager
                model_manager = ModelManager()
                model_config = ModelConfig(
                    provider=ModelProvider.AZURE_OPENAI,
                    model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    max_tokens=8192,
                    temperature=0.7,
                    capabilities=[ModelCapability.TEXT_GENERATION],
                    cost_per_token=0.00003,
                    rate_limit_rpm=60,
                    fallback_model=None
                )
                azure_model = AzureOpenAIModel(model_config)
                model_manager.router.register_model("azure-gpt", azure_model, model_config)
                
                progress.update(task, description="Initializing memory systems...")
                
                # Setup memory manager with story-specific path
                story_data_path = story_path / "memory_data"
                story_data_path.mkdir(exist_ok=True)
                
                memory_manager = EnhancedMemoryManager(
                    base_path=story_data_path,
                    model_manager=model_manager
                )
                
                progress.update(task, description="Starting chapter generation...")
                
                # Generate chapter number
                chapter_num = len(story_metadata.get("chapters", [])) + 1
                chapter_id = f"chapter_{chapter_num:02d}"
                
                # Start generation session
                session_id = await memory_manager.start_chapter_generation(
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    expected_length=2000
                )
                
                progress.update(task, description="Gathering story context...")
                
                # Get context from previous chapters if any
                context_query = f"{story_metadata['title']} {chapter_title}"
                if story_metadata.get("notes"):
                    context_query += f" {story_metadata['notes']}"
                    
                context = memory_manager.hierarchical_memory.get_context_for_generation(
                    query=context_query,
                    max_tokens=6000
                )
                
                progress.update(task, description="Generating chapter content...")
                
                # Build generation prompt
                generation_prompt = self._build_chapter_prompt(
                    story_metadata, chapter_title, chapter_num, chapter_prompt, context
                )
                
                # Generate the chapter
                response = await model_manager.generate_text(
                    prompt=generation_prompt,
                    capability=ModelCapability.TEXT_GENERATION,
                    max_tokens=3000,
                    temperature=0.8,
                    context={"chapter_id": chapter_id, "story_id": story_id}
                )
                
                if not response.success:
                    raise Exception(f"Generation failed: {response.error_message}")
                
                progress.update(task, description="Processing and saving chapter...")
                
                # Process the generated content
                await memory_manager.process_generated_content(
                    content=response.generated_text,
                    content_metadata={
                        "chapter_id": chapter_id,
                        "title": chapter_title,
                        "story_id": story_id
                    }
                )
                
                # Save chapter file (sanitize title for filename)
                safe_title = re.sub(r"[^a-z0-9_\-]+", "_", chapter_title.lower())
                safe_title = safe_title.strip("_") or "chapter"
                chapter_filename = f"{chapter_id}_{safe_title}.md"
                chapter_path = story_path / "chapters" / chapter_filename
                
                # Prepare chapter content
                generated_content = response.generated_text
                chapter_word_count = len(generated_content.split())
                
                # Post-generation translation if requested
                if translate_to:
                    progress.update(task, description=f"Translating chapter to {translate_to}...")
                    
                    try:
                        from novel_memory.translation import StoryTranslator, TranslationRequest
                        
                        translator = StoryTranslator(model_manager)
                        translation_request = TranslationRequest(
                            content=generated_content,
                            target_language=translate_to,
                            preserve_formatting=True,
                            context={"chapter_id": chapter_id, "story_id": story_id}
                        )
                        
                        translation_response = await translator.translate_story(translation_request)
                        
                        if translation_response.success:
                            generated_content = translation_response.translated_content
                            console.print(f"[green]✓ Chapter translated to {translation_response.target_language}[/green]")
                            
                            # Update filename to indicate translation
                            base_filename = chapter_filename.replace('.md', '')
                            chapter_filename = f"{base_filename}_{translate_to.lower()}.md"
                            chapter_path = story_path / "chapters" / chapter_filename
                        else:
                            console.print(f"[yellow]⚠ Translation failed: {translation_response.error_message}[/yellow]")
                            console.print(f"[yellow]Continuing with original content[/yellow]")
                    
                    except Exception as trans_error:
                        console.print(f"[yellow]⚠ Translation error: {str(trans_error)}[/yellow]")
                        console.print(f"[yellow]Continuing with original content[/yellow]")
                
                # Create chapter content with proper YAML frontmatter
                yaml_fields = [
                    f"title: {chapter_title}",
                    f"chapter: {chapter_num}",
                    f"story: {story_metadata['title']}",
                    f"generated: {datetime.now().isoformat()}",
                    f"word_count: {chapter_word_count}"
                ]
                
                if translate_to:
                    yaml_fields.append(f"translated_to: {translate_to}")
                
                chapter_content = f"""---
{chr(10).join(yaml_fields)}
---

# Chapter {chapter_num}: {chapter_title}

{generated_content}
"""
                
                with open(chapter_path, 'w', encoding='utf-8') as f:
                    f.write(chapter_content)
                
                # Update story metadata
                story_metadata["chapters"].append({
                    "id": chapter_id,
                    "title": chapter_title,
                    "filename": chapter_filename,
                    "word_count": chapter_word_count,
                    "created": datetime.now().isoformat(),
                    **({"translated_to": translate_to} if translate_to else {})
                })
                story_metadata["word_count"] += chapter_word_count
                story_metadata["last_modified"] = datetime.now().isoformat()
                
                with open(metadata_path, 'w') as f:
                    json.dump(story_metadata, f, indent=2)
                
                # End generation session
                await memory_manager.end_chapter_generation(
                    final_word_count=chapter_word_count,
                    model_calls=1,
                    total_tokens=response.tokens_used
                )
                
                progress.update(task, description="Chapter generation complete!")
            
            console.print(f"[green]✓ Generated chapter '{chapter_title}' ({chapter_word_count} words)[/green]")
            if translate_to:
                console.print(f"[blue]✓ Translated to {translate_to}[/blue]")
            console.print(f"[blue]Saved to: {chapter_path}[/blue]")
            
            return str(chapter_path)
            
        except Exception as e:
            console.print(f"[red]✗ Error generating chapter: {str(e)}[/red]")
            raise
    
    async def generate_complete_novel(self, story_id: str, novel_concept: str, num_chapters: int = 12, translate_to: str = None) -> str:
        """Generate a complete novel autonomously based on a concept"""
        
        story_path = self.stories_path / story_id
        if not story_path.exists():
            raise ValueError(f"Story '{story_id}' not found")
        
        # Load story metadata
        metadata_path = story_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            story_metadata = json.load(f)
        
        console.print(f"[bold cyan]Generating complete novel: {story_metadata['title']}[/bold cyan]")
        console.print(f"[yellow]Novel concept: {novel_concept}[/yellow]")
        console.print(f"[blue]Target chapters: {num_chapters}[/blue]")
        if translate_to:
            console.print(f"[magenta]Translation target: {translate_to}[/magenta]")
        
        # Create chapter outline first
        chapter_outline = self._generate_novel_outline(novel_concept, num_chapters)
        
        console.print(f"[green]Generated novel outline with {len(chapter_outline)} chapters[/green]")
        
        # Generate each chapter
        for i, chapter_info in enumerate(chapter_outline, 1):
            chapter_title = chapter_info['title']
            chapter_prompt = chapter_info['prompt']
            
            console.print(f"\n[bold cyan]Generating Chapter {i}: {chapter_title}[/bold cyan]")
            
            try:
                chapter_path = await self.generate_chapter(story_id, chapter_title, chapter_prompt, translate_to)
                console.print(f"[green]✓ Chapter {i} completed[/green]")
            except Exception as e:
                console.print(f"[red]✗ Error in Chapter {i}: {str(e)}[/red]")
                raise
        
        console.print(f"\n[bold green]✓ Complete novel generated successfully![/bold green]")
        console.print(f"[blue]Total chapters: {num_chapters}[/blue]")
        if translate_to:
            console.print(f"[magenta]✓ All chapters translated to {translate_to}[/magenta]")
        
        return str(story_path)
        
    async def translate_existing_content(self, story_id: str, chapter_id: str, target_language: str) -> str:
        """Translate existing chapter content to target language"""
        
        story_path = self.stories_path / story_id
        if not story_path.exists():
            raise ValueError(f"Story '{story_id}' not found")
            
        # Load story metadata
        metadata_path = story_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            story_metadata = json.load(f)
            
        # Find the chapter
        chapter_info = None
        for chapter in story_metadata.get('chapters', []):
            if chapter['id'] == chapter_id:
                chapter_info = chapter
                break
                
        if not chapter_info:
            raise ValueError(f"Chapter '{chapter_id}' not found in story '{story_id}'")
            
        # Read the existing chapter file
        chapter_filename = chapter_info['filename']
        chapter_path = story_path / "chapters" / chapter_filename
        
        if not chapter_path.exists():
            raise ValueError(f"Chapter file not found: {chapter_path}")
            
        with open(chapter_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
            
        console.print(f"[cyan]Translating existing chapter: {chapter_info['title']}[/cyan]")
        console.print(f"[yellow]Target language: {target_language}[/yellow]")
        
        try:
            # Import the enhanced memory system for model access
            from novel_memory.memory.enhanced_manager import EnhancedMemoryManager
            from novel_memory.models.abstraction_layer import ModelManager, ModelCapability, ModelConfig, ModelProvider, AzureOpenAIModel
            
            # Initialize systems
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                task = progress.add_task("Initializing translation systems...", total=None)
                
                # Setup model manager
                model_manager = ModelManager()
                model_config = ModelConfig(
                    provider=ModelProvider.AZURE_OPENAI,
                    model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    max_tokens=8192,
                    temperature=0.3,
                    capabilities=[ModelCapability.TEXT_GENERATION],
                    cost_per_token=0.00003,
                    rate_limit_rpm=60,
                    fallback_model=None
                )
                azure_model = AzureOpenAIModel(model_config)
                model_manager.router.register_model("azure-gpt", azure_model, model_config)
                
                progress.update(task, description=f"Translating to {target_language}...")
                
                # Import and use translator
                from novel_memory.translation import StoryTranslator, TranslationRequest
                
                translator = StoryTranslator(model_manager)
                translation_request = TranslationRequest(
                    content=original_content,
                    target_language=target_language,
                    preserve_formatting=True,
                    context={"chapter_id": chapter_id, "story_id": story_id}
                )
                
                translation_response = await translator.translate_story(translation_request)
                
                if not translation_response.success:
                    raise Exception(f"Translation failed: {translation_response.error_message}")
                    
                progress.update(task, description="Saving translated content...")
                
                # Create translated filename
                base_filename = chapter_filename.replace('.md', '')
                translated_filename = f"{base_filename}_{target_language.lower()}.md"
                translated_path = story_path / "chapters" / translated_filename
                
                # Save translated content
                with open(translated_path, 'w', encoding='utf-8') as f:
                    f.write(translation_response.translated_content)
                    
                # Update story metadata to track translation
                translation_info = {
                    "original_chapter_id": chapter_id,
                    "translated_filename": translated_filename,
                    "target_language": target_language,
                    "translation_date": datetime.now().isoformat(),
                    "tokens_used": translation_response.tokens_used
                }
                
                if "translations" not in story_metadata:
                    story_metadata["translations"] = []
                    
                story_metadata["translations"].append(translation_info)
                story_metadata["last_modified"] = datetime.now().isoformat()
                
                with open(metadata_path, 'w') as f:
                    json.dump(story_metadata, f, indent=2)
                    
                progress.update(task, description="Translation complete!")
                
            console.print(f"[green]✓ Translation completed successfully![/green]")
            console.print(f"[blue]Original: {chapter_path}[/blue]")
            console.print(f"[magenta]Translated: {translated_path}[/magenta]")
            console.print(f"[yellow]Language: {translation_response.target_language}[/yellow]")
            console.print(f"[cyan]Tokens used: {translation_response.tokens_used}[/cyan]")
            
            return str(translated_path)
            
        except Exception as e:
            console.print(f"[red]✗ Error translating chapter: {str(e)}[/red]")
            raise
        
    def _generate_novel_outline(self, concept: str, num_chapters: int) -> list:
        """Generate chapter outline for the novel"""
        
        # Pre-defined story structures based on concept analysis
        if "ramayana" in concept.lower() or "epic" in concept.lower():
            return self._create_epic_outline(concept, num_chapters)
        elif "adventure" in concept.lower():
            return self._create_adventure_outline(concept, num_chapters)
        elif "romance" in concept.lower():
            return self._create_romance_outline(concept, num_chapters)
        else:
            return self._create_general_outline(concept, num_chapters)
            
    def _create_epic_outline(self, concept: str, num_chapters: int) -> list:
        """Create outline for epic stories"""
        outline = []
        
        # Epic structure: Setup -> Call to Adventure -> Trials -> Crisis -> Resolution
        chapters_per_act = num_chapters // 5
        
        # Act 1: Setup (2 chapters)
        outline.append({"title": "The Divine Birth", "prompt": "Introduce the hero's origins and divine nature. Set the peaceful kingdom."})
        outline.append({"title": "The Prince's Virtue", "prompt": "Show the hero's noble character through trials or demonstrations of virtue."})
        
        # Act 2: Call to Adventure (2 chapters) 
        outline.append({"title": "The Sacred Test", "prompt": "Introduce the love interest and a test that proves worthiness."})
        outline.append({"title": "Destiny Fulfilled", "prompt": "The hero succeeds in the test and wins love, but fate intervenes."})
        
        # Act 3: Trials and Exile (4 chapters)
        outline.append({"title": "The Path of Exile", "prompt": "The hero is forced into exile or a difficult journey."})
        outline.append({"title": "The First Trial", "prompt": "Early challenges that test the hero's resolve and introduce enemies."})
        outline.append({"title": "The Betrayal", "prompt": "A major betrayal or loss that raises the stakes."})
        outline.append({"title": "The Abduction", "prompt": "The love interest is taken, creating the main quest."})
        
        # Act 4: The Quest (3 chapters)
        outline.append({"title": "Allies Gathered", "prompt": "The hero finds unexpected allies and begins the rescue."})
        outline.append({"title": "The Great Leap", "prompt": "A pivotal moment showing heroic devotion and courage."})
        outline.append({"title": "The Final Battle", "prompt": "The climactic confrontation with the main antagonist."})
        
        # Act 5: Resolution (1 chapter)
        outline.append({"title": "Return to Light", "prompt": "Victory, reunion, and return to peace. The hero's transformation complete."})
        
        return outline[:num_chapters]  # Trim to requested number
        
    def _create_adventure_outline(self, concept: str, num_chapters: int) -> list:
        """Create outline for adventure stories"""
        # Adventure structure: Ordinary World -> Call -> Departure -> Trials -> Return
        outline = []
        for i in range(num_chapters):
            outline.append({
                "title": f"Chapter {i+1}: Adventure Continues",
                "prompt": f"Continue the adventure story, chapter {i+1} of {num_chapters}. Advance the plot with action and discovery."
            })
        return outline
        
    def _create_romance_outline(self, concept: str, num_chapters: int) -> list:
        """Create outline for romance stories"""
        # Romance structure: Meet -> Conflict -> Separation -> Reunion
        outline = []
        for i in range(num_chapters):
            outline.append({
                "title": f"Chapter {i+1}: Hearts Entwined",
                "prompt": f"Continue the romance story, chapter {i+1} of {num_chapters}. Focus on character relationships."
            })
        return outline
        
    def _create_general_outline(self, concept: str, num_chapters: int) -> list:
        """Create general story outline"""
        outline = []
        for i in range(num_chapters):
            outline.append({
                "title": f"Chapter {i+1}: The Journey",
                "prompt": f"Continue the story based on: {concept}. This is chapter {i+1} of {num_chapters}."
            })
        return outline

    def _build_chapter_prompt(self, story_metadata: Dict, chapter_title: str, 
                             chapter_num: int, user_prompt: str, context: Any) -> str:
        """Build the chapter generation prompt"""
        
        prompt = f"""You are writing Chapter {chapter_num} of the novel "{story_metadata['title']}"

STORY CONTEXT:
- Title: {story_metadata['title']}
- Genre: {story_metadata.get('genre', 'literary fiction')}
- Previous chapters: {len(story_metadata.get('chapters', []))}
- Current word count: {story_metadata.get('word_count', 0)}

CHAPTER DETAILS:
- Chapter {chapter_num}: {chapter_title}
"""

        if user_prompt:
            prompt += f"""
CHAPTER DIRECTION:
{user_prompt}
"""

        if story_metadata.get('notes'):
            prompt += f"""
STORY NOTES:
{story_metadata['notes']}
"""

        prompt += f"""
CRITICAL WORD COUNT CONSTRAINT:
- Write EXACTLY one complete chapter
- Target word count: 1500-2000 words (STRICT - do not exceed 2000 words)
- Count your words carefully to stay within this limit

INSTRUCTIONS:
Write a compelling chapter that:
1. Advances the overall story narrative
2. Maintains consistency with previous chapters
3. Has strong character development and dialogue  
4. Ends with a natural transition or hook for the next chapter
5. MOST IMPORTANT: Stay within the 1500-2000 word limit

Begin writing the chapter now:
"""
        
        return prompt

@click.group()
def cli():
    """Novel Writer - Generate novels with AI-powered narrative memory"""
    pass

@cli.command()
def list():
    """List all existing stories"""
    writer = NovelWriter()
    stories = writer.list_stories()
    
    if not stories:
        console.print("[yellow]No stories found.[/yellow]")
        return
    
    table = Table(title="Your Stories")
    table.add_column("Title", style="cyan")
    table.add_column("ID", style="green")
    table.add_column("Genre", style="magenta")
    table.add_column("Chapters", justify="right", style="blue")
    table.add_column("Words", justify="right", style="yellow")
    table.add_column("Status", style="red")
    
    for story in stories:
        meta = story['metadata']
        table.add_row(
            meta.get('title', 'Unknown'),
            story['id'],
            meta.get('genre', 'Unknown'),
            str(len(meta.get('chapters', []))),
            str(meta.get('word_count', 0)),
            meta.get('status', 'Unknown')
        )
    
    console.print(table)

@cli.command()
@click.argument('story_id')
@click.argument('concept')
@click.option('--chapters', default=12, help='Number of chapters to generate')
@click.option('--translate', help='Translate the generated novel to specified language (e.g., fr, hi, ja)')
def generate(story_id: str, concept: str, chapters: int, translate: str = None):
    """Generate a complete novel autonomously from a concept"""
    writer = NovelWriter()
    
    console.print(Panel.fit(f"[bold blue]Generating Complete Novel: {concept}"))
    
    try:
        novel_path = asyncio.run(writer.generate_complete_novel(story_id, concept, chapters, translate))
        console.print(f"[green]✓ Complete novel generated successfully![/green]")
        console.print(f"[blue]Novel saved to: {novel_path}[/blue]")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to generate novel: {str(e)}[/red]")


@cli.command()
@click.argument('story_id')
@click.argument('target_language')
@click.option('--chapters', help='Comma-separated list of chapter IDs to translate (default: all)')
def translate_story(story_id: str, target_language: str, chapters: str = None):
    """Translate multiple chapters or entire story to the specified language"""
    writer = NovelWriter()
    
    console.print(Panel.fit(f"[bold blue]Translating Story: {story_id} to {target_language}"))
    
    try:
        # Load story metadata to get chapter list
        story_path = writer.stories_path / story_id
        metadata_path = story_path / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            story_metadata = json.load(f)
        
        # Determine which chapters to translate
        if chapters:
            chapter_ids = [ch.strip() for ch in chapters.split(',')]
        else:
            chapter_ids = [ch['id'] for ch in story_metadata.get('chapters', [])]
        
        console.print(f"[yellow]Translating {len(chapter_ids)} chapters...[/yellow]")
        
        successful_translations = 0
        failed_translations = 0
        
        for i, chapter_id in enumerate(chapter_ids, 1):
            console.print(f"\n[cyan]Translating {i}/{len(chapter_ids)}: {chapter_id}[/cyan]")
            
            try:
                translated_path = asyncio.run(writer.translate_existing_content(story_id, chapter_id, target_language))
                console.print(f"[green]✓ {chapter_id} translated successfully[/green]")
                successful_translations += 1
            except Exception as e:
                console.print(f"[red]✗ Failed to translate {chapter_id}: {str(e)}[/red]")
                failed_translations += 1
                
        console.print(f"\n[bold green]Translation Summary:[/bold green]")
        console.print(f"[green]✓ Successful: {successful_translations}[/green]")
        if failed_translations > 0:
            console.print(f"[red]✗ Failed: {failed_translations}[/red]")
            
    except Exception as e:
        console.print(f"[red]✗ Failed to translate story: {str(e)}[/red]")

if __name__ == "__main__":
    cli()