
import asyncio
import os
import shutil
from pathlib import Path
from novel_writer import NovelWriter
from rich.console import Console

console = Console()

async def run_verification():
    # 1. Setup Dummy Story
    story_id = "test_art_story"
    base_path = Path.cwd() / "stories" / story_id
    if base_path.exists():
        shutil.rmtree(base_path)
    
    base_path.mkdir(parents=True, exist_ok=True)
    (base_path / "chapters").mkdir(exist_ok=True)
    
    # Metadata
    import json
    metadata = {
        "title": "Test Art Story",
        "id": story_id,
        "chapters": [
            {
                "id": "chapter_01",
                "title": "The Visual Scene",
                "filename": "chapter_01_visual.md",
                "word_count": 50
            }
        ]
    }
    with open(base_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
        
    # Chapter Content
    original_content = """---
title: The Visual Scene
chapter: 1
---

# The Visual Scene

The sunset painted the sky in violent shades of orange and purple. Below, the cyberpunk city hummed with neon energy. Rain slicked the pavement, reflecting the holographic advertisements that towered above the lonely detective.
"""
    with open(base_path / "chapters" / "chapter_01_visual.md", "w") as f:
        f.write(original_content)
        
    console.print("[yellow]Created dummy story for art verification.[/yellow]")
    
    # 2. Run Art Director
    writer = NovelWriter()
    console.print("[cyan]Running generate_image_prompts...[/cyan]")
    
    try:
        results = await writer.generate_image_prompts(story_id, ["chapter_01"])
        
        console.print(f"[green]Art Director returned results: {len(results)}[/green]")
        console.print(f"Prompt: {results.get('chapter_01')}")
        
        # 3. Verify File Update
        with open(base_path / "chapters" / "chapter_01_visual.md", "r") as f:
            new_content = f.read()
            
        console.print("\n[bold]File Content Preview:[/bold]")
        console.print(new_content[:500])
        
        if "image_prompt:" in new_content:
             console.print("\n[green]✓ Image prompt found in frontmatter[/green]")
        else:
             console.print("\n[red]✗ Image prompt MISSING from file[/red]")

    except Exception as e:
        console.print(f"[red]Verification Failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_verification())
