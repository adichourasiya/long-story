
import asyncio
import os
import shutil
from pathlib import Path
from novel_writer import NovelWriter
from rich.console import Console

console = Console()

async def run_verification():
    # 1. Setup Dummy Story
    story_id = "test_proofread_story"
    base_path = Path.cwd() / "stories" / story_id
    if base_path.exists():
        shutil.rmtree(base_path)
    
    base_path.mkdir(parents=True, exist_ok=True)
    (base_path / "chapters").mkdir(exist_ok=True)
    
    # Metadata
    import json
    metadata = {
        "title": "Test Story",
        "id": story_id,
        "chapters": [
            {
                "id": "chapter_01",
                "title": "The Badly Written Start",
                "filename": "chapter_01_the_badly_written_start.md",
                "word_count": 50
            }
        ]
    }
    with open(base_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
        
    # Chapter Content
    original_content = """---
title: The Badly Written Start
chapter: 1
---

# The Badly Written Start

Once upon a time their was a hero named Bob. Bob was cool and he liked to fight dragons but he was lazzy. One day a dragon come and burninate the village. Bob said "oh no" and went back to sleep. The End?
"""
    with open(base_path / "chapters" / "chapter_01_the_badly_written_start.md", "w") as f:
        f.write(original_content)
        
    console.print("[yellow]Created dummy story for verification.[/yellow]")
    
    # 2. Run Proofreader
    writer = NovelWriter()
    console.print("[cyan]Running proofread_chapter...[/cyan]")
    
    try:
        new_path = await writer.proofread_chapter(story_id, "chapter_01")
        
        console.print(f"[green]Proofreading call returned: {new_path}[/green]")
        
        # 3. Verify Results
        with open(new_path, 'r') as f:
            new_content = f.read()
            
        console.print("\n[bold]New Content Preview:[/bold]")
        console.print(new_content[:500] + "...")
        
        if "image_prompt" in new_content:
             console.print("\n[green]✓ Image prompt found in frontmatter[/green]")
        else:
             console.print("\n[red]✗ Image prompt MISSING[/red]")
             
        # Check filename change
        if "badly_written" not in new_path and "chapter_01" in new_path:
             console.print("[green]✓ Filename updated based on new title (presumably)[/green]")
        else:
             console.print(f"[yellow]Filename check: {new_path} (might be same if title didn't change enough)[/yellow]")

    except Exception as e:
        console.print(f"[red]Verification Failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_verification())
