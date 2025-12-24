#!/usr/bin/env python3
"""
Consumer-focused interactive CLI for creative flow.

Navigation: arrow keys + Enter only.

This is a thin UI wrapper around the existing `NovelWriter` implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path
import json
import time

# Ensure local package imports work without editable install
sys.path.insert(0, str(Path.cwd() / "src"))

try:
    from prompt_toolkit.shortcuts import radiolist_dialog, input_dialog, message_dialog
except Exception:
    print("This interactive CLI requires 'prompt_toolkit'. Install with:\n  pip install prompt_toolkit\n")
    raise

from rich.console import Console

from novel_writer import NovelWriter

console = Console()


def slugify(text: str) -> str:
    s = text.lower().strip()
    s = "_".join(s.split())
    # safe fallback + timestamp
    return f"{s[:40]}_{int(time.time())}"


def main():
    while True:
        choice = radiolist_dialog(
            title="What would you like to do?",
            text="",
            values=[
                ("generate", "Generate a new novel"),
                ("edit", "Edit/Rewrite existing chapter"),
                ("translate", "Translate a story"),
                ("exit", "Exit"),
            ],
        ).run()

        if choice is None or choice == "exit":
            console.print("Goodbye — keep writing!")
            return

        if choice == "generate":
            concept = input_dialog(title="Generate a New Novel", text="Enter a short concept or seed idea:").run()
            if not concept:
                message_dialog(title="Cancelled", text="No concept entered. Returning to menu.").run()
                continue

            # Prompt for number of chapters
            chapters_str = input_dialog(
                title="Chapter Count", 
                text="How many chapters? (Default: 12)",
                default="12"
            ).run()
            
            try:
                num_chapters = int(chapters_str) if chapters_str else 12
            except ValueError:
                num_chapters = 12

            # Prompt for chapter size
            size_choice = radiolist_dialog(
                title="Chapter Size",
                text="Select target length per chapter:",
                values=[
                    ("small", "Small (800-1000 words)"),
                    ("medium", "Medium (1500-2000 words)"),
                    ("large", "Large (2500-3500 words)"),
                    ("super_large", "Super Large (4000-5000 words)")
                ],
                default="medium"
            ).run()
            
            chapter_size = size_choice if size_choice else "medium"

            # Create story metadata automatically
            story_id = slugify(concept)
            story_dir = Path.cwd() / "stories" / story_id
            story_dir.mkdir(parents=True, exist_ok=True)
            # ensure chapters directory exists so generation can write files
            (story_dir / "chapters").mkdir(parents=True, exist_ok=True)
            metadata_path = story_dir / "metadata.json"

            metadata = {
                "title": concept,
                "story_id": story_id,
                "genre": "fiction",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "last_modified": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "chapters": [],
                "characters": {},
                "word_count": 0,
                "status": "active",
                "notes": "",
                "translations": []
            }

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            console.print(f"[bold green]Starting generation — your story will be saved automatically.[/bold green]")
            console.print("Writing chapters. Progress messages will appear below — please wait.")

            writer = NovelWriter()
            try:
                # Safe defaults: 12 chapters, no translation
                import asyncio
                asyncio.run(writer.generate_complete_novel(story_id, concept, num_chapters=num_chapters, translate_to=None, chapter_size=chapter_size))
                message_dialog(title="Done", text="Novel generation finished and saved.").run()
            except Exception as e:
                message_dialog(title="Error", text=f"Generation failed. Try again later. ({e})").run()

        elif choice == "edit":
            writer = NovelWriter()
            stories = writer.list_stories()
            values = [ (s['id'], f"{s['metadata'].get('title','Untitled')} ({len(s['metadata'].get('chapters',[]))} chapters)") for s in stories ]
            if not values:
                message_dialog(title="No stories", text="No stories found. Generate one first.").run()
                continue

            selected_id = radiolist_dialog(title="Select a story to edit", values=values).run()
            if not selected_id:
                continue

            # Load chapters for selection
            meta = next(s['metadata'] for s in stories if s['id'] == selected_id)
            chapters_list = meta.get('chapters', [])
            if not chapters_list:
                message_dialog(title="No chapters", text="No chapters found to edit.").run()
                continue
                
            ch_values = [(ch['id'], f"{i+1}. {ch['title']}") for i, ch in enumerate(chapters_list)]
            selected_chapter = radiolist_dialog(title="Select Chapter to Rewrite", values=ch_values).run()
            
            if not selected_chapter:
                continue
                
            # Get Rewrite Instructions
            instruction = input_dialog(
                title="Rewrite Instructions", 
                text=f"How should we change this chapter?\n(e.g., 'Make dialogue snappier', 'Change the ending to...')",
            ).run()
            
            if not instruction:
                continue

            console.print(f"[bold green]Rewriting chapter '{selected_chapter}'...[/bold green]")
            
            try:
                import asyncio
                asyncio.run(writer.rewrite_chapter(selected_id, selected_chapter, instruction))
                message_dialog(title="Success", text="Chapter rewritten successfully.").run()
            except Exception as e:
                message_dialog(title="Error", text=f"Failed to rewrite chapter: {e}").run()

        elif choice == "translate":
            writer = NovelWriter()
            stories = writer.list_stories()
            values = [ (s['id'], f"{s['metadata'].get('title','Untitled')} ({len(s['metadata'].get('chapters',[]))} chapters)") for s in stories ]
            if not values:
                message_dialog(title="No stories", text="No stories available to translate. Generate one first.").run()
                continue

            selected = radiolist_dialog(title="Select a story to translate", values=values).run()
            if not selected:
                continue

            scope = radiolist_dialog(title="Translation Scope", values=[('entire','Entire story'), ('select','Select chapters')]).run()
            chapter_ids = None
            if scope == 'select':
                # Let the user pick chapters interactively rather than typing numbers
                meta = next(s['metadata'] for s in stories if s['id'] == selected)
                chapters_list = meta.get('chapters', [])
                if not chapters_list:
                    message_dialog(title="No chapters", text="No chapters found for this story.").run()
                    continue

                chapter_ids = []
                # build selectable values (chapter_id, display)
                base_values = [(ch['id'], f"{i+1}. {ch['title']}") for i, ch in enumerate(chapters_list)]
                # loop to allow multiple selections
                while True:
                    values = [('__done__', 'Done selecting')] + base_values
                    pick = radiolist_dialog(title="Select a chapter (choose 'Done selecting' when finished)", values=values).run()
                    if not pick or pick == '__done__':
                        break
                    if pick not in chapter_ids:
                        chapter_ids.append(pick)

            language = radiolist_dialog(title="Target Language", values=[
                ('Hindi','Hindi'), ('French','French'), ('Spanish','Spanish'), ('Japanese','Japanese'), ('Chinese','Chinese')
            ]).run()

            if not language:
                continue

            console.print(f"Translating — this will run after generation completes.\nTarget: {language}")

            # Perform translation using existing writer methods
            try:
                meta = next(s['metadata'] for s in stories if s['id'] == selected)
                import asyncio
                if chapter_ids is None:
                    for ch in meta.get('chapters', []):
                        try:
                            asyncio.run(writer.translate_existing_content(selected, ch['id'], language))
                        except Exception as te:
                            console.print(f"[red]Translation failed for {ch['id']}: {te}[/red]")
                else:
                    for ch_id in chapter_ids:
                        try:
                            asyncio.run(writer.translate_existing_content(selected, ch_id, language))
                        except Exception as te:
                            console.print(f"[red]Translation failed for {ch_id}: {te}[/red]")

                message_dialog(title="Done", text="Translation finished and saved.").run()
            except Exception as e:
                message_dialog(title="Error", text=f"Translation failed. Try again later. ({e})").run()


if __name__ == '__main__':
    main()
