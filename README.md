# Novel Memory Architecture

AI-powered novel writing system with hierarchical memory management and Azure OpenAI integration.

## Quick Start

```bash
# Create your first story
uv run python novel_writer.py create "My Story Title"

# Generate a complete novel autonomously  
uv run python novel_writer.py generate my_story_title "Epic fantasy adventure about a prince's journey"

# Generate with translation to French
uv run python novel_writer.py generate my_story_title "Epic fantasy adventure" --translate fr

# OR generate chapters individually
uv run python novel_writer.py chapter my_story_title "Chapter 1: The Beginning"

# Generate and translate a single chapter to Hindi
uv run python novel_writer.py chapter my_story_title "Chapter 2: The Journey" --translate hi

# Check progress
uv run python novel_writer.py list
uv run python novel_writer.py status my_story_title
```

## Commands

- `translate-story story_id target_language [--chapters "ch1,ch2"]` - **NEW**: Translate multiple chapters or entire story
- `list` - List all stories
- `status story_id` - Show story details

### Translation Feature

The translation feature now works in **two modes**:

#### 1. **Translation During Generation** (Post-Processing)
```bash
# Generate new content with translation
uv run python novel_writer.py chapter my_story "New Chapter" --translate hi
Automatic installer helper
-------------------------
If you don't have `prompt_toolkit` installed, run the included helper (from the repository root):

PowerShell:
```powershell
& .\.venv\Scripts\Activate.ps1
.\scripts\install_prompt_toolkit.ps1
```

Or run the Python helper:
```powershell
& .\.venv\Scripts\Activate.ps1
python scripts\install_prompt_toolkit.py
```
uv run python novel_writer.py generate my_story "Adventure tale" --translate fr
```

#### 2. **Translation of Existing Content** ⭐ **NEW**
```bash
# Translate a specific existing chapter
uv run python novel_writer.py translate the_mahabharata_saga chapter_01 hi
uv run python novel_writer.py translate the_mahabharata_saga chapter_01 French

# Translate entire story
uv run python novel_writer.py translate-story the_mahabharata_saga hi

# Translate specific chapters only  
uv run python novel_writer.py translate-story the_mahabharata_saga ja --chapters "chapter_01,chapter_02,chapter_03"
```

**Supported formats:**
- ISO codes: `fr`, `es`, `de`, `ja`, `zh`, `hi`, `ar`
- Full names: `French`, `Spanish`, `German`, `Japanese`, `Chinese`, `Hindi`, `Arabic`

**Examples for Existing Mahabharata Chapters:**
```bash
# Translate Chapter 1 to Hindi
uv run python novel_writer.py translate the_mahabharata_saga chapter_01 hi

# Translate all chapters to French  
uv run python novel_writer.py translate-story the_mahabharata_saga French

# Translate just the first 3 chapters to Japanese
uv run python novel_writer.py translate-story the_mahabharata_saga ja --chapters "chapter_01,chapter_02,chapter_03"
```

**Important Notes:**
- Translated files are saved with language suffix (e.g., `chapter_01_the_divine_birth_hi.md`)
- Original files are never modified
- Translation preserves formatting, dialogue, and narrative structure
- Story metadata tracks all translations

## Configuration

Set up your `.env` file with Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1
EMBEDDING_MODEL=text-embedding-3-large
```

## Features

- **Smart Memory**: Each story maintains its own context and continuity
- **Character Tracking**: Automatically tracks characters across chapters
- **Story Organization**: Clean directory structure for each novel
- **Quality Generation**: High-quality narrative with consistent tone

## Project Structure

```
stories/
└── your_story/
    ├── chapters/        # Generated chapters
    ├── metadata.json    # Story tracking
    ├── outline.md       # Story planning
    └── memory_data/     # AI memory system
```

## License

MIT