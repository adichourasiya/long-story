# Novel Memory Architecture

AI-powered novel writing system with hierarchical memory management and Azure OpenAI integration.

## Features

- **Smart Memory**: Each story maintains its own context and continuity
- **Character Tracking**: Automatically tracks characters across chapters
- **Story Organization**: Clean directory structure for each novel
- **Quality Generation**: High-quality narrative with consistent tone
- **Translation Support**: Translate chapters or entire stories to multiple languages
- **Cross-Platform**: Works on macOS, Linux, and Windows

## Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd long-story
   ```

2. **Set up environment**
   
   Create a `.env` file with your Azure OpenAI credentials:
   ```env
   AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1
   EMBEDDING_MODEL=text-embedding-3-large
   ```

3. **Install dependencies**
   
   Using `uv` (recommended):
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .\.venv\Scripts\Activate.ps1  # On Windows
   
   pip install -e .
   ```

## Usage

There are two ways to use this system:

### 1. Interactive CLI (Recommended for Beginners)

Launch the interactive menu-driven interface:

**macOS/Linux:**
```bash
./scripts/launch.sh
```

**Windows:**
```powershell
.\scripts\launch.ps1
```

-   **Interactive Mode**: Run `./scripts/launch.sh` (or `launch.ps1` on Windows) to start the interactive menu.
    -   **Generate**: Create a new novel.
    -   **Edit**: Rewrite existing chapters based on new instructions.
    -   **Translate**: Translate stories or chapters.

### 2. Command-Line Interface (Advanced)

Use the command-line interface for more control:

-   **CLI Commands**:
    -   **Generate**: `uv run python novel_writer.py generate <story_id> <concept> --chapters 5`
    -   **Edit**: `uv run python novel_writer.py edit <story_id> <chapter_id> <instruction>`
    -   **Translate**: `uv run python novel_writer.py translate_story <story_id> <lang>`

#### List Stories
```bash
uv run python novel_writer.py list
```

#### Edit / Rewrite Chapter

Rewrite an existing chapter with new instructions:

```bash
uv run python novel_writer.py edit <story_id> <chapter_id> "Make the dialogue funnier and add a cliffhanger"
```

#### Generate a Complete Novel
```bash
# Basic usage
uv run python novel_writer.py generate my_story_title "Epic fantasy adventure about a prince's journey"

# With custom chapter count
uv run python novel_writer.py generate my_story_title "Epic fantasy adventure" --chapters 15

# With translation to French
uv run python novel_writer.py generate my_story_title "Epic fantasy adventure" --translate fr
```

#### Translate Existing Content

Translate an entire story:
```bash
uv run python novel_writer.py translate-story the_mahabharata_saga hi
```

Translate specific chapters:
```bash
uv run python novel_writer.py translate-story the_mahabharata_saga ja --chapters "chapter_01,chapter_02,chapter_03"
```

## Translation Feature

### Supported Languages

- **ISO codes**: `fr`, `es`, `de`, `ja`, `zh`, `hi`, `ar`
- **Full names**: `French`, `Spanish`, `German`, `Japanese`, `Chinese`, `Hindi`, `Arabic`

### Translation Modes

#### 1. During Generation (Post-Processing)
Generate new content and translate it immediately:
```bash
uv run python novel_writer.py generate my_story "Adventure tale" --translate fr
```

#### 2. Translate Existing Content
Translate chapters that have already been generated:
```bash
# Translate a specific chapter
uv run python novel_writer.py translate-story my_story chapter_01 hi

# Translate entire story
uv run python novel_writer.py translate-story my_story French

# Translate specific chapters only
uv run python novel_writer.py translate-story my_story ja --chapters "chapter_01,chapter_02"
```

**Important Notes:**
- Translated files are saved with language suffix (e.g., `chapter_01_the_divine_birth_hi.md`)
- Original files are never modified
- Translation preserves formatting, dialogue, and narrative structure
- Story metadata tracks all translations

## Project Structure

```
long-story/
├── novel_ai.py              # Interactive CLI interface
├── novel_writer.py          # Command-line interface
├── scripts/
│   ├── launch.sh           # macOS/Linux launcher
│   └── launch.ps1          # Windows launcher
├── src/
│   └── novel_memory/       # Core memory architecture
│       ├── memory/         # Memory management modules
│       ├── models/         # AI model abstraction layer
│       ├── translation/    # Translation system
│       ├── agents/         # AI agents
│       └── observability/  # System monitoring
├── stories/                # Generated novels (gitignored)
│   └── your_story/
│       ├── chapters/       # Generated chapters
│       ├── metadata.json   # Story tracking
│       └── memory_data/    # AI memory system
├── pyproject.toml          # Project configuration
└── README.md              # This file
```

## Troubleshooting

### "prompt_toolkit not found" Error

If you see an error about `prompt_toolkit` when running the interactive CLI:

**macOS/Linux:**
```bash
source .venv/bin/activate
pip install prompt_toolkit
```

**Windows:**
```powershell
& .\.venv\Scripts\Activate.ps1
pip install prompt_toolkit
```

### "python: command not found" on macOS

On macOS, use `python3` instead of `python`:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Or use the provided launch scripts which handle this automatically.

### Virtual Environment Not Activating

If the virtual environment doesn't activate automatically:

1. **Create it manually:**
   ```bash
   python3 -m venv .venv  # macOS/Linux
   python -m venv .venv   # Windows
   ```

2. **Activate it:**
   ```bash
   source .venv/bin/activate  # macOS/Linux
   .\.venv\Scripts\Activate.ps1  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

### Azure OpenAI Connection Issues

Ensure your `.env` file has the correct credentials:
- `AZURE_OPENAI_ENDPOINT` should include `https://` and end with `.openai.azure.com`
- `AZURE_OPENAI_API_KEY` should be your valid API key
- `AZURE_OPENAI_DEPLOYMENT_NAME` should match your deployment (e.g., `gpt-4.1`)

## Development

### Requirements

- Python 3.10 or higher
- Azure OpenAI API access
- `uv` package manager (optional but recommended)

### Running Tests

Currently, this project does not have automated tests. Manual testing is performed by:
1. Running the interactive CLI
2. Generating a test story
3. Verifying chapter generation and translation

## License

MIT

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- All features are documented
- Manual testing is performed before submitting