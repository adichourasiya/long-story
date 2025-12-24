"""
Novel Memory Architecture - Core package initialization.

This package provides a comprehensive long-form narrative generation system
with hierarchical memory management, Azure OpenAI integration, and intelligent
consistency enforcement.
"""

from .memory.manager import NarrativeMemoryManager
from .memory.chromadb_manager import ChromaDBManager
from .memory.markdown_persistence import MarkdownStoryPersistence
# Temporarily disable agent imports due to langchain version issues
# from .agents.azure_router import AzureModelRouter, AzureOpenAIConfig
# from .agents.novel_agent import NovelWritingAgent

__version__ = "0.1.0"
__author__ = "Novel Memory Architecture Team"

__all__ = [
    "NarrativeMemoryManager",
    "ChromaDBManager", 
    "MarkdownStoryPersistence",
    # "AzureModelRouter",
    # "AzureOpenAIConfig",
    # "NovelWritingAgent",
]