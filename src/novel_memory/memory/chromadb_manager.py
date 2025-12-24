"""
ChromaDB Manager for Narrative Memory Architecture.

Handles ChromaDB collections for narrative elements with local SQLite persistence.
Provides semantic memory storage and retrieval for characters, plot threads,
world building, chapters, and themes.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


@dataclass
class NarrativeCollectionSchema:
    """Schema definition for narrative collections."""
    
    name: str
    description: str
    metadata_schema: Dict[str, type]


class ChromaDBManager:
    """Manage ChromaDB collections for narrative memory storage."""
    
    # Collection schemas for narrative elements
    COLLECTIONS = {
        'characters': NarrativeCollectionSchema(
            name='characters',
            description='Character profiles, development arcs, and relationships',
            metadata_schema={
                'character_name': str,
                'importance': float,
                'story_arc': str,
                'chapter_introduced': int,
                'last_appearance': int,
                'status': str,
                'traits': list,
                'relationships': dict
            }
        ),
        
        'plot_threads': NarrativeCollectionSchema(
            name='plot_threads',
            description='Plot lines, conflicts, and story progression',
            metadata_schema={
                'thread_name': str,
                'thread_type': str,
                'status': str,
                'priority': float,
                'chapters_span': list,
                'resolution_target': int,
                'characters_involved': list,
                'conflict_type': str,
                'stakes': str
            }
        ),
        
        'world_building': NarrativeCollectionSchema(
            name='world_building',
            description='Setting details, rules, and established facts',
            metadata_schema={
                'element_type': str,
                'scope': str,
                'importance': float,
                'established_chapter': int,
                'consistency_level': str,
                'related_elements': list,
                'verification_required': bool
            }
        ),
        
        'chapters': NarrativeCollectionSchema(
            name='chapters',
            description='Chapter summaries and key events',
            metadata_schema={
                'chapter_number': int,
                'title': str,
                'word_count': int,
                'pov_character': str,
                'setting': str,
                'key_events': list,
                'character_development': dict,
                'plot_advancement': list,
                'tension_level': float,
                'emotional_tone': str
            }
        ),
        
        'themes': NarrativeCollectionSchema(
            name='themes',
            description='Recurring motifs and symbolic elements',
            metadata_schema={
                'theme_name': str,
                'theme_type': str,
                'first_introduction': int,
                'development_chapters': list,
                'symbolic_objects': list,
                'character_associations': list,
                'resolution_status': str
            }
        )
    }
    
    def __init__(self, db_path: str = "./chroma_db", embedding_model: str = "text-embedding-3-large"):
        """
        Initialize ChromaDB manager with local SQLite persistence.
        
        Args:
            db_path: Path to ChromaDB database directory
            embedding_model: Azure OpenAI embedding model name
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        
        # Initialize Azure OpenAI client for embeddings
        self.azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Initialize ChromaDB client with SQLite persistence
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        self.collections = {}
        self._initialize_collections()
        
        logger.info(f"ChromaDB initialized at {db_path} with Azure OpenAI model {embedding_model}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Azure OpenAI."""
        try:
            response = self.azure_client.embeddings.create(
                model=self.embedding_model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Fallback to simple hash-based embedding for development
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            # Convert hash to simple numeric embedding
            hash_bytes = hash_obj.digest()
            embedding = [float(b) / 255.0 for b in hash_bytes[:384]]  # 384-dim fallback
            return embedding
    
    def _initialize_collections(self) -> None:
        """Initialize all narrative collections."""
        for collection_name, schema in self.COLLECTIONS.items():
            try:
                collection = self.client.get_collection(collection_name)
                logger.info(f"Found existing collection: {collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "description": schema.description,
                        "hnsw:space": "cosine",
                        "hnsw:M": 16,
                        "hnsw:ef_construction": 200,
                        "hnsw:ef": 100
                    }
                )
                logger.info(f"Created new collection: {collection_name}")
            
            self.collections[collection_name] = collection
    
    def get_collection(self, name: str):
        """Get a specific collection by name."""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' not found. Available: {list(self.collections.keys())}")
        return self.collections[name]
    
    def store_narrative_element(
        self, 
        collection_name: str,
        element_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store a narrative element in the appropriate collection.
        
        Args:
            collection_name: Name of the collection
            element_id: Unique identifier for the element
            content: Text content of the element
            metadata: Element metadata
        """
        # Validate metadata against schema
        self._validate_metadata(collection_name, metadata)
        
        # Generate embedding
        embedding = self._generate_contextual_embedding(collection_name, content, metadata)
        
        # Store in ChromaDB
        collection = self.get_collection(collection_name)
        collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[element_id]
        )
        
        logger.debug(f"Stored element '{element_id}' in collection '{collection_name}'")
    
    def update_narrative_element(
        self,
        collection_name: str,
        element_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update an existing narrative element."""
        collection = self.get_collection(collection_name)
        
        update_data = {"ids": [element_id]}
        
        if content is not None:
            # Get current metadata for embedding regeneration
            current_metadata = self.get_element_metadata(collection_name, element_id)
            if metadata:
                current_metadata.update(metadata)
            
            embedding = self._generate_embedding(collection_name, content, current_metadata)
            update_data["documents"] = [content]
            update_data["embeddings"] = [embedding]
        
        if metadata is not None:
            self._validate_metadata(collection_name, metadata)
            update_data["metadatas"] = [metadata]
        
        collection.update(**update_data)
        logger.debug(f"Updated element '{element_id}' in collection '{collection_name}'")
    
    def get_element_by_id(self, collection_name: str, element_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific element by ID."""
        collection = self.get_collection(collection_name)
        
        try:
            results = collection.get(
                ids=[element_id],
                include=['documents', 'metadatas']
            )
            
            if results['documents']:
                return {
                    'id': element_id,
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
        except Exception as e:
            logger.warning(f"Error retrieving element '{element_id}': {e}")
        
        return None
    
    def get_element_metadata(self, collection_name: str, element_id: str) -> Dict[str, Any]:
        """Get metadata for a specific element."""
        element = self.get_element_by_id(collection_name, element_id)
        return element['metadata'] if element else {}
    
    def search_similar_elements(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar narrative elements."""
        collection = self.get_collection(collection_name)
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Search with optional filtering
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )
        
        return self._format_search_results(results)
    
    def get_elements_by_metadata(
        self,
        collection_name: str,
        where_filter: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve elements by metadata filtering."""
        collection = self.get_collection(collection_name)
        
        results = collection.get(
            where=where_filter,
            limit=limit,
            include=['documents', 'metadatas']
        )
        
        return [
            {
                'id': id_val,
                'content': doc,
                'metadata': meta
            }
            for id_val, doc, meta in zip(
                results['ids'],
                results['documents'],
                results['metadatas']
            )
        ]
    
    def search_narrative_context(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        max_results_per_collection: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search across multiple collections for comprehensive context."""
        if collections is None:
            collections = list(self.collections.keys())
        
        context = {}
        
        for collection_name in collections:
            try:
                results = self.search_similar_elements(
                    collection_name,
                    query,
                    n_results=max_results_per_collection
                )
                context[collection_name] = results
            except Exception as e:
                logger.error(f"Error searching collection '{collection_name}': {e}")
                context[collection_name] = []
        
        return context
    
    def get_character_context(self, character_name: str) -> Dict[str, Any]:
        """Get comprehensive context for a specific character."""
        character_filter = {"character_name": {"$eq": character_name}}
        
        # Get character profile
        character_info = self.get_elements_by_metadata('characters', character_filter, limit=1)
        
        # Get plot threads involving this character
        plot_filter = {"characters_involved": {"$in": [character_name]}}
        related_plots = self.get_elements_by_metadata('plot_threads', plot_filter)
        
        # Get chapters where character appears
        chapter_filter = {"pov_character": {"$eq": character_name}}
        pov_chapters = self.get_elements_by_metadata('chapters', chapter_filter)
        
        return {
            'character_profile': character_info[0] if character_info else None,
            'related_plots': related_plots,
            'pov_chapters': pov_chapters
        }
    
    def delete_element(self, collection_name: str, element_id: str) -> None:
        """Delete a narrative element."""
        collection = self.get_collection(collection_name)
        collection.delete(ids=[element_id])
        logger.debug(f"Deleted element '{element_id}' from collection '{collection_name}'")
    
    def batch_store_elements(
        self,
        collection_name: str,
        elements: List[Dict[str, Any]]
    ) -> None:
        """Batch store multiple elements for better performance."""
        collection = self.get_collection(collection_name)
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for element in elements:
            ids.append(element['id'])
            documents.append(element['content'])
            metadatas.append(element['metadata'])
            
            # Generate embedding
            embedding = self._generate_contextual_embedding(
                collection_name, 
                element['content'], 
                element['metadata']
            )
            embeddings.append(embedding)
        
        # Batch insert
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Batch stored {len(elements)} elements in collection '{collection_name}'")
    
    def reset_collections(self) -> None:
        """Reset all collections (useful for testing)."""
        for collection_name in list(self.collections.keys()):
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
            except ValueError:
                pass
        
        self.collections.clear()
        self._initialize_collections()
        logger.info("All collections reset")
    
    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections."""
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {
                    'document_count': count,
                    'description': self.COLLECTIONS[name].description
                }
            except Exception as e:
                stats[name] = {'error': str(e)}
        
        return stats
    
    def _generate_embedding(
        self, 
        collection_name: str, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> List[float]:
        """Generate embedding optimized for specific narrative element type."""
        enriched_content = self._enrich_content(collection_name, content, metadata)
        return self._generate_embedding(enriched_content)
    
    def _enrich_content(
        self, 
        collection_name: str, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> str:
        """Enrich content with metadata for better semantic understanding."""
        enrichers = {
            'characters': self._enrich_character_content,
            'plot_threads': self._enrich_plot_content,
            'world_building': self._enrich_world_content,
            'chapters': self._enrich_chapter_content,
            'themes': self._enrich_theme_content
        }
        
        enricher = enrichers.get(collection_name, lambda c, m: c)
        return enricher(content, metadata)
    
    def _enrich_character_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Enrich character content with traits and relationships."""
        traits = ', '.join(metadata.get('traits', []))
        relationships = '; '.join([
            f"{name}: {relation}" 
            for name, relation in metadata.get('relationships', {}).items()
        ])
        
        enriched = f"Character: {content}"
        if traits:
            enriched += f" Traits: {traits}"
        if relationships:
            enriched += f" Relationships: {relationships}"
        
        return enriched
    
    def _enrich_plot_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Enrich plot content with thread information."""
        thread_type = metadata.get('thread_type', '')
        conflict_type = metadata.get('conflict_type', '')
        stakes = metadata.get('stakes', '')
        
        enriched = f"Plot thread ({thread_type}): {content}"
        if conflict_type:
            enriched += f" Conflict: {conflict_type}"
        if stakes:
            enriched += f" Stakes: {stakes}"
        
        return enriched
    
    def _enrich_world_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Enrich world-building content with context."""
        element_type = metadata.get('element_type', '')
        scope = metadata.get('scope', '')
        
        return f"World element ({element_type}, {scope}): {content}"
    
    def _enrich_chapter_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Enrich chapter content with summary information."""
        chapter_num = metadata.get('chapter_number', '')
        pov = metadata.get('pov_character', '')
        tone = metadata.get('emotional_tone', '')
        
        enriched = f"Chapter {chapter_num} ({pov} POV): {content}"
        if tone:
            enriched += f" Tone: {tone}"
        
        return enriched
    
    def _enrich_theme_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Enrich theme content with symbolic associations."""
        theme_type = metadata.get('theme_type', '')
        symbols = ', '.join(metadata.get('symbolic_objects', []))
        
        enriched = f"Theme ({theme_type}): {content}"
        if symbols:
            enriched += f" Symbols: {symbols}"
        
        return enriched
    
    def _validate_metadata(self, collection_name: str, metadata: Dict[str, Any]) -> None:
        """Validate metadata against collection schema."""
        schema = self.COLLECTIONS[collection_name].metadata_schema
        
        for key, expected_type in schema.items():
            if key in metadata:
                value = metadata[key]
                if not isinstance(value, expected_type):
                    if expected_type == list and not isinstance(value, list):
                        metadata[key] = [value] if value else []
                    elif expected_type == dict and not isinstance(value, dict):
                        raise ValueError(f"Metadata '{key}' must be a dict, got {type(value)}")
                    elif expected_type in (int, float, str):
                        try:
                            metadata[key] = expected_type(value)
                        except (ValueError, TypeError):
                            raise ValueError(f"Cannot convert '{key}' to {expected_type}")
    
    def _format_search_results(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format ChromaDB search results into structured format."""
        formatted = []
        
        if not raw_results['documents'] or not raw_results['documents'][0]:
            return formatted
        
        for i in range(len(raw_results['documents'][0])):
            formatted.append({
                'id': raw_results['ids'][0][i],
                'content': raw_results['documents'][0][i],
                'metadata': raw_results['metadatas'][0][i],
                'similarity_score': 1 - raw_results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted