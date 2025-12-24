"""
Hierarchical Memory Architecture with multi-tier memory system
Implements working memory, episodic memory, semantic memory, and procedural memory
"""
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import chromadb
from chromadb.config import Settings
import threading
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class MemoryTier(Enum):
    """Memory hierarchy tiers"""
    WORKING = "working"  # Current chapter context
    EPISODIC = "episodic"  # Chapter-level memories
    SEMANTIC = "semantic"  # Character/world knowledge
    PROCEDURAL = "procedural"  # Story rules/patterns

class ProtectionLevel(Enum):
    """Information protection levels"""
    CRITICAL = "critical"  # Never summarize away
    IMPORTANT = "important"  # Preserve with high priority
    NORMAL = "normal"  # Standard summarization
    DISPOSABLE = "disposable"  # Can be safely compressed

@dataclass
class MemoryNode:
    """Individual memory element with metadata"""
    id: str
    content: str
    tier: MemoryTier
    protection_level: ProtectionLevel
    tags: Set[str]
    importance_score: float
    creation_time: datetime
    last_accessed: datetime
    access_count: int
    dependencies: Set[str]  # IDs of related memories
    checksum: str
    version: int = 1

    def __post_init__(self):
        if not self.checksum:
            self.checksum = hashlib.sha256(self.content.encode()).hexdigest()
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        if isinstance(self.dependencies, list):
            self.dependencies = set(self.dependencies)

@dataclass
class NarrativeArc:
    """Protected narrative arc information"""
    id: str
    name: str
    description: str
    planned_chapters: List[str]
    foreshadowing_elements: List[str]
    resolution_chapter: Optional[str]
    current_status: str  # planned, active, resolved
    importance_score: float
    creation_time: datetime = None
    dependencies: Set[str] = None

    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = datetime.now()
        if self.dependencies is None:
            self.dependencies = set()

@dataclass
class CharacterState:
    """Protected character development tracking"""
    character_id: str
    traits: Dict[str, Any]
    relationships: Dict[str, float]  # character_id -> relationship_strength
    arc_progress: Dict[str, float]  # arc_id -> completion_percentage
    last_appearance: Optional[str]  # chapter_id
    planned_development: List[str]
    emotional_state: Dict[str, float] = None
    location_history: List[str] = None

    def __post_init__(self):
        if self.emotional_state is None:
            self.emotional_state = {}
        if self.location_history is None:
            self.location_history = []

class ConsistencyViolation:
    """Represents a narrative consistency violation"""
    def __init__(self, violation_type: str, description: str, severity: str, 
                 involved_memories: List[str], suggestions: List[str]):
        self.violation_type = violation_type
        self.description = description
        self.severity = severity  # low, medium, high, critical
        self.involved_memories = involved_memories
        self.suggestions = suggestions
        self.timestamp = datetime.now()

class HierarchicalMemorySystem:
    """
    Multi-tier memory system with protection and recovery capabilities
    """
    
    def __init__(self, base_path: Path, chromadb_path: Optional[Path] = None):
        self.base_path = Path(base_path)
        self.memory_path = self.base_path / "memory"
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize ChromaDB for semantic search
        if chromadb_path is None:
            chromadb_path = self.memory_path / "chromadb"
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(chromadb_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize collections for different memory tiers
        self._init_collections()
        
        # Memory state
        self.working_memory: Dict[str, MemoryNode] = {}
        self.protected_arcs: Dict[str, NarrativeArc] = {}
        self.character_states: Dict[str, CharacterState] = {}
        
        # Caches for performance
        self._semantic_cache: Dict[str, List[Dict]] = {}
        self._consistency_cache: Dict[str, List[ConsistencyViolation]] = {}
        
        # Load existing state
        self._load_persistent_state()
        
        # Performance metrics
        self.access_metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "retrieval_times": [],
            "consistency_checks": 0,
            "violations_found": 0
        }
        
        # Configuration
        self.config = {
            "max_working_memory_size": 50,
            "semantic_cache_size": 1000,
            "consistency_check_frequency": 10,
            "auto_save_interval": 300,  # 5 minutes
            "max_context_length": 128000
        }
    
    def _init_collections(self):
        """Initialize ChromaDB collections for each memory tier"""
        try:
            self.episodic_collection = self.chroma_client.get_or_create_collection(
                name="episodic_memory",
                metadata={"tier": "episodic", "description": "Chapter-level narrative memories"}
            )
            
            self.semantic_collection = self.chroma_client.get_or_create_collection(
                name="semantic_memory", 
                metadata={"tier": "semantic", "description": "Character and world knowledge"}
            )
            
            self.procedural_collection = self.chroma_client.get_or_create_collection(
                name="procedural_memory",
                metadata={"tier": "procedural", "description": "Story patterns and rules"}
            )
            
            logger.info("ChromaDB collections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collections: {e}")
            raise
    
    def _load_persistent_state(self):
        """Load persistent memory state from disk"""
        state_file = self.memory_path / "memory_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                
                # Load protected arcs
                for arc_data in state_data.get('protected_arcs', []):
                    if 'creation_time' in arc_data and isinstance(arc_data['creation_time'], str):
                        arc_data['creation_time'] = datetime.fromisoformat(arc_data['creation_time'])
                    arc = NarrativeArc(**arc_data)
                    self.protected_arcs[arc.id] = arc
                
                # Load character states
                for char_data in state_data.get('character_states', []):
                    char_state = CharacterState(**char_data)
                    self.character_states[char_state.character_id] = char_state
                
                logger.info(f"Loaded {len(self.protected_arcs)} arcs and {len(self.character_states)} character states")
                
            except Exception as e:
                logger.error(f"Failed to load persistent state: {e}")
    
    def save_persistent_state(self):
        """Save persistent memory state to disk"""
        state_file = self.memory_path / "memory_state.json"
        backup_file = self.memory_path / f"memory_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with self._lock:
                # Prepare state data
                state_data = {
                    'protected_arcs': [
                        {**asdict(arc), 'creation_time': arc.creation_time.isoformat() if arc.creation_time else None}
                        for arc in self.protected_arcs.values()
                    ],
                    'character_states': [asdict(state) for state in self.character_states.values()],
                    'timestamp': datetime.now().isoformat(),
                    'version': '2.0',
                    'checksum': self._compute_state_checksum()
                }
                
                # Create backup of existing state
                if state_file.exists():
                    state_file.rename(backup_file)
                
                # Write new state
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, indent=2, default=str)
                
                logger.info("Memory state saved successfully")
                
        except Exception as e:
            logger.error(f"Failed to save persistent state: {e}")
            # Restore backup if write failed
            if backup_file.exists():
                backup_file.rename(state_file)
            raise
    
    def _compute_state_checksum(self) -> str:
        """Compute checksum for state integrity verification"""
        state_str = json.dumps({
            'arcs': len(self.protected_arcs),
            'characters': len(self.character_states),
            'working_memory': len(self.working_memory)
        }, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def add_memory(self, content: str, tier: MemoryTier, protection_level: ProtectionLevel,
                   tags: Set[str], importance_score: float = 0.5, 
                   dependencies: Set[str] = None) -> str:
        """Add a new memory node to the system"""
        if dependencies is None:
            dependencies = set()
            
        memory_id = hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        memory_node = MemoryNode(
            id=memory_id,
            content=content,
            tier=tier,
            protection_level=protection_level,
            tags=tags,
            importance_score=importance_score,
            creation_time=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            dependencies=dependencies,
            checksum=""  # Will be computed in __post_init__
        )
        
        with self._lock:
            if tier == MemoryTier.WORKING:
                self.working_memory[memory_id] = memory_node
                # Manage working memory size
                if len(self.working_memory) > self.config["max_working_memory_size"]:
                    self._promote_working_memory()
            else:
                # Store in appropriate ChromaDB collection
                self._store_in_chromadb(memory_node)
        
        logger.debug(f"Added memory {memory_id} to tier {tier.value}")
        return memory_id
    
    def _store_in_chromadb(self, memory_node: MemoryNode):
        """Store memory node in appropriate ChromaDB collection"""
        try:
            collection_map = {
                MemoryTier.EPISODIC: self.episodic_collection,
                MemoryTier.SEMANTIC: self.semantic_collection,
                MemoryTier.PROCEDURAL: self.procedural_collection
            }
            
            collection = collection_map.get(memory_node.tier)
            if collection is None:
                logger.warning(f"No collection for tier {memory_node.tier}")
                return
            
            metadata = {
                "protection_level": memory_node.protection_level.value,
                "importance_score": memory_node.importance_score,
                "creation_time": memory_node.creation_time.isoformat(),
                "tags": list(memory_node.tags),
                "checksum": memory_node.checksum,
                "version": memory_node.version
            }
            
            collection.add(
                documents=[memory_node.content],
                metadatas=[metadata],
                ids=[memory_node.id]
            )
            
        except Exception as e:
            logger.error(f"Failed to store memory in ChromaDB: {e}")
    
    def _promote_working_memory(self):
        """Promote least recently used working memory to episodic tier"""
        if not self.working_memory:
            return
            
        # Find least recently used memory
        lru_memory = min(self.working_memory.values(), 
                        key=lambda m: m.last_accessed)
        
        # Promote to episodic memory
        lru_memory.tier = MemoryTier.EPISODIC
        self._store_in_chromadb(lru_memory)
        
        # Remove from working memory
        del self.working_memory[lru_memory.id]
        
        logger.debug(f"Promoted memory {lru_memory.id} to episodic tier")
    
    def query_memories(self, query: str, tiers: List[MemoryTier], 
                      limit: int = 10, relevance_threshold: float = 0.7) -> List[MemoryNode]:
        """Query memories across specified tiers"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{query}:{':'.join([t.value for t in tiers])}:{limit}:{relevance_threshold}"
        if cache_key in self._semantic_cache:
            self.access_metrics["cache_hits"] += 1
            return self._semantic_cache[cache_key]
        
        self.access_metrics["cache_misses"] += 1
        results = []
        
        with self._lock:
            # Query working memory if included
            if MemoryTier.WORKING in tiers:
                for memory in self.working_memory.values():
                    if self._matches_query(memory, query):
                        memory.last_accessed = datetime.now()
                        memory.access_count += 1
                        results.append(memory)
            
            # Query ChromaDB collections
            for tier in tiers:
                if tier == MemoryTier.WORKING:
                    continue
                    
                collection_map = {
                    MemoryTier.EPISODIC: self.episodic_collection,
                    MemoryTier.SEMANTIC: self.semantic_collection,
                    MemoryTier.PROCEDURAL: self.procedural_collection
                }
                
                collection = collection_map.get(tier)
                if collection is None:
                    continue
                
                try:
                    query_results = collection.query(
                        query_texts=[query],
                        n_results=limit,
                        where={"protection_level": {"$ne": ProtectionLevel.DISPOSABLE.value}}
                    )
                    
                    if query_results['documents']:
                        for i, doc in enumerate(query_results['documents'][0]):
                            metadata = query_results['metadatas'][0][i]
                            memory_id = query_results['ids'][0][i]
                            
                            memory_node = MemoryNode(
                                id=memory_id,
                                content=doc,
                                tier=tier,
                                protection_level=ProtectionLevel(metadata['protection_level']),
                                tags=set(metadata.get('tags', [])),
                                importance_score=metadata['importance_score'],
                                creation_time=datetime.fromisoformat(metadata['creation_time']),
                                last_accessed=datetime.now(),
                                access_count=0,
                                dependencies=set(),
                                checksum=metadata.get('checksum', ''),
                                version=metadata.get('version', 1)
                            )
                            
                            results.append(memory_node)
                            
                except Exception as e:
                    logger.error(f"Failed to query {tier.value} collection: {e}")
        
        # Sort by relevance and importance
        results.sort(key=lambda m: m.importance_score, reverse=True)
        results = results[:limit]
        
        # Cache results
        if len(self._semantic_cache) < self.config["semantic_cache_size"]:
            self._semantic_cache[cache_key] = results
        
        retrieval_time = time.time() - start_time
        self.access_metrics["retrieval_times"].append(retrieval_time)
        
        return results
    
    def _matches_query(self, memory: MemoryNode, query: str) -> bool:
        """Simple text matching for working memory"""
        query_lower = query.lower()
        return (query_lower in memory.content.lower() or 
                any(query_lower in tag.lower() for tag in memory.tags))
    
    def add_narrative_arc(self, name: str, description: str, 
                         planned_chapters: List[str]) -> str:
        """Add a protected narrative arc"""
        arc_id = hashlib.sha256(f"{name}{description}".encode()).hexdigest()[:16]
        
        arc = NarrativeArc(
            id=arc_id,
            name=name,
            description=description,
            planned_chapters=planned_chapters,
            foreshadowing_elements=[],
            resolution_chapter=None,
            current_status="planned",
            importance_score=0.8  # High importance by default
        )
        
        with self._lock:
            self.protected_arcs[arc_id] = arc
        
        logger.info(f"Added narrative arc: {name}")
        return arc_id
    
    def update_character_state(self, character_id: str, **kwargs) -> None:
        """Update character state with new information"""
        with self._lock:
            if character_id not in self.character_states:
                self.character_states[character_id] = CharacterState(
                    character_id=character_id,
                    traits={},
                    relationships={},
                    arc_progress={},
                    last_appearance=None,
                    planned_development=[]
                )
            
            character = self.character_states[character_id]
            for key, value in kwargs.items():
                if hasattr(character, key):
                    setattr(character, key, value)
        
        logger.debug(f"Updated character state for {character_id}")
    
    def check_consistency(self) -> List[ConsistencyViolation]:
        """Check for narrative consistency violations"""
        self.access_metrics["consistency_checks"] += 1
        violations = []
        
        # Check character consistency
        violations.extend(self._check_character_consistency())
        
        # Check arc consistency
        violations.extend(self._check_arc_consistency())
        
        # Check temporal consistency
        violations.extend(self._check_temporal_consistency())
        
        self.access_metrics["violations_found"] += len(violations)
        return violations
    
    def _check_character_consistency(self) -> List[ConsistencyViolation]:
        """Check for character-related inconsistencies"""
        violations = []
        
        for char_id, char_state in self.character_states.items():
            # Check for contradictory traits
            trait_memories = self.query_memories(
                f"character {char_id} personality trait",
                [MemoryTier.SEMANTIC, MemoryTier.EPISODIC],
                limit=20
            )
            
            # Simple contradiction detection (can be enhanced)
            trait_keywords = ['brave', 'cowardly', 'kind', 'cruel', 'smart', 'stupid']
            contradictions = []
            
            for memory in trait_memories:
                content_lower = memory.content.lower()
                found_traits = [trait for trait in trait_keywords if trait in content_lower]
                
                if 'brave' in found_traits and 'cowardly' in found_traits:
                    contradictions.append(('brave', 'cowardly'))
                if 'kind' in found_traits and 'cruel' in found_traits:
                    contradictions.append(('kind', 'cruel'))
                if 'smart' in found_traits and 'stupid' in found_traits:
                    contradictions.append(('smart', 'stupid'))
            
            for contradiction in contradictions:
                violation = ConsistencyViolation(
                    violation_type="character_trait_contradiction",
                    description=f"Character {char_id} has contradictory traits: {contradiction[0]} vs {contradiction[1]}",
                    severity="medium",
                    involved_memories=[m.id for m in trait_memories],
                    suggestions=[
                        f"Clarify character development that explains the change",
                        f"Choose one consistent trait for {char_id}",
                        f"Show character growth from {contradiction[0]} to {contradiction[1]}"
                    ]
                )
                violations.append(violation)
        
        return violations
    
    def _check_arc_consistency(self) -> List[ConsistencyViolation]:
        """Check for narrative arc inconsistencies"""
        violations = []
        
        for arc_id, arc in self.protected_arcs.items():
            if arc.current_status == "resolved" and arc.resolution_chapter is None:
                violation = ConsistencyViolation(
                    violation_type="unresolved_arc",
                    description=f"Arc '{arc.name}' marked as resolved but no resolution chapter specified",
                    severity="high",
                    involved_memories=[arc_id],
                    suggestions=[
                        f"Specify the resolution chapter for arc '{arc.name}'",
                        f"Change arc status to 'active' if not yet resolved"
                    ]
                )
                violations.append(violation)
        
        return violations
    
    def _check_temporal_consistency(self) -> List[ConsistencyViolation]:
        """Check for temporal/sequence inconsistencies"""
        violations = []
        
        # Get all episodic memories sorted by creation time
        episodic_memories = self.query_memories(
            "chapter",
            [MemoryTier.EPISODIC],
            limit=100
        )
        
        # Simple temporal check - could be enhanced
        chapter_numbers = []
        for memory in episodic_memories:
            if 'chapter' in memory.content.lower():
                # Extract chapter numbers (basic implementation)
                import re
                matches = re.findall(r'chapter (\d+)', memory.content.lower())
                if matches:
                    chapter_numbers.extend([int(num) for num in matches])
        
        # Check for gaps in chapter sequence
        if chapter_numbers:
            chapter_numbers.sort()
            for i in range(len(chapter_numbers) - 1):
                if chapter_numbers[i + 1] - chapter_numbers[i] > 1:
                    violation = ConsistencyViolation(
                        violation_type="chapter_sequence_gap",
                        description=f"Gap in chapter sequence: {chapter_numbers[i]} to {chapter_numbers[i + 1]}",
                        severity="low",
                        involved_memories=[],
                        suggestions=[
                            "Verify chapter numbering is correct",
                            "Add missing chapters if needed"
                        ]
                    )
                    violations.append(violation)
        
        return violations
    
    def get_context_for_generation(self, query: str, max_tokens: int = None) -> Dict[str, Any]:
        """Get optimized context for content generation"""
        if max_tokens is None:
            max_tokens = self.config["max_context_length"]
        
        # Get relevant memories from all tiers
        memories = self.query_memories(
            query,
            [MemoryTier.WORKING, MemoryTier.EPISODIC, MemoryTier.SEMANTIC],
            limit=50
        )
        
        # Prioritize by protection level and importance
        critical_memories = [m for m in memories if m.protection_level == ProtectionLevel.CRITICAL]
        important_memories = [m for m in memories if m.protection_level == ProtectionLevel.IMPORTANT]
        normal_memories = [m for m in memories if m.protection_level == ProtectionLevel.NORMAL]
        
        # Build context respecting token limits
        context = {
            "critical_information": [m.content for m in critical_memories[:10]],
            "important_context": [m.content for m in important_memories[:15]],
            "background": [m.content for m in normal_memories[:20]],
            "active_arcs": [
                {"name": arc.name, "description": arc.description, "status": arc.current_status}
                for arc in self.protected_arcs.values()
                if arc.current_status == "active"
            ],
            "character_states": {
                char_id: {
                    "traits": state.traits,
                    "recent_development": state.planned_development[-3:] if state.planned_development else []
                }
                for char_id, state in self.character_states.items()
            }
        }
        
        return context
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance and usage metrics"""
        avg_retrieval_time = (
            sum(self.access_metrics["retrieval_times"]) / len(self.access_metrics["retrieval_times"])
            if self.access_metrics["retrieval_times"] else 0
        )
        
        return {
            "memory_counts": {
                "working": len(self.working_memory),
                "protected_arcs": len(self.protected_arcs),
                "character_states": len(self.character_states)
            },
            "performance": {
                "cache_hit_rate": self.access_metrics["cache_hits"] / 
                                 max(self.access_metrics["cache_hits"] + self.access_metrics["cache_misses"], 1),
                "average_retrieval_time": avg_retrieval_time,
                "total_queries": self.access_metrics["cache_hits"] + self.access_metrics["cache_misses"]
            },
            "consistency": {
                "checks_performed": self.access_metrics["consistency_checks"],
                "violations_found": self.access_metrics["violations_found"]
            }
        }