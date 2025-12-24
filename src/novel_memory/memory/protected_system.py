"""
Protected Information System
Ensures critical narrative elements survive summarization and context compression
"""
import json
import logging
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class ProtectedElementType(Enum):
    """Types of protected narrative elements"""
    CHARACTER_ARC = "character_arc"
    PLOT_THREAD = "plot_thread"  
    CHEKOVS_GUN = "chekovs_gun"
    FORESHADOWING = "foreshadowing"
    WORLD_RULE = "world_rule"
    RELATIONSHIP = "relationship"
    SECRET = "secret"
    PROPHECY = "prophecy"
    RECURRING_MOTIF = "recurring_motif"

@dataclass
class ProtectedElement:
    """A protected narrative element that must survive summarization"""
    id: str
    element_type: ProtectedElementType
    title: str
    description: str
    associated_characters: List[str]
    chapter_references: List[str]
    importance_level: int  # 1-10, 10 being critical
    resolution_status: str  # introduced, developing, resolved, abandoned
    resolution_chapter: Optional[str]
    foreshadowing_chapters: List[str]
    dependencies: List[str]  # IDs of other protected elements
    tags: Set[str]
    creation_time: datetime
    last_modified: datetime
    payload: Dict[str, Any]  # Additional element-specific data

    def __post_init__(self):
        if isinstance(self.tags, list):
            self.tags = set(self.tags)

@dataclass
class NarrativeKnowledgeGraph:
    """Represents relationships between narrative elements"""
    nodes: Dict[str, Dict[str, Any]]  # node_id -> properties
    edges: Dict[str, List[Dict[str, Any]]]  # from_node_id -> [edge_data]
    
    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any]):
        """Add a node to the knowledge graph"""
        self.nodes[node_id] = {
            "type": node_type,
            "properties": properties,
            "created": datetime.now().isoformat()
        }
    
    def add_edge(self, from_node: str, to_node: str, edge_type: str, 
                 properties: Optional[Dict[str, Any]] = None):
        """Add an edge between nodes"""
        if from_node not in self.edges:
            self.edges[from_node] = []
        
        edge_data = {
            "to": to_node,
            "type": edge_type,
            "properties": properties or {},
            "created": datetime.now().isoformat()
        }
        self.edges[from_node].append(edge_data)
    
    def get_connected_nodes(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Get all nodes connected to the given node"""
        connected = []
        
        # Outgoing connections
        for edge in self.edges.get(node_id, []):
            if edge_type is None or edge["type"] == edge_type:
                connected.append(edge["to"])
        
        # Incoming connections
        for from_node, edges in self.edges.items():
            for edge in edges:
                if edge["to"] == node_id:
                    if edge_type is None or edge["type"] == edge_type:
                        connected.append(from_node)
        
        return list(set(connected))

class ProtectedInformationSystem:
    """
    Manages protected narrative elements that must survive summarization
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.protected_path = self.base_path / "protected"
        self.protected_path.mkdir(parents=True, exist_ok=True)
        
        self.protected_elements: Dict[str, ProtectedElement] = {}
        self.knowledge_graph = NarrativeKnowledgeGraph(nodes={}, edges={})
        
        self._load_protected_state()
    
    def _load_protected_state(self):
        """Load protected elements from disk"""
        elements_file = self.protected_path / "protected_elements.json"
        graph_file = self.protected_path / "knowledge_graph.json"
        
        # Load protected elements
        if elements_file.exists():
            try:
                with open(elements_file, 'r', encoding='utf-8') as f:
                    elements_data = json.load(f)
                
                for elem_data in elements_data:
                    # Convert datetime strings back to datetime objects
                    elem_data['creation_time'] = datetime.fromisoformat(elem_data['creation_time'])
                    elem_data['last_modified'] = datetime.fromisoformat(elem_data['last_modified'])
                    elem_data['element_type'] = ProtectedElementType(elem_data['element_type'])
                    
                    element = ProtectedElement(**elem_data)
                    self.protected_elements[element.id] = element
                
                logger.info(f"Loaded {len(self.protected_elements)} protected elements")
                
            except Exception as e:
                logger.error(f"Failed to load protected elements: {e}")
        
        # Load knowledge graph
        if graph_file.exists():
            try:
                with open(graph_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                    self.knowledge_graph = NarrativeKnowledgeGraph(**graph_data)
                
                logger.info(f"Loaded knowledge graph with {len(self.knowledge_graph.nodes)} nodes")
                
            except Exception as e:
                logger.error(f"Failed to load knowledge graph: {e}")
    
    def save_protected_state(self):
        """Save protected elements and knowledge graph to disk"""
        elements_file = self.protected_path / "protected_elements.json"
        graph_file = self.protected_path / "knowledge_graph.json"
        
        try:
            # Save protected elements
            elements_data = []
            for element in self.protected_elements.values():
                elem_dict = asdict(element)
                elem_dict['creation_time'] = element.creation_time.isoformat()
                elem_dict['last_modified'] = element.last_modified.isoformat()
                elem_dict['element_type'] = element.element_type.value
                elem_dict['tags'] = list(element.tags)
                elements_data.append(elem_dict)
            
            with open(elements_file, 'w', encoding='utf-8') as f:
                json.dump(elements_data, f, indent=2, default=str)
            
            # Save knowledge graph
            with open(graph_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.knowledge_graph), f, indent=2, default=str)
            
            logger.info("Protected state saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save protected state: {e}")
            raise
    
    def add_protected_element(self, element_type: ProtectedElementType, title: str,
                            description: str, associated_characters: List[str] = None,
                            importance_level: int = 5, tags: Set[str] = None,
                            payload: Dict[str, Any] = None) -> str:
        """Add a new protected narrative element"""
        if associated_characters is None:
            associated_characters = []
        if tags is None:
            tags = set()
        if payload is None:
            payload = {}
        
        element_id = hashlib.sha256(f"{title}{description}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        element = ProtectedElement(
            id=element_id,
            element_type=element_type,
            title=title,
            description=description,
            associated_characters=associated_characters,
            chapter_references=[],
            importance_level=importance_level,
            resolution_status="introduced",
            resolution_chapter=None,
            foreshadowing_chapters=[],
            dependencies=[],
            tags=tags,
            creation_time=datetime.now(),
            last_modified=datetime.now(),
            payload=payload
        )
        
        self.protected_elements[element_id] = element
        
        # Add to knowledge graph
        self.knowledge_graph.add_node(
            element_id,
            element_type.value,
            {
                "title": title,
                "importance": importance_level,
                "status": "introduced"
            }
        )
        
        # Create relationships with characters
        for char_id in associated_characters:
            self.knowledge_graph.add_edge(
                element_id, char_id, "involves_character"
            )
        
        logger.info(f"Added protected element: {title} ({element_type.value})")
        return element_id
    
    def update_element_status(self, element_id: str, status: str, 
                            chapter_id: Optional[str] = None):
        """Update the resolution status of a protected element"""
        if element_id not in self.protected_elements:
            logger.warning(f"Element {element_id} not found")
            return
        
        element = self.protected_elements[element_id]
        element.resolution_status = status
        element.last_modified = datetime.now()
        
        if status == "resolved" and chapter_id:
            element.resolution_chapter = chapter_id
        
        # Update knowledge graph
        if element_id in self.knowledge_graph.nodes:
            self.knowledge_graph.nodes[element_id]["properties"]["status"] = status
        
        logger.debug(f"Updated element {element.title} status to {status}")
    
    def add_chapter_reference(self, element_id: str, chapter_id: str, 
                            reference_type: str = "mention"):
        """Add a chapter reference to a protected element"""
        if element_id not in self.protected_elements:
            logger.warning(f"Element {element_id} not found")
            return
        
        element = self.protected_elements[element_id]
        if chapter_id not in element.chapter_references:
            element.chapter_references.append(chapter_id)
            element.last_modified = datetime.now()
        
        if reference_type == "foreshadowing":
            element.foreshadowing_chapters.append(chapter_id)
        
        # Add edge in knowledge graph
        self.knowledge_graph.add_edge(
            element_id, chapter_id, reference_type
        )
        
        logger.debug(f"Added chapter reference {chapter_id} to element {element.title}")
    
    def add_element_dependency(self, element_id: str, depends_on_id: str,
                             dependency_type: str = "depends_on"):
        """Add a dependency between protected elements"""
        if element_id not in self.protected_elements or depends_on_id not in self.protected_elements:
            logger.warning(f"One or both elements not found: {element_id}, {depends_on_id}")
            return
        
        element = self.protected_elements[element_id]
        if depends_on_id not in element.dependencies:
            element.dependencies.append(depends_on_id)
            element.last_modified = datetime.now()
        
        # Add edge in knowledge graph
        self.knowledge_graph.add_edge(
            element_id, depends_on_id, dependency_type
        )
        
        logger.debug(f"Added dependency: {element.title} depends on {self.protected_elements[depends_on_id].title}")
    
    def get_critical_elements_for_chapter(self, chapter_id: str) -> List[ProtectedElement]:
        """Get all critical elements relevant to a specific chapter"""
        relevant_elements = []
        
        for element in self.protected_elements.values():
            # Element is directly referenced in this chapter
            if chapter_id in element.chapter_references:
                relevant_elements.append(element)
                continue
            
            # Element is being developed and hasn't been resolved
            if (element.resolution_status in ["introduced", "developing"] and 
                element.importance_level >= 7):
                relevant_elements.append(element)
                continue
            
            # Element has dependencies that are active
            for dep_id in element.dependencies:
                if dep_id in self.protected_elements:
                    dep_element = self.protected_elements[dep_id]
                    if (dep_element.resolution_status in ["introduced", "developing"] and
                        chapter_id in dep_element.chapter_references):
                        relevant_elements.append(element)
                        break
        
        # Sort by importance level
        relevant_elements.sort(key=lambda e: e.importance_level, reverse=True)
        return relevant_elements
    
    def get_unresolved_elements(self) -> List[ProtectedElement]:
        """Get all unresolved protected elements"""
        unresolved = []
        
        for element in self.protected_elements.values():
            if element.resolution_status in ["introduced", "developing"]:
                unresolved.append(element)
        
        # Sort by importance and creation time
        unresolved.sort(key=lambda e: (e.importance_level, e.creation_time), reverse=True)
        return unresolved
    
    def get_elements_by_type(self, element_type: ProtectedElementType) -> List[ProtectedElement]:
        """Get all elements of a specific type"""
        return [e for e in self.protected_elements.values() if e.element_type == element_type]
    
    def get_elements_by_character(self, character_id: str) -> List[ProtectedElement]:
        """Get all elements associated with a character"""
        return [e for e in self.protected_elements.values() 
                if character_id in e.associated_characters]
    
    def get_element_network(self, element_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get the network of connected elements for an element"""
        if element_id not in self.protected_elements:
            return {}
        
        visited = set()
        network = {"nodes": {}, "edges": []}
        
        def traverse(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return
            
            visited.add(current_id)
            
            if current_id in self.protected_elements:
                element = self.protected_elements[current_id]
                network["nodes"][current_id] = {
                    "title": element.title,
                    "type": element.element_type.value,
                    "importance": element.importance_level,
                    "status": element.resolution_status
                }
                
                # Add dependencies
                for dep_id in element.dependencies:
                    network["edges"].append({
                        "from": current_id,
                        "to": dep_id,
                        "type": "depends_on"
                    })
                    traverse(dep_id, depth + 1)
            
            # Add knowledge graph connections
            connected = self.knowledge_graph.get_connected_nodes(current_id)
            for connected_id in connected:
                if connected_id in self.protected_elements:
                    network["edges"].append({
                        "from": current_id,
                        "to": connected_id,
                        "type": "connected"
                    })
                    traverse(connected_id, depth + 1)
        
        traverse(element_id, 0)
        return network
    
    def validate_element_integrity(self) -> List[Dict[str, Any]]:
        """Validate integrity of protected elements"""
        issues = []
        
        for element in self.protected_elements.values():
            # Check for orphaned dependencies
            for dep_id in element.dependencies:
                if dep_id not in self.protected_elements:
                    issues.append({
                        "type": "orphaned_dependency",
                        "element_id": element.id,
                        "element_title": element.title,
                        "missing_dependency": dep_id,
                        "severity": "medium"
                    })
            
            # Check for unresolved high-importance elements
            if (element.importance_level >= 8 and 
                element.resolution_status in ["introduced", "developing"] and
                len(element.chapter_references) == 0):
                issues.append({
                    "type": "high_importance_unused",
                    "element_id": element.id,
                    "element_title": element.title,
                    "importance": element.importance_level,
                    "severity": "high"
                })
            
            # Check for abandoned elements
            days_since_update = (datetime.now() - element.last_modified).days
            if (days_since_update > 30 and 
                element.resolution_status == "introduced" and
                len(element.chapter_references) <= 1):
                issues.append({
                    "type": "potentially_abandoned",
                    "element_id": element.id,
                    "element_title": element.title,
                    "days_since_update": days_since_update,
                    "severity": "low"
                })
        
        return issues
    
    def generate_protection_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report on protected elements"""
        total_elements = len(self.protected_elements)
        by_type = {}
        by_status = {}
        by_importance = {i: 0 for i in range(1, 11)}
        
        for element in self.protected_elements.values():
            # Count by type
            type_name = element.element_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
            
            # Count by status
            by_status[element.resolution_status] = by_status.get(element.resolution_status, 0) + 1
            
            # Count by importance
            by_importance[element.importance_level] += 1
        
        # Get integrity issues
        integrity_issues = self.validate_element_integrity()
        
        return {
            "total_elements": total_elements,
            "by_type": by_type,
            "by_status": by_status,
            "by_importance": by_importance,
            "integrity_issues": len(integrity_issues),
            "critical_issues": len([i for i in integrity_issues if i["severity"] == "high"]),
            "knowledge_graph_nodes": len(self.knowledge_graph.nodes),
            "knowledge_graph_edges": sum(len(edges) for edges in self.knowledge_graph.edges.values())
        }