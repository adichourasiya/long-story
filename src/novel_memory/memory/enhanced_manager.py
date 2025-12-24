"""
Enhanced Memory Manager
Integrates all memory components with the new hierarchical architecture
"""
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from .hierarchical_memory import HierarchicalMemorySystem, MemoryTier, ProtectionLevel, MemoryNode
from .protected_system import ProtectedInformationSystem, ProtectedElementType
from .adaptive_summarizer import AdaptiveSummarizer, SummaryContext, CompressionLevel
from ..observability.system import ObservabilitySystem, EventType, Severity
from ..models.abstraction_layer import ModelManager, ModelCapability

logger = logging.getLogger(__name__)

class EnhancedMemoryManager:
    """
    Unified memory management system integrating all components
    """
    
    def __init__(self, base_path: Path, model_manager: ModelManager):
        self.base_path = Path(base_path)
        self.model_manager = model_manager
        
        # Initialize subsystems
        self.hierarchical_memory = HierarchicalMemorySystem(
            base_path=self.base_path / "hierarchical",
            chromadb_path=self.base_path / "chromadb"
        )
        
        self.protected_system = ProtectedInformationSystem(
            base_path=self.base_path / "protected"
        )
        
        self.adaptive_summarizer = AdaptiveSummarizer()
        
        self.observability = ObservabilitySystem(
            base_path=self.base_path / "logs",
            enable_detailed_logging=True
        )
        
        # Memory coordination state
        self.active_chapter_id: Optional[str] = None
        self.generation_session_id: Optional[str] = None
        
        # Configuration
        self.config = {
            "auto_summarization_threshold": 5000,  # words
            "consistency_check_frequency": 5,       # chapters
            "max_working_memory_items": 50,
            "protection_scan_enabled": True,
            "adaptive_compression": True
        }
        
        logger.info("Enhanced Memory Manager initialized")
    
    async def start_chapter_generation(self, chapter_id: str, 
                                     chapter_title: str = "",
                                     expected_length: int = 3000) -> str:
        """Start a new chapter generation session"""
        self.active_chapter_id = chapter_id
        
        # Start observability session
        self.generation_session_id = self.observability.start_generation_session(chapter_id)
        
        with self.observability.track_operation("start_chapter", "memory_manager", 
                                               {"chapter_id": chapter_id, "title": chapter_title}):
            
            # Prepare chapter context
            context = await self._prepare_chapter_context(chapter_id, expected_length)
            
            # Check for protected elements relevant to this chapter
            critical_elements = self.protected_system.get_critical_elements_for_chapter(chapter_id)
            
            # Store chapter start in working memory
            chapter_start_memory = f"Starting chapter: {chapter_title} (ID: {chapter_id})"
            self.hierarchical_memory.add_memory(
                content=chapter_start_memory,
                tier=MemoryTier.WORKING,
                protection_level=ProtectionLevel.IMPORTANT,
                tags={"chapter_start", chapter_id},
                importance_score=0.8
            )
            
            self.observability.log_event(
                event_type=EventType.STATE_CHANGE,
                severity=Severity.INFO,
                component="memory_manager",
                operation="chapter_started",
                metadata={
                    "chapter_id": chapter_id,
                    "critical_elements_count": len(critical_elements),
                    "context_size": len(str(context))
                }
            )
            
            return self.generation_session_id
    
    async def _prepare_chapter_context(self, chapter_id: str, expected_length: int) -> Dict[str, Any]:
        """Prepare comprehensive context for chapter generation"""
        with self.observability.track_operation("prepare_context", "memory_manager"):
            
            # Get relevant memories from hierarchical system
            context_query = f"chapter {chapter_id} narrative context"
            relevant_memories = self.hierarchical_memory.query_memories(
                query=context_query,
                tiers=[MemoryTier.WORKING, MemoryTier.EPISODIC, MemoryTier.SEMANTIC],
                limit=30,
                relevance_threshold=0.6
            )
            
            # Get critical protected elements
            critical_elements = self.protected_system.get_critical_elements_for_chapter(chapter_id)
            
            # Get unresolved elements that might need attention
            unresolved_elements = self.protected_system.get_unresolved_elements()
            
            # Build context
            context = {
                "chapter_info": {
                    "chapter_id": chapter_id,
                    "expected_length": expected_length,
                    "preparation_time": datetime.now().isoformat()
                },
                "narrative_memories": [
                    {
                        "content": memory.content,
                        "importance": memory.importance_score,
                        "protection_level": memory.protection_level.value,
                        "tags": list(memory.tags)
                    }
                    for memory in relevant_memories
                ],
                "critical_elements": [
                    {
                        "title": elem.title,
                        "description": elem.description,
                        "type": elem.element_type.value,
                        "status": elem.resolution_status,
                        "importance": elem.importance_level
                    }
                    for elem in critical_elements
                ],
                "unresolved_elements": [
                    {
                        "title": elem.title,
                        "type": elem.element_type.value,
                        "importance": elem.importance_level,
                        "days_since_creation": (datetime.now() - elem.creation_time).days
                    }
                    for elem in unresolved_elements[:5]  # Top 5 most important
                ],
                "character_states": dict(self.hierarchical_memory.character_states),
                "active_arcs": [
                    arc for arc in self.hierarchical_memory.protected_arcs.values()
                    if arc.current_status == "active"
                ]
            }
            
            return context
    
    async def process_generated_content(self, content: str, 
                                      content_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process newly generated content through the memory system"""
        if content_metadata is None:
            content_metadata = {}
        
        if not self.active_chapter_id:
            raise ValueError("No active chapter session")
        
        with self.observability.track_operation("process_content", "memory_manager", 
                                               {"content_length": len(content)}):
            
            processing_results = {}
            
            # 1. Analyze content for summarization needs
            summary_context = self.adaptive_summarizer.analyze_content(
                content=content,
                chapter_id=self.active_chapter_id,
                protected_elements=[elem.title for elem in 
                                  self.protected_system.get_critical_elements_for_chapter(self.active_chapter_id)],
                key_characters=list(self.hierarchical_memory.character_states.keys())
            )
            
            # 2. Scan for protected elements
            if self.config["protection_scan_enabled"]:
                protection_scan = await self._scan_for_protected_elements(content)
                processing_results["protection_scan"] = protection_scan
            
            # 3. Update character states
            character_updates = await self._extract_character_updates(content)
            processing_results["character_updates"] = character_updates
            
            # 4. Add to working memory
            memory_id = self.hierarchical_memory.add_memory(
                content=content,
                tier=MemoryTier.WORKING,
                protection_level=ProtectionLevel.NORMAL,
                tags={self.active_chapter_id, "generated_content"},
                importance_score=0.6
            )
            
            # 5. Check if summarization is needed
            if len(content.split()) > self.config["auto_summarization_threshold"]:
                summary_result = await self._auto_summarize_content(summary_context)
                processing_results["summarization"] = summary_result
            
            # 6. Update session metrics
            if self.generation_session_id:
                self.observability.add_session_event(
                    self.generation_session_id, 
                    "content_processed", 
                    {"word_count": len(content.split()), "memory_id": memory_id}
                )
            
            self.observability.log_event(
                event_type=EventType.STATE_CHANGE,
                severity=Severity.INFO,
                component="memory_manager",
                operation="content_processed",
                metadata={
                    "chapter_id": self.active_chapter_id,
                    "content_word_count": len(content.split()),
                    "memory_id": memory_id,
                    "processing_components": list(processing_results.keys())
                }
            )
            
            return processing_results
    
    async def _scan_for_protected_elements(self, content: str) -> Dict[str, Any]:
        """Scan content for new protected elements to track"""
        scan_results = {
            "new_elements_found": [],
            "existing_elements_referenced": [],
            "recommendations": []
        }
        
        # Simple keyword-based detection (could be enhanced with ML)
        element_keywords = {
            ProtectedElementType.CHEKOVS_GUN: ["weapon", "item", "object", "mysterious", "ancient"],
            ProtectedElementType.FORESHADOWING: ["hint", "foreshadow", "ominous", "portent", "sign"],
            ProtectedElementType.SECRET: ["secret", "hidden", "concealed", "whispered", "confidential"],
            ProtectedElementType.PROPHECY: ["prophecy", "foretold", "destiny", "fate", "oracle"],
            ProtectedElementType.RECURRING_MOTIF: ["symbol", "pattern", "recurring", "repeated", "motif"]
        }
        
        content_lower = content.lower()
        
        for element_type, keywords in element_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    # Extract surrounding context
                    sentences = content.split('. ')
                    relevant_sentences = [s for s in sentences if keyword in s.lower()]
                    
                    if relevant_sentences:
                        scan_results["recommendations"].append({
                            "type": element_type.value,
                            "keyword": keyword,
                            "context": relevant_sentences[0][:200] + "...",
                            "suggestion": f"Consider tracking this {element_type.value.replace('_', ' ')}"
                        })
        
        # Check references to existing elements
        for element in self.protected_system.protected_elements.values():
            if element.title.lower() in content_lower:
                scan_results["existing_elements_referenced"].append({
                    "element_id": element.id,
                    "title": element.title,
                    "type": element.element_type.value
                })
                
                # Update element's chapter references
                self.protected_system.add_chapter_reference(
                    element.id, 
                    self.active_chapter_id
                )
        
        return scan_results
    
    async def _extract_character_updates(self, content: str) -> Dict[str, Any]:
        """Extract character development information from content"""
        updates = {}
        
        for char_id in self.hierarchical_memory.character_states.keys():
            char_mentions = []
            
            # Find character mentions
            sentences = content.split('. ')
            for sentence in sentences:
                if char_id.lower() in sentence.lower():
                    char_mentions.append(sentence.strip())
            
            if char_mentions:
                # Use AI to analyze character development
                analysis_prompt = f"""
                Analyze the following character mentions for {char_id} and identify:
                1. Emotional state changes
                2. Relationship developments
                3. Character growth or regression
                4. New traits or behaviors
                
                Character mentions:
                {' | '.join(char_mentions)}
                
                Provide analysis in JSON format with keys: emotions, relationships, growth, traits
                """
                
                try:
                    analysis_response = await self.model_manager.generate_text(
                        prompt=analysis_prompt,
                        capability=ModelCapability.CHARACTER_DEVELOPMENT,
                        max_tokens=500,
                        temperature=0.3
                    )
                    
                    if analysis_response.success:
                        # Parse AI analysis (simplified - would need robust JSON parsing)
                        updates[char_id] = {
                            "mentions_count": len(char_mentions),
                            "ai_analysis": analysis_response.generated_text,
                            "last_appearance": self.active_chapter_id
                        }
                        
                        # Update character state in memory
                        self.hierarchical_memory.update_character_state(
                            char_id,
                            last_appearance=self.active_chapter_id
                        )
                
                except Exception as e:
                    logger.warning(f"Failed to analyze character {char_id}: {e}")
                    updates[char_id] = {
                        "mentions_count": len(char_mentions),
                        "error": str(e)
                    }
        
        return updates
    
    async def _auto_summarize_content(self, context: SummaryContext) -> Dict[str, Any]:
        """Auto-summarize content when it exceeds threshold"""
        with self.observability.track_operation("auto_summarize", "memory_manager"):
            
            if self.config["adaptive_compression"]:
                summary = self.adaptive_summarizer.create_layered_summary(context)
            else:
                # Fallback to simple summarization
                words = context.content.split()
                target_length = len(words) // 3  # 33% compression
                summary = " ".join(words[:target_length]) + "..."
            
            # Store summary in episodic memory
            summary_memory_id = self.hierarchical_memory.add_memory(
                content=f"[SUMMARY] {summary}",
                tier=MemoryTier.EPISODIC,
                protection_level=ProtectionLevel.IMPORTANT,
                tags={context.chapter_id, "summary", "auto_generated"},
                importance_score=0.7
            )
            
            return {
                "summary": summary,
                "original_length": context.word_count,
                "summary_length": len(summary.split()),
                "compression_ratio": len(summary.split()) / context.word_count,
                "summary_memory_id": summary_memory_id
            }
    
    async def run_consistency_check(self, scope: str = "full") -> Dict[str, Any]:
        """Run comprehensive consistency check"""
        with self.observability.track_operation("consistency_check", "memory_manager", 
                                               {"scope": scope}):
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "scope": scope,
                "violations": [],
                "warnings": [],
                "recommendations": []
            }
            
            # 1. Check hierarchical memory consistency
            memory_violations = self.hierarchical_memory.check_consistency()
            results["violations"].extend([
                {
                    "source": "hierarchical_memory",
                    "type": v.violation_type,
                    "description": v.description,
                    "severity": v.severity,
                    "suggestions": v.suggestions
                }
                for v in memory_violations
            ])
            
            # 2. Check protected element integrity
            protection_issues = self.protected_system.validate_element_integrity()
            results["warnings"].extend([
                {
                    "source": "protected_system",
                    "type": issue["type"],
                    "description": f"Element '{issue.get('element_title', 'unknown')}': {issue['type']}",
                    "severity": issue["severity"]
                }
                for issue in protection_issues
            ])
            
            # 3. Cross-system consistency checks
            cross_system_issues = await self._check_cross_system_consistency()
            results["violations"].extend(cross_system_issues)
            
            # 4. Generate recommendations
            if scope == "full":
                recommendations = await self._generate_consistency_recommendations(results)
                results["recommendations"] = recommendations
            
            self.observability.log_event(
                event_type=EventType.CONSISTENCY_CHECK,
                severity=Severity.WARNING if results["violations"] else Severity.INFO,
                component="memory_manager",
                operation="consistency_check",
                metadata={
                    "violations_count": len(results["violations"]),
                    "warnings_count": len(results["warnings"]),
                    "scope": scope
                }
            )
            
            return results
    
    async def _check_cross_system_consistency(self) -> List[Dict[str, Any]]:
        """Check consistency between different memory systems"""
        issues = []
        
        # Check if protected elements are properly referenced in hierarchical memory
        for element in self.protected_system.protected_elements.values():
            if element.chapter_references:
                # Search for mentions in hierarchical memory
                search_results = self.hierarchical_memory.query_memories(
                    query=element.title,
                    tiers=[MemoryTier.EPISODIC, MemoryTier.SEMANTIC],
                    limit=5
                )
                
                if not search_results:
                    issues.append({
                        "source": "cross_system",
                        "type": "missing_memory_reference",
                        "description": f"Protected element '{element.title}' has chapter references but no corresponding memories",
                        "severity": "medium",
                        "suggestions": [
                            f"Add memories for protected element '{element.title}'",
                            f"Verify chapter references for element '{element.title}'"
                        ]
                    })
        
        return issues
    
    async def _generate_consistency_recommendations(self, check_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate AI-powered recommendations for consistency issues"""
        if not check_results["violations"] and not check_results["warnings"]:
            return [{"type": "success", "message": "No consistency issues found"}]
        
        recommendations = []
        
        # Group issues by type
        issue_types = {}
        for violation in check_results["violations"]:
            issue_type = violation["type"]
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(violation)
        
        # Generate recommendations for each issue type
        for issue_type, violations in issue_types.items():
            if len(violations) > 2:  # Pattern detected
                recommendations.append({
                    "type": "pattern_detected",
                    "issue_type": issue_type,
                    "count": len(violations),
                    "recommendation": f"Multiple {issue_type} issues detected. Consider systematic review of {issue_type.replace('_', ' ')}.",
                    "priority": "high" if len(violations) > 5 else "medium"
                })
        
        return recommendations
    
    async def end_chapter_generation(self, final_word_count: int = 0, 
                                   model_calls: int = 0, 
                                   total_tokens: int = 0) -> Dict[str, Any]:
        """End the current chapter generation session"""
        if not self.active_chapter_id or not self.generation_session_id:
            raise ValueError("No active chapter session")
        
        with self.observability.track_operation("end_chapter", "memory_manager"):
            
            # Run final consistency check
            consistency_results = await self.run_consistency_check("chapter")
            
            # Promote working memory to appropriate tiers
            self._promote_session_memories()
            
            # Save all system states
            self.hierarchical_memory.save_persistent_state()
            self.protected_system.save_protected_state()
            
            # End observability session
            self.observability.end_generation_session(
                self.generation_session_id,
                word_count=final_word_count,
                model_calls=model_calls,
                total_tokens=total_tokens
            )
            
            results = {
                "chapter_id": self.active_chapter_id,
                "session_id": self.generation_session_id,
                "final_word_count": final_word_count,
                "consistency_check": consistency_results,
                "memory_metrics": self.hierarchical_memory.get_metrics(),
                "protection_report": self.protected_system.generate_protection_report(),
                "summarization_metrics": self.adaptive_summarizer.get_summarization_metrics()
            }
            
            # Reset session state
            self.active_chapter_id = None
            self.generation_session_id = None
            
            return results
    
    def _promote_session_memories(self):
        """Promote important working memories to permanent storage"""
        important_memories = [
            memory for memory in self.hierarchical_memory.working_memory.values()
            if memory.protection_level in [ProtectionLevel.CRITICAL, ProtectionLevel.IMPORTANT]
        ]
        
        for memory in important_memories:
            # Promote to episodic memory
            memory.tier = MemoryTier.EPISODIC
            self.hierarchical_memory._store_in_chromadb(memory)
        
        # Clear working memory of promoted items
        for memory in important_memories:
            if memory.id in self.hierarchical_memory.working_memory:
                del self.hierarchical_memory.working_memory[memory.id]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "active_chapter": self.active_chapter_id,
            "active_session": self.generation_session_id,
            "memory_system": self.hierarchical_memory.get_metrics(),
            "protection_system": self.protected_system.generate_protection_report(),
            "summarization_metrics": self.adaptive_summarizer.get_summarization_metrics(),
            "system_health": self.observability.get_system_health(),
            "model_status": self.model_manager.get_system_status()
        }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            # Save final states
            self.hierarchical_memory.save_persistent_state()
            self.protected_system.save_protected_state()
            
            # Close model manager
            await self.model_manager.close()
            
            logger.info("Enhanced Memory Manager cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise