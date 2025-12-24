"""
Model-Agnostic Abstraction Layer
Provides unified interface for different AI models with graceful degradation
"""
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import aiohttp
import time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Supported model providers"""
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    MOCK = "mock"

class ModelCapability(Enum):
    """Model capabilities"""
    TEXT_GENERATION = "text_generation"
    SUMMARIZATION = "summarization"
    DIALOGUE = "dialogue"
    WORLD_BUILDING = "world_building"
    CHARACTER_DEVELOPMENT = "character_development"
    CONSISTENCY_CHECK = "consistency_check"

@dataclass
class ModelConfig:
    """Configuration for a model"""
    provider: ModelProvider
    model_name: str
    api_key: Optional[str]
    endpoint: Optional[str]
    max_tokens: int
    temperature: float
    capabilities: List[ModelCapability]
    cost_per_token: float
    rate_limit_rpm: int
    fallback_model: Optional[str]

@dataclass
class GenerationRequest:
    """Request for text generation"""
    prompt: str
    max_tokens: int
    temperature: float
    stop_sequences: List[str]
    context: Dict[str, Any]
    capability_needed: ModelCapability

@dataclass
class GenerationResponse:
    """Response from text generation"""
    generated_text: str
    model_used: str
    provider: ModelProvider
    tokens_used: int
    response_time_ms: float
    success: bool
    error_message: Optional[str]
    metadata: Dict[str, Any]

class CircuitBreaker:
    """Circuit breaker for handling API failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class ModelInterface(ABC):
    """Abstract interface for AI models"""
    
    @abstractmethod
    async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using the model"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[ModelCapability]:
        """Get model capabilities"""
        pass
    
    @abstractmethod
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost for token usage"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if model is healthy and available"""
        pass

class AzureOpenAIModel(ModelInterface):
    """Azure OpenAI model implementation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker()
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Azure OpenAI"""
        if not self.circuit_breaker.can_execute():
            return GenerationResponse(
                generated_text="",
                model_used=self.config.model_name,
                provider=self.config.provider,
                tokens_used=0,
                response_time_ms=0,
                success=False,
                error_message="Circuit breaker is open",
                metadata={}
            )
        
        start_time = time.time()
        
        try:
            session = await self._get_session()
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.config.api_key
            }
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a creative writing assistant."},
                    {"role": "user", "content": request.prompt}
                ],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stop": request.stop_sequences if request.stop_sequences else None
            }
            
            async with session.post(
                f"{self.config.endpoint}/openai/deployments/{self.config.model_name}/chat/completions?api-version=2024-02-15-preview",
                headers=headers,
                json=payload
            ) as response:
                
                response_time_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    generated_text = data["choices"][0]["message"]["content"]
                    tokens_used = data.get("usage", {}).get("total_tokens", 0)
                    
                    self.circuit_breaker.record_success()
                    
                    return GenerationResponse(
                        generated_text=generated_text,
                        model_used=self.config.model_name,
                        provider=self.config.provider,
                        tokens_used=tokens_used,
                        response_time_ms=response_time_ms,
                        success=True,
                        error_message=None,
                        metadata={
                            "finish_reason": data["choices"][0].get("finish_reason"),
                            "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                            "completion_tokens": data.get("usage", {}).get("completion_tokens", 0)
                        }
                    )
                else:
                    error_data = await response.json()
                    error_message = error_data.get("error", {}).get("message", "Unknown error")
                    
                    self.circuit_breaker.record_failure()
                    
                    return GenerationResponse(
                        generated_text="",
                        model_used=self.config.model_name,
                        provider=self.config.provider,
                        tokens_used=0,
                        response_time_ms=response_time_ms,
                        success=False,
                        error_message=f"HTTP {response.status}: {error_message}",
                        metadata={"status_code": response.status}
                    )
        
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.circuit_breaker.record_failure()
            
            return GenerationResponse(
                generated_text="",
                model_used=self.config.model_name,
                provider=self.config.provider,
                tokens_used=0,
                response_time_ms=response_time_ms,
                success=False,
                error_message=str(e),
                metadata={"exception_type": type(e).__name__}
            )
    
    def get_capabilities(self) -> List[ModelCapability]:
        """Get Azure OpenAI capabilities"""
        return self.config.capabilities
    
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost for token usage"""
        return tokens * self.config.cost_per_token
    
    def health_check(self) -> bool:
        """Check Azure OpenAI health"""
        return self.circuit_breaker.state != "open"

class MockModel(ModelInterface):
    """Mock model for testing"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate mock text"""
        await asyncio.sleep(0.1)  # Simulate API delay
        
        mock_text = f"[MOCK GENERATION] Request: {request.prompt[:50]}..."
        
        return GenerationResponse(
            generated_text=mock_text,
            model_used=self.config.model_name,
            provider=self.config.provider,
            tokens_used=len(mock_text.split()),
            response_time_ms=100,
            success=True,
            error_message=None,
            metadata={"mock": True}
        )
    
    def get_capabilities(self) -> List[ModelCapability]:
        """Get mock capabilities"""
        return self.config.capabilities
    
    def estimate_cost(self, tokens: int) -> float:
        """Estimate mock cost"""
        return 0.0
    
    def health_check(self) -> bool:
        """Mock health check"""
        return True

class ModelRouter:
    """Routes requests to appropriate models with fallback logic"""
    
    def __init__(self):
        self.models: Dict[str, ModelInterface] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.routing_rules: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {}
    
    def register_model(self, model_id: str, model: ModelInterface, config: ModelConfig):
        """Register a model with the router"""
        self.models[model_id] = model
        self.model_configs[model_id] = config
        self.performance_metrics[model_id] = []
        
        logger.info(f"Registered model {model_id} ({config.provider.value})")
    
    def add_routing_rule(self, capability: ModelCapability, 
                        preferred_model: str, fallback_models: List[str],
                        conditions: Dict[str, Any] = None):
        """Add a routing rule for specific capabilities"""
        if conditions is None:
            conditions = {}
        
        rule = {
            "capability": capability,
            "preferred_model": preferred_model,
            "fallback_models": fallback_models,
            "conditions": conditions
        }
        
        self.routing_rules.append(rule)
        logger.info(f"Added routing rule for {capability.value}")
    
    def _select_model(self, request: GenerationRequest) -> str:
        """Select the best model for a request"""
        # Find applicable routing rules
        applicable_rules = [
            rule for rule in self.routing_rules
            if rule["capability"] == request.capability_needed
        ]
        
        if not applicable_rules:
            # No specific rule, use first available model with capability
            for model_id, model in self.models.items():
                if request.capability_needed in model.get_capabilities():
                    return model_id
            
            # Fallback to any model
            return list(self.models.keys())[0] if self.models else None
        
        # Use first applicable rule
        rule = applicable_rules[0]
        
        # Check if preferred model is healthy and meets conditions
        preferred = rule["preferred_model"]
        if preferred in self.models and self.models[preferred].health_check():
            # Check conditions
            conditions = rule["conditions"]
            
            if "max_tokens" in conditions and request.max_tokens > conditions["max_tokens"]:
                # Preferred model can't handle this request size
                pass
            else:
                return preferred
        
        # Try fallback models
        for fallback in rule["fallback_models"]:
            if fallback in self.models and self.models[fallback].health_check():
                return fallback
        
        # Last resort: any healthy model
        for model_id, model in self.models.items():
            if model.health_check():
                return model_id
        
        return None
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using the best available model"""
        selected_model_id = self._select_model(request)
        
        if selected_model_id is None:
            return GenerationResponse(
                generated_text="",
                model_used="none",
                provider=ModelProvider.MOCK,
                tokens_used=0,
                response_time_ms=0,
                success=False,
                error_message="No healthy models available",
                metadata={}
            )
        
        model = self.models[selected_model_id]
        
        start_time = time.time()
        response = await model.generate_text(request)
        
        # Record performance metrics
        if response.success:
            self.performance_metrics[selected_model_id].append(response.response_time_ms)
            
            # Keep only recent metrics
            if len(self.performance_metrics[selected_model_id]) > 100:
                self.performance_metrics[selected_model_id] = self.performance_metrics[selected_model_id][-100:]
        
        logger.info(f"Generated text using {selected_model_id}: success={response.success}, tokens={response.tokens_used}")
        
        return response
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all registered models"""
        status = {}
        
        for model_id, model in self.models.items():
            config = self.model_configs[model_id]
            metrics = self.performance_metrics[model_id]
            
            avg_response_time = sum(metrics) / len(metrics) if metrics else 0
            
            status[model_id] = {
                "provider": config.provider.value,
                "model_name": config.model_name,
                "healthy": model.health_check(),
                "capabilities": [cap.value for cap in model.get_capabilities()],
                "average_response_time_ms": avg_response_time,
                "total_requests": len(metrics),
                "rate_limit_rpm": config.rate_limit_rpm,
                "cost_per_token": config.cost_per_token
            }
        
        return status
    
    def optimize_routing(self):
        """Optimize routing rules based on performance data"""
        # Analyze performance metrics
        model_performance = {}
        
        for model_id, metrics in self.performance_metrics.items():
            if metrics:
                model_performance[model_id] = {
                    "avg_response_time": sum(metrics) / len(metrics),
                    "success_rate": len([m for m in metrics if m > 0]) / len(metrics),
                    "total_requests": len(metrics)
                }
        
        # Sort models by performance score
        def performance_score(model_id):
            perf = model_performance.get(model_id, {"avg_response_time": 10000, "success_rate": 0})
            return perf["success_rate"] * 100 - perf["avg_response_time"] / 1000
        
        best_models = sorted(model_performance.keys(), key=performance_score, reverse=True)
        
        # Update routing rules to prefer better performing models
        for rule in self.routing_rules:
            # Reorder fallback models by performance
            all_models = [rule["preferred_model"]] + rule["fallback_models"]
            capable_models = [
                model_id for model_id in all_models 
                if model_id in self.models and 
                rule["capability"] in self.models[model_id].get_capabilities()
            ]
            
            # Sort by performance
            capable_models.sort(key=performance_score, reverse=True)
            
            if capable_models:
                rule["preferred_model"] = capable_models[0]
                rule["fallback_models"] = capable_models[1:]
        
        logger.info("Optimized routing rules based on performance data")

class ModelManager:
    """High-level manager for model abstraction layer"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.router = ModelRouter()
        self.config_path = config_path
        
        if config_path and config_path.exists():
            self.load_configuration()
    
    def load_configuration(self):
        """Load model configuration from file"""
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
        
        # Register models from config
        for model_config in config_data.get("models", []):
            config = ModelConfig(
                provider=ModelProvider(model_config["provider"]),
                model_name=model_config["model_name"],
                api_key=model_config.get("api_key"),
                endpoint=model_config.get("endpoint"),
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
                capabilities=[ModelCapability(cap) for cap in model_config["capabilities"]],
                cost_per_token=model_config["cost_per_token"],
                rate_limit_rpm=model_config["rate_limit_rpm"],
                fallback_model=model_config.get("fallback_model")
            )
            
            # Create model instance based on provider
            if config.provider == ModelProvider.AZURE_OPENAI:
                model = AzureOpenAIModel(config)
            elif config.provider == ModelProvider.MOCK:
                model = MockModel(config)
            else:
                logger.warning(f"Unsupported provider: {config.provider}")
                continue
            
            self.router.register_model(model_config["id"], model, config)
        
        # Add routing rules from config
        for rule_config in config_data.get("routing_rules", []):
            self.router.add_routing_rule(
                capability=ModelCapability(rule_config["capability"]),
                preferred_model=rule_config["preferred_model"],
                fallback_models=rule_config["fallback_models"],
                conditions=rule_config.get("conditions", {})
            )
    
    async def generate_text(self, prompt: str, capability: ModelCapability,
                          max_tokens: int = 1000, temperature: float = 0.7,
                          context: Dict[str, Any] = None) -> GenerationResponse:
        """High-level text generation interface"""
        if context is None:
            context = {}
        
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=[],
            context=context,
            capability_needed=capability
        )
        
        return await self.router.generate(request)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        model_status = self.router.get_model_status()
        
        healthy_models = sum(1 for status in model_status.values() if status["healthy"])
        total_models = len(model_status)
        
        return {
            "total_models": total_models,
            "healthy_models": healthy_models,
            "system_health": "good" if healthy_models == total_models else "degraded" if healthy_models > 0 else "critical",
            "models": model_status,
            "routing_rules": len(self.router.routing_rules)
        }
    
    async def close(self):
        """Clean up resources"""
        for model in self.router.models.values():
            if hasattr(model, 'session') and model.session:
                await model.session.close()