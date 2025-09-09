"""
MeshAI Core Library
Shared utilities and libraries for MeshAI services
"""

__version__ = "1.0.0"
__author__ = "MeshAI Labs"

# Import key components for easier access
from .auth.api_key_validator import (
    MeshAIAPIKeyValidator,
    ValidationResult,
    APIKeyType,
    CircuitBreaker
)

__all__ = [
    "MeshAIAPIKeyValidator",
    "ValidationResult", 
    "APIKeyType",
    "CircuitBreaker"
]