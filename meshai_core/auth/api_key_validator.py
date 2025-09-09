"""
MeshAI Shared API Key Validation Library
Provides consistent API key validation across all services with caching and connection pooling
"""

import os
import logging
import hashlib
import asyncio
from typing import Dict, Optional, Any, Tuple, List
from datetime import datetime, timedelta
from enum import Enum
import asyncpg
import bcrypt
import redis.asyncio as redis
from cachetools import TTLCache
import json

logger = logging.getLogger(__name__)


class APIKeyType(Enum):
    """Types of API keys in the system"""
    USER = "user"  # User API keys (msk_*)
    SERVICE = "service"  # Internal service keys
    ADMIN = "admin"  # Admin/operational keys


class ValidationResult:
    """Result of API key validation"""
    def __init__(self, 
                 valid: bool,
                 key_type: Optional[APIKeyType] = None,
                 user_id: Optional[str] = None,
                 tenant_id: Optional[str] = None,
                 email: Optional[str] = None,
                 permissions: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None,
                 cached: bool = False):
        self.valid = valid
        self.key_type = key_type
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.email = email
        self.permissions = permissions or {}
        self.error = error
        self.cached = cached
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching"""
        return {
            "valid": self.valid,
            "key_type": self.key_type.value if self.key_type else None,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "email": self.email,
            "permissions": self.permissions,
            "error": self.error,
            "cached": self.cached
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create from dictionary (from cache)"""
        return cls(
            valid=data["valid"],
            key_type=APIKeyType(data["key_type"]) if data.get("key_type") else None,
            user_id=data.get("user_id"),
            tenant_id=data.get("tenant_id"),
            email=data.get("email"),
            permissions=data.get("permissions", {}),
            error=data.get("error"),
            cached=True
        )


class CircuitBreaker:
    """Circuit breaker for database failures"""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.state == "open":
            if self.last_failure_time:
                time_since_failure = (datetime.utcnow() - self.last_failure_time).seconds
                if time_since_failure > self.recovery_timeout:
                    self.state = "half-open"
                    return False
            return True
        return False
    
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open (testing recovery)"""
        return self.state == "half-open"


class MeshAIAPIKeyValidator:
    """
    Shared API key validator for all MeshAI services
    Features:
    - 16-character prefix validation
    - Bcrypt hash verification
    - Multi-layer caching (local + Redis)
    - Connection pooling
    - Circuit breaker for reliability
    - Support for user, service, and admin keys
    """
    
    # Configuration
    API_KEY_PREFIX = "msk_"
    PREFIX_LENGTH = 16  # 16 characters for security
    BCRYPT_ROUNDS = 12
    
    # Cache configuration
    LOCAL_CACHE_SIZE = 1000
    LOCAL_CACHE_TTL = 60  # 1 minute local cache
    REDIS_CACHE_TTL = 300  # 5 minutes Redis cache
    NEGATIVE_CACHE_TTL = 30  # 30 seconds for failed validations
    
    def __init__(self,
                 db_url: Optional[str] = None,
                 redis_url: Optional[str] = None,
                 service_keys: Optional[List[str]] = None,
                 admin_keys: Optional[List[str]] = None,
                 enable_caching: bool = True,
                 enable_circuit_breaker: bool = True):
        """
        Initialize the validator
        
        Args:
            db_url: PostgreSQL connection URL
            redis_url: Redis connection URL
            service_keys: List of valid service-to-service keys
            admin_keys: List of admin/operational keys
            enable_caching: Enable caching layers
            enable_circuit_breaker: Enable circuit breaker for DB failures
        """
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.service_keys = set(service_keys or os.getenv("SERVICE_API_KEYS", "").split(","))
        self.admin_keys = set(admin_keys or os.getenv("ADMIN_API_KEYS", "").split(","))
        self.enable_caching = enable_caching
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Remove empty strings from key sets
        self.service_keys.discard("")
        self.admin_keys.discard("")
        
        # Connection pools
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Local cache
        self.local_cache = TTLCache(
            maxsize=self.LOCAL_CACHE_SIZE,
            ttl=self.LOCAL_CACHE_TTL
        ) if enable_caching else None
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        
        # Statistics
        self.stats = {
            "validations": 0,
            "cache_hits": 0,
            "db_queries": 0,
            "failures": 0
        }
    
    async def initialize(self):
        """Initialize database and Redis connections"""
        try:
            # Initialize database pool
            if self.db_url:
                self.db_pool = await asyncpg.create_pool(
                    self.db_url,
                    min_size=2,
                    max_size=10,
                    command_timeout=10
                )
                logger.info("Database connection pool initialized")
            
            # Initialize Redis client
            if self.redis_url and self.enable_caching:
                self.redis_client = await redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Redis connection initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            # Don't fail if Redis is unavailable, just disable distributed caching
            if "redis" in str(e).lower():
                self.redis_client = None
                logger.warning("Redis unavailable, using local cache only")
            else:
                raise
    
    async def close(self):
        """Close all connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.close()
    
    def _get_cache_key(self, api_key: str) -> str:
        """Generate cache key for API key"""
        # Use hash for security (don't store raw keys in cache)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return f"api_key:v1:{key_hash[:16]}"
    
    async def _get_from_cache(self, api_key: str) -> Optional[ValidationResult]:
        """Get validation result from cache"""
        if not self.enable_caching:
            return None
        
        cache_key = self._get_cache_key(api_key)
        
        # Check local cache first
        if self.local_cache and cache_key in self.local_cache:
            self.stats["cache_hits"] += 1
            logger.debug(f"Local cache hit for key {api_key[:16]}...")
            return ValidationResult.from_dict(self.local_cache[cache_key])
        
        # Check Redis cache
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    self.stats["cache_hits"] += 1
                    logger.debug(f"Redis cache hit for key {api_key[:16]}...")
                    result = ValidationResult.from_dict(json.loads(cached_data))
                    
                    # Update local cache
                    if self.local_cache:
                        self.local_cache[cache_key] = result.to_dict()
                    
                    return result
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        return None
    
    async def _set_cache(self, api_key: str, result: ValidationResult):
        """Set validation result in cache"""
        if not self.enable_caching:
            return
        
        cache_key = self._get_cache_key(api_key)
        result_dict = result.to_dict()
        
        # Determine TTL based on result
        ttl = self.REDIS_CACHE_TTL if result.valid else self.NEGATIVE_CACHE_TTL
        
        # Update local cache
        if self.local_cache:
            self.local_cache[cache_key] = result_dict
        
        # Update Redis cache
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(result_dict)
                )
            except Exception as e:
                logger.warning(f"Failed to set Redis cache: {e}")
    
    async def _validate_user_key_from_db(self, api_key: str) -> ValidationResult:
        """Validate user API key against database"""
        if not self.db_pool:
            return ValidationResult(
                valid=False,
                error="Database connection not available"
            )
        
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.is_open():
            logger.warning("Circuit breaker open, database unavailable")
            return ValidationResult(
                valid=False,
                error="Service temporarily unavailable"
            )
        
        try:
            # Extract prefix (16 characters)
            key_prefix = api_key[:self.PREFIX_LENGTH] if len(api_key) >= self.PREFIX_LENGTH else api_key
            
            async with self.db_pool.acquire() as conn:
                # Query for API keys with matching prefix
                # Note: Column names match the actual database schema
                query = """
                    SELECT 
                        ak.id as api_key_id,
                        ak."keyHash" as key_hash,
                        ak."keyPrefix" as key_prefix,
                        ak.permissions,
                        ak.status as key_status,
                        ak."expiresAt" as expires_at,
                        u.id as user_id,
                        u.email as user_email,
                        u.status as user_status
                    FROM api_keys ak
                    JOIN users u ON ak."userId" = u.id
                    WHERE ak."keyPrefix" = $1
                        AND ak.status = 'ACTIVE'
                        AND u.status = 'ACTIVE'
                        AND (ak."expiresAt" IS NULL OR ak."expiresAt" > NOW())
                """
                
                rows = await conn.fetch(query, key_prefix)
                
                # Check each potential match with bcrypt
                for row in rows:
                    try:
                        # Verify the full key against the hash
                        if bcrypt.checkpw(api_key.encode(), row['key_hash'].encode()):
                            # Parse permissions if stored as JSON
                            permissions = {}
                            if row['permissions']:
                                try:
                                    permissions = json.loads(row['permissions'])
                                except:
                                    permissions = {"raw": row['permissions']}
                            
                            # Record success
                            if self.circuit_breaker:
                                self.circuit_breaker.record_success()
                            
                            return ValidationResult(
                                valid=True,
                                key_type=APIKeyType.USER,
                                user_id=str(row['user_id']),
                                email=row['user_email'],
                                permissions=permissions
                            )
                    except Exception as e:
                        logger.warning(f"Error checking bcrypt hash: {e}")
                        continue
                
                # No valid match found
                return ValidationResult(
                    valid=False,
                    error="Invalid API key"
                )
        
        except Exception as e:
            logger.error(f"Database validation error: {e}")
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            return ValidationResult(
                valid=False,
                error="Validation service error"
            )
        finally:
            self.stats["db_queries"] += 1
    
    async def validate(self, api_key: str) -> ValidationResult:
        """
        Validate an API key
        
        Args:
            api_key: The API key to validate
            
        Returns:
            ValidationResult with validation status and metadata
        """
        self.stats["validations"] += 1
        
        # Basic validation
        if not api_key:
            self.stats["failures"] += 1
            return ValidationResult(valid=False, error="API key required")
        
        # Check cache first
        cached_result = await self._get_from_cache(api_key)
        if cached_result:
            return cached_result
        
        # Determine key type and validate accordingly
        result = None
        
        # Check if it's a user API key
        if api_key.startswith(self.API_KEY_PREFIX):
            result = await self._validate_user_key_from_db(api_key)
        
        # Check if it's a service key
        elif api_key in self.service_keys:
            result = ValidationResult(
                valid=True,
                key_type=APIKeyType.SERVICE
            )
        
        # Check if it's an admin key
        elif api_key in self.admin_keys:
            result = ValidationResult(
                valid=True,
                key_type=APIKeyType.ADMIN
            )
        
        # Invalid key format
        else:
            result = ValidationResult(
                valid=False,
                error="Invalid API key format"
            )
        
        # Update cache with result
        await self._set_cache(api_key, result)
        
        # Update stats
        if not result.valid:
            self.stats["failures"] += 1
        
        return result
    
    async def invalidate_cache(self, api_key: str):
        """Invalidate cache for a specific API key"""
        if not self.enable_caching:
            return
        
        cache_key = self._get_cache_key(api_key)
        
        # Remove from local cache
        if self.local_cache and cache_key in self.local_cache:
            del self.local_cache[cache_key]
        
        # Remove from Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(cache_key)
            except Exception as e:
                logger.warning(f"Failed to invalidate Redis cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        stats = self.stats.copy()
        if self.stats["validations"] > 0:
            stats["cache_hit_rate"] = self.stats["cache_hits"] / self.stats["validations"]
            stats["failure_rate"] = self.stats["failures"] / self.stats["validations"]
        else:
            stats["cache_hit_rate"] = 0
            stats["failure_rate"] = 0
        
        if self.circuit_breaker:
            stats["circuit_breaker_state"] = self.circuit_breaker.state
        
        return stats


# FastAPI dependency for easy integration
async def get_validator() -> MeshAIAPIKeyValidator:
    """Get or create validator instance for FastAPI dependency injection"""
    if not hasattr(get_validator, "_instance"):
        validator = MeshAIAPIKeyValidator()
        await validator.initialize()
        get_validator._instance = validator
    return get_validator._instance


# Cleanup function for FastAPI shutdown
async def cleanup_validator():
    """Cleanup validator on application shutdown"""
    if hasattr(get_validator, "_instance"):
        await get_validator._instance.close()
        delattr(get_validator, "_instance")