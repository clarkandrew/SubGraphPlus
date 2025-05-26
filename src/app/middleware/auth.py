"""
API Key management and authentication middleware for SubgraphRAG+ API
"""

import os
import time
import hashlib
import logging
import secrets
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

logger = logging.getLogger(__name__)

# API Key header scheme
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


@dataclass
class APIKey:
    """API Key data structure"""
    key_id: str
    key_hash: str
    name: str
    created_at: str
    last_used: Optional[str] = None
    expires_at: Optional[str] = None
    is_active: bool = True
    permissions: List[str] = None
    rate_limit_override: Optional[int] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = ["read", "write"]
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'APIKey':
        """Create APIKey from dictionary"""
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if API key is expired"""
        if not self.expires_at:
            return False
        
        try:
            expires = datetime.fromisoformat(self.expires_at)
            return datetime.now() > expires
        except ValueError:
            return False
    
    def has_permission(self, permission: str) -> bool:
        """Check if API key has specific permission"""
        return permission in self.permissions
    
    def update_last_used(self):
        """Update last used timestamp"""
        self.last_used = datetime.now().isoformat()


class APIKeyManager:
    """Manages API keys for authentication and authorization"""
    
    def __init__(self, storage_path: str = "data/api_keys.json"):
        """
        Initialize API key manager
        
        Args:
            storage_path: Path to store API keys
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache of API keys
        self.api_keys: Dict[str, APIKey] = {}
        
        # Load existing keys
        self.load_keys()
        
        # Ensure default development key exists
        self._ensure_default_key()
        
        logger.info(f"API Key manager initialized with {len(self.api_keys)} keys")
    
    def _ensure_default_key(self):
        """Ensure a default development API key exists"""
        default_key = os.getenv('API_KEY_SECRET', 'default_key_for_dev_only')
        
        # Check if default key already exists
        default_hash = self._hash_key(default_key)
        existing_key = None
        
        for key_data in self.api_keys.values():
            if key_data.key_hash == default_hash:
                existing_key = key_data
                break
        
        if not existing_key:
            # Create default key
            key_id = "default_dev_key"
            api_key = APIKey(
                key_id=key_id,
                key_hash=default_hash,
                name="Default Development Key",
                created_at=datetime.now().isoformat(),
                permissions=["read", "write", "admin"],
                metadata={"type": "development", "auto_created": True}
            )
            
            self.api_keys[key_id] = api_key
            self.save_keys()
            
            logger.info("Created default development API key")
    
    def _hash_key(self, key: str) -> str:
        """Hash an API key for secure storage"""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def load_keys(self):
        """Load API keys from storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                self.api_keys = {}
                for key_id, key_data in data.items():
                    self.api_keys[key_id] = APIKey.from_dict(key_data)
                
                logger.info(f"Loaded {len(self.api_keys)} API keys from storage")
            else:
                logger.info("No existing API keys found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            self.api_keys = {}
    
    def save_keys(self):
        """Save API keys to storage"""
        try:
            data = {}
            for key_id, api_key in self.api_keys.items():
                data[key_id] = api_key.to_dict()
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("API keys saved to storage")
            
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")
    
    def generate_key(
        self,
        name: str,
        permissions: List[str] = None,
        expires_in_days: Optional[int] = None,
        rate_limit_override: Optional[int] = None,
        metadata: Dict = None
    ) -> tuple[str, str]:
        """
        Generate a new API key
        
        Args:
            name: Human-readable name for the key
            permissions: List of permissions for the key
            expires_in_days: Number of days until expiration
            rate_limit_override: Custom rate limit for this key
            metadata: Additional metadata
            
        Returns:
            Tuple of (key_id, raw_key)
        """
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        key_hash = self._hash_key(raw_key)
        
        # Generate unique key ID
        key_id = f"key_{secrets.token_hex(8)}"
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = (datetime.now() + timedelta(days=expires_in_days)).isoformat()
        
        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at,
            permissions=permissions or ["read", "write"],
            rate_limit_override=rate_limit_override,
            metadata=metadata or {}
        )
        
        # Store the key
        self.api_keys[key_id] = api_key
        self.save_keys()
        
        logger.info(f"Generated new API key: {key_id} ({name})")
        
        return key_id, raw_key
    
    def validate_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key
        
        Args:
            raw_key: The raw API key to validate
            
        Returns:
            APIKey object if valid, None otherwise
        """
        if not raw_key:
            return None
        
        key_hash = self._hash_key(raw_key)
        
        # Find matching key
        for api_key in self.api_keys.values():
            if api_key.key_hash == key_hash:
                # Check if key is active
                if not api_key.is_active:
                    logger.warning(f"Inactive API key used: {api_key.key_id}")
                    return None
                
                # Check if key is expired
                if api_key.is_expired():
                    logger.warning(f"Expired API key used: {api_key.key_id}")
                    return None
                
                # Update last used timestamp
                api_key.update_last_used()
                self.save_keys()
                
                return api_key
        
        return None
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke an API key
        
        Args:
            key_id: ID of the key to revoke
            
        Returns:
            True if key was revoked, False if not found
        """
        if key_id in self.api_keys:
            self.api_keys[key_id].is_active = False
            self.save_keys()
            logger.info(f"Revoked API key: {key_id}")
            return True
        
        return False
    
    def delete_key(self, key_id: str) -> bool:
        """
        Delete an API key permanently
        
        Args:
            key_id: ID of the key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        if key_id in self.api_keys:
            del self.api_keys[key_id]
            self.save_keys()
            logger.info(f"Deleted API key: {key_id}")
            return True
        
        return False
    
    def list_keys(self) -> List[Dict]:
        """
        List all API keys (without sensitive data)
        
        Returns:
            List of API key information
        """
        keys = []
        for api_key in self.api_keys.values():
            key_info = {
                "key_id": api_key.key_id,
                "name": api_key.name,
                "created_at": api_key.created_at,
                "last_used": api_key.last_used,
                "expires_at": api_key.expires_at,
                "is_active": api_key.is_active,
                "permissions": api_key.permissions,
                "rate_limit_override": api_key.rate_limit_override,
                "is_expired": api_key.is_expired()
            }
            keys.append(key_info)
        
        return keys
    
    def cleanup_expired_keys(self) -> int:
        """
        Remove expired API keys
        
        Returns:
            Number of keys removed
        """
        expired_keys = []
        
        for key_id, api_key in self.api_keys.items():
            if api_key.is_expired():
                expired_keys.append(key_id)
        
        for key_id in expired_keys:
            del self.api_keys[key_id]
        
        if expired_keys:
            self.save_keys()
            logger.info(f"Cleaned up {len(expired_keys)} expired API keys")
        
        return len(expired_keys)


# Global API key manager instance
api_key_manager = APIKeyManager()


async def get_api_key(api_key: str = Depends(api_key_header)) -> str:
    """
    Dependency to validate API key
    
    Args:
        api_key: API key from header
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail={
                "code": "MISSING_API_KEY",
                "message": "API key is required. Please provide X-API-KEY header."
            }
        )
    
    # Validate the API key
    key_data = api_key_manager.validate_key(api_key)
    
    if not key_data:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_API_KEY",
                "message": "Invalid or expired API key."
            }
        )
    
    return api_key


async def get_api_key_with_permission(permission: str):
    """
    Create a dependency that requires specific permission
    
    Args:
        permission: Required permission
        
    Returns:
        Dependency function
    """
    async def check_permission(api_key: str = Depends(get_api_key)) -> str:
        key_data = api_key_manager.validate_key(api_key)
        
        if not key_data or not key_data.has_permission(permission):
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail={
                    "code": "INSUFFICIENT_PERMISSIONS",
                    "message": f"API key does not have required permission: {permission}"
                }
            )
        
        return api_key
    
    return check_permission


# Convenience dependencies for common permissions
require_read_permission = get_api_key_with_permission("read")
require_write_permission = get_api_key_with_permission("write")
require_admin_permission = get_api_key_with_permission("admin")


def get_client_info(request: Request, api_key: str = Depends(get_api_key)) -> Dict:
    """
    Get client information for logging and monitoring
    
    Args:
        request: FastAPI request object
        api_key: Validated API key
        
    Returns:
        Dictionary with client information
    """
    key_data = api_key_manager.validate_key(api_key)
    
    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    
    return {
        "api_key_id": key_data.key_id if key_data else "unknown",
        "api_key_name": key_data.name if key_data else "unknown",
        "client_ip": client_ip,
        "user_agent": request.headers.get("User-Agent", "unknown"),
        "permissions": key_data.permissions if key_data else []
    } 