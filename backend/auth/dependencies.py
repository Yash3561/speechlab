"""
FastAPI Dependencies for Authentication

Provides dependency injection for protected routes.
"""

from typing import Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.auth import (
    User,
    UserRole,
    TokenPayload,
    decode_token,
    get_user_by_id,
    can_manage_users,
    can_create_experiments,
    can_delete_experiments,
)
from backend.core.logging import logger


# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    """
    Get the current authenticated user from JWT token.
    
    Usage:
        @router.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"user": user.email}
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user_data = get_user_by_id(payload.sub)
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    return User(
        id=user_data["id"],
        email=user_data["email"],
        name=user_data["name"],
        role=user_data["role"],
        team=user_data.get("team"),
        created_at=user_data["created_at"],
        last_login=user_data.get("last_login"),
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[User]:
    """
    Get the current user if authenticated, None otherwise.
    Useful for routes that work differently for authenticated users.
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_role(allowed_roles: List[UserRole]):
    """
    Dependency factory for role-based access control.
    
    Usage:
        @router.delete("/experiments/{id}")
        async def delete_experiment(
            user: User = Depends(require_role([UserRole.ADMIN]))
        ):
            ...
    """
    async def role_checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {[r.value for r in allowed_roles]}",
            )
        return user
    
    return role_checker


# Convenience dependencies
async def require_admin(user: User = Depends(get_current_user)) -> User:
    """Require admin role."""
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def require_user_or_admin(user: User = Depends(get_current_user)) -> User:
    """Require user or admin role (not viewer)."""
    if user.role == UserRole.VIEWER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write access required. Viewers have read-only access.",
        )
    return user


# Permission check dependencies
async def can_create(user: User = Depends(get_current_user)) -> User:
    """Check if user can create experiments."""
    if not can_create_experiments(user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to create experiments",
        )
    return user


async def can_delete(user: User = Depends(get_current_user)) -> User:
    """Check if user can delete experiments."""
    if not can_delete_experiments(user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can delete experiments",
        )
    return user
