"""
Authentication API Endpoints

Login, signup, token refresh, and user management.
"""

from typing import List
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from backend.auth import (
    User,
    UserCreate,
    UserLogin,
    UserRole,
    Token,
    authenticate_user,
    create_user,
    create_tokens,
    decode_token,
    list_users,
    get_user_by_id,
)
from backend.auth.dependencies import (
    get_current_user,
    require_admin,
)
from backend.core.logging import logger


router = APIRouter()


# ============================================================
# Pydantic Response Models
# ============================================================

class AuthResponse(BaseModel):
    """Authentication response with user and tokens."""
    user: User
    tokens: Token


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str


# ============================================================
# Auth Endpoints
# ============================================================

@router.post("/login", response_model=AuthResponse)
async def login(credentials: UserLogin):
    """
    Login with email and password.
    
    Returns access and refresh tokens.
    
    Demo credentials:
    - admin@speechlab.dev / admin123 (Admin)
    - user@speechlab.dev / user123 (User)
    - viewer@speechlab.dev / viewer123 (Viewer)
    """
    user = authenticate_user(credentials.email, credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    tokens = create_tokens(user)
    
    logger.info(f"User logged in: {user.email}")
    
    return AuthResponse(user=user, tokens=tokens)


@router.post("/signup", response_model=AuthResponse)
async def signup(user_data: UserCreate):
    """
    Register a new user account.
    
    New users get the 'user' role by default.
    """
    try:
        # Force new signups to be regular users (not admin)
        if user_data.role == UserRole.ADMIN:
            user_data.role = UserRole.USER
        
        user = create_user(user_data)
        tokens = create_tokens(user)
        
        logger.info(f"New user registered: {user.email}")
        
        return AuthResponse(user=user, tokens=tokens)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """
    Refresh access token using refresh token.
    """
    payload = decode_token(refresh_token)
    
    if not payload or payload.type.value != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    user_data = get_user_by_id(payload.sub)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    user = User(
        id=user_data["id"],
        email=user_data["email"],
        name=user_data["name"],
        role=user_data["role"],
        team=user_data.get("team"),
        created_at=user_data["created_at"],
    )
    
    return create_tokens(user)


@router.get("/me", response_model=User)
async def get_current_user_info(user: User = Depends(get_current_user)):
    """
    Get current user's information.
    
    Requires authentication.
    """
    return user


@router.post("/logout", response_model=MessageResponse)
async def logout(user: User = Depends(get_current_user)):
    """
    Logout current user.
    
    Note: JWT tokens are stateless, so this just logs the action.
    Client should discard the token.
    """
    logger.info(f"User logged out: {user.email}")
    return MessageResponse(message="Logged out successfully")


# ============================================================
# User Management (Admin Only)
# ============================================================

@router.get("/users", response_model=List[User])
async def list_all_users(admin: User = Depends(require_admin)):
    """
    List all users.
    
    Admin only.
    """
    return list_users()


@router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str, admin: User = Depends(require_admin)):
    """
    Get a specific user by ID.
    
    Admin only.
    """
    user_data = get_user_by_id(user_id)
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
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
        is_active=user_data.get("is_active", True),
    )


@router.post("/users", response_model=User)
async def create_new_user(
    user_data: UserCreate,
    admin: User = Depends(require_admin),
):
    """
    Create a new user (admin can set any role).
    
    Admin only.
    """
    try:
        user = create_user(user_data)
        logger.info(f"Admin {admin.email} created user: {user.email}")
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
