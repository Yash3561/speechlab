"""
Authentication Module for SpeechLab

JWT-based authentication with role-based access control (RBAC).
Supports: Admin, User, Viewer roles.
"""

from typing import Optional, List
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, EmailStr
import secrets
import hashlib

from backend.core.config import settings
from backend.core.logging import logger

# JWT imports with graceful fallback
try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    HAS_AUTH_DEPS = True
except ImportError:
    HAS_AUTH_DEPS = False
    logger.warning("Auth dependencies not installed. Install: pip install python-jose[cryptography] passlib[bcrypt]")


# ============================================================
# Constants & Configuration
# ============================================================

SECRET_KEY = getattr(settings, 'secret_key', secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 7


# ============================================================
# Enums & Models
# ============================================================

class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"      # Full access - manage users, delete experiments
    USER = "user"        # Create/edit own experiments, view all
    VIEWER = "viewer"    # Read-only access


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


class User(BaseModel):
    """User model."""
    id: str
    email: EmailStr
    name: str
    role: UserRole
    team: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


class UserCreate(BaseModel):
    """User registration model."""
    email: EmailStr
    password: str
    name: str
    role: UserRole = UserRole.USER
    team: Optional[str] = None


class UserLogin(BaseModel):
    """Login request model."""
    email: EmailStr
    password: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # user_id
    email: str
    role: UserRole
    type: TokenType
    exp: datetime
    iat: datetime


# ============================================================
# Password Hashing
# ============================================================

if HAS_AUTH_DEPS:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
else:
    pwd_context = None


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    if pwd_context:
        return pwd_context.hash(password)
    # Fallback to simple hash (NOT for production)
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    if pwd_context:
        return pwd_context.verify(plain_password, hashed_password)
    # Fallback
    return hash_password(plain_password) == hashed_password


# ============================================================
# JWT Token Management
# ============================================================

def create_access_token(
    user_id: str,
    email: str,
    role: UserRole,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT access token."""
    if not HAS_AUTH_DEPS:
        # Mock token for development
        return f"mock_access_{user_id}_{secrets.token_urlsafe(16)}"
    
    now = datetime.utcnow()
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    
    payload = {
        "sub": user_id,
        "email": email,
        "role": role.value,
        "type": TokenType.ACCESS.value,
        "exp": expire,
        "iat": now,
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """Create a JWT refresh token."""
    if not HAS_AUTH_DEPS:
        return f"mock_refresh_{user_id}_{secrets.token_urlsafe(16)}"
    
    now = datetime.utcnow()
    expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "sub": user_id,
        "type": TokenType.REFRESH.value,
        "exp": expire,
        "iat": now,
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_tokens(user: User) -> Token:
    """Create access and refresh tokens for a user."""
    access_token = create_access_token(user.id, user.email, user.role)
    refresh_token = create_refresh_token(user.id)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


def decode_token(token: str) -> Optional[TokenPayload]:
    """Decode and validate a JWT token."""
    if not HAS_AUTH_DEPS:
        # Mock decode for development
        if token.startswith("mock_access_"):
            parts = token.split("_")
            return TokenPayload(
                sub=parts[2] if len(parts) > 2 else "user_001",
                email="demo@speechlab.dev",
                role=UserRole.ADMIN,
                type=TokenType.ACCESS,
                exp=datetime.utcnow() + timedelta(hours=24),
                iat=datetime.utcnow(),
            )
        return None
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenPayload(
            sub=payload["sub"],
            email=payload.get("email", ""),
            role=UserRole(payload.get("role", "viewer")),
            type=TokenType(payload["type"]),
            exp=datetime.fromtimestamp(payload["exp"]),
            iat=datetime.fromtimestamp(payload["iat"]),
        )
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None


# ============================================================
# In-Memory User Store (replace with DB in production)
# ============================================================

users_db: dict[str, dict] = {}


def _seed_demo_users():
    """Seed demo users for development."""
    from datetime import timedelta as td
    
    demo_users = [
        {
            "id": "user_001",
            "email": "admin@speechlab.dev",
            "password_hash": hash_password("admin123"),
            "name": "Admin User",
            "role": UserRole.ADMIN,
            "team": "Platform",
            "created_at": datetime.utcnow() - td(days=30),
            "is_active": True,
        },
        {
            "id": "user_002",
            "email": "user@speechlab.dev",
            "password_hash": hash_password("user123"),
            "name": "ML Engineer",
            "role": UserRole.USER,
            "team": "ML",
            "created_at": datetime.utcnow() - td(days=15),
            "is_active": True,
        },
        {
            "id": "user_003",
            "email": "viewer@speechlab.dev",
            "password_hash": hash_password("viewer123"),
            "name": "Data Analyst",
            "role": UserRole.VIEWER,
            "team": "Analytics",
            "created_at": datetime.utcnow() - td(days=7),
            "is_active": True,
        },
    ]
    
    for user in demo_users:
        users_db[user["id"]] = user
    
    logger.info(f"Seeded {len(demo_users)} demo users")


# Seed on import
_seed_demo_users()


# ============================================================
# User Operations
# ============================================================

def get_user_by_email(email: str) -> Optional[dict]:
    """Get user by email."""
    for user in users_db.values():
        if user["email"] == email:
            return user
    return None


def get_user_by_id(user_id: str) -> Optional[dict]:
    """Get user by ID."""
    return users_db.get(user_id)


def create_user(user_create: UserCreate) -> User:
    """Create a new user."""
    # Check if email exists
    if get_user_by_email(user_create.email):
        raise ValueError("Email already registered")
    
    user_id = f"user_{secrets.token_hex(8)}"
    
    user_data = {
        "id": user_id,
        "email": user_create.email,
        "password_hash": hash_password(user_create.password),
        "name": user_create.name,
        "role": user_create.role,
        "team": user_create.team,
        "created_at": datetime.utcnow(),
        "is_active": True,
    }
    
    users_db[user_id] = user_data
    logger.info(f"Created user: {user_create.email}")
    
    return User(
        id=user_id,
        email=user_create.email,
        name=user_create.name,
        role=user_create.role,
        team=user_create.team,
        created_at=user_data["created_at"],
    )


def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password."""
    user_data = get_user_by_email(email)
    
    if not user_data:
        return None
    
    if not verify_password(password, user_data["password_hash"]):
        return None
    
    if not user_data.get("is_active", True):
        return None
    
    # Update last login
    user_data["last_login"] = datetime.utcnow()
    
    return User(
        id=user_data["id"],
        email=user_data["email"],
        name=user_data["name"],
        role=user_data["role"],
        team=user_data.get("team"),
        created_at=user_data["created_at"],
        last_login=user_data["last_login"],
    )


def list_users() -> List[User]:
    """List all users (admin only)."""
    return [
        User(
            id=u["id"],
            email=u["email"],
            name=u["name"],
            role=u["role"],
            team=u.get("team"),
            created_at=u["created_at"],
            last_login=u.get("last_login"),
            is_active=u.get("is_active", True),
        )
        for u in users_db.values()
    ]


# ============================================================
# Permission Checks
# ============================================================

def can_manage_users(role: UserRole) -> bool:
    """Check if role can manage users."""
    return role == UserRole.ADMIN


def can_create_experiments(role: UserRole) -> bool:
    """Check if role can create experiments."""
    return role in [UserRole.ADMIN, UserRole.USER]


def can_delete_experiments(role: UserRole) -> bool:
    """Check if role can delete experiments."""
    return role == UserRole.ADMIN


def can_view_experiments(role: UserRole) -> bool:
    """Check if role can view experiments."""
    return True  # All roles can view
