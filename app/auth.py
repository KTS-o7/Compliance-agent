from __future__ import annotations
from typing import Optional

USERS: dict[str, dict] = {
    "junior": {"password": "junior123", "role": "junior"},
    "senior": {"password": "senior123", "role": "senior"},
}

ROLE_TAGS: dict[str, list[str]] = {
    "junior": ["junior", "all"],
    "senior": ["junior", "senior", "all"],
}


def authenticate(username: str, password: str) -> Optional[str]:
    """Returns role string if valid credentials, None otherwise."""
    user = USERS.get(username)
    if user and user["password"] == password:
        return user["role"]
    return None


def allowed_role_tags(role: str) -> list[str]:
    return ROLE_TAGS.get(role, [])


def can_access_admin(role: str) -> bool:
    return role == "senior"
