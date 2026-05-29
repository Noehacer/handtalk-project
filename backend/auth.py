from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

import config
from logger import get_logger
from database import create_user, get_user

log     = get_logger(__name__)
_pwd    = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)


def _hash(password: str) -> str:
    return _pwd.hash(password)


def _verify(plain: str, hashed: str) -> bool:
    return _pwd.verify(plain, hashed)


def _create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=config.TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": username, "exp": expire}, config.SECRET_KEY, algorithm=config.ALGORITHM)


def _decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


def register(username: str, password: str) -> bool:
    return create_user(username, _hash(password))


def login(username: str, password: str) -> Optional[str]:
    user = get_user(username)
    if not user or not _verify(password, user["hashed_password"]):
        log.warning(f"Login fallido para: {username}")
        return None
    log.info(f"Login exitoso: {username}")
    return _create_token(username)


def verify_token(token: str) -> str:
    """Validates a JWT token and returns the username. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Token inválido.")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado.")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token requerido. Usa el header: Authorization: Bearer <token>",
        )
    username = _decode_token(credentials.credentials)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado.",
        )
    return username
