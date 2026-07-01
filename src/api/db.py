import hashlib
import os
import secrets
import sqlite3
from pathlib import Path

DB_PATH = Path(
    os.environ.get(
        "USERS_DB_PATH",
        str(Path(__file__).resolve().parent.parent.parent / "config" / "users.db"),
    )
)

# Seeded on first run only, so an existing DB is never overwritten.
DEFAULT_USERS = [
    ("admin", "admin", "admin"),
    ("user", "user", "viewer"),
]

_PBKDF2_ITERATIONS = 100_000


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash BLOB NOT NULL,
                salt BLOB NOT NULL,
                role TEXT NOT NULL
            )
            """
        )
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            for username, password, role in DEFAULT_USERS:
                salt = secrets.token_bytes(16)
                conn.execute(
                    "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
                    (username, _hash_password(password, salt), salt, role),
                )
        conn.commit()
    finally:
        conn.close()


def get_user(username: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT username, password_hash, salt, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"username": row[0], "password_hash": row[1], "salt": row[2], "role": row[3]}


def verify_password(password: str, user: dict) -> bool:
    return _hash_password(password, user["salt"]) == user["password_hash"]
