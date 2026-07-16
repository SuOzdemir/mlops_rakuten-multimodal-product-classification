import hashlib
import os
import secrets

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://api:api@postgres:5432/api")

# Seeded on first run only, so an existing table is never overwritten.
DEFAULT_USERS = [
    ("admin", "adminadmin", "admin"),
    ("user", "user", "viewer"),
]

_PBKDF2_ITERATIONS = 100_000


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)


def _connect():
    return psycopg2.connect(DATABASE_URL)


def init_db() -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash BYTEA NOT NULL,
                    salt BYTEA NOT NULL,
                    role TEXT NOT NULL
                )
                """
            )
            cur.execute("SELECT COUNT(*) FROM users")
            count = cur.fetchone()[0]
            if count == 0:
                for username, password, role in DEFAULT_USERS:
                    salt = secrets.token_bytes(16)
                    cur.execute(
                        "INSERT INTO users (username, password_hash, salt, role) VALUES (%s, %s, %s, %s)",
                        (username, _hash_password(password, salt), salt, role),
                    )
        conn.commit()
    finally:
        conn.close()


def get_user(username: str) -> dict | None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username, password_hash, salt, role FROM users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"username": row[0], "password_hash": bytes(row[1]), "salt": bytes(row[2]), "role": row[3]}


def verify_password(password: str, user: dict) -> bool:
    return _hash_password(password, user["salt"]) == user["password_hash"]
