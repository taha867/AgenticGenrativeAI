"""Per-user content-addressed uploads (SHA-256) and legacy-safe path helpers."""
import hashlib
import re
from pathlib import Path

from fastapi import HTTPException

from backend.api.schemas.documents import validate_user_id_field
from backend.core.constants import UPLOAD_ALLOWED_EXTENSIONS

_SHA256_HEX = re.compile(r"^[a-f0-9]{64}$")


def sha256_bytes(data: bytes) -> str:
    """Return lowercase hex SHA-256 of data."""
    return hashlib.sha256(data).hexdigest()


def _normalize_suffix(original_name: str) -> str:
    """Return lowercase extension including dot, restricted to allowed set."""
    name = (original_name or "file").strip() or "file"
    ext = Path(name).suffix.lower()
    if ext == ".markdown":
        ext = ".md"
    if ext not in UPLOAD_ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(UPLOAD_ALLOWED_EXTENSIONS))
        raise HTTPException(400, f"Allowed formats: {allowed}")
    return ext


def content_addressed_path(uploads_dir: str, user_id: str, content_hash: str, suffix: str) -> Path:
    """
    Return absolute path for stored object: {uploads_dir}/{user_id}/objects/aa/bb/{hash}{suffix}.
    Does not create directories.
    """
    uid = validate_user_id_field(user_id)
    if not _SHA256_HEX.match(content_hash):
        raise HTTPException(400, "Invalid content hash")
    base = Path(uploads_dir).resolve()
    user_root = (base / uid).resolve()
    if not str(user_root).startswith(str(base)):
        raise HTTPException(400, "Invalid path")
    return (user_root / "objects" / content_hash[:2] / content_hash[2:4] / f"{content_hash}{suffix}").resolve()


def content_hash_from_path(file_path: Path) -> str:
    """
    If basename stem is a 64-char hex SHA-256, return it; else hash file contents.
    """
    stem = file_path.stem
    if _SHA256_HEX.match(stem) and file_path.is_file():
        return stem
    data = file_path.read_bytes()
    return sha256_bytes(data)


def save_upload_if_new(
    uploads_dir: str,
    user_id: str,
    original_filename: str,
    content: bytes,
) -> tuple[Path, str, bool]:
    """
    Write content to a content-addressed path under uploads_dir/user_id/objects/...
    Returns (absolute_path, content_hash, deduplicated).
    """
    suffix = _normalize_suffix(original_filename)
    content_hash = sha256_bytes(content)
    dest = content_addressed_path(uploads_dir, user_id, content_hash, suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        existing = dest.read_bytes()
        if existing == content:
            return dest, content_hash, True
        # Extremely rare: hash collision; overwrite not safe — store with conflict suffix (should not happen for SHA-256)
        raise HTTPException(409, "Storage conflict: same hash path with different content")
    dest.write_bytes(content)
    return dest, content_hash, False


def relative_path_from_uploads(uploads_dir: str, absolute_file: Path) -> str:
    """Path relative to uploads_dir using POSIX separators."""
    base = Path(uploads_dir).resolve()
    resolved = absolute_file.resolve()
    return str(resolved.relative_to(base)).replace("\\", "/")
