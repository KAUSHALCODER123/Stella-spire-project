"""Durable storage for uploaded documents.

Uploads are written to local disk *and*, when Supabase is configured, pushed
to object storage. The local copy is the working copy the pipeline reads; the
remote copy is the system of record that survives a restart, a redeploy, or a
container with an ephemeral filesystem.

Remote storage is strictly best-effort. A Supabase outage, a missing bucket or
a policy that rejects the write must never fail a candidate's application --
the local copy still processes fine, the failure is logged, and the interface
says whether the document was archived. Losing durability is a problem worth
telling someone about; losing the upload in front of the user is worse.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

TIMEOUT = 20


@dataclass
class StoredFile:
    """Where a document ended up."""

    local_path: Path
    filename: str
    bucket: Optional[str] = None
    key: Optional[str] = None
    public_url: Optional[str] = None
    error: Optional[str] = None

    @property
    def archived(self) -> bool:
        return self.key is not None and self.error is None


class SupabaseStorage:
    """The slice of the Supabase Storage REST API this app needs."""

    def __init__(self, url: str, key: str) -> None:
        self.base = url.rstrip("/")
        self.key = key

    @property
    def configured(self) -> bool:
        return bool(self.base and self.key)

    def _headers(self, content_type: Optional[str] = None, upsert: bool = True) -> dict:
        headers = {
            "apikey": self.key,
            "Authorization": "Bearer {}".format(self.key),
        }
        if content_type:
            headers["Content-Type"] = content_type
        if upsert:
            headers["x-upsert"] = "true"
        return headers

    def upload(self, bucket: str, key: str, payload: bytes, content_type: Optional[str] = None) -> str:
        """Store bytes. Returns the object key. Raises on failure."""
        url = "{}/storage/v1/object/{}/{}".format(self.base, bucket, key.lstrip("/"))
        request = urllib.request.Request(
            url, data=payload, method="POST",
            headers=self._headers(content_type or "application/octet-stream"),
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            if response.status not in (200, 201):
                raise RuntimeError("upload returned HTTP {}".format(response.status))
        return key

    def download(self, bucket: str, key: str) -> bytes:
        url = "{}/storage/v1/object/{}/{}".format(self.base, bucket, key.lstrip("/"))
        request = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.read()

    def public_url(self, bucket: str, key: str) -> str:
        return "{}/storage/v1/object/public/{}/{}".format(self.base, bucket, key.lstrip("/"))

    def check(self) -> dict:
        """Is the connection usable? Used by scripts/check_setup."""
        if not self.configured:
            return {"ok": False, "detail": "SUPABASE_URL or SUPABASE_KEY not set"}
        try:
            request = urllib.request.Request(
                "{}/storage/v1/bucket".format(self.base), headers=self._headers(), method="GET")
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                buckets = json.loads(response.read().decode("utf-8"))
            return {"ok": True, "buckets": [b.get("name") for b in buckets if isinstance(b, dict)]}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:200]
            # 401/403 on the bucket list is expected with an anon key -- listing
            # buckets is an admin operation. It does not mean uploads will fail.
            if exc.code in (400, 401, 403):
                return {"ok": True, "buckets": None,
                        "detail": "Cannot list buckets with this key (HTTP {}). "
                                  "Normal for an anon key; uploads are tested separately.".format(exc.code)}
            return {"ok": False, "detail": "HTTP {}: {}".format(exc.code, body)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "detail": "{}: {}".format(type(exc).__name__, exc)}


_client: Optional[SupabaseStorage] = None


def get_storage() -> Optional[SupabaseStorage]:
    global _client
    if _client is None:
        from app.config import settings
        if not (settings.supabase_url and settings.supabase_key):
            return None
        _client = SupabaseStorage(settings.supabase_url, settings.supabase_key)
    return _client


def store_document(*, payload: bytes, filename: str, kind: str, local_path: Path) -> StoredFile:
    """Write locally, then archive remotely if Supabase is configured.

    `kind` is "resume" or "jd" and selects the bucket.
    """
    from app.config import settings

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(payload)

    stored = StoredFile(local_path=local_path, filename=filename)

    client = get_storage()
    if client is None or not client.configured:
        return stored

    bucket = settings.supabase_bucket_jds if kind == "jd" else settings.supabase_bucket_resumes
    key = local_path.name  # already uuid-prefixed and sanitised
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    try:
        client.upload(bucket, key, payload, content_type)
        stored.bucket, stored.key = bucket, key
        stored.public_url = client.public_url(bucket, key)
        log.info("archived %s to supabase://%s/%s", filename, bucket, key)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        stored.error = "HTTP {} from Supabase: {}".format(exc.code, body)
        log.warning("supabase upload failed for %s: %s", filename, stored.error)
    except Exception as exc:  # noqa: BLE001 - never fail the user's upload
        stored.error = "{}: {}".format(type(exc).__name__, exc)
        log.warning("supabase upload failed for %s: %s", filename, stored.error)

    return stored
