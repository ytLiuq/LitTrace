from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig


class BlobRef(BaseModel):
    backend: str
    object_key: str
    bucket: str | None = None
    uri: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    content_type: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactKeyContext:
    session_id: str
    kind: str
    artifact_id: str
    filename: str
    paper_id: str | None = None
    revision: str | None = None


class ArtifactStore(Protocol):
    backend: str

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> BlobRef:
        ...

    def get_bytes(self, ref: BlobRef) -> bytes:
        ...

    def delete(self, ref: BlobRef) -> None:
        ...

    def exists(self, ref: BlobRef) -> bool:
        ...

    def signed_url(self, ref: BlobRef, expires_seconds: int = 3600) -> str:
        ...

    def ref_for_path(
        self,
        path: Path,
        key: str,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> BlobRef:
        ...


@dataclass
class LocalArtifactStore:
    root: Path
    backend: str = "local"

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> BlobRef:
        target = self._path_for_key(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return self.ref_for_path(target, key, content_type=content_type, metadata=metadata)

    def get_bytes(self, ref: BlobRef) -> bytes:
        return self._path_for_ref(ref).read_bytes()

    def delete(self, ref: BlobRef) -> None:
        path = self._path_for_ref(ref)
        if path.exists():
            path.unlink()

    def exists(self, ref: BlobRef) -> bool:
        return self._path_for_ref(ref).exists()

    def signed_url(self, ref: BlobRef, expires_seconds: int = 3600) -> str:
        # Local backend has no real signed URL. The previous behaviour
        # returned ``file:///...`` URIs which leak the operator's
        # filesystem path and silently break in browsers (file:// is
        # cross-origin blocked). Raise so the route layer can either
        # stream the bytes via an internal proxy or return 501.
        raise NotImplementedError(
            "LocalArtifactStore does not support signed_url; "
            "expose bytes via an internal streaming endpoint instead."
        )

    def ref_for_path(
        self,
        path: Path,
        key: str,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> BlobRef:
        digest: str | None = None
        size_bytes: int | None = None
        if path.exists() and path.is_file():
            data = path.read_bytes()
            digest = sha256(data).hexdigest()
            size_bytes = len(data)
        return BlobRef(
            backend=self.backend,
            object_key=key,
            uri=path.resolve().as_uri(),
            sha256=digest,
            size_bytes=size_bytes,
            content_type=content_type,
            metadata=metadata or {},
        )

    def _path_for_key(self, key: str) -> Path:
        cleaned = key.strip("/")
        if not cleaned:
            raise ValueError("Artifact object key cannot be empty.")
        return self.root / cleaned

    def _path_for_ref(self, ref: BlobRef) -> Path:
        if ref.backend != self.backend:
            raise ValueError(f"Cannot read {ref.backend!r} ref from {self.backend!r} store.")
        return self._path_for_key(ref.object_key)


@dataclass(frozen=True)
class RemoteObjectStorePlan:
    backend: str
    bucket: str
    endpoint_url: str | None = None
    region: str | None = None
    path_prefix: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class S3ArtifactStore:
    bucket: str
    endpoint_url: str | None = None
    region: str | None = None
    backend: str = "s3"
    _client: object | None = None

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "S3/MinIO object storage requires the optional storage extra: "
                    "pip install -e '.[storage]'"
                ) from exc
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                region_name=self.region,
            )
        return self._client

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> BlobRef:
        extra_args: dict[str, object] = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra_args)
        return BlobRef(
            backend=self.backend,
            bucket=self.bucket,
            object_key=key,
            uri=f"s3://{self.bucket}/{key}",
            sha256=sha256(data).hexdigest(),
            size_bytes=len(data),
            content_type=content_type,
            metadata=metadata or {},
        )

    def get_bytes(self, ref: BlobRef) -> bytes:
        response = self.client.get_object(Bucket=ref.bucket or self.bucket, Key=ref.object_key)
        return response["Body"].read()

    def delete(self, ref: BlobRef) -> None:
        self.client.delete_object(Bucket=ref.bucket or self.bucket, Key=ref.object_key)

    def exists(self, ref: BlobRef) -> bool:
        try:
            self.client.head_object(Bucket=ref.bucket or self.bucket, Key=ref.object_key)
        except Exception:
            return False
        return True

    def signed_url(self, ref: BlobRef, expires_seconds: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": ref.bucket or self.bucket, "Key": ref.object_key},
            ExpiresIn=expires_seconds,
        )

    def ref_for_path(
        self,
        path: Path,
        key: str,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> BlobRef:
        return self.put_bytes(
            key,
            path.read_bytes(),
            content_type=content_type,
            metadata=metadata,
        )


def artifact_store_from_config(config: LitTraceConfig) -> ArtifactStore:
    if config.artifact_storage.backend == "local":
        return LocalArtifactStore(config.artifact_storage.local_root)
    if config.artifact_storage.backend == "s3":
        if not config.artifact_storage.bucket:
            raise ValueError("artifact_storage.bucket is required for S3/MinIO storage.")
        return S3ArtifactStore(
            bucket=config.artifact_storage.bucket,
            endpoint_url=config.artifact_storage.endpoint_url,
            region=config.artifact_storage.region,
        )
    raise ValueError(f"Unsupported artifact_storage.backend: {config.artifact_storage.backend}")


def remote_object_store_plan(config: LitTraceConfig) -> RemoteObjectStorePlan | None:
    storage = config.artifact_storage
    if storage.backend == "local":
        return None
    if not storage.bucket:
        raise ValueError("artifact_storage.bucket is required for remote object storage.")
    return RemoteObjectStorePlan(
        backend=storage.backend,
        bucket=storage.bucket,
        endpoint_url=storage.endpoint_url,
        region=storage.region,
        path_prefix=storage.path_prefix.strip("/"),
    )


def build_artifact_object_key(config: LitTraceConfig, context: ArtifactKeyContext) -> str:
    prefix = config.artifact_storage.path_prefix.strip("/")
    pieces = [
        *([prefix] if prefix else []),
        "sessions",
        _safe_segment(context.session_id),
    ]
    if context.kind == "workspace":
        pieces.extend(["workspace", context.filename])
    elif context.kind == "workspace_snapshot":
        pieces.extend(["workspace", "snapshots", context.filename])
    elif context.kind == "messages":
        pieces.extend(["messages", context.filename])
    elif context.kind == "memory":
        pieces.extend(["memory", context.filename])
    elif context.kind == "structured_document":
        pieces.extend(["structured_documents", context.filename])
    elif context.kind == "paper_pdf":
        pieces.extend(["papers", _safe_segment(context.paper_id or context.artifact_id), "paper.pdf"])
    elif context.kind == "supplementary":
        pieces.extend(
            [
                "papers",
                _safe_segment(context.paper_id or context.artifact_id),
                "supplementary",
                context.filename,
            ]
        )
    else:
        pieces.extend(["artifacts", _safe_segment(context.artifact_id), context.filename])
    return "/".join(_safe_segment(piece) for piece in pieces if piece)


def _safe_segment(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return cleaned[:160] or "unknown"
