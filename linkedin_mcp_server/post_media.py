"""Turn remote-client media inputs into local files the composer can upload.

A remote MCP client cannot hand this server a path on its own machine, so an
attachment arrives in one of three shapes:

- ``{"url": "https://..."}`` is downloaded here. Only https, only to globally
  routable addresses, and the connection goes to the exact address that was
  validated (the resolver is not asked twice, so a rebinding DNS answer cannot
  swap a private address in between). Redirects are followed by hand, a few
  hops at most, and every hop is validated the same way.
- ``{"base64": "...", "filename": "x.pdf"}`` is decoded with a size cap.
- ``{"path": "x.pdf"}`` names a file inside the uploads directory under the
  server's data root (``~/.linkedin-mcp/uploads``). Traversal and symlinks
  that resolve outside it are refused.

Whatever the shape, the bytes are sniffed and must be what the slot expects:
an image for ``media``, a PDF/PowerPoint/Word file for ``document``. The copy
lands in a private temporary directory that the caller removes afterwards.

Video is not accepted: LinkedIn processes video asynchronously for minutes,
far past a tool call's timeout, and exposes no locale-independent signal for
"processing finished" that could be verified before posting.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import ipaddress
import logging
from pathlib import Path, PurePosixPath
import re
import shutil
import ssl
import tempfile
from typing import Any, Literal
from urllib.parse import unquote, urljoin, urlparse
import zipfile

import anyio
import httpcore

from linkedin_mcp_server.scraping.post_content import (
    PostAttachment,
    PostValidationError,
)

logger = logging.getLogger(__name__)

AttachmentKind = Literal["image", "document"]

MAX_IMAGES = 20
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_DOCUMENT_BYTES = 100 * 1024 * 1024
MAX_BASE64_DECODED_BYTES = 20 * 1024 * 1024
MAX_REDIRECTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 60.0

_EXTENSIONS: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
}
_KIND_OF: dict[str, AttachmentKind] = {
    "image/jpeg": "image",
    "image/png": "image",
    "image/gif": "image",
    "application/pdf": "document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "document",
    "application/vnd.ms-powerpoint": "document",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document",
    "application/msword": "document",
}
# A server that does not know what it is sending says one of these; the bytes
# decide then. Any other declared type has to agree with the bytes.
_GENERIC_TYPES = {"application/octet-stream", "binary/octet-stream", ""}
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")

Resolver = Callable[[str, int], Awaitable[list[str]]]


def default_uploads_root() -> Path:
    """The uploads directory beside the browser profile, under the data root."""
    from linkedin_mcp_server.session_state import auth_root_dir

    return auth_root_dir() / "uploads"


def sniff_content_type(data: bytes, *, name_hint: str = "") -> str | None:
    """Identify an accepted attachment type from its leading bytes."""
    head = data[:2048]
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if b"%PDF-" in head[:1024]:
        return "application/pdf"
    suffix = PurePosixPath(name_hint.lower()).suffix
    if head.startswith(b"PK\x03\x04"):
        return _sniff_office_zip(data)
    if head.startswith(_OLE_MAGIC):
        # Legacy Office containers share one magic number; only the name
        # separates a presentation from a document.
        if suffix == ".ppt":
            return "application/vnd.ms-powerpoint"
        if suffix == ".doc":
            return "application/msword"
    return None


def _sniff_office_zip(data: bytes) -> str | None:
    import io

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
    except (zipfile.BadZipFile, ValueError):
        return None
    if any(name.startswith("ppt/") for name in names):
        return _office("presentationml.presentation")
    if any(name.startswith("word/") for name in names):
        return _office("wordprocessingml.document")
    return None


def _office(suffix: str) -> str:
    return f"application/vnd.openxmlformats-officedocument.{suffix}"


def _safe_filename(name: str, content_type: str) -> str:
    base = PurePosixPath(unquote(name).replace("\\", "/")).name
    stem = _SAFE_FILENAME_RE.sub("_", PurePosixPath(base).stem).strip(" ._")[:80]
    return f"{stem or 'attachment'}{_EXTENSIONS[content_type]}"


def _max_bytes(kind: AttachmentKind) -> int:
    return MAX_IMAGE_BYTES if kind == "image" else MAX_DOCUMENT_BYTES


def _declared_type(value: str | None) -> str:
    return (value or "").split(";", 1)[0].strip().lower()


@dataclass(frozen=True)
class _Fetched:
    data: bytes
    declared_type: str
    name_hint: str


class MediaStaging:
    """A private temporary directory holding one request's attachments."""

    def __init__(self) -> None:
        # mkdtemp creates the directory 0700 and owns its name; nothing
        # user-supplied is ever joined onto it except a sanitised basename.
        self.directory = Path(tempfile.mkdtemp(prefix="linkedin-mcp-post-"))
        self._count = 0

    def write(
        self,
        data: bytes,
        *,
        kind: AttachmentKind,
        name_hint: str,
        source: str,
        declared_type: str = "",
    ) -> PostAttachment:
        limit = _max_bytes(kind)
        if len(data) > limit:
            raise PostValidationError(
                f"{source}: {len(data)} bytes exceeds the {limit}-byte limit for "
                f"a {kind}."
            )
        content_type = sniff_content_type(data, name_hint=name_hint)
        if content_type is None or _KIND_OF[content_type] != kind:
            expected = (
                "a JPEG, PNG or GIF image"
                if kind == "image"
                else "a PDF, PowerPoint (.pptx/.ppt) or Word (.docx/.doc) file"
            )
            raise PostValidationError(f"{source}: content is not {expected}.")
        declared = _declared_type(declared_type)
        if declared not in _GENERIC_TYPES and declared != content_type:
            raise PostValidationError(
                f"{source}: declared content type {declared!r} does not match the "
                f"content ({content_type})."
            )
        self._count += 1
        filename = _safe_filename(name_hint or "attachment", content_type)
        path = self.directory / f"{self._count:02d}-{filename}"
        path.write_bytes(data)
        return PostAttachment(
            kind=kind,
            path=str(path),
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            source=source,
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


@contextmanager
def media_staging() -> Iterator[MediaStaging]:
    staging = MediaStaging()
    try:
        yield staging
    finally:
        staging.cleanup()


def _is_public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast


async def _system_resolver(host: str, port: int) -> list[str]:
    infos = await anyio.getaddrinfo(host, port, proto=6)
    return [str(info[4][0]) for info in infos]


class _PinnedBackend(httpcore.AsyncNetworkBackend):
    """Connect only to the address validated for the one expected host."""

    def __init__(self, inner: Any, host: str, address: str):
        self._inner = inner
        self._host = host
        self._address = address

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        if host != self._host:
            raise httpcore.ConnectError(f"refusing unpinned host {host!r}")
        return await self._inner.connect_tcp(
            self._address,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, *args: Any, **kwargs: Any) -> Any:
        raise httpcore.ConnectError("unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def _validate_download_url(url: str) -> tuple[str, str, int]:
    if not isinstance(url, str) or re.search(r"[\x00-\x20\x7f\\]", url):
        raise PostValidationError(f"Media URL {url!r} is not a valid URL.")
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise PostValidationError(f"Media URL {url!r} must use https.")
    if parsed.username or parsed.password or not parsed.hostname:
        raise PostValidationError(
            f"Media URL {url!r} must name a host and carry no credentials."
        )
    try:
        port = parsed.port or 443
    except ValueError as error:
        raise PostValidationError(f"Media URL {url!r} has an invalid port.") from error
    return url, parsed.hostname, port


async def download_attachment(
    url: str,
    *,
    kind: AttachmentKind,
    resolver: Resolver | None = None,
    network_backend: httpcore.AsyncNetworkBackend | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> _Fetched:
    """Download one attachment over https from a public address, capped."""
    resolver = resolver or _system_resolver
    backend = network_backend or httpcore.AnyIOBackend()
    limit = _max_bytes(kind)
    current = url
    with anyio.fail_after(DOWNLOAD_TIMEOUT_SECONDS):
        for _hop in range(MAX_REDIRECTS + 1):
            current, host, port = _validate_download_url(current)
            try:
                addresses = await resolver(host, port)
            except OSError as error:
                raise PostValidationError(
                    f"Media URL host {host!r} did not resolve."
                ) from error
            if not addresses or not all(_is_public_address(a) for a in addresses):
                raise PostValidationError(
                    f"Media URL host {host!r} resolves to a non-public address."
                )
            pinned = _PinnedBackend(backend, host, addresses[0])
            async with httpcore.AsyncConnectionPool(
                ssl_context=ssl_context or ssl.create_default_context(),
                network_backend=pinned,
                http1=True,
                http2=False,
                retries=0,
            ) as pool:
                async with pool.stream(
                    "GET",
                    current,
                    headers=[
                        (b"Accept", b"*/*"),
                        (b"Accept-Encoding", b"identity"),
                        (b"User-Agent", b"linkedin-mcp-server media fetch"),
                    ],
                ) as response:
                    headers = {
                        key.decode("latin-1").lower(): value.decode("latin-1")
                        for key, value in response.headers
                    }
                    if response.status in {301, 302, 303, 307, 308}:
                        location = headers.get("location")
                        if not location:
                            raise PostValidationError(
                                f"Media URL {current!r} redirected without a target."
                            )
                        current = urljoin(current, location)
                        continue
                    if response.status != 200:
                        raise PostValidationError(
                            f"Media URL {current!r} answered HTTP {response.status}."
                        )
                    length = headers.get("content-length", "")
                    if length.isdigit() and int(length) > limit:
                        raise PostValidationError(
                            f"Media URL {current!r} is {length} bytes; the limit "
                            f"for a {kind} is {limit}."
                        )
                    chunks: list[bytes] = []
                    received = 0
                    async for chunk in response.aiter_stream():
                        received += len(chunk)
                        if received > limit:
                            raise PostValidationError(
                                f"Media URL {current!r} exceeds the {limit}-byte "
                                f"limit for a {kind}."
                            )
                        chunks.append(chunk)
                    name = PurePosixPath(urlparse(current).path).name
                    return _Fetched(
                        b"".join(chunks), headers.get("content-type", ""), name
                    )
        raise PostValidationError(
            f"Media URL {url!r} redirected more than {MAX_REDIRECTS} times."
        )


def _resolve_upload_path(value: str, uploads_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise PostValidationError("A media path must be a non-empty string.")
    try:
        root = uploads_root.expanduser().resolve(strict=True)
    except OSError as error:
        raise PostValidationError(
            f"The uploads directory {uploads_root} does not exist. Create it on the "
            "server and place files there, or send the file as url or base64."
        ) from error
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise PostValidationError(
            f"Media path {value!r} does not exist in the uploads directory."
        ) from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise PostValidationError(
            f"Media path {value!r} must name a file inside the uploads directory."
        )
    return resolved


def _read_capped(path: Path, limit: int, source: str) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise PostValidationError(f"{source}: file exceeds the {limit}-byte limit.")
    return data


def _decode_base64(value: str, source: str) -> bytes:
    if not isinstance(value, str):
        raise PostValidationError(f"{source}: base64 must be a string.")
    compact = "".join(value.split())
    if compact.startswith("data:") and "," in compact:
        compact = compact.split(",", 1)[1]
    if len(compact) > (MAX_BASE64_DECODED_BYTES * 4) // 3 + 8:
        raise PostValidationError(
            f"{source}: base64 payload exceeds {MAX_BASE64_DECODED_BYTES} bytes; "
            "send large files as url or path."
        )
    try:
        return base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as error:
        raise PostValidationError(f"{source}: base64 payload is invalid.") from error


def _single_source(spec: Any, source: str) -> tuple[str, str]:
    if not isinstance(spec, dict):
        raise PostValidationError(f"{source} must be an object.")
    present = [key for key in ("url", "base64", "path") if spec.get(key)]
    if len(present) != 1:
        raise PostValidationError(f"{source} needs exactly one of url, base64 or path.")
    return present[0], spec[present[0]]


async def stage_attachment(
    spec: dict[str, Any],
    *,
    kind: AttachmentKind,
    staging: MediaStaging,
    source: str,
    uploads_root: Path | None = None,
    resolver: Resolver | None = None,
    network_backend: httpcore.AsyncNetworkBackend | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> PostAttachment:
    """Validate one attachment spec and copy its bytes into the staging dir."""
    shape, value = _single_source(spec, source)
    limit = _max_bytes(kind)
    if shape == "url":
        fetched = await download_attachment(
            value,
            kind=kind,
            resolver=resolver,
            network_backend=network_backend,
            ssl_context=ssl_context,
        )
        return staging.write(
            fetched.data,
            kind=kind,
            name_hint=spec.get("filename") or fetched.name_hint,
            source=f"{source} (url)",
            declared_type=fetched.declared_type,
        )
    if shape == "base64":
        filename = spec.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            raise PostValidationError(f"{source}: base64 input needs a filename.")
        data = _decode_base64(value, source)
        return staging.write(
            data[: limit + 1],
            kind=kind,
            name_hint=filename,
            source=f"{source} (base64)",
        )
    root = uploads_root if uploads_root is not None else default_uploads_root()
    path = _resolve_upload_path(value, root)
    data = await anyio.to_thread.run_sync(_read_capped, path, limit, source)
    return staging.write(
        data, kind=kind, name_hint=path.name, source=f"{source} (path)"
    )


async def stage_post_attachments(
    media: list[dict[str, Any]] | None,
    document: dict[str, Any] | None,
    *,
    staging: MediaStaging,
    uploads_root: Path | None = None,
    resolver: Resolver | None = None,
    network_backend: httpcore.AsyncNetworkBackend | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> tuple[tuple[PostAttachment, ...], PostAttachment | None, str | None]:
    """Stage every image and the document; return them with the title."""
    media = media or []
    if len(media) > MAX_IMAGES:
        raise PostValidationError(f"A post carries at most {MAX_IMAGES} images.")
    if media and document:
        raise PostValidationError(
            "LinkedIn posts carry either images or one document, not both."
        )

    async def stage(spec: dict[str, Any], kind: AttachmentKind, source: str):
        return await stage_attachment(
            spec,
            kind=kind,
            staging=staging,
            source=source,
            uploads_root=uploads_root,
            resolver=resolver,
            network_backend=network_backend,
            ssl_context=ssl_context,
        )

    images = tuple(
        [
            await stage(spec, "image", f"media[{index}]")
            for index, spec in enumerate(media)
        ]
    )
    if not document:
        return images, None, None
    title = document.get("title") if isinstance(document, dict) else None
    if not isinstance(title, str) or not title.strip():
        raise PostValidationError("document needs a non-blank title.")
    staged = await stage(document, "document", "document")
    return images, staged, " ".join(title.split())
