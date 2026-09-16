"""Attachment staging: sniffing, caps, uploads-root confinement, safe downloads.

Downloads never reach a network: a fake resolver answers with a public address
and httpcore's mock backend serves the HTTP bytes locally.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpcore
import pytest

from linkedin_mcp_server import post_media
from linkedin_mcp_server.post_media import (
    media_staging,
    sniff_content_type,
    stage_attachment,
    stage_post_attachments,
)
from linkedin_mcp_server.scraping.post_content import PostValidationError

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PDF = b"%PDF-1.7\n" + b"0" * 32
PUBLIC_ADDRESS = "93.184.216.34"


def _response(status: str, headers: list[str], body: bytes = b"") -> list[bytes]:
    head = f"HTTP/1.1 {status}\r\n" + "".join(f"{h}\r\n" for h in headers) + "\r\n"
    return [head.encode("latin-1"), body]


class _RecordingBackend(httpcore.AsyncMockBackend):
    """A local stub that also records which address a connection went to."""

    def __init__(self, buffer: list[bytes]):
        super().__init__(buffer)
        self.connected: list[str] = []

    async def connect_tcp(
        self, host, port, timeout=None, local_address=None, socket_options=None
    ):  # type: ignore[override]
        self.connected.append(host)
        return await super().connect_tcp(
            host, port, timeout, local_address, socket_options
        )


def _resolver(mapping: dict[str, list[str]]):
    async def resolve(host: str, _port: int) -> list[str]:
        return mapping[host]

    return resolve


class TestSniffing:
    def test_known_types_are_identified_from_bytes(self):
        assert sniff_content_type(PNG) == "image/png"
        assert sniff_content_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
        assert sniff_content_type(PDF) == "application/pdf"
        assert sniff_content_type(b"MZ\x90\x00 executable") is None


class TestBase64AndPaths:
    async def test_base64_image_is_staged_with_a_safe_name(self):
        with media_staging() as staging:
            attachment = await stage_attachment(
                {"base64": base64.b64encode(PNG).decode(), "filename": "../x y.png"},
                kind="image",
                staging=staging,
                source="media[0]",
            )
            assert attachment.content_type == "image/png"
            assert Path(attachment.path).parent == staging.directory
            assert Path(attachment.path).read_bytes() == PNG
            assert attachment.filename == "x y.png"
        assert not staging.directory.exists()

    async def test_a_pdf_is_not_accepted_as_an_image(self):
        with (
            media_staging() as staging,
            pytest.raises(PostValidationError, match="not a JPEG"),
        ):
            await stage_attachment(
                {"base64": base64.b64encode(PDF).decode(), "filename": "x.png"},
                kind="image",
                staging=staging,
                source="media[0]",
            )

    async def test_exactly_one_source_is_required(self):
        with (
            media_staging() as staging,
            pytest.raises(PostValidationError, match="exactly one"),
        ):
            await stage_attachment(
                {"url": "https://a.example/x.png", "path": "x.png"},
                kind="image",
                staging=staging,
                source="media[0]",
            )

    async def test_oversized_image_is_refused(self, monkeypatch):
        monkeypatch.setattr(post_media, "MAX_IMAGE_BYTES", 16)
        with (
            media_staging() as staging,
            pytest.raises(PostValidationError, match="limit"),
        ):
            await stage_attachment(
                {"base64": base64.b64encode(PNG).decode(), "filename": "x.png"},
                kind="image",
                staging=staging,
                source="media[0]",
            )

    async def test_path_inside_uploads_root_is_read(self, tmp_path):
        (tmp_path / "deck.pdf").write_bytes(PDF)
        with media_staging() as staging:
            attachment = await stage_attachment(
                {"path": "deck.pdf"},
                kind="document",
                staging=staging,
                source="document",
                uploads_root=tmp_path,
            )
        assert attachment.content_type == "application/pdf"

    @pytest.mark.parametrize("value", ["../secret.pdf", "sub/../../secret.pdf"])
    async def test_traversal_out_of_the_uploads_root_is_refused(self, tmp_path, value):
        root = tmp_path / "uploads"
        (root / "sub").mkdir(parents=True)
        (tmp_path / "secret.pdf").write_bytes(PDF)
        with (
            media_staging() as staging,
            pytest.raises(PostValidationError, match="inside the uploads"),
        ):
            await stage_attachment(
                {"path": value},
                kind="document",
                staging=staging,
                source="document",
                uploads_root=root,
            )

    async def test_absolute_path_outside_the_root_is_refused(self, tmp_path):
        root = tmp_path / "uploads"
        root.mkdir()
        outside = tmp_path / "secret.pdf"
        outside.write_bytes(PDF)
        with (
            media_staging() as staging,
            pytest.raises(PostValidationError, match="inside the uploads"),
        ):
            await stage_attachment(
                {"path": str(outside)},
                kind="document",
                staging=staging,
                source="document",
                uploads_root=root,
            )

    async def test_symlink_escaping_the_root_is_refused(self, tmp_path):
        root = tmp_path / "uploads"
        root.mkdir()
        outside = tmp_path / "secret.pdf"
        outside.write_bytes(PDF)
        link = root / "link.pdf"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks need extra privileges on this platform")
        with (
            media_staging() as staging,
            pytest.raises(PostValidationError, match="inside the uploads"),
        ):
            await stage_attachment(
                {"path": "link.pdf"},
                kind="document",
                staging=staging,
                source="document",
                uploads_root=root,
            )

    async def test_images_and_document_together_are_refused(self):
        with (
            media_staging() as staging,
            pytest.raises(PostValidationError, match="not both"),
        ):
            await stage_post_attachments(
                [{"base64": base64.b64encode(PNG).decode(), "filename": "a.png"}],
                {
                    "base64": base64.b64encode(PDF).decode(),
                    "filename": "a.pdf",
                    "title": "T",
                },
                staging=staging,
            )


class TestDownloads:
    async def test_download_connects_to_the_validated_address(self):
        backend = _RecordingBackend(
            _response(
                "200 OK",
                ["Content-Type: image/png", f"Content-Length: {len(PNG)}"],
                PNG,
            )
        )
        with media_staging() as staging:
            attachment = await stage_attachment(
                {"url": "https://cdn.example/pic.png"},
                kind="image",
                staging=staging,
                source="media[0]",
                resolver=_resolver({"cdn.example": [PUBLIC_ADDRESS]}),
                network_backend=backend,
            )
        assert attachment.size_bytes == len(PNG)
        assert attachment.filename == "pic.png"
        assert backend.connected == [PUBLIC_ADDRESS]

    @pytest.mark.parametrize(
        "address", ["127.0.0.1", "10.0.0.5", "192.168.1.10", "169.254.169.254", "::1"]
    )
    async def test_non_public_addresses_are_refused_before_connecting(self, address):
        backend = _RecordingBackend([])
        with (
            media_staging() as staging,
            pytest.raises(PostValidationError, match="non-public"),
        ):
            await stage_attachment(
                {"url": "https://internal.example/x.png"},
                kind="image",
                staging=staging,
                source="media[0]",
                resolver=_resolver({"internal.example": [address]}),
                network_backend=backend,
            )
        assert backend.connected == []

    async def test_redirect_to_a_private_host_is_refused(self):
        backend = _RecordingBackend(
            _response(
                "302 Found",
                ["Location: https://internal.example/x.png", "Content-Length: 0"],
            )
        )
        with (
            media_staging() as staging,
            pytest.raises(PostValidationError, match="non-public"),
        ):
            await stage_attachment(
                {"url": "https://cdn.example/x.png"},
                kind="image",
                staging=staging,
                source="media[0]",
                resolver=_resolver(
                    {"cdn.example": [PUBLIC_ADDRESS], "internal.example": ["10.1.2.3"]}
                ),
                network_backend=backend,
            )
        assert backend.connected == [PUBLIC_ADDRESS]

    async def test_plain_http_is_refused(self):
        with (
            media_staging() as staging,
            pytest.raises(PostValidationError, match="https"),
        ):
            await stage_attachment(
                {"url": "http://cdn.example/x.png"},
                kind="image",
                staging=staging,
                source="media[0]",
                resolver=_resolver({}),
            )

    async def test_declared_type_contradicting_the_bytes_is_refused(self):
        backend = _RecordingBackend(
            _response(
                "200 OK",
                ["Content-Type: image/jpeg", f"Content-Length: {len(PNG)}"],
                PNG,
            )
        )
        with (
            media_staging() as staging,
            pytest.raises(PostValidationError, match="does not match"),
        ):
            await stage_attachment(
                {"url": "https://cdn.example/pic.png"},
                kind="image",
                staging=staging,
                source="media[0]",
                resolver=_resolver({"cdn.example": [PUBLIC_ADDRESS]}),
                network_backend=backend,
            )

    async def test_declared_length_over_the_cap_is_refused(self, monkeypatch):
        monkeypatch.setattr(post_media, "MAX_DOCUMENT_BYTES", 8)
        backend = _RecordingBackend(
            _response(
                "200 OK", ["Content-Type: application/pdf", "Content-Length: 999"], PDF
            )
        )
        with (
            media_staging() as staging,
            pytest.raises(PostValidationError, match="limit"),
        ):
            await stage_attachment(
                {"url": "https://cdn.example/deck.pdf"},
                kind="document",
                staging=staging,
                source="document",
                resolver=_resolver({"cdn.example": [PUBLIC_ADDRESS]}),
                network_backend=backend,
            )
