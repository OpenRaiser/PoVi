from __future__ import annotations

import concurrent.futures
import functools
import http.server
import os
import socket
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _write_placeholder_png(path: Path, width: int = 768, height: int = 1024) -> None:
    """Neutral solid-color placeholder used when a character reference image is
    blocked by a safety system, so the director graph still completes."""
    try:
        from PIL import Image
    except Exception:
        path.write_bytes(
            __import__("base64").b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            )
        )
        return
    img = Image.new("RGB", (width, height), (180, 190, 185))
    img.save(path, format="PNG")


def generate_character_reference_images(
    character_asset_plan: dict[str, Any],
    image_generator: Any,
    output_dir: Path,
) -> dict[str, Any]:
    assets_dir = output_dir / "character_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    continuity_source_by_target = {
        str(link["to_asset_id"]): str(link["from_asset_id"])
        for link in character_asset_plan.get("continuity_links", [])
    }
    indexed_assets = list(enumerate(character_asset_plan.get("character_assets", []), start=1))
    known_ids = {str(asset["asset_id"]) for _, asset in indexed_assets}
    rendered_paths: dict[str, Path] = {}
    rendered_by_id: dict[str, dict[str, Any]] = {}
    pending = list(indexed_assets)
    worker_count = max(1, int(os.environ.get("IMAGE_WORKERS", "1")))

    def render(indexed_asset: tuple[int, dict[str, Any]]) -> tuple[str, Path, dict[str, Any]]:
        index, asset = indexed_asset
        asset_id = str(asset["asset_id"])
        image_path = assets_dir / f"{asset_id}.png"
        source_asset_id = continuity_source_by_target.get(asset_id, "")
        try:
            image_generator.generate_prompt(
                prompt=str(asset["reference_prompt"]),
                output_path=image_path,
                seed_offset=10_000 + index,
                negative_prompt=str(asset["negative_prompt"]),
                reference_image=rendered_paths.get(source_asset_id),
            )
        except Exception as exc:  # safety-system rejection (e.g. Azure 400) or other failure
            msg = " ".join(str(exc).lower().split())
            if "safety" in msg or "reject" in msg or "400" in msg or "content" in msg:
                print(
                    f"[reference-assets] asset {asset_id} image generation rejected "
                    f"by safety system; emitting neutral placeholder. ({exc})",
                    flush=True,
                )
                _write_placeholder_png(image_path, width=768, height=1024)
            else:
                raise
        rendered_asset = {
            "asset_id": asset_id,
            "identity_group": str(asset["identity_group"]),
            "reference_mode": str(asset["reference_mode"]),
            "image_path": str(image_path.resolve()),
            "derived_from_asset_id": source_asset_id,
        }
        return asset_id, image_path, rendered_asset

    while pending:
        ready = [
            item
            for item in pending
            if not continuity_source_by_target.get(str(item[1]["asset_id"]))
            or continuity_source_by_target[str(item[1]["asset_id"])] not in known_ids
            or continuity_source_by_target[str(item[1]["asset_id"])] in rendered_paths
        ]
        if not ready:
            unresolved = ", ".join(str(asset["asset_id"]) for _, asset in pending)
            raise ValueError(f"Cyclic character continuity links: {unresolved}")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(worker_count, len(ready))
        ) as executor:
            results = list(executor.map(render, ready))
        for asset_id, image_path, rendered_asset in results:
            rendered_paths[asset_id] = image_path
            rendered_by_id[asset_id] = rendered_asset
        ready_ids = {str(asset["asset_id"]) for _, asset in ready}
        pending = [item for item in pending if str(item[1]["asset_id"]) not in ready_ids]

    return {
        "assets": [rendered_by_id[str(asset["asset_id"])] for _, asset in indexed_assets]
    }


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class ReferenceImageServer:
    """Publish generated references while remote video requests are running."""

    def __init__(self, root: Path, advertised_host: str = "", port: int = 0):
        self.root = root.resolve()
        self.advertised_host = advertised_host.strip() or self._detect_host()
        if self.advertised_host.startswith("127.") or self.advertised_host == "localhost":
            raise ValueError("Reference-image host must be reachable by the remote Veo service")
        handler = functools.partial(_QuietHandler, directory=str(self.root))
        self.server = http.server.ThreadingHTTPServer(("0.0.0.0", int(port)), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> ReferenceImageServer:
        self.thread.start()
        self._check_local_server()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://{self.advertised_host}:{self.server.server_port}"

    def url_for(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.root)
        quoted = "/".join(urllib.parse.quote(part) for part in relative.parts)
        return f"{self.base_url}/{quoted}"

    def _check_local_server(self) -> None:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.server.server_port}/",
            timeout=5,
        ) as response:
            if int(response.status) != 200:
                raise RuntimeError("Reference-image HTTP server failed its local health check")

    @staticmethod
    def _detect_host() -> str:
        hostname = socket.gethostname()
        try:
            addresses = socket.getaddrinfo(hostname, None, socket.AF_INET)
        except socket.gaierror:
            addresses = []
        candidates = [address[4][0] for address in addresses]
        public_candidate = next(
            (value for value in candidates if not value.startswith("127.")),
            "",
        )
        if public_candidate:
            return public_candidate

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("8.8.8.8", 80))
                route_address = str(probe.getsockname()[0])
            if route_address and not route_address.startswith("127."):
                return route_address
        except OSError:
            pass
        raise ValueError(
            "Could not detect a reference-image host reachable by Veo; "
            "set VEO_REFERENCE_IMAGE_HOST explicitly"
        )
