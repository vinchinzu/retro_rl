"""Fetch the SMZ3 combo base IPS into smz3/refs/ (gitignored)."""

from __future__ import annotations

from smz3.rom_builder import ensure_base_ips, load_base_ips  # noqa: E402

def main() -> None:
    path = ensure_base_ips()
    ips = load_base_ips(path)
    print(f"Base IPS ready: {path} ({len(ips)} bytes after gunzip)")

if __name__ == "__main__":
    main()
