"""HM-Decomp ASM parsers used as comparison targets for ROM inspection."""

from __future__ import annotations

import re
from pathlib import Path

from harvest.runtime.rom_model import MAPS_GRAPHICS_ASM_PATH

def _parse_hexish_int(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    token = token.split()[0]
    token = token.rstrip(",")
    token = token.lstrip("#")
    if token.startswith("$"):
        return int(token[1:], 16)
    if token.startswith("0x") or token.startswith("0X"):
        return int(token, 16)
    if token.isdigit():
        return int(token, 10)
    return None


def _directive_bytes(directive: str, operand_text: str) -> bytes:
    return _directive_bytes_with_fallback(directive, operand_text, fallback_values=None)


def _directive_bytes_with_fallback(
    directive: str,
    operand_text: str,
    fallback_values: list[int] | None,
) -> bytes:
    width = {"db": 1, "dw": 2, "dl": 3}[directive]
    out = bytearray()
    fallback_queue = list(fallback_values or [])
    for raw_operand in operand_text.split(","):
        value = _parse_hexish_int(raw_operand)
        if value is None and fallback_queue:
            value = fallback_queue.pop(0)
        if value is None:
            continue
        out.extend(value.to_bytes(width, "little"))
    return bytes(out)


def _directive_comment_fallback_values(raw_line: str, directive: str) -> list[int]:
    width = {"db": 1, "dw": 2, "dl": 3}[directive]
    comment_fields = raw_line.split(";")[1:]
    if len(comment_fields) < 2:
        return []

    values: list[int] = []
    for field in comment_fields[1:]:
        stripped = field.strip()
        if not stripped:
            continue
        token = stripped.split()[0].lstrip("$")
        if not re.fullmatch(r"[0-9A-Fa-f]+", token):
            continue
        value = int(token, 16)
        if value <= (1 << (width * 8)) - 1:
            values.append(value)
    return values


def parse_numeric_asm_bytes(path: Path) -> bytes:
    """Extract raw numeric bytes from asm files that use db/dw/dl data directives."""
    output = bytearray()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line
        line = line.split(";", 1)[0].strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if not parts:
            continue
        directive = parts[0].lower()
        if directive not in {"db", "dw", "dl"}:
            continue
        operand_text = parts[1] if len(parts) > 1 else ""
        fallback_values = _directive_comment_fallback_values(raw_line, directive)
        output.extend(_directive_bytes_with_fallback(directive, operand_text, fallback_values))
    return bytes(output)


def parse_maps_graphics_asm(path: Path = MAPS_GRAPHICS_ASM_PATH) -> tuple[list[str], dict[str, bytes]]:
    """Parse HM-Decomp's map graphics table into labels plus entry byte payloads."""
    table_labels: list[str] = []
    entry_bytes: dict[str, bytes] = {}
    current_label: str | None = None
    current_payload = bytearray()
    in_table = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(";", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        label_match = re.match(r"^([A-Za-z0-9_]+):", stripped)
        if label_match:
            if current_label is not None:
                entry_bytes[current_label] = bytes(current_payload)
                current_label = None
                current_payload = bytearray()

            label = label_match.group(1)
            in_table = label == "Maps_Graphics_Table"
            if not in_table:
                current_label = label
            continue

        if in_table:
            table_match = re.match(r"^dw\s+([A-Za-z_][A-Za-z0-9_]*)$", stripped)
            if table_match:
                table_labels.append(table_match.group(1))
            continue

        if current_label is None:
            continue

        parts = stripped.split(None, 1)
        directive = parts[0].lower()
        if directive not in {"db", "dw", "dl"}:
            continue
        operand_text = parts[1] if len(parts) > 1 else ""
        fallback_values = _directive_comment_fallback_values(raw_line, directive)
        current_payload.extend(_directive_bytes_with_fallback(directive, operand_text, fallback_values))

    if current_label is not None:
        entry_bytes[current_label] = bytes(current_payload)

    return table_labels, entry_bytes


def parse_labeled_data_asm(path: Path) -> dict[str, bytes]:
    """Parse contiguous db/dw/dl blocks keyed by their label."""
    blocks: dict[str, bytes] = {}
    current_label: str | None = None
    current_payload = bytearray()
    collecting = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(";", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        label_match = re.match(r"^([A-Za-z0-9_]+):", stripped)
        if label_match:
            if current_label is not None and collecting:
                blocks[current_label] = bytes(current_payload)
            current_label = label_match.group(1)
            current_payload = bytearray()
            collecting = False
            stripped = stripped.split(":", 1)[1].strip()
            if not stripped:
                continue

        if current_label is None:
            continue

        parts = stripped.split(None, 1)
        directive = parts[0].lower()
        if directive in {"db", "dw", "dl"}:
            operand_text = parts[1] if len(parts) > 1 else ""
            fallback_values = _directive_comment_fallback_values(raw_line, directive)
            current_payload.extend(_directive_bytes_with_fallback(directive, operand_text, fallback_values))
            collecting = True
            continue

        if collecting:
            blocks[current_label] = bytes(current_payload)
        current_label = None
        current_payload = bytearray()
        collecting = False

    if current_label is not None and collecting:
        blocks[current_label] = bytes(current_payload)

    return blocks

