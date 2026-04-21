from __future__ import annotations

import logging
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union


@dataclass
class LogPaths:
    root_dir: Path
    date_dir: Path
    log_file: Path


@dataclass
class LoggingConfig:
    enabled: bool = True
    level: int = logging.INFO
    base_dir: Union[str, Path] = "logs"
    session_name: Optional[str] = None
    use_utc: bool = True
    to_console: bool = True
    to_file: bool = True
    reset_handlers: bool = False


def _slugify(text: str) -> str:
    allowed = set(string.ascii_letters + string.digits + "-_")
    cleaned = "".join(ch if ch in allowed else "_" for ch in text.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "session"


def build_log_paths(
    base_dir: Union[str, Path] = "logs",
    session_name: Optional[str] = None,
    use_utc: bool = True,
    now: Optional[datetime] = None,
) -> LogPaths:
    root_dir = Path(base_dir).expanduser().resolve()
    ts = now or (datetime.now(timezone.utc) if use_utc else datetime.now())
    date_dir = root_dir / ts.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    prefix = _slugify(session_name) if session_name else "session"
    log_name = f"{prefix}_{ts.strftime('%Y%m%d_%H%M%S')}.log"
    return LogPaths(root_dir=root_dir, date_dir=date_dir, log_file=date_dir / log_name)


def _has_file_handler(logger: logging.Logger, log_path: Path) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == log_path:
                    return True
            except Exception:
                continue
    return False


def _has_console_handler(logger: logging.Logger) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            return True
    return False


def setup_logging(
    level: int = logging.INFO,
    base_dir: Union[str, Path] = "logs",
    session_name: Optional[str] = None,
    use_utc: bool = True,
    to_console: bool = True,
    to_file: bool = True,
    reset_handlers: bool = False,
) -> LogPaths:
    paths = build_log_paths(base_dir=base_dir, session_name=session_name, use_utc=use_utc)
    root = logging.getLogger()

    if reset_handlers:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if to_file and not _has_file_handler(root, paths.log_file):
        file_handler = logging.FileHandler(paths.log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if to_console and not _has_console_handler(root):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    return paths
