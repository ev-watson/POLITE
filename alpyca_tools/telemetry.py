from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional


class Telemetry:
    """Lightweight request tracing for Alpaca HTTP calls."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("alpyca_tools")

    def request_started(self, method: str, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        token = {"ts": time.time(), "method": method, "url": url}
        self.logger.debug("alpaca.request.start method=%s url=%s params=%s", method, url, params)
        return token

    def request_finished(
        self, token: Optional[Dict[str, Any]], status_code: Optional[int], error: Optional[Exception]
    ) -> None:
        if token is None:
            return
        elapsed_ms = (time.time() - token["ts"]) * 1000.0
        if error is not None:
            self.logger.warning(
                "alpaca.request.error method=%s url=%s error=%s elapsed_ms=%.1f",
                token["method"],
                token["url"],
                error,
                elapsed_ms,
            )
        else:
            self.logger.debug(
                "alpaca.request.done method=%s url=%s status=%s elapsed_ms=%.1f",
                token["method"],
                token["url"],
                status_code,
                elapsed_ms,
            )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
