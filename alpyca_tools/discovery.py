from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import List, Optional


DISCOVERY_PORT = 32227
DISCOVERY_PAYLOAD = b"alpacadiscovery1"


@dataclass
class DiscoveryResult:
    address: str
    alpaca_port: int


def discover(
    timeout_s: float = 2.0,
    broadcast_addrs: Optional[List[str]] = None,
    port: int = DISCOVERY_PORT,
) -> List[DiscoveryResult]:
    results: List[DiscoveryResult] = []
    broadcast_addrs = broadcast_addrs or ["255.255.255.255"]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout_s)

    try:
        for addr in broadcast_addrs:
            sock.sendto(DISCOVERY_PAYLOAD, (addr, port))

        while True:
            try:
                data, (addr, _) = sock.recvfrom(4096)
            except socket.timeout:
                break

            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception:
                continue

            alpaca_port = payload.get("AlpacaPort")
            if alpaca_port is None:
                continue

            results.append(DiscoveryResult(address=addr, alpaca_port=int(alpaca_port)))
    finally:
        sock.close()

    return results
