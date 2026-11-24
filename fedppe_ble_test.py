#!/usr/bin/env python3
"""Verify bidirectional BLE GATT communication with the TCC001 MCU."""

from __future__ import annotations

import argparse
import asyncio
import math
import sys

DEVICE_NAME = "TCC001-BLE"
RX_CHARACTERISTIC = "f3641401-00b0-4240-ba50-05ca45bf8abc"
TX_CHARACTERISTIC = "f3641402-00b0-4240-ba50-05ca45bf8abc"
MAX_PAYLOAD_BYTES = 20


async def run_test(args: argparse.Namespace) -> None:
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError as error:
        raise RuntimeError("Instale a dependência com: py -m pip install bleak") from error

    payload = args.message.encode("utf-8")
    if not payload or len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("A mensagem deve ter entre 1 e 20 bytes UTF-8")
    timeouts = (args.scan_timeout, args.connect_timeout, args.response_timeout)
    if not all(math.isfinite(timeout) and timeout > 0 for timeout in timeouts):
        raise ValueError("Os timeouts devem ser números finitos maiores que zero")
    
    if args.address:
        target = args.address
    else:
        print(f"Procurando o periférico BLE {args.name}...")
        target = await BleakScanner.find_device_by_filter(
            lambda _device, advertisement: advertisement.local_name == args.name,
            timeout=args.scan_timeout,
        )
        if target is None:
            raise TimeoutError(
                f"Não encontrei {args.name}; confira se o MCU está ligado e anunciando"
            )

    response_received = asyncio.Event()
    responses: list[bytes] = []

    def on_notification(_characteristic, data: bytearray) -> None:
        responses.append(bytes(data))
        response_received.set()

    async with BleakClient(target, timeout=args.connect_timeout) as client:
        await client.start_notify(TX_CHARACTERISTIC, on_notification)
        await client.write_gatt_char(RX_CHARACTERISTIC, payload, response=True)
        await asyncio.wait_for(response_received.wait(), timeout=args.response_timeout)

    response = responses[-1]
    if response != payload:
        raise RuntimeError(
            f"Resposta BLE diferente: esperado {payload!r}, recebido {response!r}"
        )

    print(f"PASS: MCU recebeu e devolveu {response.decode('utf-8')!r}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--name", default=DEVICE_NAME, help="Nome BLE anunciado pelo MCU")
    target.add_argument("--address", help="Endereço BLE do MCU (opcional)")
    parser.add_argument("--message", default="ping-tcc001")
    parser.add_argument("--scan-timeout", type=float, default=10.0)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--response-timeout", type=float, default=5.0)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    try:
        from bleak.exc import BleakError
    except ImportError:
        print("Instale a dependência com: py -m pip install bleak", file=sys.stderr)
        return 1

    try:
        asyncio.run(run_test(args))
    except (BleakError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"Erro no teste BLE: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
