#!/usr/bin/env python3
"""Train a small frozen MLP backbone and coordinate one serial FedAvg client."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Sequence

CLASS_NAMES = ("parado", "carregando", "vestindo")
CLASS_TO_FOLDER = {"parado": "still", "carregando": "carrying", "vestindo": "wearing"}
WINDOW_SIZE = 32
FEATURE_COUNT = 14
HIDDEN_1 = 16
EMBEDDING_SIZE = 8
CLASS_COUNT = 3
BACKBONE_PARAMETER_COUNT = 404
HEAD_PARAMETER_COUNT = 27


def load_model_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or artifact.get("format") != "fedppe-mlp-v1":
        raise ValueError(f"Formato de modelo incompatível: {path}")

    backbone = artifact.get("backbone")
    head = artifact.get("head")
    if not isinstance(backbone, list) or len(backbone) != BACKBONE_PARAMETER_COUNT:
        raise ValueError(f"Artefato não contém os {BACKBONE_PARAMETER_COUNT} parâmetros esperados do backbone")
    if not isinstance(head, list) or len(head) != HEAD_PARAMETER_COUNT:
        raise ValueError(f"Artefato não contém os {HEAD_PARAMETER_COUNT} parâmetros esperados da cabeça")

    try:
        artifact["backbone"] = [float(value) for value in backbone]
        artifact["head"] = [float(value) for value in head]
    except (TypeError, ValueError) as error:
        raise ValueError("Artefato contém parâmetro não numérico") from error
    if not all(math.isfinite(value) for value in artifact["backbone"] + artifact["head"]):
        raise ValueError("Artefato contém parâmetro NaN ou infinito")
    return artifact


def backbone_fingerprint(values: Sequence[float]) -> str:
    packed_values = struct.pack(f"<{len(values)}f", *values)
    return hashlib.sha256(packed_values).hexdigest()


def c_float(value: float) -> str:
    formatted = format(value, ".17g")
    if "." not in formatted and "e" not in formatted.lower():
        formatted += ".0"
    return formatted + "f"


def export_backbone(args) -> None:
    artifact = load_model_artifact(args.model)
    backbone = artifact["backbone"]
    offset = 2 * FEATURE_COUNT
    w1_count = HIDDEN_1 * FEATURE_COUNT
    b1_offset = offset + w1_count
    w2_offset = b1_offset + HIDDEN_1
    w2_count = EMBEDDING_SIZE * HIDDEN_1
    b2_offset = w2_offset + w2_count

    arrays = (
        ("g_input_mean", "FEDPPE_FEATURES", backbone[:FEATURE_COUNT], 0),
        ("g_input_inverse_std", "FEDPPE_FEATURES",
         backbone[FEATURE_COUNT:offset], 0),
        ("g_w1", "FEDPPE_HIDDEN_1][FEDPPE_FEATURES",
         backbone[offset:b1_offset], FEATURE_COUNT),
        ("g_b1", "FEDPPE_HIDDEN_1", backbone[b1_offset:w2_offset], 0),
        ("g_w2", "FEDPPE_EMBEDDING_SIZE][FEDPPE_HIDDEN_1",
         backbone[w2_offset:b2_offset], HIDDEN_1),
        ("g_b2", "FEDPPE_EMBEDDING_SIZE", backbone[b2_offset:], 0),
    )
    lines = [
        "#ifndef FEDPPE_BACKBONE_H",
        "#define FEDPPE_BACKBONE_H",
        "",
        f'#define FEDPPE_BACKBONE_FINGERPRINT "{backbone_fingerprint(backbone)}"',
        "",
    ]
    for name, dimensions, values, row_width in arrays:
        lines.append(f"static const float {name}[{dimensions}] = {{")
        if row_width:
            for start in range(0, len(values), row_width):
                row = ", ".join(c_float(value) for value in values[start:start + row_width])
                lines.append(f"  {{ {row} }},")
        else:
            for start in range(0, len(values), 6):
                row = ", ".join(c_float(value) for value in values[start:start + 6])
                lines.append(f"  {row},")
        lines.extend(("};", ""))
    lines.extend(("#endif", ""))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="ascii")
    print(f"Backbone exportado para {args.output}")
    print(f"fingerprint={backbone_fingerprint(backbone)}")


def sensor_features(window: Sequence[Sequence[float]], first_channel: int) -> list[float]:
    axes = [row[first_channel:first_channel + 3] for row in window]
    magnitudes = [math.sqrt(sum(value * value for value in sample)) for sample in axes]
    deltas = [
        math.sqrt(sum((axes[i][axis] - axes[i - 1][axis]) ** 2 for axis in range(3)))
        for i in range(1, len(axes))
    ]

    def std(values: Sequence[float]) -> float:
        mean = sum(values) / len(values)
        return math.sqrt(max(0.0, sum(value * value for value in values) / len(values) - mean * mean))

    result = [
        sum(magnitudes) / len(magnitudes),
        std(magnitudes),
        sum(deltas) / len(deltas),
        std(deltas),
    ]
    for axis in range(3):
        values = [sample[axis] for sample in axes]
        result.append(std(values))
    return [value * 0.001 for value in result]


def extract_features(window: Sequence[Sequence[float]]) -> list[float]:
    return sensor_features(window, 0) + sensor_features(window, 3)


def parse_csv_stream(stream, description: str, label: int) -> list[tuple[list[float], int]]:
    reader = csv.DictReader(stream)
    required = ("x", "y", "z", "x_g", "y_g", "z_g")
    if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
        raise ValueError(f"{description}: CSV precisa das colunas {', '.join(required)}")

    windows: list[tuple[list[float], int]] = []
    rows: list[list[float]] = []
    for line_number, row in enumerate(reader, start=2):
        try:
            sample = [float(row[name]) for name in required]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{description}:{line_number}: amostra inválida") from error
        if not all(math.isfinite(value) for value in sample):
            raise ValueError(f"{description}:{line_number}: amostra contém NaN ou infinito")
        rows.append(sample)
        if len(rows) == WINDOW_SIZE:
            windows.append((extract_features(rows), label))
            rows.clear()
    return windows


def load_dataset(dataset: Path, ppe: str) -> list[tuple[list[float], int]]:
    all_windows: list[tuple[list[float], int]] = []
    ppe_lower = ppe.lower()
    if dataset.is_file() and zipfile.is_zipfile(dataset):
        with zipfile.ZipFile(dataset) as archive:
            for label, class_name in enumerate(CLASS_NAMES):
                folder = CLASS_TO_FOLDER[class_name].lower()
                candidates = [
                    name for name in archive.namelist()
                    if name.lower().endswith(f"/{ppe_lower}/{folder}/output.csv")
                ]
                if not candidates:
                    raise FileNotFoundError(f"{ppe}/{folder}/output.csv não encontrado em {dataset}")
                for name in candidates:
                    with archive.open(name) as raw:
                        import io
                        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                            all_windows.extend(parse_csv_stream(text, name, label))
    elif dataset.is_dir():
        for label, class_name in enumerate(CLASS_NAMES):
            folder = CLASS_TO_FOLDER[class_name]
            candidates = [
                dataset / ppe / folder / "output.csv",
                dataset / folder / "output.csv",
            ]
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                raise FileNotFoundError(
                    f"Não encontrei {ppe}/{folder}/output.csv abaixo de {dataset}"
                )
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                all_windows.extend(parse_csv_stream(stream, str(path), label))
    else:
        raise FileNotFoundError(f"Dataset não encontrado: {dataset}")

    counts = [sum(label == i for _, label in all_windows) for i in range(CLASS_COUNT)]
    if any(count == 0 for count in counts):
        raise ValueError(f"Cada classe precisa ter pelo menos uma janela de 32 amostras; contagens={counts}")
    return all_windows


def make_network(rng: random.Random):
    def matrix(rows: int, columns: int, scale: float):
        return [[rng.uniform(-scale, scale) for _ in range(columns)] for _ in range(rows)]

    return {
        "w1": matrix(HIDDEN_1, FEATURE_COUNT, math.sqrt(6.0 / FEATURE_COUNT)),
        "b1": [0.0] * HIDDEN_1,
        "w2": matrix(EMBEDDING_SIZE, HIDDEN_1, math.sqrt(6.0 / HIDDEN_1)),
        "b2": [0.0] * EMBEDDING_SIZE,
        "wh": matrix(CLASS_COUNT, EMBEDDING_SIZE, math.sqrt(6.0 / EMBEDDING_SIZE)),
        "bh": [0.0] * CLASS_COUNT,
    }


def forward(model, raw_features, mean, inverse_std):
    normalized = [(raw_features[i] - mean[i]) * inverse_std[i] for i in range(FEATURE_COUNT)]
    z1 = [
        model["b1"][i] + sum(model["w1"][i][j] * normalized[j] for j in range(FEATURE_COUNT))
        for i in range(HIDDEN_1)
    ]
    h1 = [max(0.0, value) for value in z1]
    z2 = [
        model["b2"][i] + sum(model["w2"][i][j] * h1[j] for j in range(HIDDEN_1))
        for i in range(EMBEDDING_SIZE)
    ]
    embedding = [max(0.0, value) for value in z2]
    logits = [
        model["bh"][i] + sum(model["wh"][i][j] * embedding[j] for j in range(EMBEDDING_SIZE))
        for i in range(CLASS_COUNT)
    ]
    maximum = max(logits)
    exponents = [math.exp(value - maximum) for value in logits]
    total = sum(exponents)
    probabilities = [value / total for value in exponents]
    return normalized, z1, h1, z2, embedding, probabilities


def train_one(model, features, label, mean, inverse_std, learning_rate):
    normalized, z1, h1, z2, embedding, probabilities = forward(model, features, mean, inverse_std)
    grad_logits = probabilities[:]
    grad_logits[label] -= 1.0

    old_wh = [row[:] for row in model["wh"]]
    grad_embedding = [
        sum(grad_logits[c] * old_wh[c][j] for c in range(CLASS_COUNT))
        for j in range(EMBEDDING_SIZE)
    ]

    for c in range(CLASS_COUNT):
        for j in range(EMBEDDING_SIZE):
            model["wh"][c][j] -= learning_rate * grad_logits[c] * embedding[j]
        model["bh"][c] -= learning_rate * grad_logits[c]

    grad_z2 = [grad_embedding[i] if z2[i] > 0.0 else 0.0 for i in range(EMBEDDING_SIZE)]
    old_w2 = [row[:] for row in model["w2"]]
    grad_h1 = [
        sum(grad_z2[i] * old_w2[i][j] for i in range(EMBEDDING_SIZE))
        for j in range(HIDDEN_1)
    ]
    for i in range(EMBEDDING_SIZE):
        for j in range(HIDDEN_1):
            model["w2"][i][j] -= learning_rate * grad_z2[i] * h1[j]
        model["b2"][i] -= learning_rate * grad_z2[i]

    grad_z1 = [grad_h1[i] if z1[i] > 0.0 else 0.0 for i in range(HIDDEN_1)]
    for i in range(HIDDEN_1):
        for j in range(FEATURE_COUNT):
            model["w1"][i][j] -= learning_rate * grad_z1[i] * normalized[j]
        model["b1"][i] -= learning_rate * grad_z1[i]


def accuracy(model, dataset, mean, inverse_std) -> float:
    correct = 0
    for features, label in dataset:
        probabilities = forward(model, features, mean, inverse_std)[-1]
        correct += int(max(range(CLASS_COUNT), key=probabilities.__getitem__) == label)
    return correct / len(dataset) if dataset else 0.0


def flatten_backbone(model, mean, inverse_std) -> list[float]:
    values = list(mean) + list(inverse_std)
    values.extend(value for row in model["w1"] for value in row)
    values.extend(model["b1"])
    values.extend(value for row in model["w2"] for value in row)
    values.extend(model["b2"])
    if len(values) != BACKBONE_PARAMETER_COUNT:
        raise AssertionError(f"Backbone possui {len(values)} parâmetros, esperado {BACKBONE_PARAMETER_COUNT}")
    return values


def flatten_head(model) -> list[float]:
    values = [value for row in model["wh"] for value in row] + list(model["bh"])
    if len(values) != HEAD_PARAMETER_COUNT:
        raise AssertionError(f"Cabeça possui {len(values)} parâmetros, esperado {HEAD_PARAMETER_COUNT}")
    return values


def train_backbone(args) -> None:
    if args.epochs <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs e learning-rate precisam ser maiores que zero")
    dataset = load_dataset(args.dataset, args.ppe)
    train = [item for i, item in enumerate(dataset) if i % 5 != 0]
    validation = [item for i, item in enumerate(dataset) if i % 5 == 0]
    if not train:
        raise ValueError("Não há janelas suficientes para treino")

    mean = [sum(features[i] for features, _ in train) / len(train) for i in range(FEATURE_COUNT)]
    variance = [
        sum((features[i] - mean[i]) ** 2 for features, _ in train) / len(train)
        for i in range(FEATURE_COUNT)
    ]
    inverse_std = [1.0 / max(math.sqrt(value), 1e-6) for value in variance]

    rng = random.Random(args.seed)
    model = make_network(rng)
    for epoch in range(args.epochs):
        rng.shuffle(train)
        for features, label in train:
            train_one(model, features, label, mean, inverse_std, args.learning_rate)
        if validation:
            print(f"epoch={epoch + 1}/{args.epochs} validation_accuracy="
                  f"{accuracy(model, validation, mean, inverse_std):.4f}")

    artifact = {
        "format": "fedppe-mlp-v1",
        "classes": list(CLASS_NAMES),
        "window_size": WINDOW_SIZE,
        "feature_count": FEATURE_COUNT,
        "hidden_1": HIDDEN_1,
        "embedding_size": EMBEDDING_SIZE,
        "backbone": flatten_backbone(model, mean, inverse_std),
        "head": flatten_head(model),
        "training_windows": len(train),
        "validation_windows": len(validation),
        "note": "Head is a centrally trained initialization; MCU updates only this head.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Modelo salvo em {args.output}")
    print(f"janelas treino={len(train)} validação={len(validation)}")
    print(f"backbone={BACKBONE_PARAMETER_COUNT} parâmetros; cabeça inicial={HEAD_PARAMETER_COUNT}")


def fedavg(client_models: Sequence[tuple[int, Sequence[float]]]) -> list[float]:
    if not client_models:
        raise ValueError("FedAvg requer pelo menos um cliente")
    parameter_count = len(client_models[0][1])
    total_samples = 0
    aggregate = [0.0] * parameter_count
    for samples, weights in client_models:
        if samples <= 0 or len(weights) != parameter_count:
            raise ValueError("Contagem de amostras ou tamanho de modelo inválido")
        total_samples += samples
        for i, value in enumerate(weights):
            aggregate[i] += samples * value
    return [value / total_samples for value in aggregate]


class SerialLink:
    def __init__(self, port: str, baud: int):
        try:
            import serial
        except ImportError as error:
            raise RuntimeError("Instale pyserial com: pip install pyserial") from error
        self.serial = serial.Serial(port, baudrate=baud, timeout=0.2, write_timeout=2)
        time.sleep(1.5)
        self.serial.reset_input_buffer()

    def close(self):
        self.serial.close()

    def send(self, line: str):
        self.serial.write((line + "\n").encode("ascii"))

    def read_line(self) -> str:
        return self.serial.readline().decode("ascii", errors="replace").strip()

    def wait_prefix(self, prefix: str, timeout: float = 10.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.read_line()
            if not line:
                continue
            if line.startswith("FEDPPE_ERROR"):
                raise RuntimeError(f"Firmware reportou: {line}")
            if line.startswith(prefix):
                return line
            if not line.startswith("H_ACK "):
                print(f"[MCU] {line}")
        raise TimeoutError(f"Timeout aguardando {prefix}")

    def set_parameters(self, prefix: str, values: Sequence[float], ack_prefix: str):
        for index, value in enumerate(values):
            self.send(f"{prefix} {index} {value:.9g}")
            self.wait_prefix(f"{ack_prefix} {index}", timeout=3.0)

    def dump_head(self) -> tuple[int, list[float]]:
        self.send("fl_dump")
        header = self.wait_prefix("FL_MODEL_BEGIN")
        fields = header.split()
        if len(fields) != 3 or int(fields[1]) != HEAD_PARAMETER_COUNT:
            raise RuntimeError(f"Cabeçalho do modelo inválido: {header}")
        sample_count = int(fields[2])
        weights = [math.nan] * HEAD_PARAMETER_COUNT
        deadline = time.monotonic() + 10.0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timeout recebendo os parâmetros da cabeça")
            line = self.read_line()
            if not line:
                continue
            if line == "FL_MODEL_END":
                break
            if line.startswith("FEDPPE_ERROR"):
                raise RuntimeError(f"Firmware reportou: {line}")
            fields = line.split()
            if len(fields) == 3 and fields[0] == "FL_WEIGHT":
                index = int(fields[1])
                value = float(fields[2])
                if index < 0 or index >= HEAD_PARAMETER_COUNT or not math.isfinite(value):
                    raise RuntimeError(f"Peso inválido recebido: {line}")
                weights[index] = value
        if any(not math.isfinite(value) for value in weights):
            raise RuntimeError("Dump da cabeça incompleto")
        return sample_count, weights


def run_serial(args):
    artifact = load_model_artifact(args.model)

    global_head = list(artifact["head"])
    if args.state.is_file():
        state = json.loads(args.state.read_text(encoding="utf-8"))
        if state.get("format") != "fedppe-head-v1" or len(state.get("weights", [])) != HEAD_PARAMETER_COUNT:
            raise ValueError(f"Estado global incompatível: {args.state}")
        global_head = [float(value) for value in state["weights"]]
        if not all(math.isfinite(value) for value in global_head):
            raise ValueError(f"Estado global contém peso NaN ou infinito: {args.state}")

    link = SerialLink(args.port, args.baud)
    try:
        link.send("status")
        status = link.wait_prefix("FEDPPE_STATUS", timeout=10.0)
        status_fields = dict(
            field.split("=", 1)
            for field in status.split()[1:]
            if "=" in field
        )
        if status_fields.get("backbone") != f"{BACKBONE_PARAMETER_COUNT}/{BACKBONE_PARAMETER_COUNT}":
            raise RuntimeError(f"Firmware não informa um backbone fixo completo: {status}")
        expected_fingerprint = backbone_fingerprint(artifact["backbone"])
        if status_fields.get("fingerprint") != expected_fingerprint:
            raise RuntimeError(
                "O backbone do firmware não corresponde ao modelo "
                f"{args.model}; exporte o backbone novamente e regrave o firmware"
            )
        print("Backbone fixo verificado; enviando apenas a cabeça global "
              f"({HEAD_PARAMETER_COUNT} parâmetros)...")
        print("Carregando cabeça global...")
        link.set_parameters("H", global_head, "H_ACK")
        print("Pronto. Treino manual: train CLASS WINDOWS (0=parado, 1=carregando, 2=vestindo).")
        print("Comandos: train C N | infer | dump | quit")

        while True:
            command = input("fedppe> ").strip()
            if command in ("quit", "exit"):
                break
            if command == "infer":
                link.send("status")
                link.wait_prefix("FEDPPE_STATUS")
                continue
            if command == "dump":
                _, weights = link.dump_head()
                print("Cabeça atual:")
                for index, value in enumerate(weights):
                    print(f"{index}: {value:.9g}")
                continue
            parts = command.split()
            if len(parts) != 3 or parts[0] != "train":
                print("Use: train CLASS WINDOWS, dump, infer ou quit")
                continue
            label, windows = int(parts[1]), int(parts[2])
            if label not in range(CLASS_COUNT) or windows <= 0:
                print("Classe deve ser 0, 1 ou 2; janelas deve ser maior que zero.")
                continue
            print(f"Coloque a placa no estado '{CLASS_NAMES[label]}' durante a coleta.")
            link.send(f"train {label} {windows}")
            done = link.wait_prefix(
                "FL_LOCAL_DONE",
                timeout=(windows * WINDOW_SIZE / 25.0) + 45.0,
            )
            local_samples = int(done.split()[1])
            _, local_head = link.dump_head()
            global_head = fedavg([(local_samples, local_head)])
            link.set_parameters("H", global_head, "H_ACK")
            args.state.parent.mkdir(parents=True, exist_ok=True)
            args.state.write_text(
                json.dumps({
                    "format": "fedppe-head-v1",
                    "classes": list(CLASS_NAMES),
                    "weights": global_head,
                    "clients_in_round": 1,
                    "samples_in_round": local_samples,
                }, indent=2),
                encoding="utf-8",
            )
            print(f"Rodada concluída: 1 cliente real, {local_samples} janelas.")
            print("Com um cliente, FedAvg resulta nos pesos desse cliente; a média federada real exige >=2 clientes.")
    finally:
        link.close()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train-backbone", help="Treina e exporta o MLP congelado")
    train.add_argument("--dataset", type=Path, required=True,
                       help="ZIP nn_offline_training-main.zip ou diretório com os CSVs")
    train.add_argument("--ppe", default="Helmet", help="Tipo de EPI; padrão: Helmet")
    train.add_argument("--output", type=Path, default=Path("fedppe_mlp.json"))
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--seed", type=int, default=7)
    train.set_defaults(func=train_backbone)

    export = commands.add_parser(
        "export-backbone",
        help="Exporta o backbone treinado como cabeçalho C para o firmware",
    )
    export.add_argument("--model", type=Path, default=Path("fedppe_mlp.json"))
    export.add_argument("--output", type=Path, default=Path("fedppe_backbone.h"))
    export.set_defaults(func=export_backbone)

    serial = commands.add_parser("serial", help="Conecta ao MCU e coordena treino local/FedAvg")
    serial.add_argument("--port", required=True, help="Porta COM virtual ligada à VCOM do kit")
    serial.add_argument("--baud", type=int, default=115200)
    serial.add_argument("--model", type=Path, default=Path("fedppe_mlp.json"))
    serial.add_argument("--state", type=Path, default=Path("fedppe_global_head.json"))
    serial.set_defaults(func=run_serial)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, RuntimeError, TimeoutError, zipfile.BadZipFile) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
