"""Ingestão, validação, diagnóstico observacional e registro MIH-V1.

O sistema NÃO infere uma falha física confirmada a partir de sintomas.
Biblioteca padrão do Python; nunca conecta ao robô nem executa código importado.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
from statistics import median

SCHEMA = "MIH-V1"
SOURCES = {"real", "simulation", "synthetic_lab"}
FIELDS = (
    "time_ms", "motor_me_command", "motor_md_command",
    "encoder_me_ticks", "encoder_md_ticks", "imu_yaw_deg",
    "battery_voltage_v", "target_distance_m", "target_yaw_deg",
    "observed_distance_m", "pose_x_m", "pose_y_m", "pose_yaw_deg",
)
NUMERIC = set(FIELDS)
META = ("schema_version", "run_id", "source", "timestamp",
        "robot_config_id", "code_version", "experiment_id", "samples")
LABEL_PROVENANCE = {"unlabeled", "operator", "synthetic_injected", "verified_measurement"}
TERMINATION = {"unknown", "completed", "timeout", "stopped", "error"}
ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$")


class ContractError(ValueError):
    """Dados não obedecem ao contrato MIH-V1."""


def _number(value, key, index):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ContractError(f"amostra {index}: {key} booleano não é numérico")
    try:
        n = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ContractError(f"amostra {index}: {key} não é numérico") from exc
    if not math.isfinite(n):
        raise ContractError(f"amostra {index}: {key} não é finito")
    if key.endswith("_command") and not -1.0 <= n <= 1.0:
        raise ContractError(f"amostra {index}: {key} fora de [-1, 1]")
    if key == "battery_voltage_v" and not 0 <= n <= 35:
        raise ContractError(f"amostra {index}: tensão fora de [0, 35] V")
    if key == "target_distance_m" and n < 0:
        raise ContractError(f"amostra {index}: distância alvo negativa")
    return n


def validate(run):
    if not isinstance(run, dict):
        raise ContractError("a execução deve ser objeto JSON")
    for key in META:
        if key not in run:
            raise ContractError(f"metadado obrigatório ausente: {key}")
    if run["schema_version"] != SCHEMA:
        raise ContractError("schema_version incompatível: esperado MIH-V1")
    if not isinstance(run["run_id"], str) or not ID_PATTERN.fullmatch(run["run_id"]):
        raise ContractError("run_id inválido (use letras, dígitos, _.-)")
    if run["source"] not in SOURCES:
        raise ContractError("source deve ser real, simulation ou synthetic_lab")
    try:
        stamp = datetime.fromisoformat(run["timestamp"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("timezone ausente")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ContractError("timestamp deve ser ISO 8601 com timezone") from exc
    for key in ("robot_config_id", "code_version", "experiment_id"):
        if not isinstance(run[key], str) or not run[key].strip():
            raise ContractError(f"{key} obrigatório (use 'unknown' explicitamente se desconhecido)")
    prov = run.get("label_provenance", "unlabeled")
    if prov not in LABEL_PROVENANCE:
        raise ContractError("label_provenance inválido")
    if prov == "synthetic_injected" and run["source"] == "real":
        raise ContractError("dados reais não podem ter rótulo de falha injetada sintética")
    if prov == "verified_measurement" and run["source"] != "real":
        raise ContractError("verificação física exige source=real")
    if run.get("result_label") is not None and not isinstance(run["result_label"], str):
        raise ContractError("result_label deve ser string ou null")
    if run.get("termination_reason", "unknown") not in TERMINATION:
        raise ContractError("termination_reason inválido")
    if not isinstance(run["samples"], list) or not run["samples"]:
        raise ContractError("samples precisa conter ao menos uma amostra")
    if len(run["samples"]) > 200_000:
        raise ContractError("limite de 200000 amostras excedido")
    previous = -math.inf
    for i, sample in enumerate(run["samples"]):
        if not isinstance(sample, dict):
            raise ContractError(f"amostra {i}: esperado objeto")
        extra = set(sample) - NUMERIC
        if extra:
            raise ContractError(f"amostra {i}: campos não reconhecidos: {sorted(extra)}")
        if "time_ms" not in sample:
            raise ContractError(f"amostra {i}: time_ms obrigatório")
        for key, val in sample.items():
            _number(val, key, i)
        t = _number(sample["time_ms"], "time_ms", i)
        if t is None or t < 0 or t <= previous:
            raise ContractError(f"amostra {i}: time_ms deve ser crescente e >= 0")
        previous = t
    return run


def _parse_rows(text, suffix):
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ContractError("CSV sem cabeçalho")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ContractError("colunas CSV duplicadas")
        rows = list(reader)
        if any(None in row for row in rows):
            raise ContractError("CSV com número de colunas incorreto")
        return reader.fieldnames, rows
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ContractError("JSON deve conter objeto")
    if "columns" in obj and "rows" in obj:
        cols = obj["columns"]
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ContractError("columns inválido")
        if len(set(cols)) != len(cols):
            raise ContractError("columns duplicadas")
        rows = obj["rows"]
        if not isinstance(rows, list) or any(not isinstance(r, list) or len(r) != len(cols) for r in rows):
            raise ContractError("rows não correspondem a columns")
        return cols, [dict(zip(cols, row)) for row in rows]
    raise ContractError("JSON não é gravação {columns,rows}; exportação de resultado não é série temporal")


def _kind(cols):
    names = set(cols)
    if {"time_s", "x_m", "y_m", "yaw_deg"}.issubset(names):
        return "meca_sim_recording"
    if {"timestamp", "imu_yaw_deg", "encoder_md_ticks", "encoder_me_ticks",
        "potencia_md", "potencia_me"}.issubset(names):
        return "neural_lab_csv"
    if "time_ms" in names:
        if not names.issubset(NUMERIC):
            raise ContractError(f"CSV MIH contém campos não reconhecidos: {sorted(names - NUMERIC)}")
        return "mih_csv"
    raise ContractError("colunas desconhecidas; veja docs/MIH-DATA-CONTRACT-v1.md")


def import_file(path, *, source, run_id, timestamp, robot_config_id,
                code_version, experiment_id, label=None,
                label_provenance="unlabeled", termination_reason="unknown",
                notes=""):
    """Converte exportações do Meca Sim, laboratório neural e MIH CSV em MIH-V1.

    'source' é declarado pelo operador; não há verificação criptográfica de origem.
    """
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ContractError("limite de importação: 32 MiB")
    binary = path.read_bytes()
    try:
        text = binary.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ContractError("arquivo não é UTF-8") from exc
    suffix = path.suffix.lower()
    if suffix not in {".json", ".csv"}:
        raise ContractError("formatos permitidos: JSON e CSV")
    if suffix == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("schema_version") == SCHEMA:
            validate(obj)
            if obj["source"] != source:
                raise ContractError("source declarado difere do arquivo MIH-V1")
            return obj
    cols, rows = _parse_rows(text, suffix)
    kind = _kind(cols)
    if kind == "meca_sim_recording" and source != "simulation":
        raise ContractError("exportação /api/recording do Meca Sim exige source=simulation")
    if not rows:
        raise ContractError("arquivo sem amostras")
    converted = []
    mapping = {"ME_ticks": "encoder_me_ticks", "MD_ticks": "encoder_md_ticks",
               "x_m": "pose_x_m", "y_m": "pose_y_m", "yaw_deg": "pose_yaw_deg",
               "potencia_me": "motor_me_command", "potencia_md": "motor_md_command",
               "timestamp": "time_ms"}
    for i, row in enumerate(rows):
        item = {}
        for key, value in row.items():
            canonical = mapping.get(key, key)
            if canonical not in NUMERIC:
                continue  # Ex.: MS_ticks e MI_ticks não são canais diagnósticos do diferencial.
            number = _number(value, canonical, i)
            if key == "time_s" or canonical == "time_ms" and key == "time_s":
                number = number * 1000 if number is not None else None
            item[canonical] = number
        if kind == "meca_sim_recording":
            item["time_ms"] = _number(row.get("time_s"), "time_ms", i) * 1000
        converted.append(item)
    run = {
        "schema_version": SCHEMA, "run_id": run_id, "source": source,
        "timestamp": timestamp, "robot_config_id": robot_config_id,
        "code_version": code_version, "experiment_id": experiment_id,
        "result_label": label, "label_provenance": label_provenance,
        "termination_reason": termination_reason, "notes": notes,
        "import_format": kind, "source_sha256": hashlib.sha256(binary).hexdigest(),
        "samples": converted,
    }
    validate(run)
    return run


def _angle_delta(a, b):
    return (b - a + 180) % 360 - 180


def _difference(run, key, first, last):
    a = run["samples"][first].get(key)
    b = run["samples"][last].get(key)
    return None if a is None or b is None else b - a


def diagnose(run):
    """Regras conservadoras. Toda detecção é suspeita, nunca prova de falha mecânica."""
    validate(run)
    s = run["samples"]
    issues = []
    missing = [key for key in ("motor_me_command", "motor_md_command",
               "encoder_me_ticks", "encoder_md_ticks", "imu_yaw_deg")
               if sum(row.get(key) is not None for row in s) < 2]

    def add(code, message, evidence, severity="warning"):
        if code not in {v["code"] for v in issues}:
            issues.append({"code": code, "severity": severity,
                           "classification": "suspected" if severity == "warning" else "observed",
                           "message": message, "evidence": evidence})

    if run.get("termination_reason") == "timeout":
        add("TIMEOUT_REPORTED", "A execução foi marcada como timeout pelo operador/exportador.",
            {"origin": "termination_reason"}, "observed")

    if len(s) > 2:
        gaps = [s[i]["time_ms"] - s[i - 1]["time_ms"] for i in range(1, len(s))]
        dt = median(gaps)
        jumps = [round(v, 2) for v in gaps if v > max(250, 3 * dt)]
        if jumps:
            add("SAMPLING_GAP", "Intervalo irregular na captura; diagnósticos podem perder eventos.",
                {"largest_gap_ms": max(jumps), "median_gap_ms": round(dt, 2)}, "observed")

    # Janelas >=700 ms, com comandos e encoders em ambas extremidades.
    for side in ("me", "md"):
        command, encoder = f"motor_{side}_command", f"encoder_{side}_ticks"
        for start in range(len(s) - 1):
            if s[start].get(command) is None or s[start].get(encoder) is None:
                continue
            end = next((j for j in range(start + 1, len(s))
                        if s[j]["time_ms"] - s[start]["time_ms"] >= 700
                        and s[j].get(encoder) is not None), None)
            if end is None:
                break
            section = s[start:end+1]
            cmds = [v.get(command) for v in section]
            if any(v is None or abs(v) < .25 for v in cmds):
                continue
            if abs(s[end][encoder] - s[start][encoder]) <= 3:
                add(f"NO_ENCODER_PROGRESS_{side.upper()}",
                    "Potência significativa sem variação do encoder; investigar ligação, encoder, carga e motor.",
                    {"from_ms": s[start]["time_ms"], "to_ms": s[end]["time_ms"],
                     "encoder_delta_ticks": s[end][encoder] - s[start][encoder]})
            break

    # A IMU travada só pode ser suspeitada se o robô foi comandado a girar,
    # ambos encoders tiveram progresso e a leitura angular permaneceu quase fixa.
    for start in range(len(s)-1):
        end = next((j for j in range(start + 1, len(s))
                    if s[j]["time_ms"] - s[start]["time_ms"] >= 700), None)
        if end is None:
            break
        block = s[start:end+1]
        if any(any(row.get(k) is None for k in (
               "motor_me_command", "motor_md_command", "encoder_me_ticks",
               "encoder_md_ticks", "imu_yaw_deg")) for row in block):
            continue
        commands = [(row["motor_me_command"], row["motor_md_command"]) for row in block]
        if not all(l * r < -.04 and abs(l-r) >= .45 for l, r in commands):
            continue
        le = _difference(run, "encoder_me_ticks", start, end)
        ri = _difference(run, "encoder_md_ticks", start, end)
        if min(abs(le), abs(ri)) < 45:
            continue
        yaw_change = sum(_angle_delta(a["imu_yaw_deg"], b["imu_yaw_deg"])
                         for a, b in zip(block, block[1:]))
        if abs(yaw_change) < .8:
            add("IMU_NO_ANGULAR_PROGRESS",
                "Encoders avançaram durante comando de giro, mas a IMU quase não variou; verifique orientação, reset e sensor.",
                {"from_ms": s[start]["time_ms"], "to_ms": s[end]["time_ms"],
                 "imu_delta_deg": round(yaw_change, 3),
                 "encoder_me_delta": le, "encoder_md_delta": ri})
        break

    if all(any(v.get(k) is not None for v in s) for k in (
           "motor_me_command", "motor_md_command", "encoder_me_ticks", "encoder_md_ticks")):
        first, last = s[0], s[-1]
        if last["time_ms"] - first["time_ms"] >= 700:
            lcmd = [v.get("motor_me_command") for v in s]
            rcmd = [v.get("motor_md_command") for v in s]
            if all(x is not None and y is not None and x*y>.09 and abs(abs(x)-abs(y))<.10
                   for x,y in zip(lcmd,rcmd)):
                dl = _difference(run,"encoder_me_ticks",0,len(s)-1)
                dr = _difference(run,"encoder_md_ticks",0,len(s)-1)
                if dl is not None and dr is not None and max(abs(dl), abs(dr)) > 50:
                    ratio = abs(abs(dl)-abs(dr)) / max(abs(dl),abs(dr))
                    if ratio > .35:
                        add("WHEEL_PROGRESS_ASYMMETRY",
                            "Avanço assimétrico entre encoders; possíveis causas incluem carga e patinagem.",
                            {"delta_me_ticks":dl, "delta_md_ticks":dr,"relative_difference":round(ratio,3)})

    target = next((x.get("target_distance_m") for x in reversed(s)
                   if x.get("target_distance_m") is not None), None)
    actual = next((x.get("observed_distance_m") for x in reversed(s)
                   if x.get("observed_distance_m") is not None), None)
    if target is not None and actual is not None and target > .1:
        error = actual - target
        if abs(error) > max(.08, .12 * target):
            add("DISTANCE_TARGET_MISMATCH",
                "Distância observada diverge do objetivo; confirme o método de medição.",
                {"target_m":target,"observed_m":actual,"error_m":round(error,3)})
    if len(s) < 2:
        add("INSUFFICIENT_SAMPLES", "Menos de duas amostras; evolução temporal indisponível.",
            {"samples":len(s)},"observed")

    return {
        "run_id": run["run_id"], "source": run["source"], "schema_version": SCHEMA,
        "status": "anomalies_observed_or_suspected" if issues else "indeterminate" if missing else "no_anomaly_observed",
        "issues": issues, "missing_channels": missing, "sample_count": len(s),
        "duration_ms": round(s[-1]["time_ms"] - s[0]["time_ms"], 3),
        "limitations": [
            "Sinais indicativos não demonstram causa física nem dispensam verificação do robô.",
            "A ausência de alertas não garante funcionamento correto.",
            "O formato de trajetória Meca Sim contém pose (yaw real simulado), não leitura IMU.",
            "A origem real/simulation é declarada pelo importador, não autenticada.",
        ],
    }


class Registry:
    """SQLite local, unicidade por run_id; rejeita sobrescritas conflitantes."""

    def __init__(self, filename):
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, source TEXT NOT NULL, "
                        "digest TEXT NOT NULL, document TEXT NOT NULL)")
        self.db.commit()

    def add(self, run):
        validate(run)
        document = json.dumps(run, ensure_ascii=False, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(document.encode()).hexdigest()
        old = self.db.execute("SELECT digest FROM runs WHERE run_id=?", (run["run_id"],)).fetchone()
        if old:
            if old[0] != digest:
                raise ContractError("run_id já utilizado com conteúdo diferente; use outro ID")
            return False
        self.db.execute("INSERT INTO runs VALUES (?, ?, ?, ?)",
                        (run["run_id"],run["source"],digest,document))
        self.db.commit()
        return True

    def get(self, run_id):
        row = self.db.execute("SELECT document FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run_id não encontrado: {run_id}")
        obj = json.loads(row[0])
        validate(obj)
        return obj

    def list(self):
        return [dict(run_id=r[0],source=r[1])
                for r in self.db.execute("SELECT run_id,source FROM runs ORDER BY run_id")]

    def close(self):
        self.db.close()


def compare(runs):
    if len(runs) < 2:
        raise ContractError("compare requer pelo menos duas execuções")
    summaries = []
    for r in runs:
        result = diagnose(r)
        summaries.append({"run_id":r["run_id"],"source":r["source"],
                          "code_version":r["code_version"],
                          "robot_config_id":r["robot_config_id"],
                          "duration_ms":result["duration_ms"],
                          "status":result["status"],
                          "issue_codes":[v["code"] for v in result["issues"]]})
    return {"runs":summaries,
            "warning":"Comparação descritiva; não há inferência causal nem otimização sem ensaios controlados."}
