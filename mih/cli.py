"""CLI MIH-V1. Sem conexão com hardware ou dependências externas."""
import argparse
import json
from pathlib import Path
import sys

from .core import ContractError, Registry, compare, diagnose, import_file


def _json(data):
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)


def _report(run, diagnosis):
    lines = [
        f"# M.E.C.A. — Execução {run['run_id']}",
        "",
        f"- [R] Origem declarada: `{run['source']}`",
        f"- [R] Registrado em: `{run['timestamp']}`",
        f"- [R] Código: `{run['code_version']}`; configuração: `{run['robot_config_id']}`",
        f"- [R] Experimento: `{run['experiment_id']}`",
        f"- [R] Rótulo informado: `{run.get('result_label')}`; proveniência: `{run.get('label_provenance', 'unlabeled')}`",
        f"- [R] Amostras: {diagnosis['sample_count']}; duração: {diagnosis['duration_ms']} ms",
        f"- [R] Status: `{diagnosis['status']}`",
        "",
        "## Evidências e hipóteses",
    ]
    if not diagnosis["issues"]:
        lines.append("[R] Nenhum gatilho determinístico foi identificado. Isso não confirma normalidade.")
    for issue in diagnosis["issues"]:
        label = "[H]" if issue["classification"] == "suspected" else "[R]"
        lines += [f"- {label} **{issue['code']}**: {issue['message']}",
                  f"  - Evidência: `{_json(issue['evidence']).replace(chr(10), ' ')}`"]
    if diagnosis["missing_channels"]:
        lines.append("- [P] Canais insuficientes: " + ", ".join(diagnosis["missing_channels"]))
    lines += ["", "## Limitações e próximos ensaios"]
    lines.extend("- [P] " + val for val in diagnosis["limitations"])
    lines += ["- [P] Coletar nova execução com canais ausentes e verificar hipótese em bancada antes de qualquer alteração do OpMode.",
              "", "**Decisão M.E.C.A.**: corrigir ou investigar conforme evidências; não aplicar ajustes automaticamente."]
    return "\n".join(lines) + "\n"


def parser():
    p = argparse.ArgumentParser(prog="python -m mih", description="MECA Intelligence Hybrid — diagnóstico offline")
    p.add_argument("--db", default="local/mih.sqlite3", help="SQLite local (não versionar)")
    sub = p.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="Importar uma gravação CSV/JSON ou execução MIH-V1")
    imp.add_argument("file", type=Path)
    imp.add_argument("--source", required=True, choices=["real", "simulation", "synthetic_lab"])
    imp.add_argument("--run-id", required=True)
    imp.add_argument("--recorded-at", required=True, help="ISO 8601 com timezone, nunca inventar horário")
    imp.add_argument("--robot-config", required=True, help="Identificador ou 'unknown'")
    imp.add_argument("--code-version", required=True, help="Commit/versão ou 'unknown'")
    imp.add_argument("--experiment-id", required=True)
    imp.add_argument("--result-label")
    imp.add_argument("--label-provenance", default="unlabeled", choices=[
        "unlabeled", "operator", "synthetic_injected", "verified_measurement"])
    imp.add_argument("--termination", default="unknown", choices=[
        "unknown", "completed", "timeout", "stopped", "error"])
    imp.add_argument("--notes", default="")
    imp.add_argument("--report", type=Path, help="Gravar relatório Markdown em arquivo local")
    run = sub.add_parser("analyze", help="Diagnóstico de um run_id")
    run.add_argument("run_id")
    run.add_argument("--report", type=Path, help="Exportar Markdown")
    sub.add_parser("list", help="Listar execuções importadas")
    cp = sub.add_parser("compare", help="Comparar execuções descritivamente")
    cp.add_argument("run_ids", nargs="+")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        with_db = Registry(args.db)
        try:
            if args.command == "import":
                record = import_file(
                    args.file, source=args.source, run_id=args.run_id,
                    timestamp=args.recorded_at, robot_config_id=args.robot_config,
                    code_version=args.code_version, experiment_id=args.experiment_id,
                    label=args.result_label, label_provenance=args.label_provenance,
                    termination_reason=args.termination, notes=args.notes)
                created = with_db.add(record)
                report = diagnose(record)
                output = {"stored":created,"diagnosis":report}
                if args.report:
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    args.report.write_text(_report(record,report), encoding="utf-8")
                    output["report_file"] = str(args.report)
            elif args.command == "analyze":
                record = with_db.get(args.run_id)
                report = diagnose(record)
                output = report
                if args.report:
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    args.report.write_text(_report(record,report), encoding="utf-8")
                    output["report_file"] = str(args.report)
            elif args.command == "list":
                output = {"runs":with_db.list()}
            else:
                output = compare([with_db.get(i) for i in args.run_ids])
        finally:
            with_db.close()
        print(_json(output))
        return 0
    except (ContractError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"ERRO MIH: {exc}", file=sys.stderr)
        return 2
