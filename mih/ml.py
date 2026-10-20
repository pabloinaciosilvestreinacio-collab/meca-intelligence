"""Classificação offline supervisionada: avaliação agrupada; nunca implantação no robô.

Scikit-learn é dependência OPCIONAL. Ausência de dados suficientes => não treina.
Labels humanos requerem revisão explícita, e simulação nunca conta como validação real.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from statistics import mean
import time

from .core import ContractError, diagnose, validate

NORMAL = {"normal"}
FAULTS = {"anomaly", "anomalia", "imu_travada", "encoder_esquerdo",
          "encoder_direito", "motor_sem_progresso", "desvio_angular",
          "timeout", "patinagem_suspeita"}
NEEDED_CHANNELS = ("motor_me_command", "motor_md_command", "encoder_me_ticks",
                   "encoder_md_ticks", "imu_yaw_deg")
MIN_RUNS_PER_CLASS = 12
MIN_SESSIONS = 6


def _label(value):
    if not isinstance(value, str):
        return None
    v = value.casefold().strip()
    if v in NORMAL:
        return 0
    if v in FAULTS:
        return 1
    return None


def _features(run):
    rows = run["samples"]
    if len(rows) < 5 or rows[-1]["time_ms"]-rows[0]["time_ms"] < 700:
        return None
    valid = [s for s in rows if all(s.get(k) is not None for k in NEEDED_CHANNELS)]
    if len(valid)/len(rows) < .90 or len(valid) < 5:
        return None
    a,b=valid[0],valid[-1]
    dt=(b["time_ms"]-a["time_ms"])/1000
    if dt < .7:
        return None
    command_me=mean(abs(s["motor_me_command"]) for s in valid)
    command_md=mean(abs(s["motor_md_command"]) for s in valid)
    delta_me=abs(b["encoder_me_ticks"]-a["encoder_me_ticks"])
    delta_md=abs(b["encoder_md_ticks"]-a["encoder_md_ticks"])
    angles=[s["imu_yaw_deg"] for s in valid]
    angular_progress=sum(abs((end-start+180)%360-180) for start,end in zip(angles,angles[1:]))
    gaps=[v["time_ms"]-u["time_ms"] for u,v in zip(rows,rows[1:])]
    return [
        command_me, command_md,
        mean(abs(s["motor_me_command"]-s["motor_md_command"]) for s in valid),
        delta_me/dt, delta_md/dt, angular_progress/dt,
        abs(delta_me-delta_md)/max(1,delta_me,delta_md),
        max(gaps), dt,
    ]


def prepare(runs, domain="real"):
    if domain not in {"real", "simulation", "synthetic_lab"}:
        raise ContractError("domínio ML inválido")
    accepted=[]
    skipped=Counter()
    seen_ids=set()
    seen_signals=set()
    for run in runs:
        validate(run)
        if run["source"] != domain:
            skipped["different_domain"] += 1
            continue
        if run["run_id"] in seen_ids:
            skipped["duplicate_run_id"] += 1
            continue
        seen_ids.add(run["run_id"])
        label=_label(run.get("result_label"))
        if label is None:
            skipped["unreviewed_or_unknown_label"] += 1
            continue
        origin=run.get("label_provenance","unlabeled")
        if domain=="real" and origin!="verified_measurement":
            skipped["physical_label_unverified"] += 1
            continue
        if domain=="synthetic_lab" and origin!="synthetic_injected":
            skipped["synthetic_label_unverified"] += 1
            continue
        if domain=="simulation" and origin not in {"synthetic_injected","operator"}:
            skipped["simulation_label_unverified"] += 1
            continue
        session=run.get("session_id")
        if not isinstance(session,str) or not session.strip():
            skipped["missing_session_id"] += 1
            continue
        features=_features(run)
        if features is None:
            skipped["missing_signals_or_short_run"] += 1
            continue
        signal_hash=hashlib.sha256(json.dumps(run["samples"],sort_keys=True).encode()).hexdigest()
        if signal_hash in seen_signals:
            skipped["duplicate_signal"] += 1
            continue
        seen_signals.add(signal_hash)
        accepted.append({"run_id":run["run_id"],"session_id":session,
                         "label":label,"features":features,"run":run})
    return accepted,dict(skipped)


def readiness(runs, domain="real"):
    rows,skipped=prepare(runs,domain=domain)
    labels=Counter(v["label"] for v in rows)
    groups=set(v["session_id"] for v in rows)
    class_sessions={c:len({v["session_id"] for v in rows if v["label"]==c}) for c in (0,1)}
    reasons=[]
    for c,name in [(0,"normal"),(1,"anomaly")]:
        if labels[c] < MIN_RUNS_PER_CLASS:
            reasons.append(f"{name}: mínimo {MIN_RUNS_PER_CLASS} execuções independentes rotuladas")
        if class_sessions[c] < 3:
            reasons.append(f"{name}: mínimo 3 sessões independentes com a classe")
    if len(groups) < MIN_SESSIONS:
        reasons.append(f"mínimo {MIN_SESSIONS} sessões distintas para divisão por grupo")
    if domain!="real":
        reasons.append("avaliação sintética somente; não valida o Mequinha físico")
    return {
        "domain":domain,"eligible_runs":len(rows),"independent_sessions":len(groups),
        "labels":{"normal":labels[0],"anomaly":labels[1]},
        "class_sessions":{"normal":class_sessions[0],"anomaly":class_sessions[1]},
        "skipped":skipped,"training_ready":not reasons or (
            domain!="real" and all("avaliação sintética somente" in r for r in reasons)),
        "limitations":reasons,
        "source_policy":"real exige verified_measurement; synthetic_lab exige synthetic_injected",
    }


def evaluate(runs, domain="real"):
    """Retorna métricas do holdout por sessão. Nunca ajusta motores ou modelos persistentes."""
    audit=readiness(runs,domain=domain)
    if not audit["training_ready"]:
        raise ContractError("treino bloqueado: " + "; ".join(audit["limitations"]))
    rows,_=prepare(runs,domain=domain)
    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ContractError("instale scikit-learn opcional para avaliação: pip install scikit-learn") from exc
    X=[x["features"] for x in rows]
    y=[x["label"] for x in rows]
    groups=[x["session_id"] for x in rows]
    splitter=GroupShuffleSplit(n_splits=64,test_size=.25,random_state=32578)
    selection=None
    for train,test in splitter.split(X,y,groups):
        if set(y[i] for i in train)=={0,1} and set(y[i] for i in test)=={0,1}:
            selection=(list(train),list(test))
            break
    if selection is None:
        raise ContractError("não foi possível obter holdout com ambas as classes sem misturar sessões")
    train,test=selection
    train_sessions=set(groups[i] for i in train)
    test_sessions=set(groups[i] for i in test)
    assert train_sessions.isdisjoint(test_sessions)
    models={
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,random_state=32578,class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=80,max_depth=6,random_state=32578,class_weight="balanced",n_jobs=1),
    }
    results={}
    for name,model in models.items():
        start=time.perf_counter()
        model.fit([X[i] for i in train],[y[i] for i in train])
        pred=model.predict([X[i] for i in test])
        elapsed_ms=(time.perf_counter()-start)*1000
        p,r,f,_=precision_recall_fscore_support([y[i] for i in test],pred,labels=[0,1],zero_division=0)
        results[name]={
            "accuracy":float(accuracy_score([y[i] for i in test],pred)),
            "confusion_matrix":confusion_matrix([y[i] for i in test],pred,labels=[0,1]).tolist(),
            "precision":list(map(float,p)),"recall":list(map(float,r)),"f1":list(map(float,f)),
            "fit_and_inference_ms":round(elapsed_ms,2),
        }
    return {
        "domain":domain,"status":"experimental_grouped_holdout",
        "class_order":["normal","anomaly"],
        "train_runs":len(train),"test_runs":len(test),
        "train_sessions":sorted(train_sessions),"test_sessions":sorted(test_sessions),
        "random_state":32578,"features":["cmd_me_abs_mean","cmd_md_abs_mean",
          "cmd_diff_mean","encoder_me_ticks_per_s","encoder_md_ticks_per_s",
          "imu_abs_deg_per_s","encoder_progress_imbalance","largest_gap_ms","duration_s"],
        "models":results,
        "warning":"Holdout agrupado não é validação física externa; não automatizar decisões competitivas.",
    }
