"""Gates supervisionados: sem treinamento nem acurácia sintética fabricada."""
import unittest
from mih.ml import readiness, prepare, _features
from mih.core import ContractError

def case(i, label="normal", source="real", provenance="verified_measurement",
         session_id=None):
    samples = [{
        "time_ms":t*100,
        "motor_me_command":.4,
        "motor_md_command":-.4,
        "encoder_me_ticks":t*(30+i),
        "encoder_md_ticks":-t*(31+i),
        "imu_yaw_deg":t*(3 if label=="normal" else 0),
    } for t in range(11)]
    return {
        "schema_version":"MIH-V1","run_id":f"run-{i}",
        "source":source,"timestamp":"2026-10-19T18:00:00-03:00",
        "robot_config_id":"bench","code_version":"test",
        "experiment_id":"fixture", "session_id":session_id or f"session-{i//4}",
        "result_label":label,"label_provenance":provenance,"samples":samples,
    }

class MLTests(unittest.TestCase):
    def test_no_training_on_empty_runs(self):
        r=readiness([])
        self.assertFalse(r["training_ready"])
        self.assertEqual(r["eligible_runs"],0)

    def test_unreviewed_real_labels_never_count(self):
        rows=[case(i,provenance="operator") for i in range(36)]
        r=readiness(rows)
        self.assertEqual(r["eligible_runs"],0)
        self.assertEqual(r["skipped"]["physical_label_unverified"],36)

    def test_grouped_independence_gate(self):
        rows=[case(i,label="normal" if i%2 else "anomalia",session_id="same-day") for i in range(36)]
        r=readiness(rows)
        self.assertFalse(r["training_ready"])
        self.assertEqual(r["independent_sessions"],1)

    def test_sufficient_reviewed_runs_ready_without_training(self):
        rows=[case(i,label="normal" if i%2 else "anomalia",session_id=f"session-{i//4}") for i in range(36)]
        r=readiness(rows)
        self.assertTrue(r["training_ready"])
        self.assertEqual(r["eligible_runs"],36)

    def test_no_cross_domain_mix(self):
        rows=[case(i,label="normal" if i%2 else "anomalia",source="synthetic_lab",
                   provenance="synthetic_injected",session_id=f"session-{i//4}") for i in range(36)]
        real=readiness(rows,domain="real")
        sim=readiness(rows,domain="synthetic_lab")
        self.assertEqual(real["eligible_runs"],0)
        self.assertTrue(sim["training_ready"])
        self.assertIn("avaliação sintética somente",sim["limitations"][0])

    def test_duplicate_signals_are_excluded(self):
        a=case(1)
        b={**a,"run_id":"duplicated"}
        _,skipped=prepare([a,b])
        self.assertEqual(skipped["duplicate_signal"],1)

if __name__ == "__main__":
    unittest.main()
