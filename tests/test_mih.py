"""Testes offline com dados artificiais, nunca medições do Mequinha."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from mih.core import ContractError, Registry, compare, diagnose, import_file, validate


def run(samples, **overrides):
    obj = {
        "schema_version":"MIH-V1", "run_id":"synthetic-001",
        "source":"synthetic_lab","timestamp":"2026-10-19T18:00:00-03:00",
        "robot_config_id":"test-fixture", "code_version":"unittest",
        "experiment_id":"test", "result_label":None,"label_provenance":"unlabeled",
        "termination_reason":"unknown", "samples":samples,
    }
    obj.update(overrides)
    return obj


class ContractTests(unittest.TestCase):
    def test_valid_sparse_execution_is_not_classified_normal(self):
        record = run([{"time_ms":0},{"time_ms":50}])
        self.assertEqual(diagnose(record)["status"],"indeterminate")

    def test_invalid_time_and_nonfinite_rejected(self):
        r=run([{"time_ms":0},{"time_ms":0}])
        with self.assertRaises(ContractError): validate(r)
        r=run([{"time_ms":0,"imu_yaw_deg":float("nan")}])
        with self.assertRaises(ContractError): validate(r)

    def test_incorrect_provenance_rejected(self):
        r=run([{"time_ms":0}],source="real",label_provenance="synthetic_injected")
        with self.assertRaises(ContractError): validate(r)

    def test_command_range_and_tz(self):
        r=run([{"time_ms":0,"motor_me_command":1.2}])
        with self.assertRaises(ContractError): validate(r)
        r=run([{"time_ms":0}],timestamp="2026-10-19T12:00:00")
        with self.assertRaises(ContractError): validate(r)

    def test_empty_and_out_of_order(self):
        with self.assertRaises(ContractError): validate(run([]))
        with self.assertRaises(ContractError): validate(run([{"time_ms":5},{"time_ms":3}]))


class DiagnosticTests(unittest.TestCase):
    @staticmethod
    def movement(me=lambda i:i*30, md=lambda i:i*30,
                 yaw=lambda i:i*4, cm=.6, cd=-.6):
        return [{"time_ms":i*100,"motor_me_command":cm,
                 "motor_md_command":cd,"encoder_me_ticks":me(i),
                 "encoder_md_ticks":md(i),"imu_yaw_deg":yaw(i)}
                for i in range(15)]

    def test_imu_suspected_when_rotation_observable(self):
        issues=diagnose(run(self.movement(yaw=lambda i:0)))["issues"]
        self.assertIn("IMU_NO_ANGULAR_PROGRESS",[v["code"] for v in issues])
        self.assertTrue(all(v["classification"]!="confirmed_physical_cause" for v in issues))

    def test_rotating_imu_not_flagged(self):
        issues=diagnose(run(self.movement()))["issues"]
        self.assertNotIn("IMU_NO_ANGULAR_PROGRESS",[v["code"] for v in issues])

    def test_stuck_encoder_detected(self):
        issues=diagnose(run(self.movement(me=lambda i:0)))["issues"]
        self.assertIn("NO_ENCODER_PROGRESS_ME",[v["code"] for v in issues])

    def test_no_imu_claim_from_pose_only(self):
        rows=[{"time_ms":i*50,"pose_x_m":1.2,"pose_y_m":i*.02,
               "pose_yaw_deg":0,"encoder_me_ticks":i*10,
               "encoder_md_ticks":i*10} for i in range(20)]
        result=diagnose(run(rows,source="simulation"))
        self.assertIn("imu_yaw_deg",result["missing_channels"])
        self.assertNotIn("IMU_NO_ANGULAR_PROGRESS",[v["code"] for v in result["issues"]])

    def test_stall_starting_late(self):
        rows=self.movement(me=lambda i: i*30 if i<=5 else 150)
        issues=diagnose(run(rows))["issues"]
        self.assertIn("NO_ENCODER_PROGRESS_ME",[v["code"] for v in issues])

    def test_late_imu_freeze(self):
        rows=self.movement(yaw=lambda i: i*4 if i<=5 else 20)
        issues=diagnose(run(rows))["issues"]
        self.assertIn("IMU_NO_ANGULAR_PROGRESS",[v["code"] for v in issues])

    def test_timeout_and_gap(self):
        rows=[{"time_ms":0},{"time_ms":50},{"time_ms":1000}]
        codes=[v["code"] for v in diagnose(run(rows,termination_reason="timeout"))["issues"]]
        self.assertIn("TIMEOUT_REPORTED",codes)
        self.assertIn("SAMPLING_GAP",codes)

    def test_distance_anomaly_requires_explicit_measurement(self):
        r=run([{"time_ms":0},{"time_ms":100,"target_distance_m":2,"observed_distance_m":.8}])
        self.assertIn("DISTANCE_TARGET_MISMATCH",[v["code"] for v in diagnose(r)["issues"]])
        r=run([{"time_ms":0},{"time_ms":100,"target_distance_m":2}])
        self.assertNotIn("DISTANCE_TARGET_MISMATCH",[v["code"] for v in diagnose(r)["issues"]])


class ImportTests(unittest.TestCase):
    def test_meca_sim_csv_maps_pose_not_imu(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"trajectory.csv"
            f.write_text("time_s,x_m,y_m,yaw_deg,ME_ticks,MD_ticks,MI_ticks,MS_ticks\n"+
                         "0,1,1,5,0,0,0,0\n0.05,1,1.1,6,10,10,2,0\n",encoding="utf8")
            kwargs=dict(source="simulation",run_id="sim-1",timestamp="2026-10-19T18:00:00-03:00",
                        robot_config_id="unknown",code_version="unknown",experiment_id="run")
            r=import_file(f,**kwargs)
            self.assertEqual(r["import_format"],"meca_sim_recording")
            self.assertEqual(r["samples"][1]["time_ms"],50)
            self.assertEqual(r["samples"][1]["pose_yaw_deg"],6)
            self.assertNotIn("imu_yaw_deg",r["samples"][1])
            with self.assertRaises(ContractError): import_file(f,**{**kwargs,"source":"real"})

    def test_browser_lab_csv_retains_synthetic_identity(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"s.csv"
            f.write_text("timestamp,imu_yaw_deg,encoder_md_ticks,encoder_me_ticks,potencia_md,potencia_me\n"+
                         "0,0,0,0,0.5,-0.5\n50,0,10,10,0.5,-0.5\n",encoding="utf8")
            r=import_file(f,source="synthetic_lab",run_id="lab-1",
                          timestamp="2026-10-19T18:00:00-03:00",robot_config_id="browser",
                          code_version="lab",experiment_id="fixture",
                          label="imu_travada",label_provenance="synthetic_injected")
            self.assertEqual(r["samples"][1]["motor_md_command"],.5)
            self.assertEqual(r["source"],"synthetic_lab")

    def test_json_recording(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"sim.json"
            f.write_text(json.dumps({"columns":["time_s","x_m","y_m","yaw_deg","ME_ticks","MD_ticks"],
                "rows":[[0,1,1,0,0,0],[.05,1,1.05,0,10,10]]}),encoding="utf8")
            r=import_file(f,source="simulation",run_id="json-1",
                          timestamp="2026-10-19T18:00:00-03:00",robot_config_id="r",
                          code_version="c",experiment_id="e")
            self.assertEqual(len(r["samples"]),2)

    def test_duplicate_csv_headers_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"bad.csv"
            f.write_text("time_ms,time_ms\n0,1\n",encoding="utf8")
            with self.assertRaises(ContractError):
                import_file(f,source="real",run_id="bad-1",
                    timestamp="2026-10-19T18:00:00-03:00",robot_config_id="x",
                    code_version="x",experiment_id="x")


class RegistryTests(unittest.TestCase):
    def test_idempotent_and_conflicting_data(self):
        with tempfile.TemporaryDirectory() as d:
            registry=Registry(Path(d)/"runs.sqlite3")
            r=run([{"time_ms":0}])
            self.assertTrue(registry.add(r))
            self.assertFalse(registry.add(r))
            self.assertEqual(registry.get("synthetic-001")["source"],"synthetic_lab")
            with self.assertRaises(ContractError):
                registry.add(run([{"time_ms":0},{"time_ms":10}]))
            other=run([{"time_ms":0}],run_id="synthetic-002")
            registry.add(other)
            self.assertEqual(len(compare([r,other])["runs"]),2)
            registry.close()


if __name__ == "__main__":
    unittest.main()
