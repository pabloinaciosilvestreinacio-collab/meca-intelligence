"""Smoke test público da CLI via subprocess, sem robô e sem rede."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CliTests(unittest.TestCase):
    def test_end_to_end_import_analysis_and_ml_gate(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            csv_file=folder/"meca-sim-trajetoria.csv"
            csv_file.write_text("time_s,x_m,y_m,yaw_deg,ME_ticks,MD_ticks\n"+
                                "0,1.0,1.0,0,0,0\n0.05,1.0,1.05,0,10,10\n",
                                encoding="utf-8")
            base=[sys.executable,"-m","mih","--db",str(folder/"runs.sqlite3")]
            action=base+["import",str(csv_file),"--source","simulation",
                         "--run-id","fixture-sim-1",
                         "--recorded-at","2026-10-19T20:00:00-03:00",
                         "--robot-config","unknown","--code-version","unknown",
                         "--experiment-id","smoke","--report",str(folder/"report.md")]
            p=subprocess.run(action,capture_output=True,text=True,check=False)
            self.assertEqual(p.returncode,0,p.stderr)
            payload=json.loads(p.stdout)
            self.assertEqual(payload["diagnosis"]["status"],"indeterminate")
            self.assertIn("imu_yaw_deg",payload["diagnosis"]["missing_channels"])
            self.assertTrue((folder/"report.md").exists())
            p=subprocess.run(base+["analyze","fixture-sim-1"],
                             capture_output=True,text=True,check=False)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads(p.stdout)["source"],"simulation")
            p=subprocess.run(base+["ml-check","--domain","real"],
                             capture_output=True,text=True,check=False)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertFalse(json.loads(p.stdout)["training_ready"])


if __name__=="__main__":
    unittest.main()
