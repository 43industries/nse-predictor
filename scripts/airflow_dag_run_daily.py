"""Optional Airflow DAG mirroring scripts/run_daily.py (blueprint scheduler alternative).

Copy into your Airflow dags/ folder and set NSE_PREDICTOR_ROOT.
Does nothing unless Airflow is installed and the DAG is loaded.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except Exception:  # noqa: BLE001
    DAG = None  # type: ignore


ROOT = Path(os.environ.get("NSE_PREDICTOR_ROOT", Path(__file__).resolve().parents[1]))
PY = os.environ.get("NSE_PREDICTOR_PYTHON", "python")

if DAG is not None:
    default_args = {
        "owner": "nse_predictor",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    }
    with DAG(
        dag_id="nse_predictor_run_daily",
        default_args=default_args,
        description="NSE daily collect/ingest/QA/score/monitor",
        schedule_interval="30 17 * * 1-5",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["nse", "research"],
    ) as dag:
        BashOperator(
            task_id="run_daily",
            bash_command=f'cd "{ROOT}" && "{PY}" scripts/run_daily.py',
        )
