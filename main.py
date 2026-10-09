
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

scripts = [
    "src/inspect_data.py",
    "src/train_model.py",
    "src/evaluate_model.py",
    "src/predict.py"
]

for script in scripts:
    print(f"\n{'=' * 60}")
    print(f"Running {script}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT
    )

    if result.returncode != 0:
        print(f"\nERROR: {script} failed.")
        sys.exit(result.returncode)

print("\nDISEASE PREDICTION PROJECT COMPLETED SUCCESSFULLY")
