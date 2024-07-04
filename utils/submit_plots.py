import os
import subprocess
from pathlib import Path


# directory_path = Path("analysis/12.07")
# ckpts = list(directory_path.rglob("*.pt"))
# for ckpt in ckpts:
#     print(ckpt)

#     subprocess.run(["python", "utils/generate_plots_refactor.py", f"root={ckpt}"])
#     # subprocess.run(["sbatch", script_path])

PATH = "analysis/12.07"
root = Path(PATH)
for item in root.iterdir():
    if item.is_dir() and item.name != "plots":
        subprocess.run(
            [
                "python",
                "utils/generate_plots_refactor.py",
                "--config-path",
                f"{item.parent.absolute()}",
                f"+root={item}",
            ]
        )
