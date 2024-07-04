import subprocess
from pathlib import Path

directory_path = Path("runs/2024/03.25")
sh_files = list(directory_path.rglob("*run*.sh"))

# Print the scripts to console
for script_path in sh_files:
    print("Script:", script_path)

# Ask for confirmation
confirmation = input("Do you want to proceed? [Y/n]: ").strip()

# Check user's response
if confirmation.lower() == "y" or confirmation == "":
    for script_path in sh_files:
        subprocess.run(["sbatch", script_path])
else:
    print("Operation aborted.")
