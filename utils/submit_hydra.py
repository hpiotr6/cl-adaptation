import subprocess
from pathlib import Path
import time


def run_background_process(command, logfile):
    subprocess.Popen(command, stdout=logfile, stderr=logfile)


def submit(directory, yaml_files, logfile_root):
    if not logfile_root.exists():
        logfile_root.mkdir(parents=True)
    print("Do you want to proceed?")

    for yaml_file in yaml_files:
        command = [
            "python3.10",
            "-m",
            "src.main_incremental",
            "-cp",
            f"../{directory}",
            "-cn",
            yaml_file.stem,  # getting the file stem (name without extension)
        ]

        log_filename = f"{yaml_file.stem}.log"
        logfile_path = Path(logfile_root, log_filename)

        confirmation = input(f"{yaml_file}: [Y/n]: ").strip()

        if confirmation.lower() == "y" or confirmation == "":
            with open(logfile_path, "w") as logfile:
                run_background_process(command, logfile)
        else:
            print("skipped")
            continue


def main():
    directory = Path("runs/2025/01.15")
    yaml_files = directory.glob("*.yaml")
    logfile_root = Path(directory, "logs")
    files = list(yaml_files)
    for script_path in files:
        print("Script: ", script_path)

    confirmation = input("Do you want to proceed? [Y/n]: ").strip()

    # Check user's response
    if confirmation.lower() == "y" or confirmation == "":
        submit(directory, files, logfile_root)
    else:
        print("Operation aborted.")


if __name__ == "__main__":
    main()
