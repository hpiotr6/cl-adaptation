from collections import defaultdict
import os
from pathlib import Path
from pprint import pprint
import re

import yaml

# Define the directory path and search pattern
directory_path = "results/2024/04.19/13-45-43"
search_text = "Exception ignored in: <function Logger.__del__"

# Find files matching the pattern
files = Path(directory_path).rglob(pattern="stderr*")

# Define a regular expression pattern to match the search text
pattern = re.compile(search_text)

broken_params = []
# Iterate through each file and search for the text
for file_path in files:
    with file_path.open("r") as file:
        for line_number, line in enumerate(file, start=1):
            if pattern.search(line):
                print(
                    f"Found in file: {file_path}, line: {line_number}, content: {line.strip()}"
                )
                yaml_file = file_path.parent.parent / ".hydra" / "overrides.yaml"
                broken_params.append(yaml.safe_load(yaml_file.open("r")))
                break
# pprint(broken_params)

grouped_data = defaultdict(list)

# Iterate through the data and group it based on the first element of each sublist
for sublist in broken_params:
    key = sublist[0]
    grouped_data[key].append(sublist[1:])

# Convert defaultdict to dict
grouped_data = dict(grouped_data)

pprint(grouped_data)
