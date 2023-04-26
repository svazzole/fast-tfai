import shutil
import subprocess
import sys

import tomli

with open("pyproject.toml", "rb") as f:
    toml_dict = tomli.load(f)

version = toml_dict.get("tool").get("poetry").get("version")

command = "docker"
if shutil.which("podman"):
    command = "podman"

if len(sys.argv) == 2:
    if sys.argv[1] == "build":
        subprocess.run(
            [command, "build", "-f", "Dockerfile", "-t", f"inspair:v{version}", "."]
        )
    elif sys.argv[1] == "run":
        subprocess.run(
            [
                command,
                "run",
                "-i",
                "-v",
                "/path/to/dataset/:/data",
                "-v",
                "/path/to/yaml:/app/conf/params.yaml",
                f"inspair:v{version}",
            ]
        )
    elif sys.argv[1] == "run-gpu":
        subprocess.run(
            [
                command,
                "run",
                "-i ",
                "-v",
                "/path/to/dataset/:/data",
                "-v",
                "/path/to/yaml:/app/conf/params.yaml",
                f"inspair:v{version}",
            ]
        )
    else:
        print("run or build?")
else:
    print("Invalid command...")
