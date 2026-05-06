import subprocess
import sys

SERVERS = [
    "qwen35_server.py",
    # "gemma4_server.py",       # uncomment to deploy
    # "gemma3_server.py",       # uncomment to deploy
    # "gemma4_server_fp8.py",   # uncomment to deploy
]

for server in SERVERS:
    print(f"\n>>> Deploying {server}")
    result = subprocess.run(["modal", "deploy", server])
    if result.returncode != 0:
        print(f"FAILED: {server}")
        sys.exit(1)
