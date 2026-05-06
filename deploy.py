import subprocess
import sys

MODELS = [
    "qwen35_model.py",
    # "gemma4_model.py",        # uncomment to deploy
    # "gemma4_fp8_model.py",    # uncomment to deploy
]

for model in MODELS:
    print(f"\n>>> Deploying {model}")
    result = subprocess.run(["modal", "deploy", model])
    if result.returncode != 0:
        print(f"FAILED: {model}")
        sys.exit(1)
