import subprocess
import sys

MODELS = [
    "gemma4_e4b_model.py",
    # "qwen35_model.py",        # uncomment to deploy
    # "gemma4_model.py",        # uncomment to deploy
    # "gemma4_fp8_model.py",    # uncomment to deploy
]

for model in MODELS:
    print(f"\n>>> Deploying {model}")
    result = subprocess.run(["modal", "deploy", model])
    if result.returncode != 0:
        print(f"FAILED: {model}")
        sys.exit(1)
