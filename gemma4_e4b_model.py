import subprocess
import socket
import modal

MODEL_NAME = "google/gemma-4-E4B-it"
VLLM_PORT = 8000
MINUTES = 60

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm>=0.9.0",
        "huggingface-hub>=0.30.0",
        "transformers>=4.50.0",
        "requests",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "VLLM_COMPILE_CACHE_DIR": "/tmp/vllm_compile_cache",
    })
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("gemma4-e4b-inference")

with vllm_image.imports():
    import requests


def wait_for_server(port: int, timeout: int = 600) -> None:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"vLLM server did not start within {timeout}s")


def _warmup_inference():
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8,
    }
    for _ in range(3):
        requests.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=120,
        ).raise_for_status()


def _sleep():
    requests.post(f"http://localhost:{VLLM_PORT}/sleep?level=1").raise_for_status()


def _wake():
    requests.post(f"http://localhost:{VLLM_PORT}/wake_up").raise_for_status()


@app.cls(
    image=vllm_image,
    gpu="A100-40GB",
    scaledown_window=5 * MINUTES,
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    min_containers=0,
    max_containers=2,
)
@modal.concurrent(max_inputs=32)
class VllmServerGemma4E4B:
    @modal.enter(snap=True)
    def start(self):
        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--uvicorn-log-level=info",
            "--served-model-name", MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--quantization", "fp8",
            "--max-model-len", "32768",
            "--gpu-memory-utilization", "0.90",
            "--max-num-batched-tokens", "32768",
            "--max-num-seqs", "16",
            "--enable-chunked-prefill",
            "--trust-remote-code",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "gemma4",
            "--enable-sleep-mode",
        ]
        print("Starting vLLM:", " ".join(cmd))
        self.vllm_proc = subprocess.Popen(cmd)
        wait_for_server(VLLM_PORT)
        _warmup_inference()
        _sleep()

    @modal.enter(snap=False)
    def wake_up(self):
        _wake()
        wait_for_server(VLLM_PORT)

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.vllm_proc.terminate()
