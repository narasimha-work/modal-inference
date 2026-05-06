import subprocess
import socket
import modal

MODEL_NAME = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
VLLM_PORT = 8000
MINUTES = 60

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm>=0.9.0",
        "huggingface-hub>=0.30.0",
        "transformers>=5.2.0",
        "requests",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("qwen35-inference")


def wait_for_server(port: int, timeout: int = 1200) -> None:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"vLLM server did not start within {timeout}s")


@app.cls(
    image=vllm_image,
    gpu="H100",
    scaledown_window=5 * MINUTES,
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    min_containers=0,
    max_containers=2,
)
@modal.concurrent(max_inputs=32)
class VllmServerQwen35:
    @modal.enter()
    def start(self):
        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--uvicorn-log-level=info",
            "--served-model-name", MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            # No --quantization: running full BF16 for quality testing
            "--max-model-len", "32768",
            "--gpu-memory-utilization", "0.95",
            "--max-num-batched-tokens", "32768",
            "--max-num-seqs", "8",
            "--enable-chunked-prefill",
            "--trust-remote-code",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--gdn-prefill-backend", "triton",
        ]
        print("Starting vLLM:", " ".join(cmd))
        self.vllm_proc = subprocess.Popen(cmd)
        wait_for_server(VLLM_PORT)

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.vllm_proc.terminate()
