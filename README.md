# modal-inference

Serverless LLM inference on Modal using vLLM. Runs on H100 GPUs and auto-scales to zero when idle.

## Models

| File | Model | Precision | Context |
|------|-------|-----------|---------|
| `gemma4_model.py` | google/gemma-4-26B-A4B-it | BF16 | 16K |
| `gemma4_fp8_model.py` | google/gemma-4-26B-A4B-it | FP8 | 16K |
| `qwen35_model.py` | Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled | BF16 | 32K |

## Setup

```bash
pip install -r requirements.txt
modal token new
```

## Deploy

Edit `deploy.py` and uncomment the models you want to deploy:

```python
MODELS = [
    "qwen35_model.py",
    # "gemma4_model.py",
    # "gemma4_fp8_model.py",
]
```

Push to `main` — CI deploys automatically.

## Adding a New Model

1. Copy an existing file:
   ```bash
   cp qwen35_model.py llama4_model.py
   ```

2. Change `MODEL_NAME` at the top

3. Add it to `deploy.py` and push

## Inference

```bash
curl https://<your-modal-endpoint>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL_NAME>",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 200
  }'
```
