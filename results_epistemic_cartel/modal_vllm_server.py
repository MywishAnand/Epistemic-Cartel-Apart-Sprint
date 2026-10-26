"""
Modal Dual-Model OpenAI-Compatible vLLM Server for the Epistemic Cartel Tournament.
Hosts Qwen/Qwen2.5-14B-Instruct-AWQ (or Qwen/Qwen2.5-7B-Instruct) and
meta-llama/Llama-3.1-8B-Instruct on Modal GPUs with automatic scale-to-zero
so your $50 Modal credit budget lasts the entire sprint.

Usage on your MacBook Air:
    1. pip install modal
    2. modal setup
    3. modal deploy modal_vllm_server.py
    4. Copy the printed HTTPS endpoint URL into your environment:
       export MODAL_BASE_URL="https://<your-username>--epistemic-cartel-vllm-serve.modal.run/v1"
"""

import os
import modal

app = modal.App("epistemic-cartel-vllm")

# Persistent volume to cache HuggingFace weights across container restarts
hf_cache_vol = modal.Volume.from_name("epistemic-cartel-hf-cache", create_if_missing=True)
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.3",
        "fastapi[standard]>=0.115.0",
        "huggingface_hub[hf_transfer]>=0.26.0",
        "pydantic>=2.9.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

QWEN_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-AWQ"
LLAMA_MODEL_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"


@app.cls(
    image=vllm_image,
    gpu="L40S",  # L40S (48GB VRAM) is ~40% cheaper than A100 on Modal (~$1.35/hr vs $2.50+/hr)
    max_containers=1,  # Prevent Modal from auto-spawning multiple parallel GPU containers
    scaledown_window=60,  # Scale down after 60s of inactivity instead of 300s to save credits
    timeout=1800,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
class DualModelEngine:
    @modal.enter()
    def load_engines(self):
        from vllm import LLM

        print(f"Loading Qwen model: {QWEN_MODEL_ID}...")
        self.qwen_llm = LLM(
            model=QWEN_MODEL_ID,
            quantization="awq",
            dtype="half",
            gpu_memory_utilization=0.45,
            max_model_len=4096,
            enforce_eager=True,
        )
        print(f"Loading Llama model: {LLAMA_MODEL_ID}...")
        self.llama_llm = LLM(
            model=LLAMA_MODEL_ID,
            dtype="half",
            gpu_memory_utilization=0.45,
            max_model_len=4096,
            enforce_eager=True,
        )
        hf_cache_vol.commit()

    @modal.method()
    def generate_chat(
        self,
        model_family: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        from vllm import SamplingParams

        llm = self.qwen_llm if "qwen" in model_family.lower() else self.llama_llm
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        outputs = llm.chat(messages=messages, sampling_params=sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text


web_app = modal.asgi_app()


@app.function(image=vllm_image, timeout=1800)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import time

    api = FastAPI(title="Epistemic Cartel Modal vLLM Router")
    engine = DualModelEngine()

    @api.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": "qwen2.5-14b", "object": "model"},
                {"id": "llama-3.1-8b", "object": "model"},
            ],
        }

    @api.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        model = payload.get("model", "qwen2.5-14b")
        messages = payload.get("messages", [])
        temperature = float(payload.get("temperature", 0.7))
        max_tokens = int(payload.get("max_tokens", 500))

        text = engine.generate_chat.remote(
            model_family=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return JSONResponse(
            {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    return api
