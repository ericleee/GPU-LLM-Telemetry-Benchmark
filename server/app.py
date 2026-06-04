"""
Inference server + control panel.

Serves a *selectable* LLM on the GPU, exposes serving + GPU metrics, and runs an
in-process load generator you can Start/Stop from a small web UI at "/"
(the external k6 test remains the CLI load tool).

Endpoints
  GET  /                 control panel (HTML)
  GET  /status           model + GPU + load-test state (the UI polls this)
  GET  /models           available + loaded model
  POST /load_model       {model_id} -> switch the served model (async)
  POST /generate         {prompt, max_tokens} -> text + timing
  POST /load_test/start  {concurrency|ramp, max_tokens} -> start in-process load
  POST /load_test/stop   -> stop it
  GET  /healthz          liveness
  GET  /metrics          Prometheus exposition

Concurrency: an asyncio.Semaphore bounds GPU work; blocking .generate() runs in a
worker thread so the event loop stays free and the queue gauge is truthful.
"""

import asyncio
import logging
import os
import threading
import time

import pynvml
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────
# Models sized for a 12 GB RTX 3060 (fp16). 3B is the heavy one most likely to
# push power/heat toward throttling.
AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct",          "label": "Qwen2.5-0.5B · light (~1 GB)"},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct",          "label": "Qwen2.5-1.5B · medium (~3 GB)"},
    {"id": "Qwen/Qwen2.5-3B-Instruct",            "label": "Qwen2.5-3B · heavy (~6 GB)"},
    {"id": "microsoft/Phi-3.5-mini-instruct",     "label": "Phi-3.5-mini 3.8B · heaviest fp16 (~7.6 GB)"},
    {"id": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit","label": "Qwen2.5-7B · 4-bit · the realistic 12 GB workload (~5 GB)", "quant": True},
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  "label": "TinyLlama-1.1B · alt arch (~2 GB)"},
]
DEFAULT_MODEL_ID   = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
GPU_CONCURRENCY    = int(os.getenv("GPU_CONCURRENCY", "2"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "64"))
LOAD_TEST_PROMPT   = "Explain what a GPU does and why it is useful for AI, in two sentences."
RAMP_STAGES        = [1, 5, 10, 20]
RAMP_STAGE_SECONDS = int(os.getenv("STAGE_SECONDS", "40"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("inference_server")

# ── Metrics ──────────────────────────────────────────────────────────────────
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
m_requests   = Counter("inference_requests_total", "Total /generate requests", ["status"])
m_latency    = Histogram("inference_request_latency_seconds", "End-to-end /generate latency (s)",
                         buckets=LATENCY_BUCKETS)
m_inflight   = Gauge("inference_inflight_requests", "Generations currently running on the GPU")
m_queue      = Gauge("inference_queue_depth", "Requests waiting for a GPU slot")
m_tokens     = Counter("inference_tokens_total", "Total generated tokens")
m_tok_per_s  = Gauge("inference_tokens_per_second", "Tokens/sec of the most recent request")
m_model_info = Gauge("inference_model_info", "Loaded model (value always 1)", ["model"])
m_test_run   = Gauge("inference_load_test_running", "1 if the in-process load test is running")
m_test_vus   = Gauge("inference_load_test_concurrency", "Target VUs of the in-process load test")

# ── Model state (mutable so it can be swapped at runtime) ─────────────────────
state = {"model_id": None, "tokenizer": None, "model": None, "status": "init"}  # status: init|switching|ready|error
state_lock = threading.Lock()
gpu_semaphore = asyncio.Semaphore(GPU_CONCURRENCY)
_last_tokens_per_s = 0.0

# ── NVML (read GPU temp/power directly so the panel is self-contained) ────────
try:
    pynvml.nvmlInit()
    _nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception as exc:                                   # noqa: BLE001
    _nvml = None
    log.warning("NVML init failed (GPU readout disabled): %s", exc)


def gpu_stats():
    if _nvml is None:
        return {}
    try:
        return {
            "temp_c":  pynvml.nvmlDeviceGetTemperature(_nvml, pynvml.NVML_TEMPERATURE_GPU),
            "power_w": round(pynvml.nvmlDeviceGetPowerUsage(_nvml) / 1000.0, 1),
            "util_pct": pynvml.nvmlDeviceGetUtilizationRates(_nvml).gpu,
            "sm_mhz":  pynvml.nvmlDeviceGetClockInfo(_nvml, pynvml.NVML_CLOCK_SM),
            "vram_used_gb": round(pynvml.nvmlDeviceGetMemoryInfo(_nvml).used / 1e9, 2),
        }
    except Exception:                                      # noqa: BLE001
        return {}


# ── Model load / switch ──────────────────────────────────────────────────────
def _is_quant(model_id):
    return any(m["id"] == model_id and m.get("quant") for m in AVAILABLE_MODELS)


def _load_blocking(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if _is_quant(model_id):
        # Pre-quantized bnb-4bit checkpoint: the quant config lives in the model
        # config and bitsandbytes does the 4-bit load. Place it with device_map
        # (a quantized model can't be moved with .to() / given a dtype).
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to("cuda")
    model.eval()
    return tokenizer, model


def switch_model(model_id: str):
    """Unload the current model, load `model_id`. Blocking — call off the loop."""
    with state_lock:
        state["status"] = "switching"
        m_model_info._metrics.clear()  # drop the old model's info series
        if state["model"] is not None:
            state["model"] = None
            state["tokenizer"] = None
            torch.cuda.empty_cache()
        log.info("Loading model: %s", model_id)
        try:
            tokenizer, model = _load_blocking(model_id)
        except Exception as exc:                            # noqa: BLE001
            log.warning("Failed to load %s: %s", model_id, exc)
            state["status"] = "error"
            raise
        state.update(model_id=model_id, tokenizer=tokenizer, model=model, status="ready")
        m_model_info.labels(model=model_id).set(1)
    # warm up outside the lock-sensitive path
    try:
        _generate_sync("Hello", 8)
        log.info("Model ready + warmed: %s", model_id)
    except Exception as exc:                                # noqa: BLE001
        log.warning("Warm-up failed (continuing): %s", exc)


def _generate_sync(prompt: str, max_new_tokens: int):
    tokenizer, model = state["tokenizer"], state["model"]
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to("cuda")
    prompt_tokens = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output[0][prompt_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, int(prompt_tokens), int(generated_ids.shape[-1])


# Load the default model at startup (blocking, before uvicorn serves).
switch_model(DEFAULT_MODEL_ID)


# ── API ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="GPU Inference Server", version="2.0")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = DEFAULT_MAX_TOKENS


class LoadModelRequest(BaseModel):
    model_id: str


class LoadTestRequest(BaseModel):
    concurrency: int = 20
    ramp: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model": state["model_id"], "model_status": state["status"]}


@app.get("/models")
async def models():
    return {"available": AVAILABLE_MODELS, "loaded": state["model_id"], "status": state["status"]}


@app.get("/status")
async def status():
    return {
        "model_id": state["model_id"],
        "model_status": state["status"],
        "available_models": AVAILABLE_MODELS,
        "gpu_concurrency": GPU_CONCURRENCY,
        "load_test": {
            "running": load_state["running"],
            "concurrency": load_state["concurrency"],
            "requests": load_state["requests"],
            "mode": load_state["mode"],
        },
        "last_tokens_per_s": round(_last_tokens_per_s, 1),
        "gpu": gpu_stats(),
    }


@app.post("/load_model")
async def load_model(req: LoadModelRequest):
    if req.model_id not in [m["id"] for m in AVAILABLE_MODELS]:
        raise HTTPException(status_code=400, detail="unknown model_id")
    await _stop_load_test()                      # switching while loaded is messy
    state["status"] = "switching"

    async def _do_switch():
        try:
            await run_in_threadpool(switch_model, req.model_id)
        except Exception:                        # noqa: BLE001
            state["status"] = "error"

    asyncio.create_task(_do_switch())
    return {"status": "switching", "model_id": req.model_id}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if state["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"model {state['status']}")

    global _last_tokens_per_s
    start = time.perf_counter()
    m_queue.inc()
    async with gpu_semaphore:
        m_queue.dec()
        m_inflight.inc()
        try:
            text, prompt_tokens, generated_tokens = await run_in_threadpool(
                _generate_sync, req.prompt, req.max_tokens
            )
            latency = time.perf_counter() - start
            tokens_per_s = generated_tokens / latency if latency > 0 else 0.0
            m_latency.observe(latency)
            m_tokens.inc(generated_tokens)
            m_tok_per_s.set(tokens_per_s)
            _last_tokens_per_s = tokens_per_s
            m_requests.labels(status="ok").inc()
            return {
                "model": state["model_id"],
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "latency_s": round(latency, 4),
                "tokens_per_s": round(tokens_per_s, 2),
                "text": text,
            }
        except Exception:
            m_requests.labels(status="error").inc()
            log.exception("Generation failed")
            raise
        finally:
            m_inflight.dec()


# ── In-process load generator (drives /generate; same metrics as k6) ──────────
load_state = {"running": False, "concurrency": 0, "requests": 0, "mode": "idle", "manager": None}


async def _worker(max_tokens: int):
    while load_state["running"]:
        try:
            await generate(GenerateRequest(prompt=LOAD_TEST_PROMPT, max_tokens=max_tokens))
            load_state["requests"] += 1
        except Exception:                        # 503 during model switch, etc.
            await asyncio.sleep(0.1)


async def _manager(stages, stage_seconds, max_tokens):
    workers: list[asyncio.Task] = []
    try:
        for target in stages:
            if not load_state["running"]:
                break
            while len(workers) < target and load_state["running"]:
                workers.append(asyncio.create_task(_worker(max_tokens)))
            load_state["concurrency"] = target
            m_test_vus.set(target)
            for _ in range(stage_seconds):       # hold the stage
                if not load_state["running"]:
                    break
                await asyncio.sleep(1)
        while load_state["running"]:             # hold at peak until stopped
            await asyncio.sleep(1)
    finally:
        load_state["running"] = False
        for w in workers:
            w.cancel()
        load_state["concurrency"] = 0
        m_test_vus.set(0)
        m_test_run.set(0)


async def _stop_load_test():
    if load_state["running"]:
        load_state["running"] = False
        mgr = load_state.get("manager")
        if mgr:
            try:
                await asyncio.wait_for(mgr, timeout=5)
            except Exception:                    # noqa: BLE001
                pass
    m_test_run.set(0)
    m_test_vus.set(0)


@app.post("/load_test/start")
async def start_load_test(req: LoadTestRequest):
    if load_state["running"]:
        return {"status": "already_running", "concurrency": load_state["concurrency"]}
    load_state.update(running=True, requests=0, concurrency=0,
                      mode=("ramp" if req.ramp else "constant"))
    stages = RAMP_STAGES if req.ramp else [max(1, req.concurrency)]
    m_test_run.set(1)
    load_state["manager"] = asyncio.create_task(_manager(stages, RAMP_STAGE_SECONDS, req.max_tokens))
    return {"status": "started", "mode": load_state["mode"]}


@app.post("/load_test/stop")
async def stop_load_test():
    await _stop_load_test()
    return {"status": "stopped", "requests": load_state["requests"]}


if __name__ == "__main__":
    log.info("Inference server on http://%s:%d  (GPU concurrency=%d)", HOST, PORT, GPU_CONCURRENCY)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
