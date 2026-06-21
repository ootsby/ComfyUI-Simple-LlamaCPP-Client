import base64
import io
import json
import os
import re
from typing import Any, Dict, Optional, Tuple, List

import requests
from PIL import Image

try:
    import torch
except Exception:
    torch = None

try:
    import comfy.utils
except Exception:
    comfy = None

DEFAULT_URL = "http://127.0.0.1:8888"

_THINK_TAG_RE = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _extract_thinking(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    parts = _THINK_TAG_RE.findall(text)
    thinking_text = "\n\n".join(p.strip() for p in parts)
    clean_text = _THINK_TAG_RE.sub("", text).strip()
    return clean_text, thinking_text


def _fix_mojibake(text: str) -> str:
    if not text:
        return ""
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def _replace_smart_quotes(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
    )


def _postprocess_text(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "fix_mojibake":
        return _fix_mojibake(text)
    if mode == "ascii_quotes":
        return _replace_smart_quotes(text)
    if mode == "fix_mojibake+ascii_quotes":
        return _replace_smart_quotes(_fix_mojibake(text))
    return text


def _clean_answer(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip()
    t = re.sub(r"^\s*(final|answer)\s*:\s*", "", t, flags=re.IGNORECASE)
    if t.startswith("```") and t.endswith("```"):
        t = t.strip("`").strip()
    return t.strip()


def _image_tensor_to_base64(image_tensor) -> Optional[str]:
    if torch is None:
        return None
    img = image_tensor[0].detach().cpu()
    arr = img.numpy()
    if arr.max() <= 1.5:
        arr = (arr * 255)
    arr = arr.clip(0, 255).astype("uint8")
    pil_img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _file_to_base64(filepath: str) -> Optional[str]:
    if not filepath or not os.path.isfile(filepath):
        return None
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    k = (api_key or "").strip()
    if k:
        headers["Authorization"] = f"Bearer {k}"
    return headers


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_json_from_text(text: str) -> Optional[Any]:
    t = str(text).strip()
    t = re.sub(r"^```(?:json)?\s*\n?", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    result = _safe_json_loads(t)
    if result is not None:
        return result
    for ch in ("{", "["):
        idx = t.find(ch)
        if idx == -1:
            continue
        for end in range(len(t), idx, -1):
            candidate = t[idx:end]
            result = _safe_json_loads(candidate)
            if result is not None:
                return result
    return None


def _make_progress():
    if comfy is None:
        return None
    try:
        pb = comfy.utils.ProgressBar(1000)
        state = {"i": 0}

        def tick():
            state["i"] = min(state["i"] + 1, 1000)
            pb.update_absolute(state["i"])

        return tick
    except Exception:
        return None


def _iter_sse_lines(resp):
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue
        yield line


def _parse_stream_and_accumulate(resp, tick=None) -> Tuple[str, str, Dict[str, Any]]:
    full_content = ""
    full_reasoning = ""
    meta = {}

    for line in _iter_sse_lines(resp):
        if tick:
            tick()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        chunk = _safe_json_loads(data_str)
        if not chunk:
            continue
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        full_content += delta.get("content", "")
        full_reasoning += delta.get("reasoning_content", "") or delta.get("reasoning", "") or delta.get("thoughts", "")

        message = choices[0].get("message", {})
        if message:
            full_content += message.get("content", "")
            full_reasoning += message.get("reasoning_content", "") or message.get("reasoning", "") or message.get("thoughts", "")

        if not meta and chunk.get("id"):
            meta["id"] = chunk["id"]
        if not meta and chunk.get("model"):
            meta["model"] = chunk["model"]

    if not full_reasoning and full_content:
        clean, thinking = _extract_thinking(full_content)
        if thinking:
            full_reasoning = thinking
            full_content = clean

    return full_content, full_reasoning, meta


class LlamaCppRouterClient:
    CATEGORY = "LLM / Simple llama.cpp"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "thinking", "json", "raw", "model_used")
    OUTPUT_TOOLTIPS = (
        "Clean response text with thinking blocks removed",
        "Extracted thinking/reasoning content (empty if model produced none)",
        "Parsed JSON object (empty if json_mode is off or no JSON found)",
        "Full raw server response as JSON string",
        "The model name that was actually used for this request",
    )
    FUNCTION = "generate"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {
                    "default": DEFAULT_URL,
                    "tooltip": "llama.cpp server base URL (router mode or single-model)",
                }),
                "model": ("STRING", {
                    "default": "",
                    "tooltip": "Model name — click Fetch Models button to pick from the server",
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                    "tooltip": "System prompt sent before the user message",
                }),
                "prompt": ("STRING", {
                    "default": "Hello!",
                    "multiline": True,
                    "tooltip": "User message / question",
                }),
                "stream": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Stream SSE",
                    "label_off": "Single response",
                    "tooltip": "Enable SSE streaming for real-time token output",
                }),
                "max_tokens": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 131072,
                    "tooltip": "Max tokens to generate (0 = server default)",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Sampling temperature",
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "RNG seed (-1 = random)",
                }),
                "json_mode": ("BOOLEAN", {
                    "default": False,
                    "label_on": "JSON output",
                    "label_off": "Free text",
                    "tooltip": "Force JSON-only output via response_format",
                }),
                "unload_after_generate": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Unload model after",
                    "label_off": "Keep model loaded",
                    "tooltip": "Call /models/unload on the server after generation",
                }),
                "text_postprocess": (["fix_mojibake", "none", "ascii_quotes", "fix_mojibake+ascii_quotes"], {
                    "default": "fix_mojibake",
                    "tooltip": "Text post-processing mode",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image for vision models (first frame used)",
                }),
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional path to audio file (WAV/MP3) for audio models",
                }),
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional path to video file for video models",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key for Bearer token auth (leave empty if none)",
                }),
                "json_schema_hint": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "JSON schema hint appended to system prompt when json_mode is on",
                }),
                "stop_custom": ("STRING", {
                    "default": "",
                    "tooltip": "Custom stop string (leave empty for none)",
                }),
            },
        }

    def generate(
        self,
        server_url,
        model,
        system_prompt,
        prompt,
        stream,
        max_tokens,
        temperature,
        seed,
        json_mode,
        unload_after_generate,
        text_postprocess,
        image=None,
        audio_path="",
        video_path="",
        api_key="",
        json_schema_hint="",
        stop_custom="",
    ):
        base_url = server_url.rstrip("/")
        headers = _build_headers(api_key)

        model_used = (model or "").strip()
        if not model_used:
            model_used = "local-model"

        if not prompt.strip():
            return ("", "", "", "", model_used)

        system_text = system_prompt.strip()
        if json_mode:
            json_instruction = "You must respond with valid JSON only. No other text."
            if json_schema_hint.strip():
                json_instruction += f"\nFollow this JSON schema:\n{json_schema_hint.strip()}"
            system_text = f"{system_text}\n\n{json_instruction}" if system_text else json_instruction

        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})

        user_content = []
        user_content.append({"type": "text", "text": prompt})

        if image is not None:
            img_b64 = _image_tensor_to_base64(image)
            if img_b64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })

        audio_b64 = _file_to_base64(audio_path.strip()) if audio_path.strip() else None
        if audio_b64:
            ext = os.path.splitext(audio_path.strip())[1].lower().lstrip(".")
            fmt = ext if ext in ("wav", "mp3", "flac", "ogg") else "wav"
            user_content.append({
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": fmt},
            })

        video_b64 = _file_to_base64(video_path.strip()) if video_path.strip() else None
        if video_b64:
            user_content.append({
                "type": "input_video",
                "input_video": {"data": video_b64},
            })

        if len(user_content) == 1:
            user_content = user_content[0]["text"]

        messages.append({"role": "user", "content": user_content})

        payload: Dict[str, Any] = {
            "model": model_used,
            "messages": messages,
        }

        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if temperature != 0.8:
            payload["temperature"] = temperature
        if seed != -1:
            payload["seed"] = seed
        if stop_custom.strip():
            payload["stop"] = [stop_custom.strip()]
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        tick = _make_progress() if stream else None

        try:
            if stream:
                payload["stream"] = True
                r = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=300,
                    stream=True,
                )
                r.raise_for_status()
                full_content, full_reasoning, meta = _parse_stream_and_accumulate(r, tick)
                raw_json = json.dumps({
                    "content": full_content,
                    "reasoning": full_reasoning,
                    "meta": meta,
                }, ensure_ascii=False)
            else:
                r = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=300,
                )
                r.raise_for_status()
                resp_json = r.json()
                raw_json = json.dumps(resp_json, ensure_ascii=False)
                choice = (resp_json.get("choices") or [{}])[0]
                message = choice.get("message", {})
                full_content = message.get("content", "") or choice.get("text", "")
                full_reasoning = (
                    message.get("reasoning_content", "")
                    or message.get("reasoning", "")
                    or message.get("thoughts", "")
                )
                if not full_reasoning and full_content:
                    clean, thinking = _extract_thinking(full_content)
                    if thinking:
                        full_reasoning = thinking
                        full_content = clean

        except Exception as exc:
            err = f"[LlamaCpp ERROR] {exc}"
            if unload_after_generate:
                try:
                    requests.post(f"{base_url}/models/unload", json={"model": model_used}, timeout=5)
                except Exception:
                    pass
            return (err, "", "", "", model_used)

        answer = _clean_answer(full_content)
        answer = _postprocess_text(answer, text_postprocess)
        thinking_text = _postprocess_text(full_reasoning, text_postprocess)

        json_str = ""
        if json_mode:
            parsed = _extract_json_from_text(full_content)
            if parsed is not None:
                json_str = json.dumps(parsed, ensure_ascii=False)

        if unload_after_generate:
            try:
                requests.post(f"{base_url}/models/unload", json={"model": model_used}, timeout=5)
            except Exception:
                pass

        return (answer, thinking_text, json_str, raw_json, model_used)


class LlamaCppModelSelector:
    CATEGORY = "LLM / Simple llama.cpp"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_name",)
    FUNCTION = "select"
    OUTPUT_TOOLTIPS = ("Selected model name to wire to LlamaCppRouterClient",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {
                    "default": DEFAULT_URL,
                    "tooltip": "llama.cpp server base URL",
                }),
                "model": ("STRING", {
                    "default": "",
                    "tooltip": "Model name — click Fetch Models button to pick from the server",
                }),
            }
        }

    def select(self, server_url: str, model: str):
        return (model,)


NODE_CLASS_MAPPINGS = {
    "LlamaCppRouterClient": LlamaCppRouterClient,
    "LlamaCppModelSelector": LlamaCppModelSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCppRouterClient": "🦙 Llama.cpp Router Client (Chat + Vision + Audio + Video + Stream)",
    "LlamaCppModelSelector": "🦙 Llama.cpp Model Selector",
}
