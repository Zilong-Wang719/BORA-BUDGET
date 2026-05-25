from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Union


PROMPT_TEMPLATES = {
    "default": {
        "use_chat": False,
    },
    "qwen_chat": {
        "use_chat": True,
    },
}

AUTO_DEVICE_MAP_VALUES = {"auto", "balanced", "balanced_low_0", "sequential"}


def detect_prompt_template(model_name: str) -> str:
    name = model_name.lower()
    if "qwen" in name:
        return "qwen_chat"
    return "default"


def resolve_torch_dtype(value: str) -> Any:
    lowered = str(value).lower()
    if lowered == "auto":
        return "auto"
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if lowered not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {value}")
    return mapping[lowered]


def _ensure_visible_devices(backend_cfg: dict[str, Any]) -> None:
    visible_devices = backend_cfg.get("cuda_visible_devices")
    if visible_devices is None:
        return
    visible = str(visible_devices)
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, ""}:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _ensure_tokenizer_padding(tokenizer: Any) -> Any:
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    return tokenizer


def _render_prompt_with_tokenizer(
    tokenizer: Any,
    *,
    prompt_template: str,
    system_prompt: str | None,
    user_prompt: str,
    enable_thinking: bool | None = None,
) -> str:
    template = PROMPT_TEMPLATES.get(prompt_template, PROMPT_TEMPLATES["default"])
    if template["use_chat"] and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Older tokenizer templates may not expose Qwen3's enable_thinking flag.
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)
    parts = []
    if system_prompt:
        parts.append(system_prompt)
    parts.append(user_prompt)
    return "\n\n".join(parts).strip() + "\n"


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int


class TransformersBackend:
    backend_name = "hf_transformers"

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "auto",
        prompt_template: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        torch_dtype: str = "auto",
        max_context_tokens: int = 4096,
        repetition_penalty: float = 1.0,
        cuda_visible_devices: str | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.trust_remote_code = bool(trust_remote_code)
        self.torch_dtype = torch_dtype
        self.max_context_tokens = int(max_context_tokens)
        self.repetition_penalty = float(repetition_penalty)
        self.cuda_visible_devices = cuda_visible_devices
        self.enable_thinking = enable_thinking
        self.prompt_template = (
            detect_prompt_template(model_name)
            if prompt_template == "auto"
            else prompt_template
        )
        self._tokenizer = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        _ensure_visible_devices({"cuda_visible_devices": self.cuda_visible_devices})
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self._generation_config_cls = GenerationConfig
        self._tokenizer = _ensure_tokenizer_padding(
            AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
            )
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "dtype": resolve_torch_dtype(self.torch_dtype),
            "local_files_only": self.local_files_only,
        }
        device_name = str(self.device).lower()
        if device_name in AUTO_DEVICE_MAP_VALUES:
            model_kwargs["device_map"] = self.device
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if device_name not in AUTO_DEVICE_MAP_VALUES:
            self._model = self._model.to(self.device)
        self._model.eval()

    def _model_input_device(self) -> Any:
        self._lazy_load()
        assert self._model is not None
        return next(self._model.parameters()).device

    def render_prompt(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        enable_thinking: bool | None = None,
    ) -> str:
        self._lazy_load()
        assert self._tokenizer is not None
        return _render_prompt_with_tokenizer(
            self._tokenizer,
            prompt_template=self.prompt_template,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enable_thinking=self.enable_thinking if enable_thinking is None else enable_thinking,
        )

    def generate_text(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float | None = None,
    ) -> GenerationResult:
        self._lazy_load()
        assert self._model is not None
        assert self._tokenizer is not None
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_tokens,
        )
        prompt_tokens = int(inputs["input_ids"].shape[1])
        device = self._model_input_device()
        inputs = {key: value.to(device) for key, value in inputs.items()}

        do_sample = temperature > 1e-5
        generation_config = self._generation_config_cls.from_model_config(self._model.config)
        generation_config.max_new_tokens = int(max_new_tokens)
        generation_config.do_sample = do_sample
        generation_config.pad_token_id = self._tokenizer.pad_token_id
        generation_config.eos_token_id = self._tokenizer.eos_token_id
        generation_config.repetition_penalty = (
            float(repetition_penalty)
            if repetition_penalty is not None
            else self.repetition_penalty
        )
        if do_sample:
            generation_config.temperature = float(temperature)
            generation_config.top_p = float(top_p)
        else:
            generation_config.temperature = 1.0
            generation_config.top_p = 1.0
            generation_config.top_k = 0

        import torch

        start = perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, generation_config=generation_config)
        latency_ms = int((perf_counter() - start) * 1000)
        completion_ids = output_ids[0][prompt_tokens:]
        completion_tokens = int(len(completion_ids))
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
        )


class VllmBackend:
    backend_name = "vllm"

    def __init__(
        self,
        *,
        model_name: str,
        prompt_template: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        torch_dtype: str = "auto",
        max_context_tokens: int = 4096,
        repetition_penalty: float = 1.0,
        cuda_visible_devices: str | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        max_num_seqs: int = 8,
        enable_thinking: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.local_files_only = bool(local_files_only)
        self.trust_remote_code = bool(trust_remote_code)
        self.torch_dtype = torch_dtype
        self.max_context_tokens = int(max_context_tokens)
        self.repetition_penalty = float(repetition_penalty)
        self.cuda_visible_devices = cuda_visible_devices
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.enforce_eager = bool(enforce_eager)
        self.max_num_seqs = int(max_num_seqs)
        self.enable_thinking = enable_thinking
        self.prompt_template = (
            detect_prompt_template(model_name)
            if prompt_template == "auto"
            else prompt_template
        )
        self._tokenizer = None
        self._engine = None
        self._sampling_params_cls = None

    def _lazy_load(self) -> None:
        if self._tokenizer is not None and self._engine is not None:
            return
        _ensure_visible_devices({"cuda_visible_devices": self.cuda_visible_devices})
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        dtype = resolve_torch_dtype(self.torch_dtype)
        self._tokenizer = _ensure_tokenizer_padding(
            AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
            )
        )
        self._sampling_params_cls = SamplingParams
        self._engine = LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            trust_remote_code=self.trust_remote_code,
            dtype=dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_context_tokens,
            enforce_eager=self.enforce_eager,
            max_num_seqs=self.max_num_seqs,
            skip_tokenizer_init=False,
        )

    def render_prompt(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        enable_thinking: bool | None = None,
    ) -> str:
        self._lazy_load()
        assert self._tokenizer is not None
        return _render_prompt_with_tokenizer(
            self._tokenizer,
            prompt_template=self.prompt_template,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enable_thinking=self.enable_thinking if enable_thinking is None else enable_thinking,
        )

    def generate_text(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float | None = None,
    ) -> GenerationResult:
        self._lazy_load()
        assert self._engine is not None
        assert self._sampling_params_cls is not None
        do_sample = temperature > 1e-5
        sampling_params = self._sampling_params_cls(
            n=1,
            max_tokens=int(max_new_tokens),
            temperature=float(temperature) if do_sample else 0.0,
            top_p=float(top_p) if do_sample else 1.0,
            repetition_penalty=(
                float(repetition_penalty)
                if repetition_penalty is not None
                else self.repetition_penalty
            ),
            skip_special_tokens=True,
        )
        start = perf_counter()
        request_output = self._engine.generate([prompt], sampling_params, use_tqdm=False)[0]
        latency_ms = int((perf_counter() - start) * 1000)
        best = request_output.outputs[0]
        prompt_tokens = len(getattr(request_output, "prompt_token_ids", []) or [])
        completion_tokens = len(getattr(best, "token_ids", []) or [])
        return GenerationResult(
            text=best.text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
        )


BackendType = Union[TransformersBackend, VllmBackend]

_BACKEND_CACHE: dict[tuple[Any, ...], BackendType] = {}


def resolve_backend_config(config: dict[str, Any], section: str) -> dict[str, Any]:
    shared = dict(config.get("llm", {}))
    scoped = dict(config.get(section, {}))
    merged = {**shared, **scoped}
    if "model_name" not in merged:
        raise ValueError(f"{section}.model_name (or llm.model_name) is required for LLM backends")
    merged["backend"] = str(merged.get("backend", "hf_transformers"))
    return merged


def get_llm_backend(config: dict[str, Any], section: str) -> BackendType:
    backend_cfg = resolve_backend_config(config, section)
    backend_name = str(backend_cfg.get("backend", "hf_transformers"))
    key = (
        backend_name,
        backend_cfg["model_name"],
        backend_cfg.get("device", "auto"),
        backend_cfg.get("prompt_template", "auto"),
        bool(backend_cfg.get("local_files_only", True)),
        bool(backend_cfg.get("trust_remote_code", True)),
        str(backend_cfg.get("torch_dtype", "auto")),
        int(backend_cfg.get("max_context_tokens", 4096)),
        float(backend_cfg.get("repetition_penalty", 1.0)),
        backend_cfg.get("cuda_visible_devices"),
        int(backend_cfg.get("tensor_parallel_size", 1)),
        float(backend_cfg.get("gpu_memory_utilization", 0.9)),
        bool(backend_cfg.get("enforce_eager", False)),
        int(backend_cfg.get("max_num_seqs", 8)),
        backend_cfg.get("enable_thinking"),
    )
    if key in _BACKEND_CACHE:
        return _BACKEND_CACHE[key]

    common_kwargs = {
        "model_name": str(backend_cfg["model_name"]),
        "prompt_template": str(backend_cfg.get("prompt_template", "auto")),
        "local_files_only": bool(backend_cfg.get("local_files_only", True)),
        "trust_remote_code": bool(backend_cfg.get("trust_remote_code", True)),
        "torch_dtype": str(backend_cfg.get("torch_dtype", "auto")),
        "max_context_tokens": int(backend_cfg.get("max_context_tokens", 4096)),
        "repetition_penalty": float(backend_cfg.get("repetition_penalty", 1.0)),
        "cuda_visible_devices": (
            str(backend_cfg["cuda_visible_devices"])
            if backend_cfg.get("cuda_visible_devices") is not None
            else None
        ),
        "enable_thinking": (
            bool(backend_cfg["enable_thinking"])
            if backend_cfg.get("enable_thinking") is not None
            else None
        ),
    }

    if backend_name in {"hf_transformers", "transformers"}:
        backend: BackendType = TransformersBackend(
            **common_kwargs,
            device=str(backend_cfg.get("device", "auto")),
        )
    elif backend_name == "vllm":
        backend = VllmBackend(
            **common_kwargs,
            tensor_parallel_size=int(backend_cfg.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(backend_cfg.get("gpu_memory_utilization", 0.9)),
            enforce_eager=bool(backend_cfg.get("enforce_eager", False)),
            max_num_seqs=int(backend_cfg.get("max_num_seqs", 8)),
        )
    else:
        raise ValueError(f"Unsupported llm backend: {backend_name}")

    _BACKEND_CACHE[key] = backend
    return backend


def get_transformers_backend(config: dict[str, Any], section: str) -> BackendType:
    return get_llm_backend(config, section)
