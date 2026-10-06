from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from experiment_io import (
    merge_resume_metadata,
    read_json,
    runtime_snapshot,
    sha256_file,
    write_json,
)
from generation import (
    DEFAULT_SYSTEM_PROMPT,
    build_answer_messages,
    context_sha256,
    reconstruct_retrieved_memories,
)


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EXTRA_FIELDS = [
    "token_count",
    "retrieved_context_token_count",
    "prompt_token_count",
    "completion_token_count",
    "generation_latency_ms",
    "total_latency_ms",
    "generated_answer",
    "answer_correct",
    "judge_model",
    "answer_model",
    "answer_model_revision",
    "quantization",
    "generation_torch_dtype",
    "max_new_tokens",
    "generation_seed",
    "context_sha256",
]
RESUME_SETTING_KEYS = (
    "model_name",
    "requested_model_revision",
    "quantization",
    "torch_dtype",
    "max_new_tokens",
    "seed",
    "device_map",
    "warmup",
    "system_prompt_sha256",
    "future_session_policy",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    _, rows = read_csv(path)
    keys = set()
    for row in rows:
        if str(row.get("generated_answer", "")).strip():
            keys.add(
                (
                    str(row["question_id"]),
                    str(row["strategy"]),
                    int(row["k"]),
                )
            )
    return keys


def resolve_dtype(torch, name: str):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def load_answer_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = resolve_dtype(torch, args.torch_dtype)
    quantization_config = None
    if args.quantization == "4bit":
        compute_dtype = torch.bfloat16 if args.torch_dtype == "auto" else dtype
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif args.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: dict[str, Any] = {
        "revision": args.model_revision,
        "device_map": args.device_map,
        "trust_remote_code": False,
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()
    return torch, tokenizer, model


def model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    raise RuntimeError("Could not determine the model input device")


def terminator_ids(tokenizer) -> list[int]:
    ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0 and eot != tokenizer.unk_token_id:
        ids.append(eot)
    return sorted(set(x for x in ids if isinstance(x, int) and x >= 0))


def generate_one(
    *,
    torch,
    tokenizer,
    model,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> tuple[str, int, int, float]:
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    encoded = encoded.to(model_input_device(model))
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit and prompt_tokens + max_new_tokens > int(context_limit):
        raise ValueError(
            f"Prompt ({prompt_tokens}) plus max_new_tokens ({max_new_tokens}) "
            f"exceeds the model context limit ({context_limit})."
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminator_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0

    generated_ids = output[0, prompt_tokens:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer, prompt_tokens, int(generated_ids.shape[-1]), latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LongMemEval answers from previously retrieved memories."
    )
    parser.add_argument("--pilot-file", required=True)
    parser.add_argument("--retrieval-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument("--k", type=int, nargs="+", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate at most this many pending rows (useful for a smoke test).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    pilot_path = Path(args.pilot_file)
    retrieval_path = Path(args.retrieval_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".metadata.json")

    records_raw = json.loads(pilot_path.read_text(encoding="utf-8"))
    records = {str(record["question_id"]): record for record in records_raw}
    input_fields, retrieval_rows = read_csv(retrieval_path)
    if not retrieval_rows:
        raise SystemExit("Retrieval-results CSV is empty")
    required_fields = {
        "question_id",
        "strategy",
        "k",
        "retrieved_memory_ids",
        "latency_ms",
    }
    missing_fields = required_fields - set(input_fields)
    if missing_fields:
        raise SystemExit(
            f"Retrieval-results CSV is missing fields: {sorted(missing_fields)}"
        )

    if args.k:
        selected_k = set(args.k)
        retrieval_rows = [row for row in retrieval_rows if int(row["k"]) in selected_k]

    retrieval_metadata_path = retrieval_path.with_suffix(".metadata.json")
    retrieval_metadata = (
        read_json(retrieval_metadata_path) if retrieval_metadata_path.exists() else {}
    )
    future_policy = retrieval_metadata.get(
        "future_session_policy",
        "exclude_strictly_later_calendar_dates; retain_same_day",
    )
    exclude_future = future_policy != "include_all"

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()

    import hashlib

    metadata = {
        "model_name": args.model_name,
        "requested_model_revision": args.model_revision,
        "quantization": args.quantization,
        "torch_dtype": args.torch_dtype,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "device_map": args.device_map,
        "warmup": not args.no_warmup,
        "k_values": sorted({int(row["k"]) for row in retrieval_rows}),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "future_session_policy": future_policy,
        "input_sha256": sha256_file(pilot_path),
        "retrieval_results_sha256": sha256_file(retrieval_path),
        "run_history": [runtime_snapshot()],
        "decoding": {"do_sample": False, "temperature": 0},
    }

    existing_metadata = None
    if args.resume:
        if not output_path.exists() or not metadata_path.exists():
            raise SystemExit("--resume requires both the existing CSV and metadata JSON")
        try:
            existing_metadata = read_json(metadata_path)
            metadata = merge_resume_metadata(
                existing_metadata,
                metadata,
                setting_keys=RESUME_SETTING_KEYS,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    elif output_path.exists():
        raise SystemExit(f"Output already exists; use --resume or a new path: {output_path}")

    done = completed_keys(output_path) if args.resume else set()
    pending = [
        row
        for row in retrieval_rows
        if (str(row["question_id"]), str(row["strategy"]), int(row["k"])) not in done
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print("All selected generation rows are already complete.")
        return

    print(f"Loading answer model: {args.model_name} ({args.quantization})")
    torch, tokenizer, model = load_answer_model(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    resolved_revision = getattr(model.config, "_commit_hash", None)
    if (
        existing_metadata
        and existing_metadata.get("resolved_model_revision")
        and resolved_revision
        and existing_metadata["resolved_model_revision"] != resolved_revision
    ):
        raise SystemExit(
            "The model revision behind the requested ref changed since the "
            "previous run. Resume with the recorded immutable revision or use "
            "a new output file."
        )
    metadata["resolved_model_revision"] = resolved_revision
    write_json(metadata_path, metadata)

    if not args.no_warmup:
        generate_one(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Reply with OK."},
            ],
            max_new_tokens=2,
        )

    output_fields = list(input_fields)
    for field in EXTRA_FIELDS:
        if field not in output_fields:
            output_fields.append(field)

    mode = "a" if args.resume else "w"
    need_header = mode == "w" or output_path.stat().st_size == 0
    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        if need_header:
            writer.writeheader()
            handle.flush()

        for index, row in enumerate(pending, start=1):
            qid = str(row["question_id"])
            if qid not in records:
                raise RuntimeError(f"Question {qid} is missing from the pilot file")
            record = records[qid]
            memories = reconstruct_retrieved_memories(
                record,
                row.get("retrieved_memory_ids", ""),
                exclude_future=exclude_future,
            )
            messages, context = build_answer_messages(
                question=str(record.get("question", "")),
                question_date=record.get("question_date"),
                retrieved_memories=memories,
                system_prompt=system_prompt,
            )
            answer, prompt_tokens, completion_tokens, generation_ms = generate_one(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
            )
            context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
            retrieval_ms = float(row.get("latency_ms") or 0.0)

            output_row = dict(row)
            output_row.update(
                {
                    "token_count": context_tokens,
                    "retrieved_context_token_count": context_tokens,
                    "prompt_token_count": prompt_tokens,
                    "completion_token_count": completion_tokens,
                    "generation_latency_ms": generation_ms,
                    "total_latency_ms": retrieval_ms + generation_ms,
                    "generated_answer": answer,
                    "answer_correct": "",
                    "judge_model": "",
                    "answer_model": args.model_name,
                    "answer_model_revision": resolved_revision or args.model_revision,
                    "quantization": args.quantization,
                    "generation_torch_dtype": args.torch_dtype,
                    "max_new_tokens": args.max_new_tokens,
                    "generation_seed": args.seed,
                    "context_sha256": context_sha256(context),
                }
            )
            writer.writerow(output_row)
            handle.flush()
            print(
                f"[{index}/{len(pending)}] {qid} {row['strategy']} k={row['k']} "
                f"prompt={prompt_tokens} completion={completion_tokens}"
            )

    print(f"Generation results: {output_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
