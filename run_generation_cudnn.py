"""Frozen local generation launcher: cuDNN prefill, math one-token decode."""
import sys
sys.dont_write_bytecode = True
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = r'D:\AIProject\model'
os.environ['HF_HUB_CACHE'] = r'D:\AIProject\model\hub'
import argparse
import hashlib
import importlib.metadata
import json
import runpy
from pathlib import Path
from datetime import datetime, timezone
import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--output', required=True)
parser.add_argument('--model-name', required=True)
parser.add_argument('--model-revision', required=True)
parser.add_argument('--quantization', required=True)
parser.add_argument('--max-new-tokens', required=True, type=int)
parser.add_argument('--seed', required=True, type=int)
parser.add_argument('--include-baselines', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--limit', type=int)
args, _ = parser.parse_known_args()
assert args.model_name == 'meta-llama/Llama-3.1-8B-Instruct'
assert args.model_revision == '0e9e39f249a16976918f6564b8830bc894c89659'
assert args.quantization == '4bit' and args.max_new_tokens == 128 and args.seed == 42 and args.include_baselines
assert '--allow-cpu-offload' not in sys.argv
repo = Path.cwd()
runner = repo / 'run_generation.py'
assert runner.is_file(), 'Run this launcher from the project directory.'
output = Path(args.output)
provenance = output.with_suffix(output.suffix + '.attention_backend.json')
settings = {
    'attention_backend': 'CUDNN_PREFILL_MATH_SINGLE_TOKEN_DECODE',
    'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(),
    'packages': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'bitsandbytes', 'accelerate']},
    'model_revision': args.model_revision,
}
if provenance.exists():
    ledger = json.loads(provenance.read_text(encoding='utf-8'))
    assert ledger['settings'] == settings, 'Backend/code/package settings changed; stop and inspect.'
else:
    assert not output.exists(), 'Existing generation output has no backend ledger; stop and inspect.'
    ledger = {'settings': settings, 'runs': []}
output.parent.mkdir(parents=True, exist_ok=True)
entry = {'started_utc': datetime.now(timezone.utc).isoformat(), 'resume': args.resume, 'limit': args.limit, 'status': 'RUNNING', 'gpu': torch.cuda.get_device_name(0)}
ledger['runs'].append(entry)
def save():
    temporary = provenance.with_suffix(provenance.suffix + '.tmp')
    temporary.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    temporary.replace(provenance)
save()
original_sdpa = functional.scaled_dot_product_attention
def routed_sdpa(*inputs, **kwargs):
    query = inputs[0] if inputs else kwargs['query']
    backend = SDPBackend.MATH if query.shape[-2] == 1 else SDPBackend.CUDNN_ATTENTION
    with sdpa_kernel(backend):
        return original_sdpa(*inputs, **kwargs)
functional.scaled_dot_product_attention = routed_sdpa
sys.path.insert(0, str(repo))
sys.argv[0] = str(runner)
print('Attention backend: CUDNN_PREFILL_MATH_SINGLE_TOKEN_DECODE', flush=True)
try:
    runpy.run_path(str(runner), run_name='__main__')
    entry['status'] = 'COMPLETE'
except BaseException as exc:
    entry['status'] = 'COMPLETE' if isinstance(exc, SystemExit) and exc.code in (None, 0) else 'FAILED'
    if entry['status'] == 'FAILED':
        entry['error'] = repr(exc)
    raise
finally:
    functional.scaled_dot_product_attention = original_sdpa
    entry['ended_utc'] = datetime.now(timezone.utc).isoformat()
    save()
