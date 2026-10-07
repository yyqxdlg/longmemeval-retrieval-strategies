# 给队友的后续执行提示词

把下面整段提示词直接发给负责本地 Llama 生成的队友或其 Codex 任务。执行者需要拥有
`meta-llama/Llama-3.1-8B-Instruct` 官方 Hugging Face 访问权限，并能使用本地 NVIDIA GPU。

```text
请接手 LongMemEval 项目的本地答案生成与待评分文件导出。检索已经完成，不要重跑或改写正式检索结果。

仓库：
https://github.com/yyqxdlg/longmemeval-retrieval-strategies

必须从包含以下提交的最新 main 开始：
45fa85e3760494a18663c554cf8a1be0777c11d9
Complete confirmatory retrieval and generation handoff

默认 Windows 路径：
- 项目父目录：D:\课程\II2201
- 仓库：D:\课程\II2201\longmemeval-retrieval-strategies
- 模型和 Hugging Face 缓存：D:\AIProject\model
- generation 环境：D:\课程\II2201\.venv-generation

如果你的磁盘布局不同，可以调整绝对路径，但不要把模型、缓存或数据集放进 Git 仓库；尽量不要在 C 盘创建大型文件。

重要限制：

1. 不调用 OpenAI API、Hugging Face Inference API 或任何远程推理服务。
2. Hugging Face 只用于从官方仓库下载模型权重；推理必须在本机 GPU 上完成。
3. 只使用官方 `meta-llama/Llama-3.1-8B-Instruct`，不得绕过 gated 权限，也不得使用来源不明的镜像或 wheel。
4. 不把 Hugging Face token、模型权重、原始数据集、API key 或缓存提交到 GitHub。
5. 不伪造 answer correctness。没有人工或官方 judge 标签时，`answer_correct` 必须保持为空。
6. 不得因为看到部分答案质量而中途修改 prompt、revision、量化或 decoding 设置。
7. 长任务必须逐行保存并使用 `--resume`。
8. 不要重跑 512 或 1024 正式检索；已提交结果就是生成输入。

已完成且已验证的输入：

- 固定样本：200 题，IE/MR/KU/TR 各 50；IE 子类型为 17/17/16。
- 固定 ID：`config/confirmatory_200_ids.json`。
- 512 主检索：`outputs/confirmatory_200/retrieval_stella512_fp32_k5_k10.csv`。
- 512 CSV：2000 行、2000 个唯一 `(question_id, strategy, k)` 键、non-finite=0。
- 512 CSV SHA256：`b3531f30d75f9f7f72170b4a8e660d2965e08bab569b850fc96544bab9d5b42c`。
- 1024 sensitivity 已完成，但不用于答案生成。
- 检索输入 `pilot_questions.json` SHA256：`586a14e73f598885be4480094a98aabd263b1378e5beee062c4304df84ba7b19`。

开始前先阅读：

- `README.md`
- `LOCAL_RUN_SUMMARY.md`
- `TEAMMATE_HANDOFF.md`
- `METHOD_CHANGES.md`
- `LIMITATIONS.md`
- `PENDING_API_EVALUATION.md`
- `run_generation.py`
- `generation.py`
- `analyze_local_generation.py`
- `longmemeval_eval.py`

一、同步和检查仓库

1. 拉取最新 `main`。
2. 检查当前分支、工作区、远端和最新提交。
3. 不覆盖已有检索 CSV、metadata、表格和图。
4. 如果远端已经在 `45fa85e` 之后前进，保留新提交并在最新 main 上工作。

二、检查机器

记录并保存：

- 操作系统
- GPU 名称和显存
- NVIDIA driver
- CUDA 和 PyTorch CUDA 版本
- Python 版本
- D 盘或模型盘剩余空间

先运行：

```powershell
nvidia-smi
python --version
```

模型盘应至少预留 25 GB。不要更新显卡驱动。

三、准备 generation 环境

优先使用独立环境 `D:\课程\II2201\.venv-generation`。如果不存在，创建 Python 3.11 环境。根据本机 GPU 和驱动，从 PyTorch 官方说明选择匹配的 CUDA PyTorch；不要破坏 retrieval 环境。

安装并验证：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-generation.txt
hf --help
```

至少需要：Transformers >= 4.43.2、Accelerate、官方 bitsandbytes、safetensors、huggingface_hub。先验证 CUDA 矩阵运算为 finite，并能构造 NF4 `BitsAndBytesConfig`。

四、恢复被 Git 忽略的 200 条完整问题记录

仓库不会包含原始 LongMemEval 数据集。把 classic cleaned LongMemEval 数据放在：

`data/longmemeval-cleaned/`

然后运行：

```powershell
python prepare_pilot_data.py `
  --data-root data/longmemeval-cleaned `
  --per-task 50 `
  --seed 42 `
  --stratify-ie `
  --output-dir outputs/confirmatory_200
```

必须验证：

- `outputs/confirmatory_200/pilot_questions.json` 正好 200 条；
- 四个任务各 50 条；
- IE 子类型为 17/17/16；
- 有序的 question ID、task type、question type 与 `config/confirmatory_200_ids.json` 完全一致；
- 文件 SHA256 等于 `586a14e73f598885be4480094a98aabd263b1378e5beee062c4304df84ba7b19`。

任一项不同都停止，不要继续生成，也不要修改已提交的检索 CSV。

五、登录并下载官方 Llama

需要用户本人在安全终端中登录。不要让用户把 token 发到聊天中，也不要把 token 放进命令行参数历史。

```powershell
$env:HF_HOME = 'D:\AIProject\model'
$env:HF_HUB_CACHE = 'D:\AIProject\model\hub'
hf auth login
hf auth whoami
$snapshot = hf download meta-llama/Llama-3.1-8B-Instruct --revision main
$revision = Split-Path $snapshot -Leaf
$snapshot
$revision
```

检查完整模型文件和磁盘空间。`$revision` 必须是下载解析出的 immutable commit；后续正式运行统一使用该值，不能继续使用会变化的 `main`。

如果官方访问仍未获批或被拒绝，停止并准确报告权限状态，不要绕过。

六、本地模型加载冒烟

先只加载模型并生成一句很短的回答。要求：

- 4-bit NF4；
- `device_map=auto`；
- `do_sample=False`；
- seed 42；
- 很小的 `max_new_tokens`；
- 输出实际显存、设备分配和 resolved revision；
- 不允许 CPU/disk offload；
- 不访问远程 inference endpoint。

如果 4-bit 加载失败，保存完整错误、显存和软件版本并报告。不要静默切换正式配置。只有在明确决定采用其他官方支持方案后，所有正式条件才能统一使用同一配置。

七、真实输入冒烟与 resume

先对正式输出运行一条 pending row：

```powershell
python run_generation.py `
  --pilot-file outputs/confirmatory_200/pilot_questions.json `
  --retrieval-results outputs/confirmatory_200/retrieval_stella512_fp32_k5_k10.csv `
  --output outputs/generation_local/generation_all_conditions.csv `
  --model-name meta-llama/Llama-3.1-8B-Instruct `
  --model-revision $revision `
  --quantization 4bit `
  --max-new-tokens 128 `
  --seed 42 `
  --include-baselines `
  --limit 1
```

检查：

- 回答非空；
- prompt 不含 gold answer、`has_answer`、task label、answer session IDs 或“正确证据”标签；
- token count、context hash、generation latency 和 resolved revision 已记录；
- 模型全部留在 GPU；
- `answer_correct` 为空。

然后用完全相同设置加 `--resume`。确认第一行被跳过且没有重复键。

八、正式本地生成

继续同一个输出文件：

```powershell
python run_generation.py `
  --pilot-file outputs/confirmatory_200/pilot_questions.json `
  --retrieval-results outputs/confirmatory_200/retrieval_stella512_fp32_k5_k10.csv `
  --output outputs/generation_local/generation_all_conditions.csv `
  --model-name meta-llama/Llama-3.1-8B-Instruct `
  --model-revision $revision `
  --quantization 4bit `
  --max-new-tokens 128 `
  --seed 42 `
  --include-baselines `
  --resume
```

正式条件为：

- recency k=5/10
- semantic k=5/10
- hybrid_raw k=5/10
- hybrid_minmax k=5/10
- hybrid_rrf k=5/10
- no_retrieval k=0
- oracle k=0

预期总数：`5 × 2 × 200 + 200 + 200 = 2400` 行。

运行结束后再次执行同一 `--resume` 命令，必须新增 0 行。

九、验证生成文件

必须验证：

- 实际 2400 行；
- 2400 个唯一 `(question_id, strategy, k)` 键；
- 每个正式条件 200 行；
- 缺失和空回答数量明确记录；
- revision、量化、seed、max_new_tokens、system prompt hash 在所有行一致；
- context hash 存在；
- token/latency 字段可解析；
- `answer_correct` 全部为空；
- no_retrieval/oracle 各只运行一次/题；
- 没有把 20 题 pilot 或 smoke 行混入正式文件。

不要因为少量空答案而修改 prompt 后局部重跑并混入同一结果；应保留诊断并先报告。

十、本地分析与人工审核模板

```powershell
python analyze_local_generation.py `
  --generation outputs/generation_local/generation_all_conditions.csv `
  --pilot-file outputs/confirmatory_200/pilot_questions.json `
  --output-dir outputs/analysis_generation `
  --manual-audit-output outputs/manual_audit/manual_audit_28.csv `
  --audit-size 28
```

只报告完成率、空答案、token、生成延迟、总延迟、输出长度和明确标为 `exploratory heuristic` 的字符串指标。不得把 normalized exact match 或字符串包含率称为官方 answer accuracy。

人工审核 CSV 保持人工正确性和备注列为空，并覆盖四种任务、不同策略、Top-5/10、no_retrieval、oracle、recall 成功/失败和长上下文案例。

十一、导出 12 个待评分 hypothesis JSONL

```powershell
python longmemeval_eval.py export `
  --input outputs/generation_local/generation_all_conditions.csv `
  --output outputs/hypotheses_pending `
  --split-by-condition
```

验证生成 12 个 JSONL，每个 200 条：五种检索策略各 k=5/10，再加 no_retrieval_k0 和 oracle_k0。每条必须包含 `question_id`、`hypothesis`、`strategy`、`k`。

本任务不运行 GPT-4o judge，不要求 `OPENAI_API_KEY`，不产生官方 accuracy。

十二、更新文档和 runtime

更新：

- `LOCAL_RUN_SUMMARY.md`：填入 immutable Llama revision、2400 行完整性、文件 hash、实际完成/未评分状态；
- `PENDING_API_EVALUATION.md`：填入 12 个 JSONL 的实际条数与路径；
- `LIMITATIONS.md`：保留量化、硬件、未评分和许可说明；
- `outputs/runtime/`：保存 generation 环境、GPU、CUDA、包版本、磁盘、开始/结束时间和 resume history。

明确说明：

- 512 是主检索；1024 只是 retrieval-only sensitivity；
- 20 题 pilot 与 200 题正式结果分开；
- 当前不能报告官方 accuracy；
- 以后只需 judge、merge 和 accuracy analysis。

十三、测试与 Git

至少运行：

```powershell
python -m unittest test_pipeline test_retrieval_strategies test_time_handling
git diff --check
```

提交前：

1. 扫描 Hugging Face token、API key 和其他密钥；
2. 确认没有 staged 模型权重、缓存、原始数据集或人工审核中的敏感数据；
3. 检查 generation CSV 和 hypothesis 文件体积合理；
4. fetch 远端并确认没有冲突；
5. 提交代码、合理大小的正式结果、分析和更新后的文档；
6. push 后确认远端 commit 与本地 HEAD 一致。

不要在只完成模型下载或 smoke test 后停止。应持续完成正式 2400 行生成、验证、分析、JSONL 导出、文档更新和 Git push。只有遇到官方模型权限、显存/磁盘不足、无法恢复的环境错误或缺失原始数据集时，才暂停并报告准确的阻塞证据。
```
