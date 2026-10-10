# OpenRouter 评测运行说明

本项目通过 OpenRouter 的 OpenAI-compatible endpoint 调用
`openai/gpt-4o-2024-08-06`。该模型快照、判分提示词、`temperature=0`、
`max_tokens=10` 和 yes/no 判定规则均与 LongMemEval 官方
`evaluate_qa.py` 保持一致。报告中应写成：

> LongMemEval official judge protocol, using GPT-4o-2024-08-06 via OpenRouter.

不要把 OpenRouter API Key 发到聊天、写入脚本、写进命令参数或提交到 Git。
脚本只读取当前进程的 `OPENROUTER_API_KEY` 环境变量。

## 1. 环境

评测脚本只使用 Python 标准库，不需要额外安装 SDK。Python 3.10 或更新版本即可。

## 2. 在当前 PowerShell 会话安全输入 Key

以下方式不会把 Key 显示在屏幕上，也不会把明文写入命令历史：

```powershell
$secret = Read-Host "OpenRouter API Key" -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $env:OPENROUTER_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
}
```

关闭这个 PowerShell 窗口后，该环境变量会消失。建议在 OpenRouter 后台为该
Key 设置较小的消费上限。

## 3. 只检查 Key，不调用模型

```powershell
python scripts/evaluate_qa_openrouter.py check-key
```

输出只显示脱敏后的 Key 标签、额度和使用量，不显示完整 Key。

## 4. 无 API 的结构检查

```powershell
.\scripts\run_openrouter_evaluation.ps1 -DryRun -Limit 0
```

该步骤会导出12个条件文件，并检查预计评测条数，不调用模型、不产生费用。

## 5. 单条 smoke test

```powershell
.\scripts\run_openrouter_evaluation.ps1 -Limit 1
```

检查输出中的 raw response 是否为 `yes` 或 `no`，并确认 partial CSV 中只有一条
新增的 `answer_correct`。

## 6. 断点续跑全部 2,400 条

```powershell
.\scripts\run_openrouter_evaluation.ps1 -Limit 0 -Resume
```

脚本逐条追加并立即刷新结果。网络中断后重复同一命令即可跳过已有结果。
脚本默认在成功请求之间等待3.2秒，以遵守新账户常见的20 requests/minute
限制；遇到429时还会读取服务端重置时间后自动重试。

默认只允许 OpenRouter 路由到 OpenAI provider，不允许换成其他 provider；这样
更接近官方评测。评测日志为：

```text
outputs/evaluation_openrouter/gpt4o_2024_08_06_all_conditions.jsonl
```

完整合并后的结果为：

```text
outputs/generation_local/generation_all_conditions_scored_openrouter.csv
```

## 安全和复现规则

- 不要提交 `.env`、API Key 或终端截图中的 Key。
- 不要把模型改成自动路由或 `:free` 模型后仍称为官方 GPT-4o Judge。
- smoke test 成功后使用 `-Resume`，避免覆盖或重复付费。
- 保留原始未评分 generation CSV；评分写入新的 CSV。
- 日志保存 requested/resolved model、原始 yes/no、延迟和 token usage，便于审计。
