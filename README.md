# mini-codex-xue

本地 AI Coding Agent —— Gemma4 8B + Ollama Tool API，Mac M5 16GB 可运行。

## 架构

```
User → Gemma4 → Tool Calls → Agent Runtime → Workspace
                       ├── read_file
                       ├── write_file
                       ├── list_files
                       ├── search_code (ripgrep)
                       ├── git_diff
                       ├── run_compile (mvn)
                       └── run_tests (mvn)
```

## 前置依赖

```bash
# Python 依赖
pip install ollama --break-system-packages

# 系统依赖
brew install ripgrep maven

# Ollama 模型
ollama pull gemma4:16k
```

## 使用

```bash
# 单次任务
python3 mini-codex-xue.py "重构 UserService.java，提取 PhoneValidator"

# 跑 benchmark
python3 mini-codex-xue.py --benchmark single_phone_validate
python3 mini-codex-xue.py --benchmark all

# 列出 benchmark 任务
python3 mini-codex-xue.py --list-tasks

# Rollback
python3 mini-codex-xue.py --rollback
```

## Benchmark 结果 (Mac M5 · Gemma4 8B · 10 tasks)

| 分类 | 通过率 | 平均工具 | 平均耗时 |
|------|--------|---------|---------|
| 单文件修改 (4) | **100%** | 8.2 | 81s |
| 跨文件重构 (4) | **75%** | 8.2 | 119s |
| 错误恢复 (2) | **100%** | 10.0 | 3470s |
| **总计 (10)** | **90%** | **8.7** | **—** |

详见 [ARTICLE.md](ARTICLE.md)

## examples/

Java 测试工作区（Maven），含中国+日本手机号校验示例。
