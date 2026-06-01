# mini-codex-xue

本地 AI Coding Agent —— Gemma4 8B + Ollama Tool API，Mac M5 16GB 可运行。

## 核心架构原则

**对于 8B 级模型，正确的架构不是：**

```
Read Large File → Rewrite Entire File  ← 上下文溢出 + 文件截断
```

**而是：**

```
Locate → Patch → Verify
```

具体工具链：

| 操作 | 错误方式 | 正确方式 |
|------|---------|---------|
| 编辑代码 | `write_file` 重写 3500 行 | `edit_file(path, old_text, new_text)` — 只传差异 |
| 添加方法 | `write_file` 重建整个文件 | `insert_method(file, after_method, source)` |
| 重命名 | 手动 search → 逐个修改 | `rename_symbol(old, new)` — 全量原子替换 |
| 找代码 | `read_file` 读整个文件 | `locate_symbol(name)` → 精确返回 file:line:context |

## 架构

```
User → Gemma4 → Tool Calls → Agent Runtime → Workspace
                       ├── edit_file (Locate→Patch→Verify)
                       ├── insert_method / rename_symbol
                       ├── locate_symbol
                       ├── read_file (offset/limit chunked)
                       ├── write_file (仅小文件)
                       ├── list_files
                       ├── search_code (ripgrep)
                       ├── fix_single_error / verify_changes (compound)
                       ├── analyze_error / root_cause_analyzer
                       ├── git_diff
                       ├── run_compile (mvn)
                       ├── run_tests (mvn)
                       └── rollback (git reset)
```

**状态机**: IDLE → OBSERVE → EXPLAIN → FIX → VERIFY → DONE

**运行时已包含**: ToolRouter (按阶段过滤工具) · ContextCompression (每5步/大文件提前触发) · FailureClustering (共享根因聚合) · AgentMemory (历史失败模式库 → 自动升级策略) · SelfBenchmark (自动生成任务+弱点分析) · NO-OP Gate (拦截"只读不写") · TaskSummaryInjection (每3步注入任务焦点)

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

# 跑 benchmark（自动记录到 Agent Memory）
python3 mini-codex-xue.py --benchmark all

# 列出 benchmark 任务
python3 mini-codex-xue.py --list-tasks

# 自评测（扫描代码库，自动生成任务，发现弱点）
python3 mini-codex-xue.py --self-benchmark --visualize

# 对外部仓库做压力测试
python3 mini-codex-xue.py -w /path/to/repo --self-benchmark

# Agent Memory 管理
python3 mini-codex-xue.py --memory-stats     # 查看历史失败模式
python3 mini-codex-xue.py --memory-reset    # 清空记忆库

# 生成 HTML 可视化报告
python3 mini-codex-xue.py --benchmark all --visualize

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

## 压力测试：JSON-java (27k LOC, 85 files, 777 tests)

将 Runtime 扔进从未见过的中型真实仓库后，发现了 8B 模型的关键瓶颈：

| 机制 | v1（无优化） | v2（4项优化后） |
|------|------------|---------------|
| 分块读取 | 一次读 3500 行 → 上下文溢出 | limit=200-500 → 逐块消化 |
| Gate 拦截 | 0 次 | **3 次** NO-OP 检测 |
| Context Compress | 0 次 | **10+ 次** (Step 2 即触发) |
| 模型行为 | 读完文件直接放弃 | Gate 拦截后被迫继续操作 |

**结论**：对于 8B 模型，`Read Large File → Rewrite Entire File` 必然失败。正确架构是 `Locate → Patch → Verify`。

## examples/

Java 测试工作区（Maven），含中国+日本手机号校验示例。
