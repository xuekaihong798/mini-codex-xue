#!/usr/bin/env python3
"""mini-codex-xue — 本地 AI Coding Agent
    Model: Gemma4 8B + Ollama Tool API  |  Mac M5 · 16GB

Architecture:
    User → Gemma4 → Tool Calls → Agent Runtime → Workspace
                                   ├── read_file
                                   ├── write_file
                                   ├── list_files
                                   ├── search_code (ripgrep)
                                   ├── git_diff
                                   ├── run_compile (mvn)
                                   ├── run_tests (mvn)
                                   └── rollback (git reset)
"""

import json, os, subprocess, sys, time
from pathlib import Path
from typing import Any
from ollama import Client

# ── Config ───────────────────────────────────────────
MODEL        = "gemma4:16k"
ALLOWED_ROOT = os.path.realpath("/Users/Admin/project")
MAX_TOKENS   = 2048
MAX_STEPS    = 20
MVN          = "/opt/homebrew/bin/mvn"

# ── Tool definitions ─────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file. Use before modifying any file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to project root, e.g. src/main/java/com/example/UserService.java"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write the full content of a file. Overwrites if exists. Provide the COMPLETE file content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to project root"},
                    "content": {"type": "string", "description": "Complete new file content"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory. Use to discover project structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path relative to project root, e.g. src/main/java/"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search code for a keyword or pattern using ripgrep. Returns file:line:content matches. Use BEFORE reading files to find relevant code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Keyword or regex pattern to search for"},
                    "path": {"type": "string", "description": "Optional subdirectory to search in, e.g. src/main/java/ (default: entire project)"}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show current git diff of all uncommitted changes. Call after write_file to verify what you changed.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_compile",
            "description": "Run 'mvn compile'. Returns {success, errors[]}. Call after making changes to verify compilation.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run 'mvn test'. Returns {passed, failed, total, errors[]}. Call AFTER compile succeeds.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rollback",
            "description": "Discard ALL uncommitted changes and restore workspace to last commit. Use when compile or tests fail and you cannot fix the errors. This is a DESTRUCTIVE operation — use only as last resort.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
]

# ── Safety ───────────────────────────────────────────
def check_path(path: str) -> tuple[bool, str]:
    try:
        real = os.path.realpath(os.path.join(ALLOWED_ROOT, path.lstrip("/")))
    except Exception:
        return False, "bad path"
    if not real.startswith(ALLOWED_ROOT + "/") and real != ALLOWED_ROOT:
        return False, f"outside root: {real}"
    return True, real

# ── Dispatch ─────────────────────────────────────────
def dispatch(tool_call: dict) -> dict:
    """Execute a tool call. Returns {result: ...} or {error: ...}."""
    fn = tool_call["function"]["name"]
    raw = tool_call["function"]["arguments"]
    try:
        args = raw if isinstance(raw, dict) else json.loads(raw)
    except (json.JSONDecodeError, KeyError):
        return {"error": "invalid arguments"}

    if fn == "read_file":
        ok, real = check_path(args["path"])
        if not ok: return {"error": real}
        try:
            with open(real) as f:
                return {"result": f.read()}
        except FileNotFoundError:
            return {"error": f"not found: {args['path']}"}

    elif fn == "write_file":
        ok, real = check_path(args["path"])
        if not ok: return {"error": real}
        content = args.get("content", "")
        os.makedirs(os.path.dirname(real), exist_ok=True)
        with open(real, "w") as f:
            f.write(content)
        return {"result": f"Written {args['path']} ({len(content)} bytes)"}

    elif fn == "list_files":
        ok, real = check_path(args["path"])
        if not ok: return {"error": real}
        try:
            result = []
            for entry in sorted(Path(real).iterdir()):
                suffix = "/" if entry.is_dir() else ""
                result.append(entry.name + suffix)
            return {"result": "\n".join(result) if result else "(empty)"}
        except FileNotFoundError:
            return {"error": f"not found: {args['path']}"}

    elif fn == "search_code":
        keyword = args.get("keyword", "")
        search_path = args.get("path", "").strip() or "."
        ok, real = check_path(search_path)
        if not ok and args.get("path"):
            return {"error": real}
        real_dir = os.path.join(ALLOWED_ROOT, search_path.lstrip("/"))
        try:
            r = subprocess.run(
                ["/opt/homebrew/bin/rg", "--no-heading", "-n", "--color", "never", keyword, real_dir],
                capture_output=True, text=True, timeout=15, cwd=ALLOWED_ROOT
            )
            out = r.stdout.strip()
            if not out:
                return {"result": f"No matches for '{keyword}'"}
            # Limit output to 100 lines / 8KB
            lines = out.split("\n")
            if len(lines) > 100:
                out = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"
            if len(out) > 8192:
                out = out[:8192] + "\n... (truncated)"
            return {"result": out}
        except subprocess.TimeoutExpired:
            return {"error": "search timed out"}
        except FileNotFoundError:
            # Fallback to grep
            try:
                r = subprocess.run(
                    ["grep", "-rn", "--color", "never", keyword, real_dir],
                    capture_output=True, text=True, timeout=15, cwd=ALLOWED_ROOT
                )
                out = r.stdout.strip()
                if not out:
                    return {"result": f"No matches for '{keyword}'"}
                lines = out.split("\n")
                if len(lines) > 100:
                    out = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"
                if len(out) > 8192:
                    out = out[:8192] + "\n... (truncated)"
                return {"result": out}
            except subprocess.TimeoutExpired:
                return {"error": "search timed out"}

    elif fn == "git_diff":
        try:
            r = subprocess.run(
                ["git", "diff", "--color=never"],
                capture_output=True, text=True, timeout=10, cwd=ALLOWED_ROOT
            )
            out = r.stdout.strip()
            return {"result": out if out else "(no changes)"}
        except Exception as e:
            return {"error": str(e)}

    elif fn == "run_compile":
        try:
            r = subprocess.run(
                [MVN, "-q", "compile"],
                capture_output=True, text=True, timeout=120, cwd=ALLOWED_ROOT
            )
            if r.returncode == 0:
                return {"result": {"success": True, "errors": []}}
            else:
                # Extract error lines
                errors = [line.strip() for line in (r.stdout + "\n" + r.stderr).split("\n")
                          if "ERROR" in line or "error:" in line.lower()]
                return {"result": {"success": False, "errors": errors[:20]}}
        except subprocess.TimeoutExpired:
            return {"error": "compile timed out"}

    elif fn == "run_tests":
        try:
            r = subprocess.run(
                [MVN, "test"],
                capture_output=True, text=True, timeout=120, cwd=ALLOWED_ROOT
            )
            out = r.stdout + r.stderr
            # Parse Maven test results
            import re
            m = re.search(r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)", out)
            if m:
                total = int(m.group(1))
                failures = int(m.group(2))
                errors = int(m.group(3))
                passed = total - failures - errors
                return {"result": {"passed": passed, "failed": failures + errors, "total": total, "errors": []}}
            # Fallback: try to find BUILD result
            if "BUILD SUCCESS" in out:
                return {"result": {"passed": 0, "failed": 0, "total": 0, "errors": [], "note": "BUILD SUCCESS but could not parse counts"}}
            # Extract error lines
            err_lines = [line.strip() for line in out.split("\n") if "ERROR" in line or "FAIL" in line]
            return {"result": {"passed": 0, "failed": 1, "total": 0, "errors": err_lines[:20]}}
        except subprocess.TimeoutExpired:
            return {"error": "tests timed out"}

    elif fn == "rollback":
        r = subprocess.run(["git", "reset", "--hard", "HEAD"],
                           capture_output=True, text=True, timeout=10, cwd=ALLOWED_ROOT)
        subprocess.run(["git", "clean", "-fd"],
                       capture_output=True, timeout=10, cwd=ALLOWED_ROOT)
        out = r.stdout.strip() or "workspace reset to HEAD"
        return {"result": out, "rollback": True}

    return {"error": f"unknown tool: {fn}"}


# ── Agent State ──────────────────────────────────────
class AgentState:
    def __init__(self):
        self.files_read: set[str] = set()
        self.files_modified: set[str] = set()
        self.tests_run: bool = False
        self.tests_passed: int | None = None
        self.tests_failed: int | None = None
        self.compile_success: bool | None = None
        self.rollbacks: int = 0

    def reset(self):
        """Reset state after rollback — all file changes are discarded."""
        self.files_read.clear()
        self.files_modified.clear()
        self.tests_run = False
        self.tests_passed = None
        self.tests_failed = None
        self.compile_success = None
        self.rollbacks += 1

    def status_text(self) -> str:
        lines = ["## Current State"]
        lines.append("### Files read:")
        for f in sorted(self.files_read):
            lines.append(f"  - {f}")
        if not self.files_read:
            lines.append("  (none)")
        lines.append("### Files modified:")
        for f in sorted(self.files_modified):
            lines.append(f"  - {f}")
        if not self.files_modified:
            lines.append("  (none)")
        lines.append("### Last compile:")
        if self.compile_success is None:
            lines.append("  not run yet")
        elif self.compile_success:
            lines.append("  PASS")
        else:
            lines.append("  FAIL - fix errors before running tests")
        lines.append("### Tests:")
        if not self.tests_run:
            lines.append("  not run yet")
        else:
            lines.append(f"  {self.tests_passed}/{self.tests_passed + self.tests_failed} passed")
        if self.rollbacks > 0:
            lines.append(f"### Rollbacks: {self.rollbacks}")
        return "\n".join(lines)


# ── Agent Loop ───────────────────────────────────────
def agent(task: str) -> dict:
    """Run agent on a task. Returns metrics dict."""
    client = Client()
    state = AgentState()
    base_system = (
        "You are an expert Java coding agent. Work step by step:\n"
        "1. SEARCH for relevant code with search_code to find files and call chains\n"
        "2. READ files you plan to modify\n"
        "3. WRITE modified files with complete content\n"
        "4. CHECK your work with git_diff\n"
        "5. COMPILE with run_compile — if it fails, fix the errors\n"
        "6. TEST with run_tests — if tests fail, fix the bugs\n"
        "7. ROLLBACK with rollback tool ONLY if you cannot fix errors after 3 attempts\n"
        "NEVER write a file without reading it first.\n"
        "NEVER skip compilation before running tests.\n"
        "Output your summary in Chinese."
    )

    metrics = {
        "tool_calls": 0,
        "files_modified": 0,
        "compile_success": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "rollbacks": 0,
        "duration": 0,
        "steps": 0,
    }
    t0 = time.time()

    messages = [
        {"role": "system", "content": base_system},
        {"role": "user", "content": task},
    ]

    for step in range(MAX_STEPS):
        # Inject state into system context each turn
        if step > 0 and messages:
            # Insert state as a system message before the last assistant message
            state_msg = {"role": "system", "content": state.status_text()}
            # Insert before the last message (which is the tool result)
            # Actually, append after the last tool result
            messages.append(state_msg)

        print(f"\n── Step {step + 1}/{MAX_STEPS} ──", flush=True)

        response = client.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            options={"num_ctx": 16384, "num_predict": MAX_TOKENS},
        )

        msg = response.message
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            content = (msg.content or "")[:500]
            print(f"  [done] {content}")
            metrics["duration"] = int(time.time() - t0)
            metrics["steps"] = step + 1
            metrics["files_modified"] = len(state.files_modified)
            metrics["rollbacks"] = state.rollbacks
            return metrics

        for tc in tool_calls:
            fn = tc.function
            name = fn.name
            args = fn.arguments if isinstance(fn.arguments, dict) else {}

            print(f"  [{name}] {json.dumps(args)[:150]}", flush=True)
            result = dispatch({"function": {"name": name, "arguments": args}})

            # Update state
            if name == "read_file":
                p = args.get("path", "")
                if p:
                    state.files_read.add(p)
            elif name == "write_file":
                p = args.get("path", "")
                if p:
                    state.files_modified.add(p)
            elif name == "run_tests":
                state.tests_run = True
                r = result.get("result", {})
                if isinstance(r, dict):
                    state.tests_passed = r.get("passed", 0)
                    state.tests_failed = r.get("failed", 0)
                    metrics["tests_passed"] = r.get("passed", 0)
                    metrics["tests_failed"] = r.get("failed", 0)
            elif name == "run_compile":
                r = result.get("result", {})
                if isinstance(r, dict):
                    state.compile_success = r.get("success", False)
                    metrics["compile_success"] = r.get("success", False)
            elif name == "rollback":
                state.reset()
                metrics["rollbacks"] = state.rollbacks
                metrics["compile_success"] = False
                metrics["tests_passed"] = 0
                metrics["tests_failed"] = 0

            metrics["tool_calls"] += 1

            preview = json.dumps(result)[:250].replace("\n", "\\n")
            print(f"    → {preview}", flush=True)

            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [tc.model_dump()],
            })
            messages.append({
                "role": "system",
                "content": f"Tool result: {json.dumps(result)}",
            })

    # Max steps hit
    metrics["duration"] = int(time.time() - t0)
    metrics["steps"] = MAX_STEPS
    metrics["files_modified"] = len(state.files_modified)
    metrics["rollbacks"] = state.rollbacks
    print("  [max_steps] Reached limit")
    return metrics


# ── Benchmark Runner ─────────────────────────────────
BENCHMARK_TASKS = {
    # ── 单文件修改 (4 tasks) ─────────────────────────
    "s1_add_method": {
        "name": "[单文件] 添加 isValidPhoneForCountry",
        "task": (
            "在 UserService.java 中添加一个新方法 isValidPhoneForCountry(String phone, String country)。\n"
            "country 参数支持 \"CN\" 和 \"JP\"，分别使用现有的中国和日本号码正则规则。\n"
            "未知 country 返回 false。编译通过后运行测试。"
        )
    },
    "s2_add_overload": {
        "name": "[单文件] formatPhone 增加重载",
        "task": (
            "修改 UserService.java 的 formatPhone 方法，增加一个重载版本 formatPhone(String phone, String country)。\n"
            "日本号码格式为 XXX-XXXX-XXXX，中国号码格式为 XXX-XXXX-XXXX。\n"
            "先 read_file 再修改。编译通过后运行测试。"
        )
    },
    "s3_add_email": {
        "name": "[单文件] 添加 isValidEmail",
        "task": (
            "在 UserService.java 中添加 isValidEmail(String email) 方法。\n"
            "规则：必须包含 @，@ 前后至少 1 个字符，总长度不超过 254。\n"
            "用 search_code 找到 UserService.java 的位置，read_file 读取后修改。\n"
            "编译通过后运行测试。"
        )
    },
    "s4_add_javadoc": {
        "name": "[单文件] 添加完整 Javadoc",
        "task": (
            "为 UserService.java 中所有 public 方法添加完整的 Javadoc 注释。\n"
            "每个方法需要 @param 和 @return 标签。不修改方法逻辑。\n"
            "编译通过即成功。"
        )
    },
    # ── 跨文件修改 (4 tasks) ─────────────────────────
    "m1_extract_validator": {
        "name": "[跨文件] 提取 PhoneValidator",
        "task": (
            "1. 创建 src/main/java/com/example/PhoneValidator.java，包含：\n"
            "   - public static boolean isValidPhone(String phone)\n"
            "   - public static String formatPhone(String phone)\n"
            "2. 修改 UserService.java，让 isValidPhone 和 formatPhone 委托给 PhoneValidator\n"
            "3. 修改 TestRunner.java 让测试仍然通过\n"
            "用 search_code 找到所有相关文件。编译通过后运行测试。"
        )
    },
    "m2_create_order_svc": {
        "name": "[跨文件] 创建 OrderService + 测试",
        "task": (
            "1. 创建 src/main/java/com/example/OrderService.java，包含：\n"
            "   - public boolean validateOrderPhone(String phone) — 调用 UserService.isValidPhone\n"
            "2. 修改 TestRunner.java，增加 2 个测试：validOrderPhone（用合法号码）和 invalidOrderPhone（用非法号码）\n"
            "编译通过后运行测试。"
        )
    },
    "m3_rename_method": {
        "name": "[跨文件] 全局重命名方法",
        "task": (
            "将 UserService.java 中的 isValidPhone 方法重命名为 validatePhoneNumber。\n"
            "必须用 search_code 搜索所有引用，同步修改 TestRunner.java 和 UserServiceTest.java。\n"
            "不能遗漏任何引用。编译通过后运行测试。"
        )
    },
    "m4_extract_string_utils": {
        "name": "[跨文件] 提取 StringUtils + 重构",
        "task": (
            "1. 创建 src/main/java/com/example/StringUtils.java，包含：\n"
            "   - public static boolean isEmpty(String s)\n"
            "   - public static String removeNonDigit(String s) — 去掉所有非数字字符\n"
            "2. 修改 UserService.java 的 normalizePhone 使用 StringUtils.removeNonDigit\n"
            "3. 修改 TestRunner.java，增加 StringUtils 的测试\n"
            "编译通过后运行测试。"
        )
    },
    # ── 错误恢复 (2 tasks) ───────────────────────────
    "e1_compile_repair": {
        "name": "[错误恢复] 编译失败 → 修复",
        "task": (
            "在 UserService.java 中把 isValidPhone 返回类型从 boolean 改成 int（制造编译错误）。\n"
            "然后运行 run_compile，确认编译失败。\n"
            "接着根据编译错误信息修回正确的 boolean 类型。\n"
            "如果修了 3 次还失败就用 rollback 回滚。\n"
            "最终必须编译通过。"
        )
    },
    "e2_test_repair": {
        "name": "[错误恢复] 测试失败 → 修复",
        "task": (
            "在 UserService.java 的 isValidPhone 方法中，故意让 080 开头的号码返回 false。\n"
            "然后运行 run_tests，确认测试失败。\n"
            "接着根据测试失败信息修复这个 bug，恢复 080 号码的支持。\n"
            "如果修了 3 次还失败就用 rollback 回滚。\n"
            "最终必须所有测试通过。"
        )
    },
}


def run_benchmark(task_key: str) -> dict:
    """Run a single benchmark task. Ensures clean git state before and after."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {BENCHMARK_TASKS[task_key]['name']}")
    print(f"Task Key:  {task_key}")
    print(f"{'='*60}")

    # Ensure clean starting state
    subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True, cwd=ALLOWED_ROOT)
    subprocess.run(["git", "clean", "-fd"], capture_output=True, cwd=ALLOWED_ROOT)

    result = agent(BENCHMARK_TASKS[task_key]["task"])

    # Rollback: restore clean state for next benchmark
    subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True, cwd=ALLOWED_ROOT)
    subprocess.run(["git", "clean", "-fd"], capture_output=True, cwd=ALLOWED_ROOT)

    return result


def run_all_benchmarks():
    """Run all benchmark tasks and print summary with category stats."""
    all_results = {}
    for key in BENCHMARK_TASKS:
        result = run_benchmark(key)
        all_results[key] = result

    # Categorize
    categories = {"s": ("单文件修改", []), "m": ("跨文件修改", []), "e": ("错误恢复", [])}

    print(f"\n{'='*70}")
    print("BENCHMARK REPORT — mini-codex-xue (Gemma4 8B · Mac M5)")
    print(f"{'='*70}")
    print(f"{'Task':45s} {'Compile':8s} {'Tests':10s} {'Tools':6s} {'Time':6s} {'Rollback':9s}")
    print("-" * 70)

    for key, r in all_results.items():
        name = BENCHMARK_TASKS[key]["name"]
        c = "PASS" if r["compile_success"] else "FAIL"
        total_tests = r["tests_passed"] + r["tests_failed"]
        t = f"{r['tests_passed']}/{total_tests}" if total_tests > 0 else "-"
        rb = str(r.get("rollbacks", 0))
        print(f"  {name:43s} {c:8s} {t:10s} {r['tool_calls']:4d}   {r['duration']:3d}s  {rb:7s}")
        cat = key[0]
        if cat in categories:
            categories[cat][1].append(r)

    # Category summary
    print(f"\n{'='*70}")
    print("CATEGORY SUMMARY")
    print(f"{'='*70}")
    total_all = 0
    passed_all = 0
    for cat_key, (cat_name, results) in categories.items():
        if not results:
            continue
        n = len(results)
        compiled = sum(1 for r in results if r["compile_success"])
        tests_ok = sum(1 for r in results if r["tests_failed"] == 0 and r["tests_passed"] > 0)
        avg_tools = sum(r["tool_calls"] for r in results) / n
        avg_time = sum(r["duration"] for r in results) / n
        total_rollbacks = sum(r.get("rollbacks", 0) for r in results)
        print(f"  {cat_name}: {n} tasks | compile {compiled}/{n} | tests {tests_ok}/{n} | avg {avg_tools:.1f} tools | avg {avg_time:.0f}s | {total_rollbacks} rollbacks")
        total_all += n
        passed_all += compiled

    print(f"\n  TOTAL: {passed_all}/{total_all} compile pass ({100*passed_all/total_all:.0f}%)")


# ── Rollback Tool ────────────────────────────────────
def rollback():
    """Reset workspace to last committed state. Called externally or by agent."""
    print("  -> Rolling back to HEAD...")
    subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True, cwd=ALLOWED_ROOT)
    subprocess.run(["git", "clean", "-fd"], capture_output=True, cwd=ALLOWED_ROOT)
    print("  -> Rollback complete.")


# ── Main ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Gemma4 Local Agent v3 — Industrial Grade")
    ap.add_argument("task", nargs="?", help="Task description (omit for benchmark mode)")
    ap.add_argument("--benchmark", "-b", choices=list(BENCHMARK_TASKS.keys()) + ["all"],
                    help="Run benchmark task(s)")
    ap.add_argument("--rollback", "-r", action="store_true", help="Rollback workspace to HEAD")
    ap.add_argument("--list-tasks", "-l", action="store_true", help="List benchmark tasks")
    args = ap.parse_args()

    if args.list_tasks:
        for k, v in BENCHMARK_TASKS.items():
            print(f"  {k}: {v['name']}")
        sys.exit(0)

    if args.rollback:
        rollback()
        sys.exit(0)

    if args.benchmark:
        if args.benchmark == "all":
            run_all_benchmarks()
        else:
            r = run_benchmark(args.benchmark)
            print(f"\nResult: {json.dumps(r, indent=2)}")
        sys.exit(0)

    if args.task:
        print(f"Model: {MODEL}  |  Root: {ALLOWED_ROOT}")
        print(f"Task:  {args.task}")
        t0 = time.time()
        result = agent(args.task)
        print(f"\nResult: {json.dumps(result, indent=2)}")
        print(f"Total: {time.time() - t0:.0f}s")
    else:
        ap.print_help()
