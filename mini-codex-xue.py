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
                                   └── run_tests (mvn)
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
        "5. COMPILE with run_compile\n"
        "6. TEST with run_tests\n"
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
    print("  [max_steps] Reached limit")
    return metrics


# ── Benchmark Runner ─────────────────────────────────
BENCHMARK_TASKS = {
    "single_phone_validate": {
        "name": "Add isValidPhoneForCountry method",
        "task": (
            "在 UserService.java 中添加一个新方法 isValidPhoneForCountry(String phone, String country)。\n"
            "country 参数支持 \"CN\" 和 \"JP\"，分别调用现有的中国和日本号码规则。\n"
            "未知 country 返回 false。\n"
            "编译通过后运行测试。"
        )
    },
    "single_format_update": {
        "name": "Update formatPhone with country param",
        "task": (
            "修改 UserService.java 的 formatPhone 方法，增加一个重载版本：\n"
            "formatPhone(String phone, String country)。\n"
            "日本号码格式为 XXX-XXXX-XXXX，中国号码格式为 XXX-XXXX-XXXX。\n"
            "需要先 read_file 再修改，修改后编译并运行测试。"
        )
    },
    "multi_cross_file": {
        "name": "Create PhoneValidator + refactor UserService",
        "task": (
            "1. 创建新文件 src/main/java/com/example/PhoneValidator.java，包含：\n"
            "   - public static boolean isValidPhone(String phone) — 支持中国和日本号码\n"
            "   - public static String formatPhone(String phone) — 格式化号码\n"
            "2. 修改 UserService.java，让 isValidPhone 和 formatPhone 委托给 PhoneValidator\n"
            "3. 修改 UserServiceTest.java 或 TestRunner.java，确保测试仍然通过\n"
            "完成后编译并运行测试验证。"
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
    """Run all benchmark tasks and print summary."""
    all_results = {}
    for key in BENCHMARK_TASKS:
        result = run_benchmark(key)
        all_results[key] = result

    # Print final report
    print(f"\n{'='*60}")
    print("BENCHMARK REPORT")
    print(f"{'='*60}")
    for key, r in all_results.items():
        name = BENCHMARK_TASKS[key]["name"]
        c = "PASS" if r["compile_success"] else "FAIL"
        t = f"{r['tests_passed']}/{r['tests_passed'] + r['tests_failed']}"
        print(f"  {name:40s}  compile:{c:5s}  tests:{t:8s}  tools:{r['tool_calls']:3d}  time:{r['duration']:3d}s")


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
