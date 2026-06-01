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

import json, os, re, subprocess, sys, time
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
    {
        "type": "function",
        "function": {
            "name": "analyze_error",
            "description": "Analyze a compile or test error to extract structured info: file path, line number, error type, and message. Call this immediately after run_compile or run_tests fails, before attempting any fix. Understanding the exact error location and type prevents guesswork.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Which tool produced the error: 'compile' or 'tests'"},
                    "error_raw": {"type": "string", "description": "The full error output. Copy the 'errors' list or raw output text from the compile/test tool result."}
                },
                "required": ["source", "error_raw"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "root_cause_analyzer",
            "description": "Perform systematic root cause analysis AFTER identify WHERE an error is. Takes analyze_error findings + the relevant code snippet, returns: (a) cause_analysis — WHY the error occurs, (b) potential_risks — what could break if fixed carelessly, (c) fix_strategy — step-by-step fix approach. Use this BEFORE write_file to avoid guesswork. write_file is BLOCKED until this returns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "error_summary": {"type": "string", "description": "The error description and location from analyze_error (e.g. 'compile_error at UserService.java:42: int cannot convert to boolean')"},
                    "code_snippet": {"type": "string", "description": "The relevant code around the error location (copy from the file you read, include 5 lines before and after the error line)"},
                    "expected_behavior": {"type": "string", "description": "What should this code do? (e.g. 'return a boolean indicating validity')"}
                },
                "required": ["error_summary", "code_snippet", "expected_behavior"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fix_single_error",
            "description": "COMPOUND TOOL: Read file, apply a targeted fix at a specific line, write file, compile — all in one atomic call. Reduces 4 round-trips to 1. Use after root_cause_analyzer has identified the exact fix. Returns: {fixed, compile_result, new_code_snippet}. If compile fails, the original file is restored.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Path to the file to fix, e.g. src/main/java/com/example/UserService.java"},
                    "line": {"type": "integer", "description": "The line number that needs to change"},
                    "old_line": {"type": "string", "description": "The EXACT current line content (used for verification and pattern matching)"},
                    "new_line": {"type": "string", "description": "The replacement line content"},
                    "reason": {"type": "string", "description": "Brief explanation of WHY this fix works (from root_cause_analyzer)"}
                },
                "required": ["file", "line", "old_line", "new_line", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_changes",
            "description": "COMPOUND TOOL: Run compile then tests as one atomic verification step. Returns combined pass/fail with details. Use after making changes to verify everything at once.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
]

# ── Task Profiles ────────────────────────────────────
# Each task type customizes the state machine's EXPLAIN/FIX/VERIFY rules.
# All profiles share the same state machine; only rules differ.
class TaskProfile:
    """Customizable rules for different task types."""
    def __init__(self, type_name: str, explain_requires_search: bool = False,
                 verify_tests_required: bool = True, strict_explain: bool = False,
                 max_fix_attempts: int = 3, pre_fix_read_required: bool = True):
        self.type_name = type_name
        self.explain_requires_search = explain_requires_search  # Must search_code before root_cause_analyzer
        self.verify_tests_required = verify_tests_required       # Tests must pass for DONE
        self.strict_explain = strict_explain                     # BLOCK write_file even in observing
        self.max_fix_attempts = max_fix_attempts                 # When to suggest rollback
        self.pre_fix_read_required = pre_fix_read_required       # Must read file before writing

    def explain_hint(self) -> str:
        """Extra instructions injected during EXPLAIN phase."""
        hints = []
        if self.explain_requires_search:
            hints.append("- CROSS-FILE task: call search_code to find ALL references before explaining the fix.")
        if self.strict_explain:
            hints.append("- ERROR RECOVERY: you MUST complete analyze_error → root_cause_analyzer before write_file.")
        hints.append("- Before editing, explain: (1) Why error (2) Which line (3) Why fix works")
        return "\n".join(hints) if hints else ""

    def verify_hint(self) -> str:
        """What's required to reach DONE."""
        if self.verify_tests_required:
            return "compile + tests both required"
        return "compile required (tests recommended)"

# Pre-built profiles
PROFILE_SINGLE_FILE = TaskProfile(
    type_name="single_file",
    explain_requires_search=False,
    verify_tests_required=True,
    strict_explain=False,
    max_fix_attempts=3,
    pre_fix_read_required=True,
)

PROFILE_CROSS_FILE = TaskProfile(
    type_name="cross_file",
    explain_requires_search=True,   # Must search for all references
    verify_tests_required=True,     # Tests required — breaking other files is high risk
    strict_explain=False,
    max_fix_attempts=3,
    pre_fix_read_required=True,
)

PROFILE_ERROR_RECOVERY = TaskProfile(
    type_name="error_recovery",
    explain_requires_search=True,   # Must understand full call chain
    verify_tests_required=True,     # Both compile + tests mandatory
    strict_explain=True,            # ⛔ write_file BLOCKED until root_cause_analyzer done
    max_fix_attempts=3,
    pre_fix_read_required=True,
)

def get_profile_for_task(task_key: str) -> TaskProfile:
    """Map benchmark task key to appropriate profile."""
    if task_key.startswith("s"):
        return PROFILE_SINGLE_FILE
    elif task_key.startswith("m"):
        return PROFILE_CROSS_FILE
    elif task_key.startswith("e"):
        return PROFILE_ERROR_RECOVERY
    return PROFILE_SINGLE_FILE  # default


# ── Agent Memory: Failure Pattern Database ───────────────
# Persisted historical failure patterns. After enough repetitions,
# the agent auto-upgrades strategy for known failure-prone areas.
#
# Example: PhoneValidator class errors occurred 17 times →
#   next task mentioning "PhoneValidator" or "refactor" →
#   automatically selects CrossFile strategy instead of SingleFile.

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")
MEMORY_FILE = os.path.join(MEMORY_DIR, "failure_patterns.json")

class FailurePatternDB:
    """Persistent store of historical failure patterns.
    Used to: (1) auto-select task profiles, (2) inject relevant
    historical context into the system prompt, (3) track improvement/decay.
    """

    def __init__(self, path: str = MEMORY_FILE):
        self.path = path
        self.patterns: dict[str, dict] = {}  # cluster_key → {count, error_type, files[], suggested_profile, first_seen, last_seen, successes, failures, strategy_upgrades}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.patterns = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.patterns = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.patterns, f, indent=2, ensure_ascii=False)

    def record(self, task_key: str, metrics: dict, task_profile: TaskProfile):
        """Record a benchmark result. Builds/updates failure patterns."""
        compile_ok = metrics.get("compile_success", False)
        tests_ok = metrics.get("tests_failed", 0) == 0 and metrics.get("tests_passed", 0) > 0
        success = compile_ok and tests_ok

        # Extract key files/classes from task key and trace
        task_name = BENCHMARK_TASKS.get(task_key, {}).get("name", task_key)
        trace = metrics.get("trace", [])

        # Collect files involved
        files_touched = set()
        for s in (trace or []):
            a = s.get("args", {})
            f = a.get("path", a.get("file", ""))
            if f:
                files_touched.add(f)

        # Extract class names from files
        classes = set()
        for f in files_touched:
            m = re.search(r"(\w+)\.java$", f)
            if m:
                classes.add(m.group(1))

        # Extract error types from the trace
        error_types = set()
        for s in (trace or []):
            if s["tool"] in ("analyze_error", "root_cause_analyzer"):
                rp = s.get("result_preview", "")
                for et in ["type_mismatch", "missing_symbol", "assertion_failure",
                           "null_reference", "parse_error", "missing_definition"]:
                    if et in rp:
                        error_types.add(et)
            if not s.get("result_success", True):
                error_types.add("compile_fail" if s["tool"] == "run_compile" else
                               "test_fail" if s["tool"] == "run_tests" else "tool_error")

        # For each class involved, update its pattern
        for cls in classes:
            key = f"class:{cls}"
            if key not in self.patterns:
                self.patterns[key] = {
                    "count": 0, "error_types": [], "files": list(files_touched),
                    "suggested_profile": task_profile.type_name,
                    "first_seen": time.strftime("%Y-%m-%d %H:%M"),
                    "last_seen": "", "successes": 0, "failures": 0,
                    "strategy_upgrades": 0, "task_keys": [],
                }
            p = self.patterns[key]
            p["count"] += 1
            p["last_seen"] = time.strftime("%Y-%m-%d %H:%M")
            p["task_keys"].append(task_key)
            if len(p["task_keys"]) > 20:
                p["task_keys"] = p["task_keys"][-20:]

            if success:
                p["successes"] += 1
            else:
                p["failures"] += 1
                for et in error_types:
                    if et not in p["error_types"]:
                        p["error_types"].append(et)

            # Auto-upgrade: if failures >= 3 and still on single_file, upgrade to cross_file
            if p["failures"] >= 3 and p["suggested_profile"] == "single_file":
                p["suggested_profile"] = "cross_file"
                p["strategy_upgrades"] += 1

        # Also record by error type
        for et in error_types:
            ekey = f"error:{et}"
            if ekey not in self.patterns:
                self.patterns[ekey] = {
                    "count": 0, "error_types": [et], "files": [],
                    "suggested_profile": "error_recovery" if "fail" in et else "cross_file",
                    "first_seen": time.strftime("%Y-%m-%d %H:%M"),
                    "last_seen": "", "successes": 0, "failures": 0,
                    "strategy_upgrades": 0, "task_keys": [],
                }
            ep = self.patterns[ekey]
            ep["count"] += 1
            ep["last_seen"] = time.strftime("%Y-%m-%d %H:%M")
            if success:
                ep["successes"] += 1
            else:
                ep["failures"] += 1

        self.save()

    def query(self, task_text: str) -> list[dict]:
        """Find relevant historical patterns for a task.
        Returns patterns sorted by relevance (how many times seen).
        """
        matches = []
        for key, p in self.patterns.items():
            # Check if any file or class in the pattern appears in the task
            for f in p.get("files", []):
                if f and f in task_text:
                    matches.append((key, p, 10))  # exact file match = high relevance
                    break
                cls = re.search(r"(\w+)\.java$", f)
                if cls and cls.group(1) in task_text:
                    matches.append((key, p, 8))  # class name match
                    break
            else:
                # Check if error type keywords appear
                for et in p.get("error_types", []):
                    if et in task_text.lower():
                        matches.append((key, p, 3))
                        break
        # Sort by relevance, then by count
        matches.sort(key=lambda x: (x[2], x[1]["count"]), reverse=True)
        return [{"key": m[0], "pattern": m[1], "relevance": m[2]} for m in matches[:10]]

    def suggest_profile(self, task_text: str, default_profile: TaskProfile) -> TaskProfile:
        """Auto-select the best TaskProfile based on historical patterns.
        If a class in the task has >=3 failures and was upgraded → use suggested profile.
        """
        matches = self.query(task_text)
        if not matches:
            return default_profile

        best = matches[0]
        pattern = best["pattern"]
        failures = pattern.get("failures", 0)
        suggested = pattern.get("suggested_profile", "")

        # Decision logic: if enough failures and suggestion differs from default
        if failures >= 3 and suggested == "cross_file" and default_profile.type_name == "single_file":
            print(f"  [memory] Auto-upgrade: {default_profile.type_name} → {suggested} "
                  f"(class {best['key'].replace('class:', '')} has {failures} historical failures "
                  f"across {pattern.get('count', 0)} tasks)", flush=True)
            return PROFILE_CROSS_FILE
        elif failures >= 5 and suggested == "error_recovery" and default_profile.type_name != "error_recovery":
            print(f"  [memory] Auto-upgrade: {default_profile.type_name} → error_recovery "
                  f"(error pattern {best['key']} seen {failures} times)", flush=True)
            return PROFILE_ERROR_RECOVERY

        return default_profile

    def context_injection(self, task_text: str) -> str:
        """Generate a system prompt injection with relevant historical context."""
        matches = self.query(task_text)
        if not matches:
            return ""

        lines = ["\n## Historical Context (from Agent Memory)"]
        shown = 0
        for m in matches[:3]:
            p = m["pattern"]
            key = m["key"]
            count = p.get("count", 0)
            failures = p.get("failures", 0)
            successes = p.get("successes", 0)
            suggested = p.get("suggested_profile", "")

            if count < 2:
                continue  # Skip patterns seen only once

            if key.startswith("class:"):
                cls = key.replace("class:", "")
                lines.append(f"- ⚠ {cls}: {count} tasks touched this class, "
                           f"{successes}✓/{failures}✗. "
                           f"Strategy: {suggested}.")
            elif key.startswith("error:"):
                et = key.replace("error:", "")
                if failures >= 3:
                    lines.append(f"- 🔁 {et}: appeared in {count} tasks ({failures} failures). "
                               f"Common failure pattern — verify carefully.")
            shown += 1

        if shown == 0:
            return ""
        return "\n".join(lines)

    def stats(self) -> dict:
        """Return summary stats about the memory database."""
        total = len(self.patterns)
        class_patterns = [p for k, p in self.patterns.items() if k.startswith("class:")]
        error_patterns = [p for k, p in self.patterns.items() if k.startswith("error:")]
        upgraded = sum(1 for p in self.patterns.values() if p.get("strategy_upgrades", 0) > 0)
        total_failures = sum(p.get("failures", 0) for p in self.patterns.values())
        total_successes = sum(p.get("successes", 0) for p in self.patterns.values())

        # Most failure-prone classes (top 5)
        top_classes = sorted(class_patterns, key=lambda p: p.get("failures", 0), reverse=True)[:5]
        top_list = [(re.search(r"class:(\w+)", f"class:{cp.get('files',[''])[0]}"),
                     cp["failures"], cp["count"])
                    for cp in top_classes]
        top_display = []
        for m, fails, cnt in top_list:
            name = m.group(1) if m else "?"
            top_display.append(f"{name}({fails}✗/{cnt}tasks)")

        return {
            "total_patterns": total,
            "class_patterns": len(class_patterns),
            "error_patterns": len(error_patterns),
            "strategy_upgrades": upgraded,
            "total_failures": total_failures,
            "total_successes": total_successes,
            "top_failure_classes": top_display,
        }


# Auto-select profile using memory (if db is available)
def get_profile_for_task_with_memory(task_key: str, task_text: str = "",
                                      db: FailurePatternDB | None = None) -> TaskProfile:
    """Get profile for task key, with optional memory-based auto-upgrade."""
    default = get_profile_for_task(task_key)
    if db and task_text:
        return db.suggest_profile(task_text, default)
    return default


# ── Tool Router ─────────────────────────────────────────
class ToolRouter:
    """Restricts available tools based on profile + phase + state.
    Reduces cognitive load: model only sees tools it should use right now.
    """
    # Tool name sets per profile
    ESSENTIAL = {"read_file", "write_file", "run_compile", "run_tests"}
    ERROR_RECOVERY = {"read_file", "write_file", "run_compile", "run_tests",
                      "analyze_error", "root_cause_analyzer"}
    CROSS_FILE = {"read_file", "write_file", "run_compile", "run_tests",
                  "search_code", "list_files", "git_diff",
                  "analyze_error", "root_cause_analyzer"}
    FULL = {t["function"]["name"] for t in TOOLS}

    @staticmethod
    def route(profile: TaskProfile, phase: str, state: "AgentState") -> list[dict]:
        """Return filtered tool list for current state."""
        if profile.strict_explain:
            # Error recovery: progressive disclosure
            if phase in ("idle", "observing"):
                # Only let them observe and detect errors
                allowed = {"read_file", "run_compile", "run_tests", "analyze_error"}
            elif phase == "explain":
                # Must call root_cause_analyzer — no write_file
                allowed = {"read_file", "analyze_error", "root_cause_analyzer"}
            elif phase == "fixing":
                # Now write_file is unlocked
                allowed = ToolRouter.ERROR_RECOVERY
            elif phase == "verifying":
                allowed = {"run_compile", "run_tests", "read_file", "write_file",
                          "analyze_error", "root_cause_analyzer"}
            else:
                allowed = ToolRouter.ERROR_RECOVERY
        elif profile.explain_requires_search:
            allowed = ToolRouter.CROSS_FILE
        else:
            allowed = ToolRouter.FULL

        return [t for t in TOOLS if t["function"]["name"] in allowed]


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
            m = re.search(r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)", out)
            if m:
                total = int(m.group(1))
                failures = int(m.group(2))
                errors = int(m.group(3))
                passed = total - failures - errors
                # Extract test failure details
                fail_lines = []
                for m2 in re.finditer(r"(?:FAILURE!|FAILED|ERROR!)\s*\n?\s*(.+?)(?=\n\s*(?:Tests run:|BUILD|$))", out, re.DOTALL):
                    fail_lines.append(m2.group(1).strip()[:200])
                if not fail_lines:
                    # Try simpler extraction: lines with <<< FAILURE or AssertionError
                    fail_lines = [line.strip()[:200] for line in out.split("\n") if "FAILURE" in line or "AssertionError" in line or "expected:" in line or "but was:" in line]
                return {"result": {"passed": passed, "failed": failures + errors, "total": total, "errors": fail_lines[:10]}}
            # Fallback: try to find BUILD result
            if "BUILD SUCCESS" in out:
                return {"result": {"passed": 0, "failed": 0, "total": 0, "errors": [], "note": "BUILD SUCCESS but could not parse counts"}}
            # Extract error lines
            err_lines = [line.strip() for line in out.split("\n") if "ERROR" in line or "FAIL" in line]
            return {"result": {"passed": 0, "failed": 1, "total": 0, "errors": err_lines[:20]}}
        except subprocess.TimeoutExpired:
            return {"error": "tests timed out"}

    elif fn == "analyze_error":
        source = args.get("source", "")
        error_raw = args.get("error_raw", "")
        if isinstance(error_raw, list):
            error_raw = "\n".join(str(e) for e in error_raw)
        elif isinstance(error_raw, dict):
            error_raw = json.dumps(error_raw)
        # Parse Java error patterns
        findings = []
        # Pattern 1: Maven compiler error  [ERROR] /path/to/File.java:[42,10] error message
        for m in re.finditer(r"\[?(?:ERROR|error)\]?\s*(/[^\s:]+\.java):[\(\[](\d+),(\d+)[\)\]]\s*(.*)", error_raw):
            findings.append({"file": m.group(1), "line": int(m.group(2)), "col": int(m.group(3)), "message": m.group(4).strip(), "type": "compile_error"})
        # Pattern 2: Maven test failure  at com.example.Test.method(File.java:42)
        for m in re.finditer(r"at\s+[\w.]+\.[\w.]+\(([\w./]+\.java):(\d+)\)", error_raw):
            path = m.group(1)
            if not any(f["file"].endswith(path.replace("/", "/")) or path.endswith(f["file"].split("/")[-1]) for f in findings):
                findings.append({"file": path, "line": int(m.group(2)), "col": 0, "message": "Test failure or assertion error at this location", "type": "test_failure"})
        # Pattern 3: Assertion error messages
        for m in re.finditer(r"(?:FAILED|AssertionError|assert\w+ failed)[:\s]*(.*)", error_raw, re.IGNORECASE):
            findings.append({"file": "", "line": 0, "col": 0, "message": m.group(1).strip()[:200], "type": "assertion_error"})
        # Pattern 4: Generic error lines with file hints
        if not findings:
            for m in re.finditer(r"(?:cannot find symbol|incompatible types|reached end of file|';' expected|')\)?\s*", error_raw):
                findings.append({"file": "", "line": 0, "col": 0, "message": m.group(0), "type": "parse_error"})
        return {"result": {"findings": findings[:10], "source": source, "suggestion": ("No fix attempt yet — read the file at the reported line, understand why the error occurs, then make ONE targeted fix." if findings else "No structured errors found. Read the raw error output carefully. Common causes: wrong file path, missing import, or logic bug missed by the regex parser.")}}

    elif fn == "root_cause_analyzer":
        error_summary = args.get("error_summary", "")
        code_snippet = args.get("code_snippet", "")
        expected = args.get("expected_behavior", "")
        # Build a structured causal analysis template the model must complete
        # Step 1: Classify the error type
        error_type = "unknown"
        if "cannot find symbol" in error_summary.lower() or "找不到符号" in error_summary:
            error_type = "missing_symbol"
        elif "incompatible types" in error_summary.lower() or "不兼容的类型" in error_summary or "cannot convert" in error_summary.lower() or "无法转换" in error_summary:
            error_type = "type_mismatch"
        elif "expected" in error_summary.lower() and "but was" in error_summary.lower():
            error_type = "assertion_failure"
        elif "null" in error_summary.lower() or "NullPointer" in error_summary:
            error_type = "null_reference"
        elif "cannot find" in error_summary.lower():
            error_type = "missing_definition"
        elif "reached end of file" in error_summary.lower():
            error_type = "parse_error"
        # Step 2: Generate causal chain template
        cause_templates = {
            "type_mismatch": "A type mismatch means: (a) a method is declared to return type X but returns type Y, OR (b) a caller expects type X but receives type Y, OR (c) the return/call types are correct but a boolean operator is wrong (e.g. 'if (x)' instead of 'if (!x)'). Check ALL callers and the method signature.",
            "missing_symbol": "The compiler cannot find a symbol (method/class/variable). Check: (a) is the name misspelled, (b) was the method renamed elsewhere but not here, (c) does the class/import exist.",
            "assertion_failure": "A test assertion failed. The test expected one value but got another. Trace the logic from the test input through the method to understand WHY the output differs from expectations. The bug is often in a condition (if/while/return) rather than the types.",
            "null_reference": "A null value was dereferenced. Track where the null originated — is a method returning null unexpectedly, or is a variable not initialized.",
            "parse_error": "Syntax error — missing semicolon, bracket, or parenthesis. Check the exact line and the lines immediately before it.",
            "missing_definition": "A class or method definition is missing. Was a file deleted or not created? Is the classpath correct?",
            "unknown": "Cannot classify error type. Read the error message character by character and compare with the code at the reported location.",
        }
        cause_analysis = cause_templates.get(error_type, cause_templates["unknown"])
        # Step 3: Generate risk assessment based on error type and code scope
        risk_templates = {
            "type_mismatch": "HIGH — changing return type may break all callers. Check every call site with search_code before changing the signature.",
            "missing_symbol": "MEDIUM — renaming or adding imports is low-risk, but renaming a public method breaks external callers.",
            "assertion_failure": "MEDIUM — changing logic in one condition may affect other test cases. Re-run ALL tests after fix.",
            "null_reference": "HIGH — adding null checks may hide upstream bugs. Verify the null source, don't just guard.",
            "parse_error": "LOW — syntax fixes are local, but check adjacent lines for cascaded errors.",
            "missing_definition": "HIGH — creating a new file/class may introduce new compilation issues. Compile immediately after creation.",
            "unknown": "MEDIUM — without clear classification, verify the fix doesn't introduce new issues by running compile+tests.",
        }
        risks = risk_templates.get(error_type, risk_templates["unknown"])
        # Step 4: Generate fix strategy based on error type
        strategy_templates = {
            "type_mismatch": "1. Use search_code to find ALL callers of the mismatched method/field. 2. Decide: change the declaration OR change the callers (not both). 3. If changing return type, check that all return statements match. 4. Make ONE consistent change across declaration + return statements.",
            "missing_symbol": "1. Check if the symbol exists but is misspelled. 2. Check if an import is missing. 3. If the symbol was recently renamed, update ALL references (use search_code).",
            "assertion_failure": "1. Trace the test input through the method step by step. 2. Identify which condition (if/return) produces the wrong value. 3. Fix the LOGIC, not the test expectation (unless the test is wrong).",
            "null_reference": "1. Find where the null value originates (search_code for the variable). 2. Decide: add null check at the source OR at the dereference site. 3. Prefer fixing the source over adding guards.",
            "parse_error": "1. Check the exact line + the 2 lines before it. 2. Common causes: missing semicolon, unmatched bracket, missing return statement. 3. Fix one syntax error at a time and recompile.",
            "missing_definition": "1. Verify the class/method name is correct. 2. If it's a new file, ensure it's in the right package/directory. 3. Create the file, then immediately compile to catch import issues.",
            "unknown": "1. Read the error line character by character. 2. Compare with the code at the reported location. 3. Understand WHAT the compiler is saying before touching any code.",
        }
        fix_strategy = strategy_templates.get(error_type, strategy_templates["unknown"])
        return {"result": {
            "pipeline": "Observe → Explain → Fix → Verify",
            "step": "EXPLAIN (do NOT write code yet)",
            "error_type": error_type,
            "cause_analysis": cause_analysis,
            "potential_risks": risks,
            "fix_strategy": fix_strategy,
            "code_snippet": code_snippet[:500],
            "expected_behavior": expected,
            "next_action": "1. Understand the cause_analysis. 2. Evaluate potential_risks — what could break? 3. Follow fix_strategy step by step. 4. Then call write_file with the corrected file. 5. Then compile + test."
        }}

    elif fn == "fix_single_error":
        file = args.get("file", "")
        line_num = args.get("line", 0)
        old_line = args.get("old_line", "")
        new_line = args.get("new_line", "")
        reason = args.get("reason", "")
        ok, real = check_path(file)
        if not ok: return {"error": real}
        # Step 1: Read current file
        try:
            with open(real) as f:
                lines = f.readlines()
        except FileNotFoundError:
            return {"error": f"not found: {file}"}
        # Step 2: Verify old_line matches at line_num
        if line_num < 1 or line_num > len(lines):
            return {"error": f"line {line_num} out of range (1-{len(lines)})"}
        current = lines[line_num - 1].rstrip("\n").rstrip("\r")
        if old_line.strip() != current.strip():
            return {"error": f"old_line mismatch. Expected: '{old_line.strip()[:80]}', Got: '{current.strip()[:80]}'"}
        # Step 3: Apply fix
        lines[line_num - 1] = new_line + "\n"
        new_content = "".join(lines)
        # Save backup
        backup = "".join(open(real).readlines()) if False else ""  # no-op, backup in memory
        # Step 4: Write and compile
        with open(real, "w") as f:
            f.write(new_content)
        r = subprocess.run(
            [MVN, "-q", "compile"],
            capture_output=True, text=True, timeout=120, cwd=ALLOWED_ROOT
        )
        if r.returncode == 0:
            return {"result": {
                "fixed": True,
                "compile": "PASS",
                "file": file,
                "line": line_num,
                "reason_applied": reason,
                "new_code_snippet": "".join(lines[max(0,line_num-4):min(len(lines),line_num+3)])
            }}
        else:
            # Restore original on failure
            with open(real, "w") as f:
                f.write("".join(lines[:line_num-1] + [old_line + "\n"] + lines[line_num:]))
            errors = [line.strip() for line in (r.stdout + "\n" + r.stderr).split("\n")
                      if "ERROR" in line or "error:" in line.lower()]
            return {"result": {
                "fixed": False,
                "compile": "FAIL",
                "errors": errors[:10],
                "note": "Original file restored. Analyze the compile errors and try again."
            }}

    elif fn == "verify_changes":
        # Step 1: Compile
        r1 = subprocess.run(
            [MVN, "-q", "compile"],
            capture_output=True, text=True, timeout=120, cwd=ALLOWED_ROOT
        )
        compile_ok = r1.returncode == 0
        compile_errors = [] if compile_ok else [
            line.strip() for line in (r1.stdout + "\n" + r1.stderr).split("\n")
            if "ERROR" in line or "error:" in line.lower()
        ][:10]
        # Step 2: Test (only if compile passes)
        test_result = None
        if compile_ok:
            r2 = subprocess.run(
                [MVN, "test"],
                capture_output=True, text=True, timeout=120, cwd=ALLOWED_ROOT
            )
            out = r2.stdout + r2.stderr
            m = re.search(r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)", out)
            if m:
                test_result = {"passed": int(m.group(1)) - int(m.group(2)) - int(m.group(3)),
                              "failed": int(m.group(2)) + int(m.group(3)),
                              "total": int(m.group(1))}
            else:
                test_result = {"passed": 0, "failed": 1 if "BUILD FAILURE" in out else 0, "total": 0}
        return {"result": {
            "compile": "PASS" if compile_ok else "FAIL",
            "compile_errors": compile_errors,
            "tests": test_result,
            "all_pass": compile_ok and (test_result and test_result.get("failed", 1) == 0) if test_result else compile_ok
        }}

    elif fn == "rollback":
        r = subprocess.run(["git", "reset", "--hard", "HEAD"],
                           capture_output=True, text=True, timeout=10, cwd=ALLOWED_ROOT)
        subprocess.run(["git", "clean", "-fd"],
                       capture_output=True, timeout=10, cwd=ALLOWED_ROOT)
        out = r.stdout.strip() or "workspace reset to HEAD"
        return {"result": out, "rollback": True}

    return {"error": f"unknown tool: {fn}"}


# ── Agent State ──────────────────────────────────────
# Workflow state machine:
#   idle ──→ observing ──→ fixing ──→ verifying ──→ done
#                │            ▲           │
#                └──→ explain ─┘           │
#                      (BLOCK write_file)  │
#                ←──── compile/test FAIL ──┘
WORKFLOW_PHASES = ["idle", "observing", "explain", "fixing", "verifying", "done"]

class AgentState:
    def __init__(self):
        self.files_read: set[str] = set()
        self.files_modified: set[str] = set()
        self.tests_run: bool = False
        self.tests_passed: int | None = None
        self.tests_failed: int | None = None
        self.compile_success: bool | None = None
        self.rollbacks: int = 0
        self.last_error: dict | None = None
        self.fix_attempts: int = 0
        self.error_signatures: list[str] = []
        self.current_phase: str = "idle"
        # Per-phase timing (ms)
        self.phase_times: dict[str, list[float]] = {"idle": [], "observing": [], "explain": [], "fixing": [], "verifying": [], "done": []}
        self._phase_enter_time: float = 0.0
        # 5 new benchmark metrics tracking
        self.gate_interventions: int = 0           # False Completion counter
        self.root_cause_hits: int = 0              # First fix after analyze_error succeeded
        self.root_cause_misses: int = 0            # First fix after analyze_error still failed
        self._pending_first_fix: bool = False      # analyze_error called, waiting for fix+compile
        self._fix_written: bool = False            # write_file executed in current analyze cycle
        self._last_write_index: int = -1           # Trace index of last write_file
        self.total_analyze_calls: int = 0          # Total analyze_error calls
        # Failure clustering
        self.failure_clusters: dict[str, list[dict]] = {}  # cluster_key → list of error dicts

    def transition(self, tool_name: str, result: dict | None = None, elapsed_ms: float = 0):
        """Drive the workflow state machine based on tool calls."""
        prev = self.current_phase
        r = result or {}

        if tool_name in ("read_file", "search_code", "list_files", "git_diff"):
            if self.current_phase == "idle":
                self.current_phase = "observing"

        elif tool_name == "analyze_error":
            self.current_phase = "explain"
            self.total_analyze_calls += 1
            self._pending_first_fix = True
            self._fix_written = False

        elif tool_name == "root_cause_analyzer":
            self.current_phase = "fixing"

        elif tool_name == "write_file":
            if self.current_phase in ("explain",):
                return False  # BLOCKED
            self.current_phase = "verifying"
            if self._pending_first_fix and not self._fix_written:
                self._fix_written = True

        elif tool_name == "run_compile":
            if isinstance(r, dict) and r.get("success", False):
                if self.current_phase != "verifying":
                    self.current_phase = "verifying"
                if self._pending_first_fix and self._fix_written:
                    self.root_cause_hits += 1
                    self._pending_first_fix = False
                    self._fix_written = False
            else:
                self.current_phase = "observing"
                if self._pending_first_fix and self._fix_written:
                    self.root_cause_misses += 1
                    self._pending_first_fix = False
                    self._fix_written = False

        elif tool_name == "run_tests":
            if isinstance(r, dict) and r.get("failed", 0) == 0 and r.get("total", 0) > 0:
                self.current_phase = "done"
            else:
                self.current_phase = "observing"

        elif tool_name == "fix_single_error":
            if self.current_phase in ("explain",):
                return False  # BLOCKED in EXPLAIN phase
            if isinstance(r, dict) and r.get("fixed", False):
                self.current_phase = "verifying"
            else:
                self.current_phase = "observing"
            if self._pending_first_fix and self._fix_written:
                if isinstance(r, dict) and r.get("fixed", False):
                    self.root_cause_hits += 1
                else:
                    self.root_cause_misses += 1
                self._pending_first_fix = False
                self._fix_written = False

        elif tool_name == "verify_changes":
            all_pass = isinstance(r, dict) and r.get("all_pass", False)
            compile_ok = isinstance(r, dict) and r.get("compile") == "PASS"
            if all_pass:
                self.current_phase = "done"
            elif compile_ok:
                self.current_phase = "verifying"
            else:
                self.current_phase = "observing"

        elif tool_name == "rollback":
            self.current_phase = "idle"

        # Track per-phase timing
        if prev != self.current_phase:
            if prev in self.phase_times:
                self.phase_times[prev].append(elapsed_ms)
            self._phase_enter_time = 0  # reset for next phase

        return True

    def record_error(self, source: str, detail: str, signature: str = ""):
        """Record the most recent error for status injection. Tracks repeating error patterns."""
        self.last_error = {"source": source, "detail": str(detail)[:300]}
        self.fix_attempts += 1
        if signature:
            self.error_signatures.append(signature)
            if len(self.error_signatures) > 10:
                self.error_signatures = self.error_signatures[-10:]
            # Cluster by extracting root key from signature
            cluster_key = self._extract_cluster_key(signature)
            if cluster_key:
                if cluster_key not in self.failure_clusters:
                    self.failure_clusters[cluster_key] = []
                self.failure_clusters[cluster_key].append({
                    "source": source, "signature": signature, "detail": str(detail)[:200]
                })

    def _extract_cluster_key(self, signature: str) -> str | None:
        """Extract root cause cluster key from error signature.
        Groups by: error source + file + error type for compile errors,
        or by assertion method for test failures."""
        if not signature:
            return None
        # compile:UserService.java:type_mismatch → cluster by file+type
        m = re.match(r"compile:.*?\[(?:ERROR|error)\]\s*(/\S+\.java)", signature)
        if m:
            return f"compile:{m.group(1)}"
        # test:first error content → cluster by test class
        m_test = re.match(r"test:(.+)", signature)
        if m_test:
            # Extract test class name if present
            cls = re.search(r"(\w+Test)", m_test.group(1))
            if cls:
                return f"test:{cls.group(1)}"
            return f"test:{m_test.group(1)[:60]}"
        # fix:... → cluster by file
        m_fix = re.match(r"fix:(.+)", signature)
        if m_fix:
            return f"fix:{m_fix.group(1)[:60]}"
        return signature[:80]

    def is_stuck_on_same_error(self) -> bool:
        """Detect if we keep hitting the same error (same signature appears 3+ times)."""
        if len(self.error_signatures) < 3:
            return False
        last3 = self.error_signatures[-3:]
        return len(set(last3)) == 1

    def clear_error(self):
        """Clear error state after a successful compile or test run."""
        self.last_error = None
        self.fix_attempts = 0
        self.error_signatures.clear()
        if self.current_phase in ("observing", "explain"):
            self.current_phase = "fixing"

    def reset(self):
        """Reset state after rollback — all file changes are discarded."""
        self.files_read.clear()
        self.files_modified.clear()
        self.tests_run = False
        self.tests_passed = None
        self.tests_failed = None
        self.compile_success = None
        self.last_error = None
        self.fix_attempts = 0
        self.error_signatures.clear()
        self.current_phase = "idle"
        self.phase_times = {p: [] for p in WORKFLOW_PHASES}
        self._phase_enter_time = 0.0
        self._pending_first_fix = False
        self._fix_written = False
        self.failure_clusters.clear()
        self.rollbacks += 1

    def error_repetition_rate(self) -> float:
        """Rate of repeated identical errors. 1.0 = every error was the same, 0.0 = all unique."""
        if len(self.error_signatures) < 2:
            return 0.0
        repeats = sum(1 for i in range(1, len(self.error_signatures)) if self.error_signatures[i] == self.error_signatures[i-1])
        return repeats / (len(self.error_signatures) - 1)

    def repeated_error_rate(self) -> float:
        """Portion of error signatures that appear >= 3 times (true stuck rate)."""
        if not self.error_signatures:
            return 0.0
        from collections import Counter
        counts = Counter(self.error_signatures)
        stuck = sum(1 for c in counts.values() if c >= 3)
        return stuck / len(counts) if counts else 0.0

    def root_cause_success_rate(self) -> float | None:
        """First-fix success rate: hits / (hits + misses). None if no cycles."""
        total = self.root_cause_hits + self.root_cause_misses
        if total == 0:
            return None
        return self.root_cause_hits / total

    def recovery_efficiency(self) -> float:
        """Recovery efficiency: 1.0 / fix_attempts if recovered, 0.0 if failed."""
        if self.compile_success and self.tests_failed == 0 and self.tests_run:
            if self.fix_attempts == 0:
                return 1.0  # no errors encountered
            return 1.0 / self.fix_attempts
        return 0.0

    def phase_summary(self) -> dict:
        """Return per-phase timing summary for benchmark metrics."""
        summary = {}
        for phase, times in self.phase_times.items():
            if times:
                summary[phase] = {"count": len(times), "total_ms": sum(times), "avg_ms": sum(times) / len(times)}
        return summary

    def status_text(self) -> str:
        lines = ["## Current State"]
        # Show workflow phase prominently
        phase_desc = {
            "idle": "IDLE — no work done yet",
            "observing": "OBSERVE — read/search code to understand",
            "explain": "⚠ EXPLAIN — MUST call root_cause_analyzer BEFORE write_file (write_file BLOCKED)",
            "fixing": "FIX — write_file ALLOWED, make targeted fix now",
            "verifying": "VERIFY — compile + test required",
            "done": "DONE — all checks passed",
        }
        lines.append(f"### Phase: {phase_desc.get(self.current_phase, self.current_phase)}")
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
        if self.last_error:
            lines.append(f"### Last Error ({self.last_error['source']}, attempt {self.fix_attempts}/3):")
            lines.append(f"  {self.last_error['detail']}")
        if self.current_phase == "explain":
            lines.append("### ⛔ write_file is BLOCKED until you call root_cause_analyzer")
            lines.append("  You MUST explain: (1) Why the error happened (2) Which line is wrong (3) Why your fix will work")
            lines.append("  Call root_cause_analyzer(error_summary, code_snippet, expected_behavior) NOW.")
        if self.is_stuck_on_same_error():
            lines.append("### ⚠ STUCK: Same error repeated 3+ times. Do NOT try the same fix again.")
            lines.append("  Step back. The root cause is different from what you think.")
            lines.append("  If root_cause_analyzer does not reveal a new approach, call rollback.")
        # Failure clustering
        multi_error_clusters = {k: v for k, v in self.failure_clusters.items() if len(v) >= 2}
        if multi_error_clusters:
            lines.append("### 🔗 Shared Root Cause Clusters:")
            for ck, errors in sorted(multi_error_clusters.items()):
                lines.append(f"  Cluster [{ck}] → {len(errors)} failures share this root cause:")
                for e in errors[-3:]:  # Show last 3
                    lines.append(f"    - {e['source']}: {e['signature'][:100]}")
                if len(errors) > 3:
                    lines.append(f"    ... and {len(errors) - 3} more")
                lines.append(f"  → FIX TOGETHER: all {len(errors)} errors in this cluster share the same root cause.")
        if self.rollbacks > 0:
            lines.append(f"### Rollbacks: {self.rollbacks}")
        return "\n".join(lines)


def _check_verification(trace_steps: list | None) -> bool:
    """Check if compile+test ran after the last write_file in the trace."""
    if not trace_steps:
        return False
    last_write_idx = -1
    for i, s in enumerate(trace_steps):
        if s["tool"] == "write_file":
            last_write_idx = i
    if last_write_idx == -1:
        return True  # No files written — nothing to verify
    has_compile = any(s["tool"] == "run_compile" for s in trace_steps[last_write_idx:])
    has_test = any(s["tool"] == "run_tests" for s in trace_steps[last_write_idx:])
    return has_compile and has_test


# ── Context Compression ──────────────────────────────
def compress_context(messages: list[dict], state: "AgentState", focus: str = "") -> list[dict]:
    """Distill message history — keep essential, prune noise.

    Strategy:
    - ALWAYS keep: system prompt (idx 0), user task (idx 1)
    - Keep last 2 state injections
    - Condense read_file results into a summary
    - Remove old tool call/result pairs for files already verified
    - If focus is set, prioritize messages referencing that symbol
    """
    if len(messages) < 8:
        return messages  # Too small to compress

    # Identify verified files (compiled + tested successfully)
    verified_files = state.files_modified.copy() if state.compile_success and state.tests_passed and state.tests_failed == 0 else set()

    # Build compressed list
    compressed = [messages[0], messages[1]]  # system + user
    read_summaries: dict[str, str] = {}
    kept_state_injections = 0
    seen_files = set()

    for i, m in enumerate(messages[2:], start=2):
        role = m.get("role", "")
        content = m.get("content", "")

        if role == "system":
            if "## Current State" in content or "Phase:" in content:
                kept_state_injections += 1
                if kept_state_injections <= 2:
                    compressed.append(m)
                continue
            # Tool results: condense read_file, pass through others
            if "Tool result:" in content and "Written" not in content:
                for fname in state.files_read:
                    if fname in content:
                        read_summaries[fname] = f"✓ read ({len(content)} chars)"
                        seen_files.add(fname)
                        break
                else:
                    # Not a read result, keep it
                    if len(content) < 2000:
                        compressed.append(m)
                    else:
                        compressed.append({"role": "system", "content": content[:1000] + "\n... (compressed)"})
                continue
            compressed.append(m)

        elif role == "assistant":
            # Keep only tool_calls, drop long text content in middle steps
            tc = m.get("tool_calls", [])
            if tc:
                compressed.append(m)
            elif content:
                # Final message or important text — keep
                compressed.append(m)

        elif role == "user":
            compressed.append(m)

    # Add compressed read summary if we condensed any reads
    if read_summaries:
        summary_lines = ["## Context Distillation — files already read:"]
        for fname, status in sorted(read_summaries.items()):
            verified = "✓ VERIFIED" if fname in verified_files else ""
            summary_lines.append(f"  {status} {verified}")
        if focus:
            summary_lines.append(f"## Current Focus: {focus}")
        # Insert after system+user
        compressed.insert(2, {"role": "system", "content": "\n".join(summary_lines)})

    return compressed


# ── Agent Loop ───────────────────────────────────────
def agent(task: str, collect_trace: bool = False, task_profile: TaskProfile | None = None,
          failure_db: "FailurePatternDB | None" = None) -> dict:
    """Run agent on a task. Returns metrics dict. If collect_trace, includes 'trace' key.

    task_profile: optional TaskProfile to customize EXPLAIN/FIX/VERIFY rules.
                  Defaults to PROFILE_SINGLE_FILE.
    failure_db: optional FailurePatternDB for historical context injection.
    """
    if task_profile is None:
        task_profile = PROFILE_SINGLE_FILE

    client = Client()
    state = AgentState()
    # Build profile-specific system prompt
    profile_rules = ""
    if task_profile.strict_explain:
        profile_rules += (
            "\n⚠ ERROR RECOVERY MODE — strict EXPLAIN enforcement:\n"
            "  - write_file BLOCKED until root_cause_analyzer completes\n"
            "  - You MUST state: (1) Why error happened (2) Which line is wrong (3) Why fix works\n"
        )
    if task_profile.explain_requires_search:
        profile_rules += (
            "\n⚠ CROSS-FILE MODE — reference checking required:\n"
            "  - Before root_cause_analyzer, call search_code to find ALL callers/references\n"
            "  - Read every file that references the symbol being changed\n"
        )
    verify_req = task_profile.verify_hint()

    # Inject historical context from memory
    memory_context = ""
    if failure_db:
        memory_context = failure_db.context_injection(task)

    base_system = (
        "You are an expert Java coding agent. The system enforces a WORKFLOW STATE MACHINE:\n"
        "\n"
        "  IDLE → OBSERVE → EXPLAIN → FIX → VERIFY → DONE\n"
        "              ↑        │                 │\n"
        "              └────────┘←── FAIL ────────┘\n"
        "\n"
        "PHASE RULES (ENFORCED — violations will be BLOCKED):\n"
        "  OBSERVE:  search_code → read_file → understand the code\n"
        "  EXPLAIN:  [MANDATORY after compile/test FAIL]\n"
        "            analyze_error → root_cause_analyzer\n"
        "            ⛔ write_file is BLOCKED in this phase\n"
        "            You MUST state: (1) Why error happened (2) Which line is wrong (3) Why fix works\n"
        "  FIX:      write_file ALLOWED now. Make ONE targeted fix.\n"
        "  VERIFY:   " + verify_req + "\n"
        "  DONE:     all passed, task complete\n"
        f"\nTASK TYPE: {task_profile.type_name}\n"
        f"Max fix attempts before rollback: {task_profile.max_fix_attempts}\n"
        + profile_rules +
        "\nERROR RECOVERY PROTOCOL (compile or test FAIL):\n"
        "  1. Phase→OBSERVE: call analyze_error(source, error_raw)\n"
        "  2. Phase→EXPLAIN: call root_cause_analyzer(error_summary, code_snippet, expected_behavior)\n"
        "     ⛔ Do NOT call write_file — it will be REJECTED\n"
        "  3. Phase→FIX:     now call write_file with the corrected code\n"
        "  4. Phase→VERIFY:  call run_compile, then run_tests\n"
        "  5. If FAIL again → back to step 1. If PASS → DONE.\n"
        "\n"
        "COMPOUND TOOLS (reduce round-trips):\n"
        "  fix_single_error(file, line, old_line, new_line, reason) — read→fix→write→compile in ONE call.\n"
        "    Use after root_cause_analyzer when you know the exact fix.\n"
        "  verify_changes() — compile+test in ONE call.\n"
        "    Use instead of separate run_compile then run_tests.\n"
        "\n"
        "CONTEXT: The system auto-compresses message history every 5 steps.\n"
        "  Verified files are distilled into summaries. Focus on current task, not past successes.\n"
        "\n"
        "FAILURE CLUSTERS: When multiple errors share a root cause, they are grouped.\n"
        "  Fix the ROOT CAUSE once rather than each symptom individually.\n"
        "\n"
        "ROLLBACK only after 3 failed fix attempts on the SAME error signature.\n"
        "NEVER write a file without reading it first.\n"
        "NEVER skip compilation before running tests.\n"
        "Output your summary in Chinese."
    )

    # Append memory context if available
    if memory_context:
        base_system += memory_context

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
    trace_steps = [] if collect_trace else None
    t0 = time.time()

    messages = [
        {"role": "system", "content": base_system},
        {"role": "user", "content": task},
    ]

    for step in range(MAX_STEPS):
        if step > 0 and messages:
            state_msg = {"role": "system", "content": state.status_text()}
            messages.append(state_msg)

        # Context compression every 5 steps (after step 5)
        if step >= 5 and step % 5 == 0:
            before = len(messages)
            # Extract focus: last method/class mentioned in recent messages
            focus = ""
            for m in reversed(messages[-6:]):
                c = m.get("content", "")
                for pattern in [r"(?:format|isValid|validate|Phone|Email|String)\w+", r"public \w+ (\w+)\("]:
                    fm = re.findall(pattern, str(c))
                    if fm:
                        focus = fm[0]
                        break
                if focus:
                    break
            messages = compress_context(messages, state, focus)
            after = len(messages)
            if before != after:
                print(f"  [compress] {before} → {after} messages ({before - after} pruned, focus={focus})", flush=True)

        print(f"\n── Step {step + 1}/{MAX_STEPS} ──", flush=True)
        step_t0 = time.time()

        # Tool routing: filter tools based on profile + phase
        active_tools = ToolRouter.route(task_profile, state.current_phase, state)
        response = client.chat(
            model=MODEL,
            messages=messages,
            tools=active_tools,
            options={"num_ctx": 16384, "num_predict": MAX_TOKENS},
        )

        msg = response.message
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            content = (msg.content or "")[:500]
            # Verification enforcement: refuse to exit with failing state
            needs_fix = False
            if state.compile_success is False:
                needs_fix = True
            elif not state.tests_run and state.files_modified and state.compile_success is True:
                needs_fix = True  # Modified files, compile passed, but never ran tests
            elif state.tests_run and state.tests_failed is not None and state.tests_failed > 0:
                needs_fix = True
            elif state.compile_success is None and state.files_modified:
                needs_fix = True  # Modified files but never compiled
            if needs_fix and step < MAX_STEPS - 1:
                state.gate_interventions += 1  # Track false completion
                pp = state.tests_passed or 0
                pf = state.tests_failed or 0
                print(f"  [gate] Refusing exit (#{state.gate_interventions}) — profile={task_profile.type_name}, phase={state.current_phase}, compile={state.compile_success}, tests={pp}/{pp + pf}")
                injection = "## STATE MACHINE: VERIFY failed — cycle back to OBSERVE\n"
                injection += f"Task type: {task_profile.type_name}. Current phase: {state.current_phase}.\n"
                injection += f"Workflow: OBSERVE → EXPLAIN → FIX → VERIFY → DONE\n"
                if state.compile_success is False:
                    injection += "- Phase→OBSERVE: compile FAILED. Call analyze_error(source='compile', error_raw=<errors>).\n"
                elif state.compile_success is None and state.files_modified:
                    injection += "- Phase→OBSERVE: files modified but never compiled. Call run_compile.\n"
                elif not state.tests_run and state.files_modified and task_profile.verify_tests_required:
                    injection += "- Phase→OBSERVE: compile passed but tests not run. Call run_tests.\n"
                if state.tests_failed is not None and state.tests_failed > 0:
                    injection += f"- Phase→OBSERVE: {state.tests_failed} test(s) FAILED. Call analyze_error(source='tests', error_raw=<errors>).\n"
                if task_profile.explain_requires_search:
                    injection += "- CROSS-FILE: call search_code to find ALL references before root_cause_analyzer.\n"
                if state.fix_attempts >= 1:
                    injection += "- Phase→EXPLAIN: call root_cause_analyzer(error_summary, code_snippet, expected_behavior).\n"
                    injection += "  ⛔ write_file is BLOCKED until root_cause_analyzer completes.\n"
                # Profile-specific explain hint
                hint = task_profile.explain_hint()
                if hint:
                    injection += f"{hint}\n"
                if state.is_stuck_on_same_error():
                    injection += "- ⚠ STUCK: same error signature 3+ times. You are repeating the broken fix.\n"
                if state.fix_attempts >= task_profile.max_fix_attempts:
                    injection += f"- {task_profile.max_fix_attempts} fix attempts reached. Call rollback.\n"
                injection += "Workflow: analyze_error → root_cause_analyzer → write_file → run_compile → run_tests."
                messages.append({"role": "system", "content": injection})
                continue
            print(f"  [done] {content}")
            # Compute verification from trace: did compile+test run after last write?
            verified_completion = _check_verification(trace_steps)
            metrics["duration"] = int(time.time() - t0)
            metrics["steps"] = step + 1
            metrics["files_modified"] = len(state.files_modified)
            metrics["rollbacks"] = state.rollbacks
            metrics["fix_attempts"] = state.fix_attempts
            metrics["error_repetition_rate"] = state.error_repetition_rate()
            metrics["phase_summary"] = state.phase_summary()
            # 5 new metrics
            metrics["verified_completion"] = verified_completion
            metrics["gate_interventions"] = state.gate_interventions
            metrics["root_cause_hits"] = state.root_cause_hits
            metrics["root_cause_misses"] = state.root_cause_misses
            metrics["root_cause_success_rate"] = state.root_cause_success_rate()
            metrics["repeated_error_rate"] = state.repeated_error_rate()
            metrics["recovery_efficiency"] = state.recovery_efficiency()
            metrics["total_analyze_calls"] = state.total_analyze_calls
            if collect_trace:
                metrics["trace"] = trace_steps
                metrics["final_message"] = content
                metrics["exit_reason"] = "done" if not needs_fix else "forced_continue"
            return metrics

        for tc in tool_calls:
            fn = tc.function
            name = fn.name
            args = fn.arguments if isinstance(fn.arguments, dict) else {}
            tc_t0 = time.time()

            # ⛔ State machine enforcement: block write_file/fix_single_error in EXPLAIN phase
            # In strict_explain mode, also block in observing phase
            blocked_phases = ["explain"]
            if task_profile.strict_explain:
                blocked_phases.append("observing")
                blocked_phases.append("idle")
            if name in ("write_file", "fix_single_error") and state.current_phase in blocked_phases:
                block_reason = (
                    f"⛔ write_file BLOCKED — current phase is {state.current_phase.upper()}. "
                )
                if state.current_phase == "explain":
                    block_reason += (
                        "You MUST call root_cause_analyzer(error_summary, code_snippet, expected_behavior) first.\n"
                        "Explain: (1) Why the error happened (2) Which line is wrong (3) Why your fix will work."
                    )
                else:
                    block_reason += (
                        "Task type is error_recovery. You MUST first run_compile to check state, "
                        "then analyze_error → root_cause_analyzer before write_file is allowed."
                    )
                blocked_msg = {"error": block_reason}
                print(f"  [⛔ BLOCKED] write_file rejected — profile={task_profile.type_name}, phase={state.current_phase}", flush=True)
                messages.append({
                    "role": "assistant", "content": "",
                    "tool_calls": [tc.model_dump()],
                })
                messages.append({
                    "role": "system",
                    "content": f"Tool result: {json.dumps(blocked_msg)}",
                })
                if collect_trace:
                    trace_steps.append({
                        "step": step + 1,
                        "tool": name,
                        "args": args,
                        "result_preview": json.dumps(blocked_msg)[:200],
                        "result_success": False,
                        "elapsed_ms": 0,
                        "timestamp": time.time(),
                    })
                continue

            print(f"  [{name}] {json.dumps(args)[:150]}", flush=True)
            result = dispatch({"function": {"name": name, "arguments": args}})

            # Drive workflow state machine
            elapsed = int((time.time() - tc_t0) * 1000)
            r_for_phase = result.get("result", {})
            prev_phase = state.current_phase
            allowed = state.transition(name, r_for_phase, elapsed)
            if not allowed:
                print(f"    ⛔ Phase transition rejected: {prev_phase} → {state.current_phase}", flush=True)
            elif state.current_phase != prev_phase:
                print(f"    🔄 Phase: {prev_phase} → {state.current_phase}", flush=True)

            # Update state
            if name == "read_file":
                p = args.get("path", "")
                if p: state.files_read.add(p)
            elif name == "write_file":
                p = args.get("path", "")
                if p: state.files_modified.add(p)
            elif name == "run_tests":
                state.tests_run = True
                r = result.get("result", {})
                if isinstance(r, dict):
                    state.tests_passed = r.get("passed", 0)
                    state.tests_failed = r.get("failed", 0)
                    metrics["tests_passed"] = r.get("passed", 0)
                    metrics["tests_failed"] = r.get("failed", 0)
                    if r.get("failed", 0) > 0:
                        # Build error signature from first error line
                        err_lines = r.get("errors", [])
                        sig = ""
                        if err_lines and isinstance(err_lines, list) and len(err_lines) > 0:
                            # Extract key info for signature: first 80 chars of first error
                            sig = f"test:{str(err_lines[0])[:80]}"
                        state.record_error("tests", json.dumps(r.get("errors", r))[:300], sig)
                    else:
                        state.clear_error()
            elif name == "run_compile":
                r = result.get("result", {})
                if isinstance(r, dict):
                    state.compile_success = r.get("success", False)
                    metrics["compile_success"] = r.get("success", False)
                    if not r.get("success", False):
                        err_lines = r.get("errors", [])
                        sig = ""
                        if err_lines and isinstance(err_lines, list) and len(err_lines) > 0:
                            sig = f"compile:{str(err_lines[0])[:80]}"
                        state.record_error("compile", json.dumps(r.get("errors", r))[:300], sig)
                    else:
                        state.clear_error()
            elif name == "fix_single_error":
                p = args.get("file", "")
                if p: state.files_modified.add(p)
                r = result.get("result", {})
                if isinstance(r, dict):
                    if r.get("fixed", False):
                        state.compile_success = True
                        metrics["compile_success"] = True
                        state.clear_error()
                    else:
                        state.compile_success = False
                        metrics["compile_success"] = False
                        err_lines = r.get("errors", [])
                        sig = f"fix:{str(err_lines[0])[:80]}" if err_lines else "fix:unknown"
                        state.record_error("compile", json.dumps(r.get("errors", r))[:300], sig)
            elif name == "verify_changes":
                r = result.get("result", {})
                if isinstance(r, dict):
                    compile_ok = r.get("compile") == "PASS"
                    state.compile_success = compile_ok
                    metrics["compile_success"] = compile_ok
                    tests = r.get("tests", {})
                    if tests:
                        state.tests_run = True
                        state.tests_passed = tests.get("passed", 0)
                        state.tests_failed = tests.get("failed", 0)
                        metrics["tests_passed"] = tests.get("passed", 0)
                        metrics["tests_failed"] = tests.get("failed", 0)
                    if r.get("all_pass", False):
                        state.clear_error()
                    elif not compile_ok:
                        sig = f"verify_compile:{str(r.get('compile_errors', [''])[0])[:80]}"
                        state.record_error("compile", str(r.get("compile_errors", ""))[:300], sig)
            elif name == "rollback":
                state.reset()
                metrics["rollbacks"] = state.rollbacks
                metrics["compile_success"] = False
                metrics["tests_passed"] = 0
                metrics["tests_failed"] = 0

            metrics["tool_calls"] += 1

            # Collect trace
            if collect_trace:
                r = result.get("result", {})
                trace_steps.append({
                    "step": step + 1,
                    "tool": name,
                    "args": args,
                    "result_preview": json.dumps(result)[:200],
                    "result_success": "error" not in result,
                    "elapsed_ms": elapsed,
                    "timestamp": time.time(),
                })

            preview = json.dumps(result)[:250].replace("\n", "\\n")
            print(f"    → {preview}", flush=True)

            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [tc.model_dump()],
            })
            messages.append({
                "role": "system",
                "content": f"Tool result: {json.dumps(result)}",
            })

    # Max steps hit
    verified_completion = _check_verification(trace_steps)
    metrics["duration"] = int(time.time() - t0)
    metrics["steps"] = MAX_STEPS
    metrics["files_modified"] = len(state.files_modified)
    metrics["rollbacks"] = state.rollbacks
    metrics["fix_attempts"] = state.fix_attempts
    metrics["error_repetition_rate"] = state.error_repetition_rate()
    metrics["phase_summary"] = state.phase_summary()
    metrics["verified_completion"] = verified_completion
    metrics["gate_interventions"] = state.gate_interventions
    metrics["root_cause_hits"] = state.root_cause_hits
    metrics["root_cause_misses"] = state.root_cause_misses
    metrics["root_cause_success_rate"] = state.root_cause_success_rate()
    metrics["repeated_error_rate"] = state.repeated_error_rate()
    metrics["recovery_efficiency"] = state.recovery_efficiency()
    metrics["total_analyze_calls"] = state.total_analyze_calls
    if collect_trace:
        metrics["trace"] = trace_steps
        metrics["final_message"] = ""
        metrics["exit_reason"] = "max_steps"
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


def run_benchmark(task_key: str, failure_db: "FailurePatternDB | None" = None) -> dict:
    """Run a single benchmark task. Ensures clean git state before and after."""
    # Auto-select profile using memory
    task_text = BENCHMARK_TASKS[task_key]["task"]
    profile = get_profile_for_task_with_memory(task_key, task_text, failure_db)

    print(f"\n{'='*60}")
    print(f"Benchmark: {BENCHMARK_TASKS[task_key]['name']}")
    print(f"Task Key:  {task_key}  |  Profile: {profile.type_name}")
    print(f"{'='*60}")

    # Ensure clean starting state
    subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True, cwd=ALLOWED_ROOT)
    subprocess.run(["git", "clean", "-fd"], capture_output=True, cwd=ALLOWED_ROOT)

    result = agent(task_text, collect_trace=True, task_profile=profile, failure_db=failure_db)

    # Rollback: restore clean state for next benchmark
    subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True, cwd=ALLOWED_ROOT)
    subprocess.run(["git", "clean", "-fd"], capture_output=True, cwd=ALLOWED_ROOT)

    # Record pattern in memory
    if failure_db:
        failure_db.record(task_key, result, profile)
        print(f"  [memory] Pattern recorded for {task_key} (profile={profile.type_name})", flush=True)

    return result


def run_all_benchmarks(collect_trace: bool = True, failure_db: "FailurePatternDB | None" = None):
    """Run all benchmark tasks and print summary with category stats."""
    all_results = {}
    for key in BENCHMARK_TASKS:
        result = run_benchmark(key, failure_db=failure_db)
        all_results[key] = result

    # Categorize
    categories = {"s": ("单文件修改", []), "m": ("跨文件修改", []), "e": ("错误恢复", [])}

    print(f"\n{'='*100}")
    print("BENCHMARK REPORT — mini-codex-xue (Gemma4 8B · Mac M5)")
    print(f"{'='*100}")
    print(f"{'Task':43s} {'Compile':8s} {'Tests':9s} {'V':3s} {'FC':4s} {'RCA':5s} {'Rep%':5s} {'Recov':5s} {'Tools':5s} {'Time':5s}")
    print("-" * 100)

    for key, r in all_results.items():
        name = BENCHMARK_TASKS[key]["name"]
        c = "PASS" if r["compile_success"] else "FAIL"
        total_tests = r["tests_passed"] + r["tests_failed"]
        t = f"{r['tests_passed']}/{total_tests}" if total_tests > 0 else "-"
        v = "✓" if r.get("verified_completion", False) else "✗"
        fc = str(r.get("gate_interventions", 0))
        # RCA: hits/misses → rate
        hits = r.get("root_cause_hits", 0); misses = r.get("root_cause_misses", 0)
        rca_rate = r.get("root_cause_success_rate")
        rca_str = f"{hits}/{hits+misses}" if (hits+misses) > 0 else "-"
        rep = f"{r.get('repeated_error_rate', 0):.0%}" if r.get('repeated_error_rate', 0) else "-"
        rec = f"{r.get('recovery_efficiency', 0):.2f}" if r.get('fix_attempts', 0) > 0 else "-"
        print(f"  {name:43s} {c:8s} {t:9s} {v:3s} {fc:4s} {rca_str:5s} {rep:5s} {rec:5s} {r['tool_calls']:4d}  {r['duration']:3d}s")
        cat = key[0]
        if cat in categories:
            categories[cat][1].append(r)

    # Category summary with 5 new quality metrics
    print(f"\n{'='*100}")
    print("CATEGORY SUMMARY — Quality Metrics")
    print(f"{'='*100}")
    print(f"{'Category':20s} {'Pass':6s} {'Verif%':7s} {'FalseC':7s} {'RCA%':7s} {'RepErr%':8s} {'Recov%':7s} {'AvgTools':9s} {'AvgTime':8s}")
    print("-" * 100)
    total_all = 0; passed_all = 0
    for cat_key, (cat_name, results) in categories.items():
        if not results:
            continue
        n = len(results)
        compiled = sum(1 for r in results if r["compile_success"])
        tests_ok = sum(1 for r in results if r["tests_failed"] == 0 and r["tests_passed"] > 0)
        avg_tools = sum(r["tool_calls"] for r in results) / n
        avg_time = sum(r["duration"] for r in results) / n
        # 5 new metrics aggregation
        verified = sum(1 for r in results if r.get("verified_completion", False))
        verif_pct = 100 * verified / n
        total_false_c = sum(r.get("gate_interventions", 0) for r in results)
        # RCA: aggregate hits and misses
        total_hits = sum(r.get("root_cause_hits", 0) for r in results)
        total_misses = sum(r.get("root_cause_misses", 0) for r in results)
        rca_pct = 100 * total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
        avg_rep_err = sum(r.get("repeated_error_rate", 0) for r in results) / n
        avg_recov = sum(r.get("recovery_efficiency", 0) for r in results) / n
        print(f"  {cat_name:20s} {compiled}/{n:3d}  {verif_pct:4.0f}%   {total_false_c:5d}   {rca_pct:4.0f}%   {avg_rep_err:5.0%}    {avg_recov:5.2f}   {avg_tools:5.1f}     {avg_time:4.0f}s")
        # Phase timing line
        phase_agg: dict[str, list[float]] = {}
        for r in results:
            for phase, info in r.get("phase_summary", {}).items():
                if phase not in phase_agg:
                    phase_agg[phase] = []
                phase_agg[phase].append(info["total_ms"])
        phase_strs = []
        for phase in ["observing", "explain", "fixing", "verifying"]:
            times = phase_agg.get(phase, [])
            if times:
                phase_strs.append(f"{phase}={sum(times)/len(times)/1000:.1f}s")
        if phase_strs:
            print(f"    Phases: {' · '.join(phase_strs)}")
        total_all += n
        passed_all += compiled

    print(f"\n  TOTAL: {passed_all}/{total_all} compile pass ({100*passed_all/total_all:.0f}%)")
    return all_results


# ── HTML Report Generator ───────────────────────────
TOOL_COLORS = {
    "search_code":  "#4CAF50", "read_file":   "#2196F3",
    "write_file":   "#FF9800", "list_files":  "#607D8B",
    "run_compile":  "#9C27B0", "run_tests":   "#E91E63",
    "git_diff":     "#00BCD4", "rollback":    "#F44336",
}
TOOL_ICONS = {
    "search_code": "🔍", "read_file": "📖", "write_file": "✏️",
    "list_files": "📁", "run_compile": "⚙️", "run_tests": "🧪",
    "git_diff": "📋", "rollback": "↩️",
}

def generate_html_report(all_results: dict, output_path: str):
    """Generate a standalone HTML benchmark report with visualizations."""
    tasks_data = []
    for key, r in all_results.items():
        cat = "单文件修改" if key.startswith("s") else "跨文件重构" if key.startswith("m") else "错误恢复"
        tasks_data.append({
            "key": key, "name": BENCHMARK_TASKS[key]["name"], "cat": cat,
            "compile": r["compile_success"], "tools": r["tool_calls"],
            "tests_passed": r.get("tests_passed", 0),
            "tests_failed": r.get("tests_failed", 0),
            "duration": r["duration"], "rollbacks": r.get("rollbacks", 0),
            "trace": r.get("trace", []), "final": r.get("final_message", ""),
            "exit": r.get("exit_reason", "?"),
            # 5 new quality metrics
            "verified": r.get("verified_completion", False),
            "gate_interventions": r.get("gate_interventions", 0),
            "root_cause_hits": r.get("root_cause_hits", 0),
            "root_cause_misses": r.get("root_cause_misses", 0),
            "root_cause_success_rate": r.get("root_cause_success_rate"),
            "repeated_error_rate": r.get("repeated_error_rate", 0),
            "recovery_efficiency": r.get("recovery_efficiency", 0),
            "total_analyze_calls": r.get("total_analyze_calls", 0),
        })

    n_total = len(tasks_data)
    n_pass = sum(1 for t in tasks_data if t["compile"])
    total_tools = sum(t["tools"] for t in tasks_data)
    total_duration = sum(t["duration"] for t in tasks_data)
    total_rollbacks = sum(t["rollbacks"] for t in tasks_data)
    # New overview aggregates
    n_verified = sum(1 for t in tasks_data if t["verified"])
    total_false_c = sum(t["gate_interventions"] for t in tasks_data)
    total_rca_hits = sum(t["root_cause_hits"] for t in tasks_data)
    total_rca_misses = sum(t["root_cause_misses"] for t in tasks_data)
    total_rca = total_rca_hits + total_rca_misses
    avg_recovery = sum(t["recovery_efficiency"] for t in tasks_data) / n_total if n_total > 0 else 0

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    # ── Build HTML ──
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>mini-codex-xue Benchmark Report</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#0d1117; color:#c9d1d9; padding:24px; }}
h1 {{ font-size:24px; margin-bottom:4px; }}
h1 span {{ color:#58a6ff; }}
.subtitle {{ color:#8b949e; margin-bottom:24px; font-size:14px; }}
.cards {{ display:grid; grid-template-columns: repeat(5,1fr); gap:14px; margin-bottom:32px; }}
.card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px; text-align:center; }}
.card .val {{ font-size:32px; font-weight:700; color:#58a6ff; }}
.card .lbl {{ font-size:12px; color:#8b949e; margin-top:4px; text-transform:uppercase; letter-spacing:1px; }}
.card.pass .val {{ color:#3fb950; }}
.card.fail .val {{ color:#f85149; }}
h2 {{ font-size:18px; margin:32px 0 12px; padding-bottom:8px; border-bottom:1px solid #30363d; }}
table {{ width:100%; border-collapse:collapse; margin-bottom:24px; }}
th, td {{ padding:10px 14px; text-align:left; border-bottom:1px solid #21262d; font-size:13px; }}
th {{ color:#8b949e; font-weight:600; }}
.bar-bg {{ background:#21262d; border-radius:4px; height:8px; overflow:hidden; }}
.bar-fg {{ background:#3fb950; height:100%; border-radius:4px; }}
.task-card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; margin-bottom:12px; overflow:hidden; }}
.task-header {{ padding:14px 18px; cursor:pointer; display:flex; justify-content:space-between; align-items:center; user-select:none; }}
.task-header:hover {{ background:#1c2129; }}
.task-header .status {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; }}
.task-header .status.pass {{ background:#3fb950; }}
.task-header .status.fail {{ background:#f85149; }}
.task-body {{ display:none; padding:0 18px 18px; }}
.task-body.open {{ display:block; }}
.timeline {{ position:relative; margin:12px 0; }}
.timeline-bar {{ display:flex; height:32px; border-radius:4px; overflow:hidden; margin:4px 0; }}
.timeline-seg {{ display:flex; align-items:center; justify-content:center; font-size:10px; color:#fff; font-weight:600; cursor:pointer; position:relative; }}
.timeline-seg:hover {{ filter:brightness(1.3); }}
.tooltip {{ display:none; position:absolute; bottom:110%; left:50%; transform:translateX(-50%); background:#1c2129; border:1px solid #30363d; border-radius:6px; padding:8px 12px; font-size:11px; white-space:nowrap; z-index:10; color:#c9d1d9; }}
.timeline-seg:hover .tooltip {{ display:block; }}
.flow {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:12px 0; padding:12px; background:#0d1117; border-radius:8px; }}
.flow-step {{ background:#161b22; border:1px solid #30363d; border-radius:6px; padding:10px 14px; font-size:12px; text-align:center; min-width:80px; }}
.flow-step.err {{ border-color:#f85149; }}
.flow-step.ok {{ border-color:#3fb950; }}
.flow-arrow {{ color:#58a6ff; font-size:18px; }}
.dep-list {{ list-style:none; padding:0; }}
.dep-list li {{ padding:6px 0; font-size:13px; font-family:monospace; }}
.dep-list .read {{ color:#2196F3; }}
.dep-list .write {{ color:#FF9800; }}
.dep-list .compile {{ color:#9C27B0; }}
.dep-list .test {{ color:#E91E63; }}
</style>
</head>
<body>
<h1>mini-codex-xue <span>Benchmark Report</span></h1>
<p class="subtitle">Gemma4 8B · Mac M5 16GB · {n_total} tasks · Generated {time.strftime('%Y-%m-%d %H:%M')}</p>

<div class="cards">
<div class="card pass"><div class="val">{n_pass}/{n_total}</div><div class="lbl">Compile Pass Rate</div></div>
<div class="card"><div class="val">{n_verified}/{n_total}</div><div class="lbl">Verification Rate</div></div>
<div class="card"><div class="val">{total_false_c}</div><div class="lbl">False Completions</div></div>
<div class="card"><div class="val">{total_rca_hits}/{total_rca}</div><div class="lbl">Root Cause Hits</div></div>
<div class="card"><div class="val">{avg_recovery:.2f}</div><div class="lbl">Avg Recovery Eff.</div></div>
</div>

<h2>Category Breakdown</h2>
<table>
<tr><th>Category</th><th>Tasks</th><th>Compile Pass</th><th>Tests Pass</th><th>Avg Tools</th><th>Avg Time</th><th>Pass Rate</th></tr>
"""
    for cat_key, cat_name in [("s","单文件修改"),("m","跨文件重构"),("e","错误恢复")]:
        cats = [t for t in tasks_data if t["cat"] == cat_name]
        if not cats: continue
        n = len(cats); cp = sum(1 for t in cats if t["compile"])
        tp = sum(1 for t in cats if t["tests_failed"]==0 and t["tests_passed"]>0)
        at = sum(t["tools"] for t in cats)/n
        ad = sum(t["duration"] for t in cats)/n
        pct = 100*cp/n
        html += f"<tr><td>{cat_name}</td><td>{n}</td><td>{cp}/{n}</td><td>{tp}/{n}</td><td>{at:.1f}</td><td>{ad:.0f}s</td><td><div class='bar-bg'><div class='bar-fg' style='width:{pct}%'></div></div>{pct:.0f}%</td></tr>\n"

    html += "</table>\n"
    # Quality Metrics table
    html += """<h2>Quality Metrics (5 new indicators)</h2>
<table>
<tr><th>Task</th><th>Verify</th><th>False C.</th><th>RCA Hits</th><th>RCA Miss</th><th>RCA%</th><th>RepErr%</th><th>RecovEff</th></tr>
"""
    for t in tasks_data:
        v = "✓" if t["verified"] else "✗"
        fc = t["gate_interventions"]
        hits = t["root_cause_hits"]; misses = t["root_cause_misses"]
        rca = f"{hits}/{hits+misses}" if (hits+misses) > 0 else "-"
        rca_pct = f"{t['root_cause_success_rate']:.0%}" if t["root_cause_success_rate"] is not None else "-"
        rer = f"{t['repeated_error_rate']:.0%}" if t["repeated_error_rate"] else "-"
        re = f"{t['recovery_efficiency']:.2f}"
        html += f"<tr><td style='font-size:12px;'>{esc(t['name'][:45])}</td><td>{v}</td><td>{fc}</td><td>{hits}</td><td>{misses}</td><td>{rca_pct}</td><td>{rer}</td><td>{re}</td></tr>\n"
    html += "</table>\n<h2>Task Details</h2>\n"

    for i, t in enumerate(tasks_data):
        trace = t["trace"]
        status_cls = "pass" if t["compile"] else "fail"
        status_label = "PASS" if t["compile"] else "FAIL"
        html += f"""
<div class="task-card">
<div class="task-header" onclick="this.nextElementSibling.classList.toggle('open')">
<div><span class="status {status_cls}"></span><strong>{esc(t['name'])}</strong></div>
<div style="color:#8b949e;font-size:12px;">{t['tools']} tools · {t['duration']}s · {status_label}</div>
</div>
<div class="task-body">
"""
        total_t = t['tests_passed'] + t['tests_failed']
        tests_display = f"{t['tests_passed']}/{total_t}" if total_t > 0 else "-"
        html += f"""<p style="font-size:13px;color:#8b949e;margin:8px 0;">Exit: {esc(t['exit'])} | Tests: {tests_display} | Rollbacks: {t['rollbacks']}</p>
"""
        # Timeline
        if trace:
            total_ms = sum(s["elapsed_ms"] for s in trace) or 1
            html += '<div class="timeline"><div class="timeline-bar">'
            for s in trace:
                pct = max(s["elapsed_ms"]/total_ms*100, 2)
                color = TOOL_COLORS.get(s["tool"], "#666")
                icon = TOOL_ICONS.get(s["tool"], "")
                args_preview = esc(json.dumps(s.get("args",{}))[:80])
                ok = "✓" if s["result_success"] else "✗"
                html += f"""<div class="timeline-seg" style="width:{pct:.1f}%;background:{color}" title="{esc(s['tool'])}: {args_preview}">
<span class="tooltip">{icon} {esc(s['tool'])} ({s['elapsed_ms']}ms) {ok}<br>{args_preview}</span>
</div>"""
            html += '</div></div>\n'

        # File dependency chain
        file_ops = [s for s in trace if s["tool"] in ("read_file","write_file","run_compile","run_tests")]
        if file_ops:
            html += '<div style="font-size:12px;color:#8b949e;margin:8px 0;">Dependency Chain:</div><ol class="dep-list">'
            for s in file_ops:
                cls = s["tool"]
                a = s.get("args", {})
                detail = a.get("path", a.get("keyword", ""))
                html += f'<li class="{cls}">[{s["step"]}] {esc(s["tool"])} → {esc(str(detail)[:100])}</li>'
            html += '</ol>'

        # Error recovery flow visualization for e1/e2
        if t["key"].startswith("e"):
            html += '<div style="font-size:12px;color:#8b949e;margin:8px 0;">Error Recovery Flow:</div>'
            html += '<div class="flow">'
            # Detect flow from trace: inject → compile/test → fail → fix → pass
            wrote_inject = False; saw_fail = False; fixed = False
            steps = []
            for s in trace:
                if s["tool"] == "write_file" and not wrote_inject:
                    steps.append(("Inject Bug", "write_file"))
                    wrote_inject = True
                elif s["tool"] in ("run_compile","run_tests") and not s["result_success"] and not saw_fail:
                    steps.append(("FAIL", "err"))
                    saw_fail = True
                elif s["tool"] == "write_file" and saw_fail and not fixed:
                    steps.append(("Fix Bug", "write_file"))
                    fixed = True
                elif s["tool"] in ("run_compile","run_tests") and s["result_success"] and fixed:
                    steps.append(("VERIFY PASS", "ok"))
            if not steps:
                for s in trace:
                    if s["tool"] == "write_file": steps.append(("Modify", "write_file"))
                    elif s["tool"] == "run_compile" and s["result_success"]: steps.append(("Compile OK", "ok"))
                    elif s["tool"] == "run_tests" and s["result_success"]: steps.append(("Tests OK", "ok"))
            for j, (label, cls) in enumerate(steps):
                if j > 0: html += '<span class="flow-arrow">→</span>'
                html += f'<div class="flow-step {cls}">{label}</div>'
            html += '</div>'

        html += '</div></div>\n'

    html += "</body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved: {output_path}")



# ── Self Benchmark ────────────────────────────────────
class SelfBenchmark:
    """Agent generates its own tasks, evaluates itself, and discovers weaknesses.

    Workflow:
      1. Scan workspace → identify classes, methods, dependencies
      2. Generate tasks of 3 types: single_file, cross_file, error_recovery
      3. Run each task with the appropriate profile
      4. Analyze results → identify weakest category, worst phase, top error patterns
      5. Output a weakness report
    """

    def __init__(self, workspace_root: str = ALLOWED_ROOT, failure_db: "FailurePatternDB | None" = None):
        self.root = workspace_root
        self.failure_db = failure_db
        self.workspace_info: dict = {}  # Populated by scan()

    def scan(self) -> dict:
        """Scan workspace to understand structure: classes, methods, dependencies."""
        info = {
            "java_files": [],
            "classes": {},
            "test_files": [],
            "dependencies": {},  # class → set of classes it imports/references
        }

        src_dir = os.path.join(self.root, "src/main/java")
        test_dir = os.path.join(self.root, "src/test/java")

        # Find all Java source files
        for d, label in [(src_dir, "main"), (test_dir, "test")]:
            if not os.path.isdir(d):
                continue
            for root, dirs, files in os.walk(d):
                for f in files:
                    if f.endswith(".java"):
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, self.root)
                        entry = {"path": rel, "full": full, "type": label}
                        info["java_files"].append(entry)
                        if label == "test":
                            info["test_files"].append(entry)

        # Parse each Java file to extract class name, methods, imports
        for entry in info["java_files"]:
            try:
                with open(entry["full"]) as f:
                    content = f.read()
            except Exception:
                continue

            # Extract class name
            cls_match = re.search(r"(?:public\s+)?class\s+(\w+)", content)
            if not cls_match:
                continue
            cls_name = cls_match.group(1)
            info["classes"][cls_name] = {
                "file": entry["path"],
                "type": entry["type"],
                "methods": [],
                "imports": [],
                "references": [],  # Other classes referenced
            }

            # Extract methods
            for m in re.finditer(r"(?:public|private|protected|static|\s)+(\w+)\s+(\w+)\s*\(([^)]*)\)", content):
                ret_type = m.group(1)
                method_name = m.group(2)
                params = m.group(3)
                if method_name not in (cls_name,):  # Skip constructor-like matches
                    info["classes"][cls_name]["methods"].append({
                        "name": method_name, "return": ret_type, "params": params
                    })

            # Extract imports and references
            for m in re.finditer(r"import\s+[\w.]+\.(\w+);", content):
                info["classes"][cls_name]["imports"].append(m.group(1))
            # Find references to other project classes
            for other_cls in info["classes"]:
                if other_cls != cls_name and other_cls in content:
                    info["classes"][cls_name]["references"].append(other_cls)

        # Build dependency graph
        for cls_name, data in info["classes"].items():
            info["dependencies"][cls_name] = list(set(
                data["imports"] + data["references"]
            ))

        self.workspace_info = info
        return info

    def generate_tasks(self) -> list[dict]:
        """Generate benchmark tasks from workspace analysis.

        Returns list of task dicts: {key, name, task_text, profile}
        """
        if not self.workspace_info:
            self.scan()

        info = self.workspace_info
        tasks = []

        # Find main classes (not test classes)
        main_classes = {k: v for k, v in info["classes"].items() if v["type"] == "main"}
        test_classes = {k: v for k, v in info["classes"].items() if v["type"] == "test"}

        if not main_classes:
            print("  [self-bench] No main classes found in workspace")
            return []

        # Pick the most important class (most methods or most references)
        primary_class = max(main_classes.items(),
                          key=lambda x: len(x[1]["methods"]) + len(x[1]["references"]))
        primary_name = primary_class[0]
        primary_data = primary_class[1]

        other_classes = [k for k in main_classes if k != primary_name]

        # ── Single-file tasks ──
        methods = primary_data["methods"]
        if methods:
            # Task 1: Add a new method
            existing_names = {m["name"] for m in methods}
            new_method_name = "getPhoneRegion"
            if new_method_name in existing_names:
                new_method_name = "extractPhonePrefix"

            tasks.append({
                "key": "self_s1",
                "name": f"[自生成·单文件] {primary_name} 添加 getPhoneRegion",
                "task": (
                    f"在 {primary_data['file']} 中添加一个新方法 getPhoneRegion(String phone)。\n"
                    f"如果 phone 以 '1' 开头返回 \"CN\"，以 '0' 开头返回 \"JP\"，否则返回 \"UNKNOWN\"。\n"
                    f"用 search_code 找到文件位置，read_file 读取后修改。编译通过后运行测试。"
                ),
                "profile": PROFILE_SINGLE_FILE,
            })

            # Task 2: Add Javadoc
            tasks.append({
                "key": "self_s2",
                "name": f"[自生成·单文件] {primary_name} 添加 Javadoc",
                "task": (
                    f"为 {primary_data['file']} 中所有 public 方法添加 Javadoc 注释。\n"
                    f"每个方法需要 @param 和 @return 标签。不修改方法逻辑。编译通过即可。"
                ),
                "profile": PROFILE_SINGLE_FILE,
            })

        # ── Cross-file tasks ──
        if len(other_classes) >= 1 and methods:
            # Task 3: Create new class that uses primary
            new_class = f"{primary_name}Helper"
            if new_class in main_classes:
                new_class = f"{primary_name}Util"

            # Find a method to extract
            target_method = methods[0]["name"] if methods else "isValidPhone"

            tasks.append({
                "key": "self_m1",
                "name": f"[自生成·跨文件] 提取 {new_class}",
                "task": (
                    f"1. 创建 src/main/java/com/example/{new_class}.java，包含从 {primary_name} 提取的工具方法。\n"
                    f"2. 修改 {primary_data['file']}，让原方法委托给 {new_class}。\n"
                    f"3. 搜索所有引用 {primary_name}.{target_method} 的文件并更新。\n"
                    f"4. 编译通过后运行测试。"
                ),
                "profile": PROFILE_CROSS_FILE,
            })

        # Task 4: Cross-file rename if we have methods
        if methods:
            # Pick a method and rename it
            rename_target = None
            for m in methods:
                if len(m["name"]) > 8 and m["name"].startswith("is"):
                    rename_target = m["name"]
                    break
            if not rename_target and methods:
                rename_target = methods[0]["name"]

            new_name = rename_target.replace("isValid", "validate") if rename_target.startswith("isValid") else rename_target + "V2"

            tasks.append({
                "key": "self_m2",
                "name": f"[自生成·跨文件] 重命名 {rename_target} → {new_name}",
                "task": (
                    f"将 {primary_data['file']} 中的 {rename_target} 方法重命名为 {new_name}。\n"
                    f"用 search_code 搜索所有引用，同步修改所有文件。不能遗漏任何引用。\n"
                    f"编译通过后运行测试。"
                ),
                "profile": PROFILE_CROSS_FILE,
            })

        # ── Error-recovery tasks ──
        if methods:
            error_method = methods[0]["name"]

            # Task 5: Break compile then fix
            tasks.append({
                "key": "self_e1",
                "name": f"[自生成·错误恢复] {primary_name} 编译失败 → 修复",
                "task": (
                    f"在 {primary_data['file']} 中把 {error_method} 返回类型改成 int（制造编译错误）。\n"
                    f"然后运行 run_compile，确认编译失败。\n"
                    f"接着根据错误信息修回正确的返回类型。\n"
                    f"如果修了 3 次还失败就用 rollback 回滚。最终必须编译通过。"
                ),
                "profile": PROFILE_ERROR_RECOVERY,
            })

            # Task 6: Break logic then fix
            tasks.append({
                "key": "self_e2",
                "name": f"[自生成·错误恢复] {primary_name} 逻辑错误 → 修复",
                "task": (
                    f"在 {primary_data['file']} 的 {error_method} 方法中，故意添加 'if (true) return false;' 作为第一行。\n"
                    f"然后运行 run_tests，确认测试失败。\n"
                    f"接着根据测试失败信息移除这行错误代码。\n"
                    f"如果修了 3 次还失败就用 rollback 回滚。最终必须所有测试通过。"
                ),
                "profile": PROFILE_ERROR_RECOVERY,
            })

        print(f"  [self-bench] Generated {len(tasks)} tasks from workspace analysis")
        for t in tasks:
            print(f"    {t['key']}: {t['name']} (profile={t['profile'].type_name})")
        return tasks

    def run(self, failure_db: "FailurePatternDB | None" = None) -> dict:
        """Run all self-generated tasks and collect results."""
        tasks = self.generate_tasks()
        if not tasks:
            return {"error": "No tasks generated", "results": {}, "weaknesses": {}}

        db = failure_db or self.failure_db
        results = {}
        for t in tasks:
            print(f"\n{'='*60}")
            print(f"Self-Bench: {t['name']}")
            print(f"{'='*60}")

            # Clean state
            subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True, cwd=self.root)
            subprocess.run(["git", "clean", "-fd"], capture_output=True, cwd=self.root)

            profile = t["profile"]
            if db:
                profile = db.suggest_profile(t["task"], profile)

            result = agent(t["task"], collect_trace=True, task_profile=profile, failure_db=db)
            results[t["key"]] = {
                "name": t["name"],
                "profile_used": profile.type_name,
                "result": result,
            }

            # Record in memory
            if db:
                db.record(t["key"], result, profile)

            # Clean state after each task
            subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True, cwd=self.root)
            subprocess.run(["git", "clean", "-fd"], capture_output=True, cwd=self.root)

        # Analyze weaknesses
        weaknesses = self.analyze_weaknesses(results)

        return {
            "tasks_generated": len(tasks),
            "results": results,
            "weaknesses": weaknesses,
        }

    def analyze_weaknesses(self, results: dict) -> dict:
        """Analyze self-benchmark results to find systematic weaknesses.

        Returns a weakness report identifying:
        - weakest category (by compile/tests pass rate)
        - worst phase (by timing)
        - most repeated error types
        - areas needing profile upgrade
        """
        if not results:
            return {"message": "No results to analyze"}

        by_category = {"single_file": [], "cross_file": [], "error_recovery": []}
        for key, r in results.items():
            res = r["result"]
            profile = r.get("profile_used", "")
            # Map profile to category
            if "single" in profile:
                cat = "single_file"
            elif "cross" in profile:
                cat = "cross_file"
            elif "error" in profile:
                cat = "error_recovery"
            else:
                cat = "single_file"
            by_category[cat].append(res)

        # Per-category stats
        cat_stats = {}
        for cat, res_list in by_category.items():
            if not res_list:
                continue
            n = len(res_list)
            compile_pass = sum(1 for r in res_list if r.get("compile_success", False))
            tests_pass = sum(1 for r in res_list
                           if r.get("tests_failed", 0) == 0 and r.get("tests_passed", 0) > 0)
            avg_tools = sum(r.get("tool_calls", 0) for r in res_list) / n
            avg_duration = sum(r.get("duration", 0) for r in res_list) / n
            total_fix = sum(r.get("fix_attempts", 0) for r in res_list)
            total_rca_hits = sum(r.get("root_cause_hits", 0) for r in res_list)
            total_rca_misses = sum(r.get("root_cause_misses", 0) for r in res_list)
            rca_rate = total_rca_hits / (total_rca_hits + total_rca_misses) if (total_rca_hits + total_rca_misses) > 0 else 0
            avg_rep_err = sum(r.get("repeated_error_rate", 0) for r in res_list) / n
            avg_gate = sum(r.get("gate_interventions", 0) for r in res_list) / n

            cat_stats[cat] = {
                "tasks": n,
                "compile_pass": f"{compile_pass}/{n}",
                "compile_pct": 100 * compile_pass / n,
                "tests_pass": f"{tests_pass}/{n}",
                "avg_tools": round(avg_tools, 1),
                "avg_duration_s": round(avg_duration, 0),
                "total_fix_attempts": total_fix,
                "rca_rate": round(rca_rate, 2),
                "avg_repeated_error_rate": round(avg_rep_err, 3),
                "avg_gate_interventions": round(avg_gate, 1),
            }

        # Find weakest category
        weakest = min(cat_stats.items(), key=lambda x: x[1]["compile_pct"]) if cat_stats else (None, None)
        # Find worst RCA
        worst_rca = min(cat_stats.items(), key=lambda x: x[1]["rca_rate"]) if cat_stats else (None, None)

        # Aggregate phase timing
        phase_totals: dict[str, list[float]] = {}
        for key, r in results.items():
            for phase, info in r["result"].get("phase_summary", {}).items():
                if phase not in phase_totals:
                    phase_totals[phase] = []
                phase_totals[phase].append(info["total_ms"] / 1000)
        worst_phase = max(phase_totals.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0) if phase_totals else (None, None)

        # Overall error type aggregation
        all_error_sigs: list[str] = []
        for key, r in results.items():
            trace = r["result"].get("trace", [])
            for s in trace:
                if s.get("result_preview"):
                    for et in ["type_mismatch", "missing_symbol", "assertion_failure",
                               "null_reference", "parse_error", "missing_definition"]:
                        if et in str(s.get("result_preview", "")):
                            all_error_sigs.append(et)
        from collections import Counter
        top_errors = Counter(all_error_sigs).most_common(3)

        # Recommendations
        recommendations = []
        if weakest and weakest[1]["compile_pct"] < 100:
            cat_name_cn = {"single_file": "单文件修改", "cross_file": "跨文件重构", "error_recovery": "错误恢复"}
            recommendations.append(
                f"⚠ 最弱类别: {cat_name_cn.get(weakest[0], weakest[0])} "
                f"(编译通过率 {weakest[1]['compile_pct']:.0f}%)。建议检查该类别对应 profile 的规则是否过严。"
            )
        if worst_rca and worst_rca[1]["rca_rate"] < 0.5:
            recommendations.append(
                f"⚠ RCA 命中率低: {worst_rca[0]} ({worst_rca[1]['rca_rate']:.0%})。"
                f"根因分析模板需要增强该类型的错误分类逻辑。"
            )
        if top_errors:
            top_err_str = ", ".join(f"{et}({c}次)" for et, c in top_errors)
            recommendations.append(f"🔁 高频错误类型: {top_err_str}。建议在 FailurePatternDB 中增加专项修复规则。")

        return {
            "weakest_category": weakest[0] if weakest else None,
            "weakest_stats": weakest[1] if weakest else None,
            "worst_rca_category": worst_rca[0] if worst_rca else None,
            "worst_rca_rate": worst_rca[1]["rca_rate"] if worst_rca else None,
            "worst_phase": worst_phase[0] if worst_phase else None,
            "worst_phase_avg_s": round(sum(worst_phase[1]) / len(worst_phase[1]), 1) if worst_phase and worst_phase[1] else 0,
            "top_error_types": [{"type": et, "count": c} for et, c in top_errors],
            "category_stats": cat_stats,
            "recommendations": recommendations,
            "overall_compile_rate": f"{sum(1 for r in results.values() if r['result'].get('compile_success', False))}/{len(results)}",
        }

    def print_weakness_report(self, results: dict):
        """Print a formatted weakness report to console."""
        w = results.get("weaknesses", {})
        cat_stats = w.get("category_stats", {})

        print(f"\n{'='*80}")
        print("SELF-BENCHMARK WEAKNESS REPORT")
        print(f"{'='*80}")

        print(f"\n📊 概览: 生成 {results.get('tasks_generated', 0)} 个自评测任务, "
              f"编译通过率 {w.get('overall_compile_rate', '?')}")

        # Category breakdown
        print(f"\n{'─'*60}")
        print(f"{'类别':16s} {'任务数':6s} {'编译':6s} {'测试':6s} {'RCA率':7s} {'平均工具':8s} {'平均耗时':8s}")
        print(f"{'─'*60}")
        cat_name_cn = {"single_file": "单文件修改", "cross_file": "跨文件重构", "error_recovery": "错误恢复"}
        for cat, stats in cat_stats.items():
            cn = cat_name_cn.get(cat, cat)
            print(f"  {cn:14s} {stats['tasks']:4d}  {stats['compile_pass']:6s} {stats['tests_pass']:6s} "
                  f"{stats['rca_rate']:5.0%}  {stats['avg_tools']:6.1f}  {stats['avg_duration_s']:6.0f}s")

        # Worst phase
        if w.get("worst_phase"):
            print(f"\n⏱ 最耗时阶段: {w['worst_phase']} (平均 {w.get('worst_phase_avg_s', 0):.1f}s)")

        # Top errors
        if w.get("top_error_types"):
            print(f"\n🔁 高频错误类型:")
            for e in w["top_error_types"]:
                print(f"    {e['type']}: {e['count']} 次")

        # Recommendations
        if w.get("recommendations"):
            print(f"\n💡 改进建议:")
            for i, rec in enumerate(w["recommendations"], 1):
                print(f"  {i}. {rec}")

        print(f"{'='*80}\n")


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
    ap.add_argument("--visualize", "-v", action="store_true", help="Generate HTML report after benchmark")
    ap.add_argument("--output", "-o", default="benchmark-report.html", help="HTML report output path")
    ap.add_argument("--self-benchmark", action="store_true",
                    help="Agent generates its own tasks, evaluates itself, discovers weaknesses")
    ap.add_argument("--memory-stats", action="store_true",
                    help="Show Agent Memory stats (failure pattern database)")
    ap.add_argument("--no-memory", action="store_true",
                    help="Disable Agent Memory for this run")
    ap.add_argument("--memory-reset", action="store_true",
                    help="Reset (clear) the failure pattern database")
    args = ap.parse_args()

    # ── Memory operations ──
    if args.memory_reset:
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
            print(f"Memory reset: {MEMORY_FILE} deleted.")
        else:
            print("Memory file does not exist.")
        sys.exit(0)

    if args.memory_stats:
        db = FailurePatternDB()
        stats = db.stats()
        print(f"\n{'='*50}")
        print("Agent Memory — Failure Pattern Database")
        print(f"{'='*50}")
        print(f"  File: {MEMORY_FILE}")
        print(f"  Total patterns: {stats['total_patterns']} ({stats['class_patterns']} class, {stats['error_patterns']} error)")
        print(f"  Strategy upgrades: {stats['strategy_upgrades']} (auto single_file → cross_file)")
        print(f"  Total failures: {stats['total_failures']}")
        print(f"  Total successes: {stats['total_successes']}")
        if stats['top_failure_classes']:
            print(f"  Top failure-prone classes:")
            for item in stats['top_failure_classes']:
                print(f"    - {item}")
        if stats['total_patterns'] == 0:
            print(f"\n  Memory is empty. Run benchmarks to populate:")
            print(f"    python3 mini-codex-xue.py --benchmark all")
        print(f"{'='*50}\n")
        sys.exit(0)

    # ── Self-Benchmark mode ──
    if args.self_benchmark:
        print(f"\n{'='*60}")
        print("SELF-BENCHMARK MODE")
        print(f"Model: {MODEL}  |  Root: {ALLOWED_ROOT}")
        print(f"{'='*60}")

        db = None if args.no_memory else FailurePatternDB()
        if db:
            stats = db.stats()
            if stats['total_patterns'] > 0:
                print(f"\n[memory] Loaded {stats['total_patterns']} patterns "
                      f"({stats['strategy_upgrades']} upgrades, "
                      f"{stats['total_failures']} failures)")
            else:
                print("\n[memory] Memory is empty — will build from scratch during this run")

        sb = SelfBenchmark(workspace_root=ALLOWED_ROOT, failure_db=db)
        results = sb.run(failure_db=db)
        sb.print_weakness_report(results)

        # Save results
        report_path = args.output.replace(".html", "-self.html")
        # Convert results to format compatible with generate_html_report
        report_data = {}
        for key, r in results.get("results", {}).items():
            report_data[key] = r["result"]

        if args.visualize and report_data:
            generate_html_report(report_data, report_path)

        print(f"\nSelf-benchmark complete. {results.get('tasks_generated', 0)} tasks run.")
        sys.exit(0)

    if args.list_tasks:
        for k, v in BENCHMARK_TASKS.items():
            print(f"  {k}: {v['name']}")
        sys.exit(0)

    if args.rollback:
        rollback()
        sys.exit(0)

    # ── Initialize memory for benchmark/normal runs ──
    db = None if args.no_memory else FailurePatternDB()
    if db and args.benchmark:
        stats = db.stats()
        if stats['total_patterns'] > 0:
            print(f"[memory] Loaded {stats['total_patterns']} patterns"
                  + (f" ({stats['strategy_upgrades']} upgrades)" if stats['strategy_upgrades'] > 0 else ""))

    if args.benchmark:
        if args.benchmark == "all":
            results = run_all_benchmarks(failure_db=db)
            if args.visualize:
                generate_html_report(results, args.output)
        else:
            r = run_benchmark(args.benchmark, failure_db=db)
            print(f"\nResult: {json.dumps(r, indent=2)}")
            if args.visualize:
                generate_html_report({args.benchmark: r}, args.output)
        sys.exit(0)

    if args.task:
        print(f"Model: {MODEL}  |  Root: {ALLOWED_ROOT}")
        print(f"Task:  {args.task}")

        # Auto-select profile with memory
        if db:
            task_key = "adhoc_" + re.sub(r'\W+', '_', args.task)[:30]
            default_profile = PROFILE_SINGLE_FILE
            profile = db.suggest_profile(args.task, default_profile)
        else:
            profile = PROFILE_SINGLE_FILE

        t0 = time.time()
        result = agent(args.task, task_profile=profile, failure_db=db)
        print(f"\nResult: {json.dumps(result, indent=2)}")
        print(f"Total: {time.time() - t0:.0f}s")
    else:
        ap.print_help()
