#!/usr/bin/env python3
"""Benchmark the running model through a real coding agent, and keep a scoreboard.

  hearthwork bench   [--agent claude|codex|opencode|aider|qwen] [--all] [--no-warmup] [--show]
  (or menu option "Benchmark the running model")

Eight prompts run as one agent conversation in a fresh folder (bench/runs/...): chat, list files, read a file,
write and run a script, edit it, fix three planted bugs, write unit tests, summarize. Each step is graded by
checking the files and running the code, not by trusting the agent's reply. Results go to bench/results.jsonl;
--show prints the scoreboard (and the latest --all summary) without running anything.

--all is the release check: every installed agent, one after another, after a warmup (a ~30K-token prompt on the
Anthropic and OpenAI endpoints, so the first agent does not pay for cold caches). Each agent runs with its own empty
config folder (CODEX_HOME / CLAUDE_CONFIG_DIR) so benchmark sessions never land in your real agent history. A run that
fails because of the agent (not a wrong answer) is retried once; both attempts go to results.jsonl. The summary table
(model, agent, score, time) is printed and saved as summary.md in the run's folder.
"""
import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from .harnesses import HARNESSES, installed, launch
from .onboard import CYAN, GREEN, RED, RESET, YELLOW, load_config
from .paths import BENCH
from .server import agent_context, served_model

RESULTS = BENCH / "results.jsonl"
BOLD, DIM = "\033[1m", "\033[2m"
BENCH_ALLOW = ["python *", "python3 *", "py *"]  # commands the other agents may run (they all need to run Python)
TIMEOUT = 20 * 60  # per prompt
WARMUP_TOKENS = 30000
CONFIG_ENV = {"claude": "CLAUDE_CONFIG_DIR", "codex": "CODEX_HOME"}  # where each agent keeps its history
REAL_DIRS = {"claude": ".claude", "codex": ".codex"}

DEMO = "The secret phrase is: copper owl builds a bridge.\n"
BUGGY = '''def average(numbers):
    total = 0
    for i in range(1, len(numbers)):
        total += numbers[i]
    return total / len(numbers)

def top_n(items, n):
    return sorted(items)[:n]

print(average([10, 20, 30]))   # should print 20.0
print(top_n([5, 1, 9, 3], 2))  # should print [9, 5]
print(average([]))             # should print 0, not crash
'''
FIZZBUZZ = ["FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i) for i in range(1, 31)]
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

PROMPTS = [
    ("chat", "hi"),
    ("list files", "List the files in this folder and tell me what each one is for."),
    ("read a file", "Read demo.txt and tell me the secret phrase in it."),
    ("write + run", "Create a file called fizzbuzz.py that prints FizzBuzz from 1 to 30, then run it with python and "
                    "show me the output."),
    ("edit + run", "Add a function is_prime(n) with a docstring to fizzbuzz.py, and at the end print all primes below "
                   "30. Run it again and show the output."),
    ("fix 3 bugs", "Run buggy.py. The comments say what each line should print. Find and fix all the bugs, then run it "
                   "again to show it is correct."),
    ("write tests", "Write unit tests for the functions in buggy.py in a file called test_buggy.py using unittest. Run "
                    "them and make sure they all pass."),
    ("summary", "Summarize which files you created or changed in this session and what each one does."),
]


def run_py(folder, *args):
    try:
        out = subprocess.run([sys.executable, *args], cwd=folder, capture_output=True, text=True, timeout=60)
        return out.returncode, out.stdout + out.stderr
    except subprocess.TimeoutExpired:
        return -1, "timeout"


def grade(step, folder, reply):
    """(points 0..1, note) for prompt `step` after the agent answered `reply`."""
    reply_l = reply.lower()
    if step == 0:
        return (1, "replied") if reply.strip() else (0, "no reply")
    if step == 1:
        ok = "demo.txt" in reply_l and "buggy.py" in reply_l
        return (1, "named both files") if ok else (0, "did not name demo.txt and buggy.py")
    if step == 2:
        return (1, "found the phrase") if "copper owl builds a bridge" in reply_l else (0, "wrong or no phrase")
    if step == 3:
        if not (folder / "fizzbuzz.py").exists():
            return 0, "fizzbuzz.py not created"
        code, out = run_py(folder, "fizzbuzz.py")
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        return (1, "output correct") if code == 0 and lines[:30] == FIZZBUZZ else (0, "wrong FizzBuzz output")
    if step == 4:
        path = folder / "fizzbuzz.py"
        if not path.exists():
            return 0, "fizzbuzz.py missing"
        try:
            funcs = {n.name: n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))) if isinstance(n, ast.FunctionDef)}
        except SyntaxError:
            return 0, "fizzbuzz.py does not parse"
        code, out = run_py(folder, "fizzbuzz.py")
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        tail = " ".join(lines[30:])
        numbers = {int(n) for n in re.findall(r"\b\d+\b", tail)}
        checks = ["is_prime" in funcs, bool(funcs.get("is_prime") and ast.get_docstring(funcs["is_prime"])),
                  code == 0 and lines[:30] == FIZZBUZZ, PRIMES <= numbers and not (numbers - PRIMES - {30})]
        notes = ["is_prime", "docstring", "FizzBuzz still right", "primes printed"]
        missing = [n for n, ok in zip(notes, checks) if not ok]
        return (1, "all correct") if not missing else (sum(checks) / 4, "missing: " + ", ".join(missing))
    if step == 5:
        code, out = run_py(folder, "buggy.py")
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        expected = ["20.0", "[9, 5]", "0"]
        right = sum(1 for got, want in zip(lines, expected) if got in (want, "0.0" if want == "0" else want))
        return (1, "all 3 bugs fixed") if code == 0 and right == 3 else (right / 3, f"{right}/3 lines right")
    if step == 6:
        path = folder / "test_buggy.py"
        if not path.exists():
            return 0, "test_buggy.py not created"
        code, out = run_py(folder, "-m", "unittest", "-v", "test_buggy")
        tests = len(re.findall(r" \.\.\. ok", out))
        if code != 0 or not tests:
            return 0, "tests fail or none ran"
        source = path.read_text(encoding="utf-8-sig")
        if not re.search(r"^\s*(from buggy import|import buggy)", source, re.M):
            return 0.5, f"{tests} tests pass, but they test a copy instead of importing buggy.py"
        return 1, f"{tests} tests pass, importing buggy.py"
    named = [f for f in ("fizzbuzz.py", "buggy.py", "test_buggy.py") if f in reply_l]
    return (1, "named all 3 files") if len(named) == 3 else (len(named) / 3, f"named {len(named)}/3 files")


class AgentError(Exception):
    """The agent itself failed (crashed, refused to start, or reached the wrong model): not a model score."""


def ask_agent(agent, config, model, folder, prompt, first, session, env=None):
    """One turn. Returns (reply, session id); raises AgentError when the agent itself fails."""
    remote = config.get("remote")
    port, context = ((None, remote.get("context", 32768)) if remote else  # context: set by main() from the host
                     (config["server"]["port"], agent_context(config)))
    if agent == "claude":
        # dontAsk: the allowed tools just run. Otherwise a user's default "auto" mode asks the (local, slow) model to
        # classify every command first; with GLM-4.7-Flash those checks timed out and blocked the commands.
        args = ["-p", prompt, "--output-format", "json", "--permission-mode", "dontAsk",
                "--allowedTools", "Read Write Edit Bash Glob Grep"]
        if not first:
            args.append("--continue")
        result = launch("claude", port, model, context, remote=remote, capture=True, cwd=folder, timeout=TIMEOUT, args=args, extra_env=env)
        try:
            data = json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise AgentError(f"Claude Code exited with {result.returncode}: {(result.stdout + result.stderr)[-400:]}")
        if data.get("is_error") and not data.get("result"):
            raise AgentError(f"Claude Code reported an error: {str(data)[:400]}")
        return data.get("result") or "", session
    if agent != "codex":
        return ask_other(agent, port, remote, context, model, folder, prompt, first, session, env)
    last = folder / ".codex-last-message.txt"
    last.unlink(missing_ok=True)  # never grade a turn on the previous turn's reply
    # `codex exec resume` has no -s flag; the sandbox is set as config, which both forms accept.
    common = ["--skip-git-repo-check", "-c", 'sandbox_mode="workspace-write"', "-o", str(last)]
    args = ["exec", *common, prompt] if first else ["exec", "resume", *common, session, prompt]
    result = launch("codex", port, model, context, remote=remote, capture=True, cwd=folder, timeout=TIMEOUT, args=args, extra_env=env)
    output = result.stdout + result.stderr
    provider = re.search(r"^provider:\s*(\S+)", output, re.M)
    if provider and provider.group(1) != "llamacpp":
        raise AgentError(f"Codex used provider '{provider.group(1)}' instead of the local model; stopped.")
    if first:
        found = re.search(r"session id:\s*([0-9a-f-]{36})", output)
        session = found.group(1) if found else None
        if not session:
            raise AgentError(f"Codex did not start a session (exit {result.returncode}): {output[-400:]}")
    if result.returncode != 0 and not last.exists():
        raise AgentError(f"Codex exited with {result.returncode}: {output[-400:]}")
    reply = last.read_text(encoding="utf-8", errors="replace") if last.exists() else ""
    return reply, session


def ask_other(agent, port, remote, context, model, folder, prompt, first, session, env):
    """One turn through an agent that is described by its registry entry (OpenCode, Aider, Qwen Code, ...)."""
    spec = HARNESSES[agent]
    handle, prompt_file = tempfile.mkstemp(prefix="hearthwork-bench-", suffix=".txt")  # outside the agent's folder
    os.close(handle)
    try:
        Path(prompt_file).write_text(prompt, encoding="utf-8")
        args, extra = spec["task"](BENCH_ALLOW, None, prompt_file, resume=session or not first, cwd=folder)
        result = launch(agent, port, model, context, remote=remote, capture=True, cwd=folder, timeout=TIMEOUT, args=args,
                        extra_env={**(env or {}), **extra}, input_text=prompt if spec["prompt"] == "stdin" else None)
    finally:
        os.unlink(prompt_file)
    reply, error = spec["result"](result.stdout, result.stderr, result.returncode, None)
    if error and not reply:
        raise AgentError(error)
    return reply, (spec["session"](result.stdout) if "session" in spec else None) or session


def isolated_env(agent, folder):
    """Env that points `agent`'s config/history folder at the empty folder `folder`. Exits if that folder could be
    the user's real one (~/.codex, ~/.claude, or wherever CODEX_HOME / CLAUDE_CONFIG_DIR already point)."""
    if agent not in CONFIG_ENV:  # the others keep their history per project folder, and every run has its own
        return {}
    var = CONFIG_ENV[agent]
    folder = Path(folder)
    real = {(Path.home() / REAL_DIRS[agent]).resolve()}
    if os.environ.get(var):
        real.add(Path(os.environ[var]).expanduser().resolve())
    mine = folder.resolve()
    if mine in real or any(r in mine.parents for r in real):
        sys.exit(f"Refusing to benchmark: the isolated {var} ({folder}) is inside your real {agent} folder.")
    folder.mkdir(parents=True, exist_ok=True)
    if agent == "codex":
        # An empty CODEX_HOME loses the user's Windows sandbox choice, and Codex then blocks every command.
        seed = windows_sandbox_config(Path(os.environ.get(var) or Path.home() / ".codex").expanduser() / "config.toml")
        if seed:
            (folder / "config.toml").write_text(seed, encoding="utf-8")
    return {var: str(folder)}


def windows_sandbox_config(path):
    """The `[windows]` table of a Codex config.toml (its sandbox setting), or ""."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    table, inside = [], False
    for line in lines:
        if line.strip().startswith("["):
            inside = line.strip() == "[windows]"
        if inside:
            table.append(line)
    return "\n".join(table).strip() + "\n" if table else ""


def warmup(port, model, remote=None):
    """Send a ~30K-token prompt through the Anthropic and OpenAI endpoints, so the first timed agent does not pay
    for cold caches and kernels. Failures only warn: the benchmark itself still tells."""
    text = "".join(f"Note {i}: the quick brown fox jumps over the lazy dog near the river bank. " for i in range(WARMUP_TOKENS // 17))
    calls = [("Anthropic /v1/messages", "/v1/messages", {"model": model, "max_tokens": 4, "messages": [{"role": "user", "content": text}]}),
             ("OpenAI /v1/responses", "/v1/responses", {"model": model, "max_output_tokens": 16, "input": text})]
    for name, path, body in calls:
        print(f"  warmup {name:<24}", end=" ", flush=True)
        t0 = time.time()
        base, key = (f"http://{remote['host']}:{remote['port']}", remote["key"]) if remote else (f"http://127.0.0.1:{port}", "local")
        request = urllib.request.Request(f"{base}{path}", json.dumps(body).encode(),
                                         {"content-type": "application/json", "x-api-key": key, "authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                response.read()
            print(f"{time.time() - t0:5.0f} s")
        except Exception as error:
            print(f"{YELLOW}skipped ({str(error)[:80]}){RESET}")


def run_once(agent, config, model, root, attempt):
    """One full 8-prompt run of `agent`. Returns the result record; raises AgentError if the agent itself fails."""
    title = HARNESSES[agent]["title"]
    name = agent if attempt == 1 else f"{agent}-retry"
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "demo.txt").write_text(DEMO, encoding="utf-8")
    (folder / "buggy.py").write_text(BUGGY, encoding="utf-8")
    env = isolated_env(agent, root / f"{name}-config")
    print(f"\n{BOLD}Benchmark{RESET}: {model} via {title}{'' if attempt == 1 else ' (retry)'}  "
          f"{DIM}(8 prompts, one conversation; folder {folder}){RESET}\n")
    steps, session, started = [], None, time.time()
    for i, (step, prompt) in enumerate(PROMPTS):
        print(f"  {i + 1}. {step:<12}", end=" ", flush=True)
        t0 = time.time()
        try:
            reply, session = ask_agent(agent, config, model, folder, prompt, i == 0, session, env)
        except subprocess.TimeoutExpired:
            reply = ""
        except AgentError:
            print()
            raise
        seconds = time.time() - t0
        points, note = grade(i, folder, reply)
        color = GREEN if points == 1 else YELLOW if points > 0 else RED
        print(f"{color}{points:>4.2g}{RESET}  {seconds:6.0f} s   {DIM}{note}{RESET}")
        steps.append({"step": step, "points": points, "seconds": round(seconds, 1), "note": note})
    return {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": model, "agent": title,
            "score": round(sum(s["points"] for s in steps), 2), "seconds": round(time.time() - started),
            "attempt": attempt, "steps": steps, "folder": str(folder)}


def save(record):
    BENCH.mkdir(exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_agent(agent, config, model, root):
    """Run `agent`, retrying once if the agent itself fails. Every attempt is recorded: a score for a finished run,
    an error entry (no score, so never on the scoreboard) for a failed one. Returns the record, or None."""
    title = HARNESSES[agent]["title"]
    for attempt in (1, 2):
        try:
            record = run_once(agent, config, model, root, attempt)
        except AgentError as error:
            print(f"\n  {RED}{error}{RESET}\n  Not scored: this is an agent problem, not a model score."
                  + ("  Retrying once." if attempt == 1 else ""))
            save({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": model, "agent": title,
                  "attempt": attempt, "error": str(error)[:400]})
            continue
        save(record)
        print(f"\n  {BOLD}Score {record['score']:.1f}/8{RESET} in {record['seconds'] / 60:.1f} min")
        return record
    return None


def summary_table(model, results):
    """Markdown for release notes. `results` is [(agent title, record or None)]."""
    rows = ["| Model | Agent | Score | Time |", "|---|---|---|---|"]
    for title, record in results:
        rows.append(f"| {model} | {title} | " + (f"{record['score']:.1f}/8 | {record['seconds'] / 60:.1f} min |" if record else "failed (agent error, retried) | - |"))
    return "\n".join(rows)


def scoreboard(summary=True):
    if not RESULTS.exists():
        print("No benchmark results yet.")
        return
    runs = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    runs = [r for r in runs if "score" in r]  # agent-error attempts have no score
    runs.sort(key=lambda r: (-r["score"], r["seconds"]))
    print(f"\n{BOLD}Scoreboard{RESET}  {DIM}({RESULTS}){RESET}")
    print(f"  {'model':<48} {'agent':<12} {'score':>7} {'time':>8}   date")
    for r in runs:
        color = GREEN if r["score"] >= 7 else YELLOW if r["score"] >= 5 else RED
        print(f"  {r['model'][:48]:<48} {r['agent']:<12} {color}{r['score']:>5.1f}/8{RESET} {r['seconds'] / 60:>6.1f} m   {r['date']}")
    summaries = sorted((BENCH / "runs").glob("*/summary.md"), key=lambda p: p.parent.name)
    if summary and summaries:
        print(f"\n{BOLD}Latest summary{RESET}  {DIM}({summaries[-1]}){RESET}\n{summaries[-1].read_text(encoding='utf-8')}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hearthwork bench", description="Benchmark the running model through a coding agent.")
    parser.add_argument("--agent", choices=sorted(HARNESSES), default="claude")
    parser.add_argument("--all", action="store_true", help="release check: every installed agent, one after another, after a warmup")
    parser.add_argument("--warmup", action="store_true", help="run the 30K-token warmup even for a single agent (--all always does)")
    parser.add_argument("--show", action="store_true", help="only print the scoreboard")
    args = parser.parse_args(argv)
    if args.show:
        return scoreboard()
    config = load_config()
    if config.get("remote"):  # the host's model; the warmup goes to it over the network too
        from .remote import require
        got = require(config)
        if not got:
            sys.exit(1)
        model, port = got[0], None
        config["remote"]["context"] = got[1]  # in memory only: ask_agent reads it
    else:
        port = config["server"]["port"]
        model = served_model(port)
    if not model:
        sys.exit("No model is running. Start one first (hearthwork menu: Start / switch model).")
    agents = [key for key in sorted(HARNESSES) if installed(key)] if args.all else [args.agent]
    if not agents:
        sys.exit("No coding agent is installed.")
    if not args.all and not installed(args.agent):
        sys.exit(f"{HARNESSES[args.agent]['title']} is not installed.")
    root = BENCH / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True)
    if args.all or args.warmup:
        print(f"\n{BOLD}Warmup{RESET}: {model}  {DIM}(~{WARMUP_TOKENS // 1000}K-token prompt on both APIs, not timed){RESET}")
        warmup(port, model, config.get("remote"))
    results = [(HARNESSES[agent]["title"], run_agent(agent, config, model, root)) for agent in agents]
    table = summary_table(model, results)
    (root / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(f"\n{BOLD}Summary{RESET}  {DIM}({root / 'summary.md'}){RESET}\n{table}")
    scoreboard(summary=False)
    if not all(record for _, record in results):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
