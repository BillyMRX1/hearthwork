#!/usr/bin/env python3
"""Benchmark the running model through a real coding agent, and keep a scoreboard.

  bench.bat / ./bench.sh / python bench.py   [--agent claude|codex] [--show]
  (or menu option "Benchmark the running model")

Eight prompts run as one agent conversation in a fresh folder (bench/runs/...): chat, list files, read a file,
write and run a script, edit it, fix three planted bugs, write unit tests, summarize. Each step is graded by
checking the files and running the code, not by trusting the agent's reply. Results go to bench/results.jsonl;
--show prints the scoreboard without running anything.
"""
import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from harnesses import HARNESSES, installed, launch
from onboard import CYAN, GREEN, HERE, RED, RESET, YELLOW, load_config
from server import served_model

BENCH = HERE / "bench"
RESULTS = BENCH / "results.jsonl"
BOLD, DIM = "\033[1m", "\033[2m"
TIMEOUT = 20 * 60  # per prompt

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


def ask_agent(agent, config, model, folder, prompt, first, session):
    """One turn. Returns (reply, session id); raises AgentError when the agent itself fails."""
    port, context = config["server"]["port"], config["server"]["context"]
    if agent == "claude":
        args = ["-p", prompt, "--output-format", "json", "--allowedTools", "Read Write Edit Bash Glob Grep"]
        if not first:
            args.append("--continue")
        result = launch("claude", port, model, context, capture=True, cwd=folder, timeout=TIMEOUT, args=args)
        try:
            data = json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise AgentError(f"Claude Code exited with {result.returncode}: {(result.stdout + result.stderr)[-400:]}")
        if data.get("is_error") and not data.get("result"):
            raise AgentError(f"Claude Code reported an error: {str(data)[:400]}")
        return data.get("result") or "", session
    last = folder / ".codex-last-message.txt"
    last.unlink(missing_ok=True)  # never grade a turn on the previous turn's reply
    # `codex exec resume` has no -s flag; the sandbox is set as config, which both forms accept.
    common = ["--skip-git-repo-check", "-c", 'sandbox_mode="workspace-write"', "-o", str(last)]
    args = ["exec", *common, prompt] if first else ["exec", "resume", *common, session, prompt]
    result = launch("codex", port, model, context, capture=True, cwd=folder, timeout=TIMEOUT, args=args)
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


def scoreboard():
    if not RESULTS.exists():
        print("No benchmark results yet.")
        return
    runs = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    runs.sort(key=lambda r: (-r["score"], r["seconds"]))
    print(f"\n{BOLD}Scoreboard{RESET}  {DIM}({RESULTS}){RESET}")
    print(f"  {'model':<48} {'agent':<12} {'score':>7} {'time':>8}   date")
    for r in runs:
        color = GREEN if r["score"] >= 7 else YELLOW if r["score"] >= 5 else RED
        print(f"  {r['model'][:48]:<48} {r['agent']:<12} {color}{r['score']:>5.1f}/8{RESET} {r['seconds'] / 60:>6.1f} m   {r['date']}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the running model through a coding agent.")
    parser.add_argument("--agent", choices=sorted(HARNESSES), default="claude")
    parser.add_argument("--show", action="store_true", help="only print the scoreboard")
    args = parser.parse_args()
    if args.show:
        return scoreboard()
    config = load_config()
    model = served_model(config["server"]["port"])
    if not model:
        sys.exit("No model is running. Start one first (hearthwork menu: Start / switch model).")
    if not installed(args.agent):
        sys.exit(f"{HARNESSES[args.agent]['title']} is not installed.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = BENCH / "runs" / f"{stamp}-{args.agent}-{model}"
    folder.mkdir(parents=True)
    (folder / "demo.txt").write_text(DEMO, encoding="utf-8")
    (folder / "buggy.py").write_text(BUGGY, encoding="utf-8")
    title = HARNESSES[args.agent]["title"]
    print(f"\n{BOLD}Benchmark{RESET}: {model} via {title}  {DIM}(8 prompts, one conversation; folder {folder}){RESET}\n")

    steps, session, started = [], None, time.time()
    for i, (name, prompt) in enumerate(PROMPTS):
        print(f"  {i + 1}. {name:<12}", end=" ", flush=True)
        t0 = time.time()
        try:
            reply, session = ask_agent(args.agent, config, model, folder, prompt, i == 0, session)
        except subprocess.TimeoutExpired:
            reply = ""
        except AgentError as error:
            print(f"\n\n  {RED}{error}{RESET}\n  Not recorded: this is an agent problem, not a model score.")
            sys.exit(1)
        seconds = time.time() - t0
        points, note = grade(i, folder, reply)
        color = GREEN if points == 1 else YELLOW if points > 0 else RED
        print(f"{color}{points:>4.2g}{RESET}  {seconds:6.0f} s   {DIM}{note}{RESET}")
        steps.append({"step": name, "points": points, "seconds": round(seconds, 1), "note": note})

    total = sum(s["points"] for s in steps)
    seconds = time.time() - started
    record = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": model, "agent": title,
              "score": round(total, 2), "seconds": round(seconds), "steps": steps, "folder": str(folder)}
    BENCH.mkdir(exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\n  {BOLD}Score {total:.1f}/8{RESET} in {seconds / 60:.1f} min")
    scoreboard()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
