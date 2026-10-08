"""Process helpers shared by `hearthwork task` and `hearthwork mcp`: parallel-session limit, a registry of running tasks,
killing process trees (and, on Windows, a Job Object so children die with the MCP server), UTF-8 console setup."""
import atexit
import os
import signal
import subprocess
import sys

from .paths import HOME

DEFAULT_SLOTS = 2
TASKS_DIR = HOME / "tasks"  # one empty file per running `hearthwork task`, named by PID


# ---------- how many tasks may run at once ----------

def parallel_slots(config):
    """Parallel sessions the model server offers. Remote mode: the host's count (the share info's "slots"), read
    defensively. THE integration point for the host's slot count: change only this function if its location differs."""
    remote = config.get("remote")
    if remote:
        info = remote.get("info") if isinstance(remote.get("info"), dict) else {}
        for source in (remote, info):
            value = source.get("slots")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        try:  # not asked yet: ask the host (session() stores its "slots" in config["remote"])
            from .remote import remote_slots, session
            session(config)
            return remote_slots(config, DEFAULT_SLOTS)
        except Exception:
            return DEFAULT_SLOTS
    value = (config.get("server") or {}).get("slots", DEFAULT_SLOTS)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else DEFAULT_SLOTS


# ---------- is a process alive ----------

def pid_alive(pid):
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


# ---------- registry of running tasks ----------

def register_task(directory=None, pid=None):
    """Record this process as a running task; the entry is removed again at exit."""
    directory = directory or TASKS_DIR
    path = directory / str(pid or os.getpid())
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        atexit.register(lambda: path.unlink(missing_ok=True))
    except OSError:
        pass
    return path


def running_tasks(directory=None, alive=pid_alive):
    """PIDs of Hearthwork tasks running now; entries of dead processes are removed."""
    directory = directory or TASKS_DIR
    found = []
    try:
        entries = list(directory.iterdir())
    except OSError:
        return found
    for entry in entries:
        if entry.name.isdigit() and alive(int(entry.name)):
            found.append(int(entry.name))
        else:
            try:
                entry.unlink()
            except OSError:
                pass
    return sorted(found)


def overload_warning(others, slots):
    """Text for stderr when starting another task would exceed the model's parallel sessions, else None."""
    if others >= slots:
        return (f"hearthwork task: {others} other Hearthwork task(s) are already running and the model has {slots} "
                f"parallel session(s); this one will wait for a free session and may be slow.")
    return None


# ---------- killing ----------

def kill_tree(process):
    """Kill a process started with new_group_flags() and everything it started."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        process.kill()
    except OSError:
        pass


def new_group_flags():
    """Popen keyword arguments that put the child in its own process group / session."""
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}


class KillOnClose:
    """Windows Job Object that kills every process in it when its last handle closes, i.e. when this process dies,
    even hard. Elsewhere (and when the job cannot be made) it does nothing; POSIX relies on explicit cleanup at exit."""

    def __init__(self):
        self.handle = None
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes

            class Basic(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class IoCounters(ctypes.Structure):
                _fields_ = [(n, ctypes.c_uint64) for n in ("a", "b", "c", "d", "e", "f")]

            class Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IoCounters),
                            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            kernel32 = ctypes.windll.kernel32
            kernel32.CreateJobObjectW.restype = ctypes.c_void_p
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return
            info = Extended()
            # KILL_ON_JOB_CLOSE | BREAKAWAY_OK (the model server a task starts must outlive it: it is started with
            # CREATE_BREAKAWAY_FROM_JOB, see server.start_background)
            info.BasicLimitInformation.LimitFlags = 0x2000 | 0x800
            if not kernel32.SetInformationJobObject(ctypes.c_void_p(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
                kernel32.CloseHandle(ctypes.c_void_p(job))
                return
            self.handle, self._ctypes, self._kernel32 = job, ctypes, kernel32
        except Exception:
            self.handle = None

    def add(self, process):
        """Put a started child in the job. True when it worked."""
        if not self.handle:
            return False
        try:
            return bool(self._kernel32.AssignProcessToJobObject(
                self._ctypes.c_void_p(self.handle), self._ctypes.c_void_p(int(process._handle))))  # Popen's handle
        except Exception:
            return False


# ---------- console encoding ----------

def console_utf8():
    try:
        import ctypes
        return ctypes.windll.kernel32.GetConsoleOutputCP() == 65001
    except Exception:
        return False


def setup_stdio():
    """Make stdin/stdout/stderr UTF-8 when they are pipes or files (a Windows pipe defaults to the ANSI code page,
    which turned "9x9" into "9?9"). A console keeps its code page unless that is UTF-8; unencodable text is replaced."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not reconfigure:
            continue
        try:
            if not stream.isatty() or (os.name == "nt" and console_utf8()):
                reconfigure(encoding="utf-8", errors="replace")
            else:
                reconfigure(errors="replace")
        except (OSError, ValueError):
            pass


def utf8_stream(stream):
    return (getattr(stream, "encoding", "") or "").lower().replace("-", "").replace("_", "") == "utf8"
