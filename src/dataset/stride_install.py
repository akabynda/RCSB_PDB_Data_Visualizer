"""Discover, validate, and install the pinned STRIDE executable."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback for library users.
    fcntl = None  # type: ignore[assignment]
import os
import platform
import re
import shutil
import subprocess
import tempfile

from src.dataset.config import (
    DEFAULT_STRIDE_INSTALL_DIR,
    LOCAL_STRIDE_CANDIDATE,
    LOGGER,
    STRIDE_REPOSITORY_URL,
    STRIDE_SETUP_TIMEOUT_SECONDS,
    STRIDE_SOURCE_REVISION,
)

_STRIDE_INSTALL_THREAD_LOCK = Lock()


def _is_usable_stride_executable(path: Path) -> bool:
    """Return whether ``path`` is a regular executable file."""
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def _managed_stride_checkout_dir(install_dir: Path) -> Path:
    """Return the versioned checkout directory for the pinned STRIDE source."""
    system = platform.system().lower() or "unknown-os"
    architecture = platform.machine().lower() or "unknown-architecture"
    platform_tag = re.sub(r"[^a-z0-9_.-]+", "_", f"{system}-{architecture}")
    return Path(install_dir).expanduser() / STRIDE_SOURCE_REVISION / platform_tag


def _managed_stride_executable_path(install_dir: Path) -> Path:
    """Return the expected executable path inside the managed checkout."""
    return _managed_stride_checkout_dir(install_dir) / "src" / "stride"


@contextmanager
def _stride_install_lock(install_dir: Path) -> Iterator[None]:
    """Serialize a managed STRIDE installation across threads and processes."""
    install_dir.mkdir(parents=True, exist_ok=True)
    with _STRIDE_INSTALL_THREAD_LOCK:
        lock_handle = (install_dir / ".install.lock").open("a+b")
        try:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()


def _run_stride_setup_command(
    command: Sequence[str],
    *,
    action: str,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded STRIDE setup command and raise an actionable error."""
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=STRIDE_SETUP_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Cannot {action}: required command {command[0]!r} was not found. "
            "Automatic STRIDE setup requires Git, GNU Make, and a C compiler."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out while trying to {action} after "
            f"{STRIDE_SETUP_TIMEOUT_SECONDS:g} seconds."
        ) from exc
    except subprocess.CalledProcessError as exc:
        output = str(exc.stderr or exc.stdout or "").strip()
        detail = f" Tool output: {output[-2000:]}" if output else ""
        raise RuntimeError(
            f"Failed to {action} (exit status {exc.returncode}).{detail}"
        ) from exc


def _build_stride_checkout(checkout_dir: Path) -> Path:
    """Build and validate STRIDE in an existing pinned source checkout."""
    make_executable = shutil.which("make")
    compiler_executable = (
        shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    )
    missing_tools = []
    if make_executable is None:
        missing_tools.append("GNU Make")
    if compiler_executable is None:
        missing_tools.append("a C compiler (gcc, cc, or clang)")
    if missing_tools:
        missing = " and ".join(missing_tools)
        raise RuntimeError(
            f"Cannot build STRIDE because {missing} is not available in PATH."
        )

    source_dir = checkout_dir / "src"
    if not (source_dir / "Makefile").is_file():
        raise RuntimeError(
            f"Cannot build STRIDE: expected source Makefile is missing at "
            f"{source_dir / 'Makefile'}."
        )
    LOGGER.info("Building STRIDE in %s", source_dir)
    _run_stride_setup_command(
        [
            str(make_executable),
            "-C",
            str(source_dir),
            f"CC={compiler_executable} -O2",
            "stride",
        ],
        action="build STRIDE",
    )
    executable = source_dir / "stride"
    if not _is_usable_stride_executable(executable):
        raise RuntimeError(
            f"STRIDE build completed without creating an executable at {executable}."
        )
    return executable


def _verify_existing_stride_checkout(checkout_dir: Path) -> None:
    """Verify an existing managed checkout before executing its Makefile."""
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError(
            "Cannot verify the existing STRIDE source because Git is not "
            "available in PATH."
        )
    revision = _run_stride_setup_command(
        [str(git_executable), "-C", str(checkout_dir), "rev-parse", "HEAD"],
        action="verify the existing STRIDE source revision",
    ).stdout.strip()
    if revision != STRIDE_SOURCE_REVISION:
        raise RuntimeError(
            f"Existing STRIDE checkout at {checkout_dir} has revision "
            f"{revision or '<empty>'}; expected {STRIDE_SOURCE_REVISION}."
        )
    status = _run_stride_setup_command(
        [
            str(git_executable),
            "-C",
            str(checkout_dir),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        action="verify the existing STRIDE source files",
    ).stdout.strip()
    if status:
        raise RuntimeError(
            f"Existing STRIDE checkout at {checkout_dir} has modified tracked "
            "files and will not be built automatically."
        )


def download_and_build_stride(install_dir: Path) -> Path:
    """Download the pinned STRIDE source, build it, and return its executable."""
    install_root = Path(install_dir).expanduser().resolve()
    checkout_dir = _managed_stride_checkout_dir(install_root)
    executable = _managed_stride_executable_path(install_root)

    with _stride_install_lock(install_root):
        if _is_usable_stride_executable(executable):
            return executable.resolve()
        if os.name == "nt":
            raise RuntimeError(
                "Automatic STRIDE builds are supported on macOS and Linux. "
                "On native Windows, use WSL or pass a built executable with "
                "--solution-nmr-monomer-stride-executable."
            )

        if checkout_dir.exists() or checkout_dir.is_symlink():
            if not checkout_dir.is_dir():
                raise RuntimeError(
                    f"Cannot install STRIDE: {checkout_dir} is not a directory."
                )
            _verify_existing_stride_checkout(checkout_dir)
            return _build_stride_checkout(checkout_dir).resolve()

        git_executable = shutil.which("git")
        if git_executable is None:
            raise RuntimeError(
                "Cannot download STRIDE because Git is not available in PATH."
            )

        LOGGER.info(
            "Downloading STRIDE revision %s into %s",
            STRIDE_SOURCE_REVISION,
            checkout_dir,
        )
        with tempfile.TemporaryDirectory(
            prefix=f".{STRIDE_SOURCE_REVISION[:12]}.",
            dir=str(install_root),
        ) as temporary_dir:
            staged_checkout = Path(temporary_dir) / "checkout"
            _run_stride_setup_command(
                [
                    str(git_executable),
                    "clone",
                    "--no-tags",
                    STRIDE_REPOSITORY_URL,
                    str(staged_checkout),
                ],
                action="download STRIDE sources",
            )
            _run_stride_setup_command(
                [
                    str(git_executable),
                    "-C",
                    str(staged_checkout),
                    "checkout",
                    "--detach",
                    STRIDE_SOURCE_REVISION,
                ],
                action="check out the pinned STRIDE revision",
            )
            revision = _run_stride_setup_command(
                [
                    str(git_executable),
                    "-C",
                    str(staged_checkout),
                    "rev-parse",
                    "HEAD",
                ],
                action="verify the STRIDE source revision",
            ).stdout.strip()
            if revision != STRIDE_SOURCE_REVISION:
                raise RuntimeError(
                    "Downloaded STRIDE source has unexpected revision "
                    f"{revision or '<empty>'}; expected {STRIDE_SOURCE_REVISION}."
                )

            staged_executable = _build_stride_checkout(staged_checkout)
            if not _is_usable_stride_executable(staged_executable):
                raise RuntimeError(
                    f"STRIDE executable validation failed at {staged_executable}."
                )
            checkout_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                staged_checkout.replace(checkout_dir)
            except OSError as exc:
                if _is_usable_stride_executable(executable):
                    return executable.resolve()
                raise RuntimeError(
                    f"Could not publish the STRIDE installation to {checkout_dir}."
                ) from exc

        LOGGER.info("Installed STRIDE executable at %s", executable)
        return executable.resolve()


def resolve_stride_executable(
    explicit_value: str,
    install_dir: Path | None = None,
) -> str | None:
    """Resolve STRIDE from an argument, PATH, or an existing local build."""
    explicit_path = explicit_value.strip()
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if _is_usable_stride_executable(path):
            return str(path)
        return None

    resolved = shutil.which("stride")
    if resolved:
        return resolved

    managed_install_dir = install_dir or DEFAULT_STRIDE_INSTALL_DIR
    managed_candidate = _managed_stride_executable_path(managed_install_dir)
    if _is_usable_stride_executable(managed_candidate):
        return str(managed_candidate.resolve())

    if _is_usable_stride_executable(LOCAL_STRIDE_CANDIDATE):
        return str(LOCAL_STRIDE_CANDIDATE.resolve())

    return None


def ensure_stride_executable(
    explicit_value: str,
    install_dir: Path | None = None,
) -> str | None:
    """Resolve STRIDE, automatically installing it only when none was specified."""
    managed_install_dir = install_dir or DEFAULT_STRIDE_INSTALL_DIR
    resolved = resolve_stride_executable(explicit_value, managed_install_dir)
    if resolved is not None or explicit_value.strip():
        return resolved
    return str(download_and_build_stride(managed_install_dir))
