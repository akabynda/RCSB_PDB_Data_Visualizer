"""Tests for automatic discovery, download, and building of STRIDE."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import src.pdb_dataset_builder as builder


def _write_executable(path: Path) -> None:
    """Create a small executable fixture at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


def _tool_path(name: str) -> str | None:
    """Return deterministic tool paths while hiding system STRIDE."""
    return {
        "git": "/usr/bin/git",
        "make": "/usr/bin/make",
        "gcc": "/usr/bin/gcc",
    }.get(name)


def test_ensure_stride_prefers_explicit_path_and_path_lookup(tmp_path: Path) -> None:
    """Never bootstrap when the user or PATH already supplies STRIDE."""
    explicit = tmp_path / "explicit-stride"
    _write_executable(explicit)
    install_dir = tmp_path / "managed"

    with patch.object(builder, "download_and_build_stride") as bootstrap:
        assert builder.ensure_stride_executable(str(explicit), install_dir) == str(
            explicit
        )
        bootstrap.assert_not_called()

    with (
        patch.object(
            builder.shutil,
            "which",
            side_effect=lambda name: "/opt/bin/stride" if name == "stride" else None,
        ),
        patch.object(builder, "download_and_build_stride") as bootstrap,
    ):
        assert builder.ensure_stride_executable("", install_dir) == "/opt/bin/stride"
        bootstrap.assert_not_called()


def test_ensure_stride_does_not_replace_invalid_explicit_path(
    tmp_path: Path,
) -> None:
    """Treat a bad explicit path as an error instead of silently downloading."""
    with patch.object(builder, "download_and_build_stride") as bootstrap:
        assert (
            builder.ensure_stride_executable(
                str(tmp_path / "missing-stride"), tmp_path / "managed"
            )
            is None
        )
        bootstrap.assert_not_called()


def test_resolve_stride_rejects_non_executable_files_and_directories(
    tmp_path: Path,
) -> None:
    """Reject path-shaped values that cannot actually be executed."""
    non_executable = tmp_path / "stride-file"
    non_executable.touch()
    directory = tmp_path / "stride-directory"
    directory.mkdir()

    with (
        patch.object(builder.shutil, "which", return_value=None),
        patch.object(builder, "LOCAL_STRIDE_CANDIDATE", non_executable),
    ):
        assert builder.resolve_stride_executable(str(non_executable), tmp_path) is None
        assert builder.resolve_stride_executable(str(directory), tmp_path) is None
        assert builder.resolve_stride_executable("", tmp_path) is None


def test_ensure_stride_bootstraps_when_all_candidates_are_absent(
    tmp_path: Path,
) -> None:
    """Route the user-selected managed root into the automatic installer."""
    install_dir = tmp_path / "managed"
    expected = install_dir / "built-stride"
    with (
        patch.object(builder.shutil, "which", return_value=None),
        patch.object(builder, "LOCAL_STRIDE_CANDIDATE", tmp_path / "legacy"),
        patch.object(
            builder, "download_and_build_stride", return_value=expected
        ) as bootstrap,
    ):
        assert builder.ensure_stride_executable("", install_dir) == str(expected)

    bootstrap.assert_called_once_with(install_dir)


def test_ensure_stride_reuses_existing_managed_binary(tmp_path: Path) -> None:
    """Prefer the managed binary to a legacy build without reinstalling."""
    executable = builder._managed_stride_executable_path(tmp_path)
    _write_executable(executable)
    legacy = tmp_path / "legacy-stride"
    _write_executable(legacy)

    with (
        patch.object(builder.shutil, "which", return_value=None),
        patch.object(builder, "LOCAL_STRIDE_CANDIDATE", legacy),
        patch.object(builder, "download_and_build_stride") as bootstrap,
    ):
        assert builder.ensure_stride_executable("", tmp_path) == str(
            executable.resolve()
        )
        bootstrap.assert_not_called()


def test_download_and_build_stride_installs_pinned_revision_once(
    tmp_path: Path,
) -> None:
    """Clone, pin, build, publish, and then reuse one managed installation."""
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": builder.STRIDE_SETUP_TIMEOUT_SECONDS,
        }
        if command[1] == "clone":
            source_dir = Path(command[-1]) / "src"
            source_dir.mkdir(parents=True)
            (source_dir / "Makefile").write_text("stride:\n", encoding="utf-8")
        elif command[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{builder.STRIDE_SOURCE_REVISION}\n",
                stderr="",
            )
        elif Path(command[0]).name == "make":
            _write_executable(Path(command[2]) / "stride")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    install_dir = tmp_path / "stride"
    with (
        patch.object(builder.shutil, "which", side_effect=_tool_path),
        patch.object(builder.subprocess, "run", side_effect=run),
    ):
        first = builder.download_and_build_stride(install_dir)
        second = builder.download_and_build_stride(install_dir)

    expected = builder._managed_stride_executable_path(install_dir).resolve()
    assert first == expected
    assert second == expected
    assert [Path(command[0]).name for command in commands] == [
        "git",
        "git",
        "git",
        "make",
    ]
    assert commands[0][1:4] == [
        "clone",
        "--no-tags",
        builder.STRIDE_REPOSITORY_URL,
    ]
    assert commands[1][-3:] == [
        "checkout",
        "--detach",
        builder.STRIDE_SOURCE_REVISION,
    ]
    assert commands[3][1] == "-C"
    assert commands[3][-1] == "stride"
    assert {path.name for path in install_dir.iterdir()} == {
        ".install.lock",
        builder.STRIDE_SOURCE_REVISION,
    }


def test_download_and_build_stride_repairs_existing_checkout(tmp_path: Path) -> None:
    """Build a downloaded checkout whose executable is absent without cloning."""
    install_dir = tmp_path / "stride"
    checkout_dir = builder._managed_stride_checkout_dir(install_dir)
    source_dir = checkout_dir / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "Makefile").write_text("stride:\n", encoding="utf-8")

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[-2:] == ["rev-parse", "HEAD"]:
            stdout = builder.STRIDE_SOURCE_REVISION
        else:
            stdout = ""
        if Path(command[0]).name == "make":
            _write_executable(source_dir / "stride")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    with (
        patch.object(builder.shutil, "which", side_effect=_tool_path) as which,
        patch.object(builder.subprocess, "run", side_effect=run) as run_command,
    ):
        assert builder.download_and_build_stride(install_dir) == (
            source_dir / "stride"
        ).resolve()

    assert [call.args[0] for call in which.call_args_list] == [
        "git",
        "make",
        "gcc",
    ]
    assert run_command.call_count == 3


def test_download_and_build_stride_reports_build_failure(tmp_path: Path) -> None:
    """Surface compiler output and leave no published first-time installation."""

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[1] == "clone":
            source_dir = Path(command[-1]) / "src"
            source_dir.mkdir(parents=True)
            (source_dir / "Makefile").write_text("stride:\n", encoding="utf-8")
        if command[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=builder.STRIDE_SOURCE_REVISION,
                stderr="",
            )
        if Path(command[0]).name == "make":
            raise subprocess.CalledProcessError(
                2, command, stderr="fixture compiler failure"
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    install_dir = tmp_path / "stride"
    with (
        patch.object(builder.shutil, "which", side_effect=_tool_path),
        patch.object(builder.subprocess, "run", side_effect=run),
        pytest.raises(RuntimeError, match="fixture compiler failure"),
    ):
        builder.download_and_build_stride(install_dir)

    assert not builder._managed_stride_checkout_dir(install_dir).exists()


def test_download_and_build_stride_reports_missing_git(tmp_path: Path) -> None:
    """Explain the missing download prerequisite before starting a clone."""
    install_dir = tmp_path / "stride"
    with (
        patch.object(builder.shutil, "which", return_value=None),
        patch.object(builder.subprocess, "run") as run_command,
        pytest.raises(RuntimeError, match="Git is not available"),
    ):
        builder.download_and_build_stride(install_dir)

    run_command.assert_not_called()
    assert not builder._managed_stride_checkout_dir(install_dir).exists()


def test_download_and_build_stride_validates_tools_revision_and_output(
    tmp_path: Path,
) -> None:
    """Reject missing prerequisites, a wrong checkout, and a missing binary."""
    install_dir = tmp_path / "stride"
    checkout_dir = builder._managed_stride_checkout_dir(install_dir)
    source_dir = checkout_dir / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "Makefile").write_text("stride:\n", encoding="utf-8")

    with (
        patch.object(builder, "_verify_existing_stride_checkout"),
        patch.object(builder.shutil, "which", return_value=None),
        pytest.raises(RuntimeError, match="GNU Make and a C compiler"),
    ):
        builder.download_and_build_stride(install_dir)

    with (
        patch.object(builder, "_verify_existing_stride_checkout"),
        patch.object(builder.shutil, "which", side_effect=_tool_path),
        patch.object(
            builder.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ),
        pytest.raises(RuntimeError, match="without creating an executable"),
    ):
        builder.download_and_build_stride(install_dir)

    broken_install_dir = tmp_path / "wrong-revision"

    def wrong_revision(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[1] == "clone":
            source = Path(command[-1]) / "src"
            source.mkdir(parents=True)
            (source / "Makefile").write_text("stride:\n", encoding="utf-8")
        stdout = "different-revision" if command[-2:] == ["rev-parse", "HEAD"] else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    with (
        patch.object(builder.shutil, "which", side_effect=_tool_path),
        patch.object(builder.subprocess, "run", side_effect=wrong_revision),
        pytest.raises(RuntimeError, match="unexpected revision"),
    ):
        builder.download_and_build_stride(broken_install_dir)

    assert not builder._managed_stride_checkout_dir(broken_install_dir).exists()


def test_stride_install_cli_directory_is_configurable(tmp_path: Path) -> None:
    """Expose a stable default and an override for managed STRIDE sources."""
    with patch("sys.argv", ["pdb_dataset_builder.py"]):
        assert (
            builder.parse_args().stride_install_dir
            == builder.DEFAULT_STRIDE_INSTALL_DIR
        )
    with patch(
        "sys.argv",
        ["pdb_dataset_builder.py", "--stride-install-dir", str(tmp_path)],
    ):
        assert builder.parse_args().stride_install_dir == tmp_path
