"""Build a pinned local TmDet and its C++ dependencies without system installs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tarfile

import requests

from src.dataset.cache import _atomic_write_json, _pdb_cache_entry_lock, _sha256_file
from src.dataset.config import LOGGER

TMDET_REVISION = "311af38c09b1ec8636810b6f9d7ab532adaa846e"
TMDET_VERSION = "4.1.4"
TMDET_BUILD_PATCH = "virtual-destructors-and-pdb-export-v1"
# Archives and hashes pin both the algorithm and its build dependencies.
TMDET_ARCHIVES = (
    ("source", f"https://github.com/brgenzim/TmDet/archive/{TMDET_REVISION}.tar.gz",
     "1e670e0ca4452804a7e30444f6e02342521a30e77782097dc35997e48705b906"),
    ("json", "https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz",
     "0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406"),
    ("eigen", "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
     "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"),
    ("gemmi", "https://github.com/project-gemmi/gemmi/archive/refs/tags/v0.7.0.tar.gz",
     "7673b005846c661196dd49ac23bac3b1acc07b3afbdaff0e5d7f0f19b491e8c4"),
    ("contrib/pugixml", "https://github.com/zeux/pugixml/archive/refs/tags/v1.14.tar.gz",
     "610f98375424b5614754a6f34a491adbddaaec074e9044577d965160ec103d2e"),
)


def _unpack_archive(archive: Path, destination: Path) -> None:
    """Strip the archive root and reject links and paths escaping the target."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            relative = Path(*parts[1:])
            if relative.is_absolute() or ".." in parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive member: {member.name}")
            path = destination / relative
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target)


def managed_tmdet_directory(install_dir: Path) -> Path:
    return Path(install_dir).resolve() / TMDET_REVISION / f"{platform.system().lower()}-{platform.machine()}"


def _patch_source(source: Path) -> None:
    """Fix undefined object deletion and erase(end) during PDB-to-CIF export."""
    for name in ("SideDetector", "Optimizer"):
        path = source / "src/Engine" / f"{name}.hpp"
        original = path.read_text()
        target = f"~{name}()"
        if original.count(target) != 1:
            raise RuntimeError(f"Unexpected upstream destructor in {path}")
        path.write_text(original.replace(target, f"virtual {target}"))
    path = source / "src/Utils/CifUtil.cpp"
    original = path.read_text()
    target = "columns.erase(columns.end());"
    if original.count(target) != 1:
        raise RuntimeError("Unexpected upstream PDB export code")
    path.write_text(original.replace(target, "columns.pop_back();"))


def ensure_tmdet_executable(explicit: str, install_dir: Path) -> Path:
    """Use an explicit binary, otherwise reuse or build the pinned managed version."""
    if explicit:
        executable = Path(shutil.which(explicit) or explicit).expanduser().resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(f"TmDet executable is unavailable: {executable}")
        return executable
    root = managed_tmdet_directory(install_dir)
    executable = root / "source/build/src/tmdet"
    manifest_path = root / "installation.json"
    with _pdb_cache_entry_lock(Path(install_dir), "tmdet-install"):
        if executable.is_file() and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, ValueError):
                manifest = {}
            if (isinstance(manifest, dict) and manifest.get("build_patch") == TMDET_BUILD_PATCH
                    and manifest.get("executable_sha256") == _sha256_file(executable)):
                return executable
        cmake = shutil.which("cmake")
        if not cmake or not (shutil.which("c++") or shutil.which("g++")):
            raise RuntimeError("Automatic TmDet installation needs CMake and a C++20 compiler; or supply --tmdet-executable")
        root.mkdir(parents=True, exist_ok=True)
        for directory, url, checksum in TMDET_ARCHIVES:
            archive = root / "archives" / f"{checksum}.tar.gz"
            if not archive.is_file() or _sha256_file(archive) != checksum:
                LOGGER.info("Downloading TmDet dependency: %s", url)
                archive.parent.mkdir(parents=True, exist_ok=True)
                response = requests.get(url, timeout=180)
                response.raise_for_status()
                temporary = archive.with_suffix(".tmp")
                temporary.write_bytes(response.content)
                if _sha256_file(temporary) != checksum:
                    temporary.unlink()
                    raise RuntimeError(f"Archive checksum mismatch: {url}")
                temporary.replace(archive)
            if directory == "source" or not (root / directory).exists():
                _unpack_archive(archive, root / directory)
        _patch_source(root / "source")
        prefix = root / "install"
        commands = []
        with (root / "build.log").open("w") as log:
            def run(command):
                commands.append(command)
                log.write(shlex.join(command) + "\n")
                log.flush()
                try:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
                except (OSError, subprocess.SubprocessError) as exc:
                    raise RuntimeError(f"TmDet build failed; see {root / 'build.log'}") from exc

            for directory, options in (
                ("json", ["-DJSON_BuildTests=OFF"]),
                ("eigen", ["-DBUILD_TESTING=OFF", "-DEIGEN_BUILD_DOC=OFF"]),
                ("gemmi", ["-DBUILD_GEMMI_PROGRAM=OFF", "-DBUILD_SHARED_LIBS=OFF", "-DUSE_PYTHON=OFF"]),
            ):
                LOGGER.info("Building local TmDet dependency: %s", directory)
                build = root / directory / "build"
                run([cmake, "-S", str(root / directory), "-B", str(build),
                     f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DCMAKE_BUILD_TYPE=Release",
                     "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", *options])
                run([cmake, "--build", str(build), "--parallel", "4"])
                run([cmake, "--install", str(build)])
            build = root / "source/build"
            # Upstream uses std::span without including <span> on libc++.
            flags = "-include span " + " ".join("-I" + shlex.quote(str(prefix / path)) for path in ("include", "include/eigen3"))
            run([cmake, "-S", str(root / "source"), "-B", str(build),
                 f"-DCMAKE_PREFIX_PATH={prefix}", f"-DCMAKE_INSTALL_PREFIX={prefix}",
                 f"-DCMAKE_INSTALL_PREFIX_LIB={prefix / 'lib'}", f"-DCMAKE_INSTALL_PREFIX_BIN={prefix / 'bin'}",
                 f"-DCMAKE_CXX_FLAGS={flags}"])
            run([cmake, "--build", str(build), "--parallel", "4"])
        if not executable.is_file():
            raise RuntimeError("TmDet build did not produce its executable")
        _atomic_write_json(manifest_path, {
            "revision": TMDET_REVISION, "version": TMDET_VERSION,
            "build_patch": TMDET_BUILD_PATCH,
            "archives": TMDET_ARCHIVES, "commands": commands,
            "executable_sha256": _sha256_file(executable),
        })
        return executable
