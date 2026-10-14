# SPDX-License-Identifier: Apache-2.0
"""The one archive extractor of the connector framework (zip-slip, links, bombs).

Archives reach a conversion from outside: result downloads of cloud solvers, bundles of
reference results, files a user received. :func:`safe_extract` is the only code that unpacks
one. It writes regular files below ``dest`` and nothing else, and it refuses the *whole*
archive (removing what it had written) as soon as one member

* has an absolute name, a Windows drive / UNC prefix, a ``..`` component or a NUL byte;
* is a symbolic link, a hard link, a device, a FIFO or any other non-regular entry (tar), or
  is flagged as a symbolic link (zip);
* would resolve outside ``dest`` (also through a symlink that already exists in ``dest``);

or when the archive as a whole exceeds a cap:

* ``max_members`` entries (:data:`MAX_MEMBERS`);
* ``max_total_bytes`` of extracted data (:data:`MAX_TOTAL_BYTES`), counted on the bytes
  actually written, not on the sizes the headers announce;
* ``max_ratio`` extracted / archive size (:data:`MAX_RATIO`), checked once more than
  :data:`RATIO_FLOOR_BYTES` have been written (tiny, highly compressible archives are fine).

Tar members additionally pass :func:`tarfile.data_filter` (the ``filter="data"`` policy of
the standard library); the explicit checks above do not depend on it. File modes, owners
and timestamps of the archive are not applied.

Formats: ``.zip``, ``.tar``, ``.tar.gz`` / ``.tgz``, ``.tar.bz2``, ``.tar.xz`` and
``.tar.zst`` / ``.tzst`` (through ``compression.zstd``, the ``zstandard`` package, the
``zstd`` program or bsdtar, whichever is present).
"""

from __future__ import annotations

import contextlib
import importlib
import shutil
import stat
import subprocess
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from frontier_connectors.errors import ConnectorError
from frontier_connectors.paths import ReferenceEscapeError, confine

__all__ = [
    "MAX_MEMBERS",
    "MAX_RATIO",
    "MAX_TOTAL_BYTES",
    "RATIO_FLOOR_BYTES",
    "ArchiveError",
    "safe_extract",
    "safe_extract_tar",
]

#: default cap on the extracted size of one archive (CFD volume results are large)
MAX_TOTAL_BYTES = 64 * 1024**3
#: default cap on extracted bytes per archive byte (float fields compress 2-20x, zeros ~1000x)
MAX_RATIO = 200.0
#: default cap on the number of entries of one archive
MAX_MEMBERS = 200_000
#: the ratio cap applies once this many bytes have been written
RATIO_FLOOR_BYTES = 8 * 1024**2
_CHUNK = 1 << 20
_ZSTD_SUFFIXES = (".tar.zst", ".tzst")


class ArchiveError(ConnectorError):
    """An archive was refused (unsafe member, cap exceeded) or could not be unpacked."""


class _Budget:
    """Running totals of one extraction, and the files to remove when it is refused."""

    def __init__(
        self,
        name: str,
        dest: Path,
        *,
        compressed_bytes: int | None,
        max_total_bytes: int,
        max_ratio: float,
        max_members: int,
    ) -> None:
        self.name = name
        self.dest = dest
        self.compressed = compressed_bytes
        self.max_total = max_total_bytes
        self.max_ratio = max_ratio
        self.max_members = max_members
        self.total = 0
        self.members = 0
        self.files = 0
        self.written: list[Path] = []

    def refuse(self, reason: str, hint: str | None = None) -> ArchiveError:
        return ArchiveError(
            f"{self.name} was not unpacked: {reason}",
            hint=hint
            or "the archive is unsafe or damaged; get it again from its source and do not "
            "unpack archives from sources you do not trust",
        )

    def member(self) -> None:
        self.members += 1
        if self.members > self.max_members:
            raise self.refuse(
                f"it holds more than {self.max_members} entries",
                "raise max_members if this is a legitimate archive",
            )

    def announce(self, size: int) -> None:
        """A size taken from a header: refuse early when it alone breaks the cap."""
        if size < 0 or self.total + size > self.max_total:
            raise self.refuse(
                f"its content is larger than the cap of {self.max_total} bytes",
                "raise max_total_bytes if this is a legitimate archive",
            )

    def add(self, n: int) -> None:
        self.total += n
        if self.total > self.max_total:
            raise self.refuse(
                f"its content is larger than the cap of {self.max_total} bytes",
                "raise max_total_bytes if this is a legitimate archive",
            )
        if (
            self.compressed
            and self.total > RATIO_FLOOR_BYTES
            and self.total > self.max_ratio * self.compressed
        ):
            raise self.refuse(
                f"it expands more than {self.max_ratio:g} times its size (decompression bomb)"
            )

    def target(self, member: str) -> Path:
        """Path below ``dest`` for a member name, or a refusal."""
        text = member.replace("\\", "/")
        if not text or "\x00" in text:
            raise self.refuse(f"member {member!r} has an empty or invalid name")
        if any(part == ".." for part in text.split("/")):
            raise self.refuse(f"member {member!r} has a '..' component")
        try:
            return confine(self.dest, text, what="archive member")
        except ReferenceEscapeError as e:
            raise self.refuse(f"member {member!r} {e.reason}") from e

    def write(self, src: IO[bytes], target: Path, limit: int | None = None) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise self.refuse(f"member {target.name!r} would replace a link or a directory")
        self.written.append(target)
        done = 0
        with open(target, "wb") as f:
            while chunk := src.read(_CHUNK):
                done += len(chunk)
                if limit is not None and done > limit:
                    raise self.refuse(f"member {target.name!r} is larger than its header says")
                self.add(len(chunk))
                f.write(chunk)
        self.files += 1

    def cleanup(self) -> None:
        for p in self.written:
            with contextlib.suppress(OSError):
                p.unlink()


def _check_tar_member(member: tarfile.TarInfo, budget: _Budget) -> Path | None:
    """Target of a regular member, ``None`` for a directory; refuses everything else."""
    name = member.name
    if member.issym() or member.islnk():
        raise budget.refuse(f"member {name!r} is a link (to {member.linkname!r})")
    if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
        raise budget.refuse(f"member {name!r} is a device or FIFO")
    if not (member.isfile() or member.isdir()):
        raise budget.refuse(f"member {name!r} is not a regular file or directory")
    target = budget.target(name)
    try:
        # in addition to the checks above: the standard library's own policy
        tarfile.data_filter(member, str(budget.dest))
    except tarfile.FilterError as e:
        raise budget.refuse(f"member {name!r} is refused by the tar data filter ({e})") from e
    return None if member.isdir() else target


def _extract_tar(tar: tarfile.TarFile, budget: _Budget) -> None:
    for member in tar:
        budget.member()
        target = _check_tar_member(member, budget)
        if target is None:
            continue
        budget.announce(member.size)
        src = tar.extractfile(member)
        if src is None:
            raise budget.refuse(f"member {member.name!r} has no data")
        with src:
            budget.write(src, target, limit=member.size)


def _extract_zip(z: zipfile.ZipFile, budget: _Budget) -> None:
    infos = z.infolist()
    for _ in infos:
        budget.member()
    announced = 0
    for info in infos:
        mode = info.external_attr >> 16
        if mode and stat.S_ISLNK(mode):
            raise budget.refuse(f"member {info.filename!r} is a symbolic link")
        if mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_IFMT(mode) == 0):
            raise budget.refuse(f"member {info.filename!r} is not a regular file or directory")
        budget.target(info.filename.rstrip("/") or info.filename)
        announced += info.file_size
    budget.announce(announced)
    if (
        budget.compressed
        and announced > RATIO_FLOOR_BYTES
        and announced > budget.max_ratio * budget.compressed
    ):
        raise budget.refuse(
            f"it expands more than {budget.max_ratio:g} times its size (decompression bomb)"
        )
    for info in infos:
        if info.is_dir():
            continue
        target = budget.target(info.filename)
        with z.open(info) as src:  # sizes in the directory are not trusted: bytes are counted
            budget.write(src, target, limit=info.file_size)


@contextlib.contextmanager
def _zstd_stream(path: Path) -> Iterator[IO[bytes]]:
    """Decompressed byte stream of a ``.zst`` file, with whatever this machine has."""
    for module in ("compression.zstd", "zstandard"):
        try:
            mod: Any = importlib.import_module(module)
        except ImportError:
            continue
        if module == "zstandard":
            with open(path, "rb") as raw, mod.ZstdDecompressor().stream_reader(raw) as stream:
                yield stream
        else:
            with mod.open(path, "rb") as stream:
                yield stream
        return
    zstd = shutil.which("zstd")
    tar = shutil.which("bsdtar") or shutil.which("tar")
    if zstd:
        cmd = [zstd, "-dc", "--", str(path)]
    elif (
        tar
        and "bsdtar"
        in subprocess.run(  # noqa: S603 - fixed program, no shell
            [tar, "--version"], capture_output=True, text=True, check=False
        ).stdout
    ):
        # bsdtar re-emits an archive as a plain tar stream; nothing is written to disk
        cmd = [tar, "-cf", "-", f"@{path}"]
    else:
        raise ArchiveError(
            f"{path} is zstd-compressed and no zstd decoder is available",
            hint="install the `zstd` program or `pip install zstandard`, or repack as .tar.gz",
        )
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)  # noqa: S603
    assert proc.stdout is not None
    try:
        yield proc.stdout
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def safe_extract_tar(
    tar: tarfile.TarFile,
    dest: Path,
    *,
    name: str = "archive",
    compressed_bytes: int | None = None,
    max_total_bytes: int = MAX_TOTAL_BYTES,
    max_ratio: float = MAX_RATIO,
    max_members: int = MAX_MEMBERS,
) -> int:
    """Extract an open tar (also a streaming one, ``mode="r|*"``) into ``dest`` under the
    rules of this module. Returns the number of files written. ``compressed_bytes`` enables
    the ratio cap for a stream whose size is known."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    budget = _Budget(
        name,
        dest.resolve(),
        compressed_bytes=compressed_bytes,
        max_total_bytes=max_total_bytes,
        max_ratio=max_ratio,
        max_members=max_members,
    )
    try:
        _extract_tar(tar, budget)
    except BaseException:
        budget.cleanup()
        raise
    return budget.files


def safe_extract(
    archive: Path | str,
    dest: Path | str,
    *,
    max_total_bytes: int = MAX_TOTAL_BYTES,
    max_ratio: float = MAX_RATIO,
    max_members: int = MAX_MEMBERS,
) -> Path:
    """Unpack ``archive`` into the directory ``dest`` (created) and return ``dest``.

    Raises :class:`ArchiveError` and leaves none of the archive's files behind when a member
    is unsafe, a cap is exceeded or the archive is damaged. See the module docstring."""
    archive, dest = Path(archive), Path(dest)
    lower = archive.name.lower()
    try:
        size = archive.stat().st_size
    except OSError as e:
        raise ArchiveError(f"{archive} could not be unpacked: {e}", hint="check the path") from e
    dest.mkdir(parents=True, exist_ok=True)
    budget = _Budget(
        str(archive),
        dest.resolve(),
        compressed_bytes=size,
        max_total_bytes=max_total_bytes,
        max_ratio=max_ratio,
        max_members=max_members,
    )
    try:
        if lower.endswith(".zip") or (
            not lower.endswith((".tar", ".tgz", ".gz", ".bz2", ".xz", *_ZSTD_SUFFIXES))
            and zipfile.is_zipfile(archive)
        ):
            with zipfile.ZipFile(archive) as z:
                _extract_zip(z, budget)
        elif lower.endswith(_ZSTD_SUFFIXES):
            with _zstd_stream(archive) as stream, tarfile.open(fileobj=stream, mode="r|") as tar:
                _extract_tar(tar, budget)
        else:
            with tarfile.open(archive) as tar:
                _extract_tar(tar, budget)
    except ArchiveError:
        budget.cleanup()
        raise
    except (tarfile.TarError, zipfile.BadZipFile, EOFError, OSError, ValueError) as e:
        budget.cleanup()
        raise ArchiveError(
            f"{archive} could not be unpacked: {e}",
            hint="the download is incomplete; delete it and download again",
        ) from e
    except BaseException:
        budget.cleanup()
        raise
    return dest
