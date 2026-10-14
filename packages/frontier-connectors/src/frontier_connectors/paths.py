# SPDX-License-Identifier: Apache-2.0
"""Confinement of file references that come from file content.

A simulation case is rarely one file: a collection names its pieces, a deck includes other
decks, a light XML file points at heavy binary data, a manifest lists arrays. Every such
reference is text an author of the case controls, so following it blindly lets a received
case read any file the converting user can read (``../../.ssh/id_ed25519``, ``/etc/passwd``,
a symlink into another project) and copy its bytes into a dataset. Connectors therefore
never build a path from file content themselves; they call :func:`resolve_in_case`.

Case root
---------
The **case root** is the one directory a conversion is allowed to read from:

* the directory itself, when the user pointed at a directory (an OpenFOAM case, a Fluent
  export folder, an Abaqus ``*_frontier`` export, a run directory);
* the directory holding the entry file, when the user pointed at a file (``run.xdmf``,
  ``case.pvd``, ``model.bdf``, ``flow.case``, ``case.foam``). A file the entry file includes
  keeps the *same* case root: an include in ``sub/part.k`` is still confined to the directory
  of the deck the user named, not to ``sub/``;
* for downloads, the download directory the caller chose (:func:`confine`).

:func:`case_root_of` computes it. The root is the path the user gave, made absolute and
resolved; a case reached through a symlinked directory is confined to the real directory.

What a reference may be
-----------------------
A reference is joined to the directory of the *referring* file (``referrer``; the case root
when omitted) and must then, after symlinks are resolved, lie inside the case root:

* absolute paths, Windows drive prefixes (``C:\\x``, ``C:x``) and UNC paths (``\\\\host\\share``,
  ``//host/share``) are refused outright, even when they happen to point into the case;
* ``..`` is allowed as long as the result stays inside (``system/../constant/x``);
* a symlink at any component whose target leaves the case root is refused, including one
  that leaves and comes back;
* an empty reference or one holding a NUL byte is refused.

The file does not have to exist: callers keep their own "missing file" messages, and write
paths (download targets) are checked before the file is created.

Not covered here: references that a third-party reader follows internally. Connectors that
hand a file to such a reader (VTK's parallel/multiblock XML readers, meshio's and VTK's XDMF
readers, HDF5 external links) scan the references first and pass each one through this module
(``frontier_connectors.paths.h5_confine_external_links`` for HDF5).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path, PureWindowsPath
from typing import Any

from frontier_connectors.errors import ConnectorError

__all__ = [
    "ReferenceEscapeError",
    "case_root_of",
    "confine",
    "first_existing_in_case",
    "h5_check_file",
    "h5_open",
    "h5_confine_external_links",
    "is_inside",
    "resolve_in_case",
]


class ReferenceEscapeError(ConnectorError):
    """A reference read from a file (or returned by a provider) points outside the case root.

    ``case_file`` is the file holding the reference (or the root for provider names),
    ``reference`` the text as written and ``case_root`` the directory it had to stay in."""

    def __init__(
        self,
        *,
        case_file: Path | str,
        reference: str,
        case_root: Path | str,
        what: str,
        reason: str,
    ) -> None:
        self.case_file = str(case_file)
        self.reference = reference
        self.case_root = str(case_root)
        self.what = what
        self.reason = reason
        super().__init__(
            f"{case_file}: {what} {reference!r} {reason}; references must stay inside the "
            f"case directory {case_root}",
            hint="copy the referenced file into the case directory and reference it with a "
            "relative path; Frontier never follows a reference that leaves the case",
        )

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict() | {
            "case_file": self.case_file,
            "reference": self.reference,
            "case_root": self.case_root,
        }


def case_root_of(source: Path | str) -> Path:
    """Case root of the path the user pointed at: the directory itself, or the directory of
    the entry file. Resolved, so a comparison against a resolved reference is meaningful."""
    p = Path(source)
    return (p if p.is_dir() else p.parent).resolve()


def is_inside(root: Path, path: Path) -> bool:
    """``path`` (after resolving symlinks) is ``root`` or below it."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):  # symlink loop
        return False


def _text(ref: str | os.PathLike[str]) -> str:
    return ref if isinstance(ref, str) else os.fspath(ref)


def _rooted(text: str) -> str | None:
    """Reason when ``text`` is absolute on any platform, else None."""
    if text.startswith(("/", "\\")):
        if text.startswith(("//", "\\\\")):
            return "is a UNC / network path"
        return "is an absolute path"
    win = PureWindowsPath(text)
    if win.drive:
        return "has a drive or UNC prefix"
    if win.is_absolute() or Path(text).is_absolute():
        return "is an absolute path"
    if text.startswith("~"):
        return "starts with a home-directory shortcut"
    return None


def _resolve(root: Path, text: str, base: Path, holder: Path, what: str) -> Path:
    def refuse(reason: str) -> ReferenceEscapeError:
        return ReferenceEscapeError(
            case_file=holder, reference=text, case_root=root, what=what, reason=reason
        )

    if not text or not text.strip():
        raise refuse("is empty")
    if "\x00" in text:
        raise refuse("contains a NUL byte")
    reason = _rooted(text)
    if reason:
        raise refuse(reason)
    try:
        walked = base.resolve()
        joined = walked / text
        # every symlink on the way must itself stay inside (one that leaves and comes back
        # is refused too); `..` is applied lexically here only to visit existing components
        for part in Path(text).parts:
            if part == "..":
                walked = walked.parent
                continue
            walked = walked / part
            if walked.is_symlink() and not walked.resolve().is_relative_to(root):
                raise refuse(f"goes through the symlink {walked}, which leaves the case directory")
        resolved = joined.resolve(strict=False)
    except (OSError, RuntimeError) as e:  # symlink loop, name too long
        raise refuse(f"cannot be resolved ({type(e).__name__})") from e
    if not resolved.is_relative_to(root):
        raise refuse("resolves outside the case directory")
    return resolved


def resolve_in_case(
    case_root: Path | str,
    ref: str | os.PathLike[str],
    *,
    what: str,
    referrer: Path | str | None = None,
) -> Path:
    """Path of ``ref`` (read from ``referrer``), guaranteed to lie inside ``case_root``.

    ``what`` names the kind of reference for the error ("XDMF DataItem", "*INCLUDE file").
    ``referrer`` is the file the reference was read from; the reference is taken relative to
    its directory. Without it the reference is relative to the case root.

    Returns the resolved path. Raises :class:`ReferenceEscapeError` otherwise; the message
    names the case file, the reference and the case root, and never the target's content.
    """
    root = Path(case_root).resolve()
    holder = Path(referrer) if referrer is not None else root
    base = holder.parent if referrer is not None else root
    return _resolve(root, _text(ref), base, holder, what)


def first_existing_in_case(
    case_root: Path | str,
    ref: str | os.PathLike[str],
    bases: Iterable[Path | str],
    *,
    what: str,
    referrer: Path | str,
) -> Path | None:
    """Search ``ref`` relative to each directory in ``bases`` (include search paths, in order)
    and return the first existing file that lies inside the case root, or None when the name
    is simply not there. A candidate that leaves the case is never opened; when no candidate
    inside the case exists and at least one left it, the reference raises instead of being
    reported as "not found" (``../x`` is fine from a subdirectory and an escape from the
    case directory itself)."""
    root = Path(case_root).resolve()
    text = _text(ref)
    escaped: ReferenceEscapeError | None = None
    for base in bases:
        try:
            cand = _resolve(root, text, Path(base), Path(referrer), what)
        except ReferenceEscapeError as e:
            escaped = escaped or e
            continue
        if cand.is_file():
            return cand
    if escaped is not None:
        raise escaped
    return None


def confine(root: Path | str, name: str | os.PathLike[str], *, what: str) -> Path:
    """Path of ``name`` below the directory ``root`` (a download or extraction directory).

    For names that do not come from a case file but from a provider API or an archive: same
    rules as :func:`resolve_in_case`, with the root as the referrer in the message."""
    return resolve_in_case(root, name, what=what)


#: how deep chains of HDF5 files linking to each other are followed when they are checked
H5_LINK_DEPTH = 8


def h5_confine_external_links(
    group: Any,
    *,
    path: Path | str,
    case_root: Path | str | None = None,
    what: str = "HDF5 link",
    _depth: int = 0,
    _seen: set[Path] | None = None,
) -> None:
    """Confine the external links of an HDF5 file (h5py ``File``/``Group``) to the case root.

    An external link names another *file* and HDF5 follows it transparently, so a dataset
    read from a received file could come from anywhere on disk. Linking to a sibling file is
    a normal layout (a CGNS solution linking to its grid file), so links are confined, not
    forbidden: the target file is resolved like every other reference (relative to the
    linking file, inside ``case_root``; default: the directory of ``path``) and is checked in
    turn. A dangling link is left to the connector, unless the name exists where HDF5 would
    search next (``$HDF5_EXT_PREFIX``, the working directory) outside the case.
    Soft links stay inside the file. The walk (``H5Lvisit``) reads link metadata only."""
    import importlib

    h5py: Any = importlib.import_module("h5py")
    external = h5py.h5l.TYPE_EXTERNAL
    holder = Path(path)
    root = Path(case_root).resolve() if case_root is not None else case_root_of(holder)
    seen = _seen if _seen is not None else {holder.resolve()}
    links: list[bytes] = []

    def visit(name: bytes, info: Any) -> None:
        if info.type == external:
            links.append(name)

    try:
        group.id.links.visit(visit, info=True)
    except (OSError, RuntimeError, ValueError):
        pass  # an unreadable link table fails where the connector reads it
    for name in links:
        where = name.decode("utf-8", "replace")
        label = f"{what} /{where}"
        try:
            raw_file, raw_path = group.id.links.get_val(name)
            target_file = raw_file.decode("utf-8", "replace")
            reference = f"{target_file}:{raw_path.decode('utf-8', 'replace')}"
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as e:
            raise ReferenceEscapeError(
                case_file=holder,
                reference="(unreadable target)",
                case_root=root,
                what=label,
                reason="is an external link that cannot be inspected",
            ) from e
        try:
            target = resolve_in_case(root, target_file, what=label, referrer=holder)
        except ReferenceEscapeError as e:
            raise ReferenceEscapeError(
                case_file=holder,
                reference=reference,
                case_root=root,
                what=label,
                reason=f"is an external link whose file {e.reason}",
            ) from e
        if not target.is_file():
            # a dangling link is the connector's to report, unless HDF5 would find the name
            # somewhere else: it goes on to try $HDF5_EXT_PREFIX and the working directory
            if any(not is_inside(root, c) for c in _h5_fallbacks(target_file) if c.exists()):
                raise ReferenceEscapeError(
                    case_file=holder,
                    reference=reference,
                    case_root=root,
                    what=label,
                    reason="is an external link to a file that is missing from the case "
                    "directory but present where HDF5 searches next (working directory)",
                )
            continue
        if target in seen:
            continue
        seen.add(target)
        if _depth >= H5_LINK_DEPTH:
            raise ReferenceEscapeError(
                case_file=holder,
                reference=reference,
                case_root=root,
                what=label,
                reason=f"is part of a chain of more than {H5_LINK_DEPTH} linked files",
            )
        try:
            linked = h5py.File(target, "r")
        except OSError:
            continue  # not HDF5: the read through the link fails on its own
        with linked:
            h5_confine_external_links(
                linked, path=target, case_root=root, what=what, _depth=_depth + 1, _seen=seen
            )


def _h5_fallbacks(name: str) -> list[Path]:
    """Where HDF5 looks for an external link's file besides the linking file's directory."""
    bases = [Path(p) for p in os.environ.get("HDF5_EXT_PREFIX", "").split(":") if p]
    return [base / name for base in [*bases, Path.cwd()]]


def h5_open(
    path: Path | str, *, case_root: Path | str | None = None, what: str = "HDF5 link"
) -> Any:
    """``h5py.File(path, "r")`` whose external links are confined to the case root (see
    :func:`h5_confine_external_links`). ``OSError`` from h5py passes through unchanged."""
    import h5py

    f = h5py.File(path, "r")
    try:
        h5_confine_external_links(f, path=path, case_root=case_root, what=what)
    except BaseException:
        f.close()
        raise
    return f


def h5_check_file(
    path: Path | str, *, case_root: Path | str | None = None, what: str = "HDF5 link"
) -> None:
    """Confine the external links of ``path`` before a third-party reader (VTK, meshio) opens
    it. Does nothing when ``path`` is not HDF5 or h5py is not installed: h5py is optional,
    and a connector whose own reader needs it has already asked for it."""
    try:
        import h5py
    except ImportError:
        return
    try:
        with open(path, "rb") as fh:
            if fh.read(8) != b"\x89HDF\r\n\x1a\n":
                return
        f = h5py.File(path, "r")
    except OSError:
        return
    with f:
        h5_confine_external_links(f, path=path, case_root=case_root, what=what)
