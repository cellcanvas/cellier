"""Config model for opening an anywidget viewer in a jupyterlab-sidecar tab.

``sidecar`` (https://github.com/jupyter-widgets/jupyterlab-sidecar) is an
optional dependency: it is only imported inside `resolve_sidecar`, so
importing this module -- or ``cellier.convenience`` at large -- never requires
it to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sidecar import Sidecar

    from cellier.convenience._launch import DisplayHandle

SidecarAnchor = Literal[
    "right",
    "split-right",
    "split-left",
    "split-top",
    "split-bottom",
    "tab-before",
    "tab-after",
]


@dataclass
class SidecarOptions:
    """Config for opening an anywidget viewer in a ``jupyterlab-sidecar`` tab.

    Passed as the ``sidecar=`` argument to
    `display`.  Mirrors the constructor of
    ``sidecar.Sidecar``.

    Parameters
    ----------
    title : str or None
        Tab title. Defaults to ``"Cellier"``; pass ``None`` explicitly to fall
        back to the ``sidecar`` package's own default title.
    anchor : SidecarAnchor
        Placement of the sidecar panel. ``"right"`` (default) docks in the
        existing right-hand panel; the ``"split-*"`` / ``"tab-*"`` values open
        a new, freely re-arrangeable JupyterLab panel instead.
    ref : SidecarOptions, DisplayHandle, or None
        Anchor the new sidecar relative to another sidecar's panel instead of
        the notebook.  Pass the `DisplayHandle`
        returned by a previous ``display(..., sidecar=...)`` call.
    """

    title: str | None = "Cellier"
    anchor: SidecarAnchor = "right"
    ref: SidecarOptions | DisplayHandle | None = None


def resolve_sidecar(sidecar: bool | SidecarOptions) -> Sidecar:
    """Build a live ``sidecar.Sidecar`` from *sidecar*.

    Parameters
    ----------
    sidecar : True or SidecarOptions
        ``True`` uses `SidecarOptions` defaults.

    Raises
    ------
    ImportError
        If the ``sidecar`` package is not installed.
    """
    try:
        from sidecar import Sidecar
    except ImportError as e:
        raise ImportError(
            "sidecar=... requires the optional 'sidecar' package "
            "(https://github.com/jupyter-widgets/jupyterlab-sidecar). "
            "Install it with `pip install sidecar`."
        ) from e

    options = sidecar if isinstance(sidecar, SidecarOptions) else SidecarOptions()

    ref = options.ref
    if isinstance(ref, SidecarOptions):
        raise TypeError(
            "SidecarOptions.ref must be a DisplayHandle from a previous "
            "display(..., sidecar=...) call, or a raw sidecar.Sidecar -- "
            "not another SidecarOptions."
        )
    if ref is not None and not isinstance(ref, Sidecar):
        # Assume a DisplayHandle; pull its live Sidecar instance off it.
        ref = ref._sidecar
        if ref is None:
            raise ValueError(
                "SidecarOptions.ref points to a DisplayHandle that was not "
                "itself displayed with sidecar=..."
            )

    kwargs: dict[str, object] = {"anchor": options.anchor, "ref": ref}
    if options.title is not None:
        # Sidecar's `title` trait is a non-nullable Unicode trait: passing
        # title=None explicitly fails traitlets validation, so it must be
        # omitted entirely to fall back to Sidecar's own default title.
        kwargs["title"] = options.title
    return Sidecar(**kwargs)
