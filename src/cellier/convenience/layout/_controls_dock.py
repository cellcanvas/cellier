"""A controls dock that follows its viewer (``plans/multi_visual_controls.md``).

An ``AppearanceControls()`` dock shows the controls for
one configured visual at a time.  When the viewer has several, a selector above
the controls chooses which; when visuals are added or removed after the layout
is shown, the dock rebuilds from the viewer's ``_controls_changed`` signal.
``AppearanceControls(presentation="collapsible_sections")`` shows every
configured visual at once instead, one collapsible section each.

Controller-aware, so it lives beside ``_walk.py`` rather than in ``_shared.py``;
every decision it makes -- which targets exist, what they are called, which one
stays selected -- is a pure function in ``_shared.py``.  Everything
toolkit-specific is reached through the host: ``live_slot`` for a region whose
contents can be replaced after it is presented, and ``backend.target_selector``
and ``backend.collapsible_section`` for the chrome around the controls.

**Every target's controls are built up front; selecting only swaps.**  On
marimo a widget can only be constructed while a cell is running -- building
one registers its ESM as a virtual file, which needs the running cell -- and a
selection arrives as a front-end message outside any cell.  So widgets are
built in :meth:`ControlsDock.refresh`, which runs from the initial render or
from an ``add_*`` / removal (both cell code), and the selector callback never
constructs anything.  Verified in a browser: building on selection raised
``AssertionError`` from marimo's ``virtual_file`` and left the dock showing the
previous visual's closed controls.  Sections are built in ``refresh`` for the
same reason, and expanding one is a front-end toggle that builds nothing.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Callable

from cellier.convenience.layout._shared import next_selection

if TYPE_CHECKING:
    from cellier.convenience._hosts import LayoutHost
    from cellier.convenience.layout._shared import ControlTarget

SELECTOR_TITLE = "Visual"
"""The selector row's label."""

OVERLAY_SELECTOR_TITLE = "Overlay"
"""The selector row's label in an overlay dock."""

APPEARANCE_PLACEHOLDER = "No visuals with appearance controls"
"""What an appearance dock says while no visual has an appearance config."""

SELECTOR = "selector"
"""The presentation that shows one target at a time, chosen by a selector."""

COLLAPSIBLE_SECTIONS = "collapsible_sections"
"""The presentation that shows every target at once, one collapsible section each."""


def _same_target(a: ControlTarget | None, b: ControlTarget | None) -> bool:
    """Whether *a* and *b* would build the same widgets.

    Deliberately not ``a == b``: a ``ControlTarget`` holds the visual model,
    and equality on models is not safe to rely on (a field ``__eq__`` that
    raises degrades a model class to identity comparison process-wide).  The
    key, the config object and the driven ids are what the build reads.
    """
    if a is None or b is None:
        return a is b
    return (
        a.key == b.key
        and a.config is b.config
        and list(a.visual_ids) == list(b.visual_ids)
    )


class ControlsDock:
    """One dock's controls, rebuilt as the viewer's configured visuals change.

    Parameters
    ----------
    viewer :
        The viewer whose recorded controls configs the dock serves.  Followed
        through ``viewer._controls_changed`` when it has one.
    host :
        The layout host composing the dock.
    resolve :
        ``viewer -> list[ControlTarget]``: the targets this dock can drive.
    build :
        ``ControlTarget -> list[widget]``: build and wire the controls for one
        target.  The dock closes them when it stops showing them.
    placeholder :
        What the dock says while it has no targets.
    presentation :
        :data:`SELECTOR` (default) or :data:`COLLAPSIBLE_SECTIONS`.
    selector_title :
        The selector row's label.  Defaults to :data:`SELECTOR_TITLE`.
    """

    def __init__(
        self,
        viewer: object,
        host: LayoutHost,
        *,
        resolve: Callable[[object], list[ControlTarget]],
        build: Callable[[ControlTarget], list],
        placeholder: str,
        presentation: str = SELECTOR,
        selector_title: str = SELECTOR_TITLE,
    ) -> None:
        self._viewer = viewer
        self._selector_title = selector_title
        self._host = host
        self._resolve = resolve
        self._build = build
        self._placeholder = placeholder
        self._presentation = presentation

        self._slot = host.live_slot()
        self._targets: list[ControlTarget] = []
        self._selected: ControlTarget | None = None
        # key -> (the target the widgets were built for, the widgets)
        self._built: dict[object, tuple[ControlTarget, list]] = {}
        self._selector = None
        # key -> (the section, the widget list it wraps); sections only.
        self._sections: dict[object, tuple[object, list]] = {}
        # Sections no longer shown, closed once the slot has let go of them.
        self._dropped_sections: list = []
        self._any_section_built = False
        self._shown: tuple | None = None
        self._closed = False

        self._changed = getattr(viewer, "_controls_changed", None)
        if self._changed is not None:
            self._changed.connect(self.refresh)
        self.refresh()

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def root(self) -> object:
        """The host item to place in the layout."""
        return self._slot.root

    @property
    def presentation(self) -> str:
        """How the dock presents several targets."""
        return self._presentation

    @property
    def targets(self) -> list[ControlTarget]:
        """The targets the dock can currently drive, in selector order."""
        return list(self._targets)

    @property
    def selected(self) -> ControlTarget | None:
        """The target whose controls are shown, or last expanded by :meth:`select`."""
        return self._selected

    @property
    def widgets(self) -> list:
        """The controls on show, without the selector or any section around them.

        With collapsible sections that is every target's controls, in target
        order, whether or not their section is expanded.
        """
        if self._presentation == COLLAPSIBLE_SECTIONS:
            return [
                widget
                for target in self._targets
                for widget in self._built.get(target.key, (None, []))[1]
            ]
        if self._selected is None:
            return []
        entry = self._built.get(self._selected.key)
        return list(entry[1]) if entry is not None else []

    @property
    def selector(self) -> object | None:
        """The selector, or ``None`` while the dock has fewer than two targets.

        Always ``None`` with collapsible sections, which need no selector.
        """
        return self._selector

    @property
    def sections(self) -> list:
        """The collapsible sections, in target order.  Empty with a selector."""
        return [
            self._sections[target.key][0]
            for target in self._targets
            if target.key in self._sections
        ]

    def refresh(self) -> None:
        """Re-resolve the targets, build what is new and release what is gone."""
        if self._closed:
            return
        targets = self._resolve(self._viewer)
        self._sync_built(targets)
        self._targets = targets
        self._selected = next_selection(self._selected, targets)
        self._render()

    def select(self, key: object) -> None:
        """Show the controls for the target recorded under *key*.

        With collapsible sections every target is already shown, so this
        expands the target's section instead.
        """
        for index, target in enumerate(self._targets):
            if target.key != key:
                continue
            if self._presentation == COLLAPSIBLE_SECTIONS:
                self._selected = target
                self._sections[key][0].set_expanded(True)
            else:
                self._on_selector(index)
            return
        raise KeyError(key)

    def close(self) -> None:
        """Stop following the viewer and release every widget the dock built."""
        if self._closed:
            return
        self._closed = True
        if self._changed is not None:
            self._changed.disconnect(self.refresh, missing_ok=True)
        for _target, widgets in self._built.values():
            _close_all(widgets)
        self._built = {}
        _close_all([section for section, _widgets in self._sections.values()])
        _close_all(self._dropped_sections)
        self._sections = {}
        self._dropped_sections = []
        if self._selector is not None:
            _close(self._selector)
            self._selector = None
        _close(self._slot)

    # ── Internals ────────────────────────────────────────────────────────────

    def _on_selector(self, index: int) -> None:
        # Never builds (see the module docstring): the widgets for every
        # target already exist, so a selection only changes which are shown.
        if self._closed or not 0 <= index < len(self._targets):
            return
        target = self._targets[index]
        if self._selected is not None and target.key == self._selected.key:
            return
        self._selected = target
        self._render()

    def _sync_built(self, targets: list[ControlTarget]) -> None:
        """Keep widgets whose target is unchanged; rebuild or release the rest."""
        kept: dict[object, tuple[ControlTarget, list]] = {}
        for target in targets:
            entry = self._built.get(target.key)
            if entry is not None and _same_target(entry[0], target):
                kept[target.key] = (target, entry[1])
        # Released before anything is built: each control emits ``closed`` and
        # the controller drops its subscriptions before a replacement
        # subscribes.
        for key, (_target, widgets) in self._built.items():
            if key not in kept or kept[key][1] is not widgets:
                _close_all(widgets)
        for target in targets:
            if target.key not in kept:
                kept[target.key] = (target, list(self._build(target)))
        self._built = {target.key: kept[target.key] for target in targets}
        if self._presentation == COLLAPSIBLE_SECTIONS:
            self._sync_sections(targets)

    def _sync_sections(self, targets: list[ControlTarget]) -> None:
        """Keep one section per target, wrapping the widgets built for it.

        A section whose widgets were kept is kept too, and retitled in case
        the target's label changed (a duplicate name added elsewhere).  One
        whose widgets were rebuilt is rebuilt around them and keeps whether it
        was expanded.  A brand-new section starts expanded only when it is the
        first this dock has built.
        """
        sections: dict[object, tuple[object, list]] = {}
        for target in targets:
            widgets = self._built[target.key][1]
            entry = self._sections.get(target.key)
            if entry is not None and entry[1] is widgets:
                entry[0].set_title(target.label)
                sections[target.key] = entry
                continue
            if entry is not None:
                expanded = bool(entry[0].expanded)
            else:
                expanded = not self._any_section_built
            section = self._host.backend.collapsible_section(
                target.label, widgets, expanded=expanded
            )
            self._any_section_built = True
            sections[target.key] = (section, widgets)
        self._dropped_sections.extend(
            entry[0]
            for key, entry in self._sections.items()
            if sections.get(key) is not entry
        )
        self._sections = sections

    def _render(self) -> None:
        dropped: list = []
        if self._presentation == COLLAPSIBLE_SECTIONS:
            items = self.sections
        else:
            items = self._selector_items(dropped)

        placeholder = None if self._targets else self._placeholder
        shown = (tuple(id(item) for item in items), placeholder)
        if shown != self._shown:
            self._slot.set(items, placeholder=placeholder)
            self._shown = shown
        # After the slot has let go of them, so the old content cannot take a
        # live selector or section down with it.
        dropped.extend(self._dropped_sections)
        self._dropped_sections = []
        _close_all(dropped)

    def _selector_items(self, dropped: list) -> list:
        """The selector, when there is a choice to make, then the selected controls.

        A selector that is no longer needed is appended to *dropped* rather
        than closed, so the caller can close it after the slot lets go.
        """
        items: list = []
        if len(self._targets) >= 2:
            labels = [target.label for target in self._targets]
            index = next(
                i
                for i, target in enumerate(self._targets)
                if target.key == self._selected.key
            )
            if self._selector is None:
                self._selector = self._host.backend.target_selector(
                    labels, index, title=self._selector_title
                )
                self._selector.selected.connect(self._on_selector)
            else:
                self._selector.set_choices(labels, index)
            items.append(self._selector)
        elif self._selector is not None:
            dropped.append(self._selector)
            self._selector = None
        items.extend(self.widgets)
        return items


def _close_all(widgets: list) -> None:
    for widget in widgets:
        _close(widget)


def _close(obj: object) -> None:
    close = getattr(obj, "close", None)
    if close is not None:
        with suppress(Exception):
            close()
