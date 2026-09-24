"""Loading indicators and loading-settings controls, decided once for both toolkits.

The Qt and anywidget indicators (``QtLoadingIndicator``,
``AnywidgetLoadingIndicator``) draw a bar and a line of text.  Which bar and
which text follows from the visual's
:class:`~cellier.events.LoadingProgress`
(``plans/progressive_loading_design_v3.md`` 5.13); this module turns one into
the other, and sums a group of visuals (an ``OrthoViewer``'s four panels
share one control) into one progress.

Nothing here imports a toolkit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, get_args, get_origin

from cellier.events import (
    LoadingConfigChangedEvent,
    LoadingConfigUpdateEvent,
    LoadingProgress,
    ResliceProgressEvent,
    SubscriptionSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from uuid import UUID

LOADING_TITLE = "Data fetch status"
"""The indicator's name: its ``DEFAULT_TITLE`` on both toolkits.

Drawn as the heading of a group around the bar and status line, never as a
label beside the bar.
"""


class IndicatorState(NamedTuple):
    """One frame of the indicator.

    Attributes
    ----------
    maximum, value : int
        The bar: target chunks needed and resident.  ``maximum`` is at least
        1: an empty plan is drawn full once complete, and empty before
        anything was planned.
    text : str
        The status line, e.g. ``"Detail: 12 / 40"``.
    busy : bool
        Anything still to load.
    """

    maximum: int
    value: int
    text: str
    busy: bool


def sum_progress(progress: Iterable[LoadingProgress]) -> LoadingProgress | None:
    """Sum several visuals' progress; ``None`` when there is none."""
    total: LoadingProgress | None = None
    for p in progress:
        if total is None:
            total = p
            continue
        total = LoadingProgress(
            *(a + b for a, b in zip(total[:8], p[:8], strict=True)),
            backstop_complete=total.backstop_complete and p.backstop_complete,
            complete=total.complete and p.complete,
            target_deferred=total.target_deferred or p.target_deferred,
        )
    return total


def indicator_state(progress: LoadingProgress | None) -> IndicatorState:
    """The bar and text for *progress*.

    - no plan yet: ``"Not loaded"``;
    - backstop still loading: ``"Overview: 3 / 8"``;
    - then the target: ``"Detail: 12 / 40"``;
    - done: ``"Loaded"``, or ``"Loaded, 2 failed"``;
    - done, but the plan was the backstop only (a slider drag with
      ``dims_drag="backstop"``): ``"Overview ready. Detail on stop."``.

    Chunks the plan dropped to fit the cache (``truncated_target``) are
    reported after either, since they will never load at this cache size.
    """
    if progress is None:
        return IndicatorState(1, 0, "Not loaded", busy=False)
    p = progress
    if not p.backstop_complete:
        text = f"Overview: {p.resident_backstop} / {p.needed_backstop}"
    elif not p.complete:
        text = f"Detail: {p.resident_target} / {p.needed_target}"
    elif p.target_deferred:
        text = "Overview ready. Detail on stop."
    else:
        text = "Loaded"
    if p.failed:
        text += f", {p.failed} failed"
    if p.truncated_target:
        text += f", {p.truncated_target} over budget"
    if p.needed_target == 0:
        maximum, value = 1, int(p.complete and not p.target_deferred)
    else:
        maximum, value = p.needed_target, p.resident_target
    busy = not p.complete or p.target_deferred
    return IndicatorState(maximum, value, text, busy=busy)


class LoadingIndicatorModel:
    """The progress of a group of visuals, updated from the bus.

    Parameters
    ----------
    visual_ids : Iterable[UUID]
        The visuals the indicator stands for.
    initial : Mapping[UUID, LoadingProgress | None]
        Their progress now (``CellierController.loading_progress``), so an
        indicator built mid-load does not wait for the next event.
    on_change : Callable[[IndicatorState], None]
        Called with the new state after every event.
    """

    def __init__(
        self,
        visual_ids: Iterable[UUID],
        initial: Mapping[UUID, LoadingProgress | None],
        on_change: Callable[[IndicatorState], None],
    ) -> None:
        self._visual_ids = tuple(visual_ids)
        self._progress: dict[UUID, LoadingProgress] = {
            vid: p for vid, p in initial.items() if p is not None
        }
        self._on_change = on_change

    @property
    def state(self) -> IndicatorState:
        """The indicator's current state."""
        return indicator_state(
            sum_progress(
                self._progress[vid] for vid in self._visual_ids if vid in self._progress
            )
        )

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``ResliceProgressEvent`` subscription per visual."""
        return [
            SubscriptionSpec(ResliceProgressEvent, self.on_progress, entity_id=vid)
            for vid in self._visual_ids
        ]

    def on_progress(self, event: ResliceProgressEvent) -> None:
        """Record *event*'s progress and redraw."""
        self._progress[event.visual_id] = event.progress
        self._on_change(self.state)


# -- the loading-settings control ----------------------------------------------------

LOADING_CONFIG_TITLE = "Progressive loading"
"""The settings control's name: its ``DEFAULT_TITLE`` on both toolkits."""

#: The row label of each ``ProgressiveLoadingConfig`` field, in display order.
_LOADING_CONFIG_LABELS: dict[str, str] = {
    "backstop": "Backstop",
    "backstop_level": "Backstop level",
    "backstop_extent": "Backstop extent",
    "backstop_max_slot_fraction": "Backstop max slots",
    "dims_drag": "Dims drag",
}

#: Shown for ``backstop_level=None``: the coarsest level.
COARSEST_LEVEL_TEXT = "coarsest"


class LoadingConfigField(NamedTuple):
    """How one ``ProgressiveLoadingConfig`` field is drawn.

    Attributes
    ----------
    name : str
        The field name.
    label : str
        The row label.
    kind : "bool", "choice", "level" or "fraction"
        The control: a checkbox, a drop-down, a level spin box whose 0
        means ``None`` (the coarsest level), or a number.
    choices : tuple[str, ...]
        The options of a ``"choice"``.
    minimum, maximum, step : float
        The range of a ``"level"`` or ``"fraction"``.
    """

    name: str
    label: str
    kind: str
    choices: tuple[str, ...] = ()
    minimum: float = 0.0
    maximum: float = 0.0
    step: float = 1.0


def loading_config_fields(n_levels: int | None = None) -> list[LoadingConfigField]:
    """Describe every ``ProgressiveLoadingConfig`` field, read off the model.

    Choices come from the ``Literal`` annotations and bounds from the field
    constraints, so the control cannot drift from the config.

    Parameters
    ----------
    n_levels : int or None
        The visual's level count, the most ``backstop_level`` may be.
        ``None`` allows up to 16.

    Returns
    -------
    list[LoadingConfigField]
        In display order.
    """
    from cellier.visuals import ProgressiveLoadingConfig

    fields = []
    for name, label in _LOADING_CONFIG_LABELS.items():
        info = ProgressiveLoadingConfig.model_fields[name]
        annotation = info.annotation
        if annotation is bool:
            fields.append(LoadingConfigField(name, label, "bool"))
        elif get_origin(annotation) is Literal:
            choices = tuple(str(choice) for choice in get_args(annotation))
            fields.append(LoadingConfigField(name, label, "choice", choices))
        elif name == "backstop_level":
            maximum = float(n_levels) if n_levels else 16.0
            fields.append(LoadingConfigField(name, label, "level", (), 0.0, maximum))
        else:
            low, high = 0.0, 1.0
            for bound in info.metadata:
                low = float(getattr(bound, "gt", getattr(bound, "ge", low)))
                high = float(getattr(bound, "le", getattr(bound, "lt", high)))
            # An exclusive lower bound of 0 is shown as its first step.
            fields.append(
                LoadingConfigField(
                    name, label, "fraction", (), max(low, 0.01), high, 0.05
                )
            )
    return fields


def to_control_value(field: str, value: Any) -> Any:
    """A config value as its control holds it (``None`` level -> 0)."""
    if field == "backstop_level":
        return 0 if value is None else int(value)
    return value


def from_control_value(field: str, value: Any) -> Any:
    """A control's value as the config field takes it (level 0 -> ``None``)."""
    if field == "backstop_level":
        return None if int(value) == 0 else int(value)
    if field == "backstop_max_slot_fraction":
        return float(value)
    if field == "backstop":
        return bool(value)
    return value


def error_message(error: BaseException) -> str:
    """The message to show for a refused edit.

    The controller's ``ValueError`` (a pydantic ``ValidationError`` for an
    invalid combination) may arrive wrapped by the signal that carried the
    edit, so the cause chain is searched for it.
    """
    seen: BaseException | None = error
    while seen is not None:
        errors = getattr(seen, "errors", None)
        if callable(errors):
            try:
                # pydantic prefixes a validator's message with its kind.
                return str(errors()[0]["msg"]).removeprefix("Value error, ")
            except Exception:  # pragma: no cover - defensive
                pass
        if isinstance(seen, ValueError | TypeError):
            return str(seen)
        seen = seen.__cause__ or seen.__context__
    return str(error)


class LoadingConfigEditor:
    """The toolkit-neutral half of a loading-settings control.

    Holds the last config the model reported, sends edits, and tells the
    widget what to show.  An edit is one ``LoadingConfigUpdateEvent`` per
    visual.  If the controller refuses it -- an invalid combination raises
    -- the widget is told to show the last config again, with the error;
    nothing is corrected.

    Parameters
    ----------
    visual_ids : Iterable[UUID]
        The visuals edited together (an ``OrthoViewer`` panel group).
    loading : Mapping[str, Any]
        Their current config, as ``ProgressiveLoadingConfig.model_dump()``.
    source_id : UUID
        The widget's id, stamped on each edit.
    emit : Callable[[LoadingConfigUpdateEvent], None]
        Sends an edit (the widget's ``changed.emit``).
    show : Callable[[dict[str, Any], str], None]
        Draws a config and an error message ("" for none).
    """

    def __init__(
        self,
        visual_ids: Iterable[UUID],
        loading: Mapping[str, Any],
        source_id: UUID,
        emit: Callable[[LoadingConfigUpdateEvent], None],
        show: Callable[[dict[str, Any], str], None],
    ) -> None:
        self._visual_ids = tuple(visual_ids)
        self.config: dict[str, Any] = dict(loading)
        self._source_id = source_id
        self._emit = emit
        self._show = show

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``LoadingConfigChangedEvent`` subscription per visual."""
        return [
            SubscriptionSpec(LoadingConfigChangedEvent, self.on_changed, entity_id=vid)
            for vid in self._visual_ids
        ]

    def edit(self, field: str, control_value: Any) -> None:
        """Send an edit of *field*; on refusal, show the last config and why."""
        value = from_control_value(field, control_value)
        if self.config.get(field) == value:
            return
        try:
            for visual_id in self._visual_ids:
                self._emit(
                    LoadingConfigUpdateEvent(
                        source_id=self._source_id,
                        visual_id=visual_id,
                        field=field,
                        value=value,
                    )
                )
        except Exception as error:
            self._show(dict(self.config), error_message(error))

    def on_changed(self, event: LoadingConfigChangedEvent) -> None:
        """The model changed (this widget's edit or anyone's): show it."""
        self.config = event.loading.model_dump()
        self._show(dict(self.config), "")
