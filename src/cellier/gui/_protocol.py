"""Backend-agnostic structural contract for cellier GUI elements.

The controller talks to widgets through one structural contract.  It imposes no
base class and says nothing about granularity: a widget may carry one field or
many.  ``CellierController.connect_widget`` wires its single ``changed`` /
``closed`` pair and its aggregated ``subscription_specs()`` regardless, which
is what lets the Qt backend stay fine-grained (per-control widgets) while the
anywidget backend uses a single composite panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uuid import UUID

    from psygnal import SignalInstance

    from cellier.events import SubscriptionSpec


@runtime_checkable
class WidgetView(Protocol):
    """Backend-agnostic contract for a cellier GUI element.

    Attributes
    ----------
    _id : UUID
        Identifies the widget; stamped as ``source_id`` on the events it emits
        and used as the ``owner_id`` for its bus subscriptions.
    changed : SignalInstance
        psygnal signal emitting a field-tagged ``*UpdateEvent`` when the user
        changes a value.
    closed : SignalInstance
        psygnal signal (no argument) emitted when the widget is closed.
    """

    _id: UUID
    changed: SignalInstance
    closed: SignalInstance

    @property
    def widget(self) -> object:
        """The toolkit element to embed (a ``QWidget`` or an anywidget)."""
        ...

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound bus subscriptions this widget requires."""
        ...

    def close(self) -> None:
        """Tear the widget down (emit ``closed`` to trigger unsubscription)."""
        ...
