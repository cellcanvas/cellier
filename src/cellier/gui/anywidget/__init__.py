"""cellier anywidget (notebook) GUI widgets.

Importing this package requires the optional ``anywidget`` dependency
(``pip install 'cellier[anywidget]'``).
"""

from cellier.gui.anywidget._container import AnywidgetBox, AnywidgetSlot
from cellier.gui.anywidget._dataset_info import (
    AnywidgetDatasetInfo,
    DatasetInfo,
    MatrixSection,
    RowSection,
    dataset_info_from_path,
)
from cellier.gui.anywidget._dims_panel import AnywidgetDimsPanel
from cellier.gui.anywidget.render import (
    AnywidgetAmbientOcclusionControls,
    AnywidgetOutlineControls,
    AnywidgetRenderConfigPanel,
    AnywidgetTemporalControls,
)

__all__ = [
    "AnywidgetAmbientOcclusionControls",
    "AnywidgetBox",
    "AnywidgetDatasetInfo",
    "AnywidgetDimsPanel",
    "AnywidgetOutlineControls",
    "AnywidgetRenderConfigPanel",
    "AnywidgetSlot",
    "AnywidgetTemporalControls",
    "DatasetInfo",
    "MatrixSection",
    "RowSection",
    "dataset_info_from_path",
]
