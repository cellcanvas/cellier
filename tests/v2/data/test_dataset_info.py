"""Every data store describes itself for the dataset-info widget.

``BaseDataStore.dataset_info`` is the seam that generalized the widget: it
used to render only rows a caller hand-wrote for a multiscale image, so a
points or graph store could not populate it at all.  These tests pin the
contract every store now shares -- and the one thing it must *not* do, which
is read array data off disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data import (
    GraphMemoryStore,
    ImageMemoryStore,
    LabelMemoryStore,
    LinesMemoryStore,
    MeshMemoryStore,
    PointsMemoryStore,
)
from cellier.data._dataset_info import (
    DatasetInfo,
    MatrixSection,
    RowSection,
    format_bytes,
    format_scale,
    source_label,
    uri_file_name,
)


def _stores():
    """One instance of every in-memory store, with distinguishable data."""
    return [
        ImageMemoryStore(data=np.arange(24, dtype=np.float32).reshape(2, 3, 4)),
        LabelMemoryStore(data=np.zeros((2, 3, 4), dtype=np.int16)),
        PointsMemoryStore(positions=np.zeros((5, 3), dtype=np.float32)),
        LinesMemoryStore(positions=np.zeros((6, 3), dtype=np.float32)),
        MeshMemoryStore(
            positions=np.zeros((6, 3), dtype=np.float32),
            indices=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        ),
        GraphMemoryStore(
            positions=np.zeros((4, 3), dtype=np.float32),
            edges=np.array([[0, 1], [1, 2]], dtype=np.int32),
        ),
    ]


def _row_dict(info: DatasetInfo) -> dict[str, str]:
    return dict(info.rows)


# ── the shared contract ──────────────────────────────────────────────────────


@pytest.mark.parametrize("store", _stores(), ids=lambda s: type(s).__name__)
def test_every_store_opens_with_its_identity(store):
    """The first section is unlabelled and names the store and its kind."""
    info = store.dataset_info()
    first = info.sections[0]
    assert isinstance(first, RowSection)
    assert first.label is None
    assert first.rows[0] == ("Name", store.name)
    assert first.rows[1][0] == "Store type"


@pytest.mark.parametrize("store", _stores(), ids=lambda s: type(s).__name__)
def test_store_type_row_is_human_readable_not_the_discriminator(store):
    """``"in-memory points"`` is what a person reads; ``points_memory`` is not."""
    label = _row_dict(store.dataset_info())["Store type"]
    assert label != store.store_type
    assert "_" not in label


@pytest.mark.parametrize("store", _stores(), ids=lambda s: type(s).__name__)
def test_every_value_is_a_string(store):
    """Both front ends display values verbatim, so coercion happens once, here."""
    for section in store.dataset_info().sections:
        if isinstance(section, RowSection):
            for label, value in section.rows:
                assert isinstance(label, str)
                assert isinstance(value, str)


@pytest.mark.parametrize("store", _stores(), ids=lambda s: type(s).__name__)
def test_no_store_reports_a_display_property(store):
    """Colour and size modes describe how a visual is *drawn*, not what is stored.

    They live on the appearance model, and a dataset-info block that repeated
    them would be stating a display decision as though it were a fact about
    the data.
    """
    rows = _row_dict(store.dataset_info())
    assert not [key for key in rows if "color" in key.lower()]
    assert not [key for key in rows if "size mode" in key.lower()]


# ── per-store content ────────────────────────────────────────────────────────


def test_image_memory_reports_shape_dtype_and_range():
    store = ImageMemoryStore(data=np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    rows = _row_dict(store.dataset_info())
    assert rows["Shape"] == "2 x 3 x 4"
    assert rows["Data type"] == "float32"
    assert rows["Value range"] == "[0, 23]"


def test_label_memory_counts_its_labels():
    data = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int16)
    rows = _row_dict(LabelMemoryStore(data=data).dataset_info())
    assert rows["Labels"] == "3"
    assert rows["Max label"] == "2"
    assert rows["Data type"] == "int16"


def test_points_reports_count_and_extent():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]], dtype=np.float32)
    rows = _row_dict(PointsMemoryStore(positions=positions).dataset_info())
    assert rows["Points"] == "2"
    assert rows["Dimensions"] == "3"
    assert rows["Extent"] == "[0, 1] x [0, 2] x [0, 4]"


def test_lines_reports_both_segments_and_vertices():
    """Two vertices per segment; showing only one of the two invites a misread."""
    rows = _row_dict(
        LinesMemoryStore(positions=np.zeros((6, 3), dtype=np.float32)).dataset_info()
    )
    assert rows["Segments"] == "3"
    assert rows["Vertices"] == "6"


def test_mesh_reports_vertices_and_faces():
    store = MeshMemoryStore(
        positions=np.zeros((6, 3), dtype=np.float32),
        indices=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
    )
    rows = _row_dict(store.dataset_info())
    assert rows["Vertices"] == "6"
    assert rows["Faces"] == "2"
    # ``ndim`` did not exist on this store before; the table needs it.
    assert store.ndim == 3
    assert rows["Dimensions"] == "3"


def test_graph_reports_structure_and_transform_section():
    store = GraphMemoryStore(
        positions=np.zeros((4, 3), dtype=np.float32),
        edges=np.array([[0, 1], [1, 2]], dtype=np.int32),
        directed=True,
        axis_scales=(2.0, 2.0, 2.0),
    )
    info = store.dataset_info()
    rows = _row_dict(info)
    assert rows["Nodes"] == "4"
    assert rows["Edges"] == "2"
    assert rows["Directed"] == "yes"
    assert rows["Slice strategy"] == "mask"

    matrices = [s for s in info.sections if isinstance(s, MatrixSection)]
    assert [s.label for s in matrices] == ["Transform"]


def test_graph_dataset_info_does_not_build_the_spatial_index():
    """Opening a panel is not a reason to pay for an index build."""
    store = GraphMemoryStore(
        positions=np.zeros((4, 3), dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int32),
    )
    store.dataset_info()
    assert store._graph is None


def test_empty_arrays_report_no_extent():
    """``min``/``max`` over zero points raises; a store with no data has no extent."""
    store = PointsMemoryStore(positions=np.zeros((0, 3), dtype=np.float32))
    assert "Extent" not in _row_dict(store.dataset_info())


# ── the flattened view ───────────────────────────────────────────────────────


def test_flattened_rows_prefix_labelled_sections():
    """Flattening must not merge two blocks into one ambiguous list."""
    info = DatasetInfo(
        sections=[
            RowSection(None, [("Name", "a")]),
            RowSection("Axes", [("z", "space")]),
        ]
    )
    assert info.rows == [("Name", "a"), ("Axes: z", "space")]


def test_flattened_rows_skip_matrix_sections():
    info = DatasetInfo(
        sections=[
            RowSection(None, [("Name", "a")]),
            MatrixSection("M", np.eye(2), ["y", "x"], ["y", "x"]),
        ]
    )
    assert info.rows == [("Name", "a")]


# ── formatting helpers ───────────────────────────────────────────────────────


def test_format_scale_collapses_the_isotropic_case():
    assert format_scale((2.0, 2.0, 2.0)) == "2x isotropic"


def test_format_scale_spells_out_anisotropy():
    """The row the examples used to hardcode as ``"2x isotropic"`` regardless."""
    assert format_scale((1.0, 2.0, 2.0)) == "1 x 2 x 2"


@pytest.mark.parametrize(
    ("n_bytes", "expected"),
    [(512, "512 B"), (2048, "2.0 KiB"), (3 * 1024**2, "3.0 MiB")],
)
def test_format_bytes(n_bytes, expected):
    assert format_bytes(n_bytes) == expected


# -- URIs and Windows paths ---------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("/data/a.zarr", "local file"),
        ("file:///data/a.zarr", "local file"),
        ("s3://bucket/a.zarr", "S3"),
        ("https://host/a.zarr", "HTTP"),
        # A drive letter is not a URL scheme.
        ("C:\\data\\a.zarr", "local file"),
        ("d:/data/a.zarr", "local file"),
    ],
)
def test_source_label(uri, expected):
    assert source_label(uri) == expected


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("/data/a.zarr", "a.zarr"),
        ("s3://bucket/path/a.zarr/", "a.zarr"),
        ("C:\\data\\a.zarr", "a.zarr"),
        ("D:\\data\\a.zarr\\", "a.zarr"),
        ("d:/data/a.zarr", "a.zarr"),
    ],
)
def test_uri_file_name(uri, expected):
    assert uri_file_name(uri) == expected
