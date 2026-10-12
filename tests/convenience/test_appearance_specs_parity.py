"""The spec builders build what the old ``appearance_specs`` built.

``plans/ordered_controls_list_implementation.md`` (Phase 2, item 6).  The
golden, ``data/appearance_specs_golden.json``, holds what ``appearance_specs``
returned for 14 visuals before the controls list replaced the config
classes.  Each case here asks for the same controls as a list, in the order
the old code built them, and must get the same specs back.

The golden is never regenerated: the code that wrote it is gone.  A
difference is a bug in a spec builder.
"""

from __future__ import annotations

import json
import warnings

import pytest

from cellier.convenience.gui._controls_config import validate_controls
from cellier.convenience.layout._shared import appearance_specs
from tests.convenience._controls_lists import PARITY_CONTROLS, PARITY_KINDS
from tests.convenience._spec_parity_cases import (
    GOLDEN_PATH,
    build_case,
    specs_as_json,
)

GOLDEN = json.loads(GOLDEN_PATH.read_text())


def test_every_golden_case_has_a_list():
    assert sorted(PARITY_CONTROLS) == sorted(GOLDEN)
    assert len(GOLDEN) == 14


@pytest.mark.parametrize("case_name", sorted(GOLDEN))
def test_the_list_builds_the_specs_of_the_golden(case_name, tmp_path):
    visual_name = case_name.split("/")[0]
    controls = validate_controls(
        PARITY_CONTROLS[case_name](), PARITY_KINDS[visual_name], case_name
    )
    with warnings.catch_warnings():
        # A phong mesh in a scene with no lights warns; not what is tested.
        warnings.simplefilter("ignore")
        case = build_case(visual_name, tmp_path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            specs, skipped = appearance_specs(
                case.visual, controls, case.store, palette=case.palette
            )
        built = specs_as_json(specs, tmp_path)
    finally:
        case.viewer.controller.close()

    assert skipped == []
    assert [spec["kind"] for spec in built] == [
        spec["kind"] for spec in GOLDEN[case_name]
    ]
    # Through JSON, as the golden went: tuples become lists either way.
    assert json.loads(json.dumps(built)) == GOLDEN[case_name]
