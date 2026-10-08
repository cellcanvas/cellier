"""The outline box and colour of the anywidget plane controls, in Qt's JS
engine.

``clipping_planes.js`` and ``render_planes.js`` are run against the fake
DOM and model of ``test_plane_position_js.py``.  Checked here: a row shows
its plane's outline, and sends ``"outline"`` and ``"outline_color"``.
"""

from __future__ import annotations

import json

import pytest

from tests.gui.test_plane_position_js import CLIPPING, FAKE_DOM, RENDER, STATIC

QtQml = pytest.importorskip("PySide6.QtQml")

SCENARIO = r"""
function row(extra) {
  var r = JSON.parse(JSON.stringify(BASE));
  for (var k in (extra || {})) r[k] = extra[k];
  return r;
}
var OFF = { enabled: false, color: [1, 1, 1, 1], width: 2 };
var RED = { enabled: true, color: [1, 0, 0, 1], width: 2 };
var el = new Element("div");
state = JSON.parse(JSON.stringify(STATE));
state.rows = [row({ outline: OFF, outline_hex: "#ffffff" })];
render({ model: model, el: el });
var box = find(el, "outline")[0];
var colour = find(el, "outline-color")[0];
function seen() { return { checked: box.checked, colour: colour.value }; }
var out = {
  type: [box.type, colour.type],
  title: [box.title, colour.title],
  off: seen(),
};

// The model changed.
push("rows", [row({ outline: RED, outline_hex: "#ff0000" })]);
out.red = seen();

// The user clears the box, then chooses a colour.
box.checked = false; box.fire("change");
out.box_edit = JSON.parse(JSON.stringify(state.edit));
colour.value = "#00ff00"; colour.fire("change");
out.colour_edit = JSON.parse(JSON.stringify(state.edit));

// A row from a Python side with no outline entry draws, with the box clear.
push("rows", [row()]);
out.bare = seen();
JSON.stringify(out);
"""


def _run(case: dict) -> dict:
    source = (STATIC / case["file"]).read_text()
    assert "export default { render };" in source
    source = source.replace("export default { render };", "")
    setup = (
        f"var BASE = {json.dumps(case['base'])};\n"
        f"var STATE = {json.dumps(case['state'])};\n"
    )
    # Kept in a name: a value outlives its engine as ``undefined``.
    engine = QtQml.QJSEngine()
    result = engine.evaluate(FAKE_DOM + source + setup + SCENARIO)
    assert not result.isError(), (
        f"line {result.property('lineNumber').toInt()}: {result.toString()}"
    )
    return json.loads(result.toString())


@pytest.mark.parametrize("case", [CLIPPING, RENDER], ids=["clipping", "render"])
def test_a_row_shows_and_sends_its_outline(case, qapp):
    from cellier.gui._clipping_planes import OUTLINE_COLOR_TOOLTIP, OUTLINE_TOOLTIP

    out = _run(case)
    assert out["type"] == ["checkbox", "color"]
    # The words are the Qt control's.
    assert out["title"] == [OUTLINE_TOOLTIP, OUTLINE_COLOR_TOOLTIP]
    assert out["off"] == {"checked": False, "colour": "#ffffff"}
    assert out["red"] == {"checked": True, "colour": "#ff0000"}
    assert out["box_edit"]["action"] == "outline"
    assert out["box_edit"]["index"] == 0
    assert out["box_edit"]["value"] is False
    assert out["colour_edit"]["action"] == "outline_color"
    assert out["colour_edit"]["value"] == "#00ff00"
    assert out["colour_edit"]["serial"] > out["box_edit"]["serial"]
    # The colour stays what it was; the box is clear.
    assert out["bare"] == {"checked": False, "colour": "#00ff00"}
