"""The position slider of the anywidget plane controls, in Qt's JS engine.

``clipping_planes.js`` and ``render_planes.js`` are run against a fake DOM
and model.  Checked here: the slider runs from 0 to the row's ``span``,
shows its ``depth``, is pinned and marked for a plane outside the box, and
sends ``"depth"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cellier.gui.anywidget.visuals as anywidget_visuals

QtQml = pytest.importorskip("PySide6.QtQml")

STATIC = Path(anywidget_visuals.__file__).parent / "static"

FAKE_DOM = r"""
function Element(tag) {
  this.tag = tag; this.children = []; this.listeners = {}; this.dataset = {};
  this.style = {}; this.attributes = {}; this.className = ""; this.textContent = "";
  this.value = ""; this.checked = false; this.disabled = false; this.parent = null;
  this.title = "";
  var classes = {};
  this.classList = {
    add: function (c) { classes[c] = true; },
    toggle: function (c, on) { if (on) classes[c] = true; else delete classes[c]; },
    contains: function (c) { return Boolean(classes[c]); },
  };
}
Element.prototype.appendChild = function (c) {
  c.parent = this; this.children.push(c); return c;
};
Element.prototype.append = function () {
  for (var i = 0; i < arguments.length; i++) {
    var c = arguments[i];
    if (typeof c === "string") {
      var t = new Element("#text"); t.textContent = c; c = t;
    }
    this.appendChild(c);
  }
};
Element.prototype.remove = function () {
  if (!this.parent) return;
  var k = this.parent.children.indexOf(this);
  if (k >= 0) this.parent.children.splice(k, 1);
};
Element.prototype.addEventListener = function (n, f) {
  (this.listeners[n] = this.listeners[n] || []).push(f);
};
Element.prototype.setAttribute = function (n, v) { this.attributes[n] = v; };
Element.prototype.fire = function (n) {
  (this.listeners[n] || []).forEach(function (f) { f({}); });
};
function find(root, role) {
  var out = [];
  (function walk(e) {
    if (e.dataset && e.dataset.role === role) out.push(e);
    (e.children || []).forEach(walk);
  })(root);
  return out;
}
var document = { createElement: function (tag) { return new Element(tag); } };
var state = {};
var handlers = {};
var model = {
  get: function (k) { return state[k]; },
  set: function (k, v) { state[k] = v; },
  save_changes: function () {},
  on: function (name, f) { (handlers[name] = handlers[name] || []).push(f); },
};
function push(k, v) {
  state[k] = v;
  (handlers["change:" + k] || []).forEach(function (f) { f(); });
}
"""

SCENARIO = r"""
function row(extra) {
  var r = JSON.parse(JSON.stringify(BASE));
  for (var k in (extra || {})) r[k] = extra[k];
  return r;
}
var el = new Element("div");
state = JSON.parse(JSON.stringify(STATE));
state.rows = [row()];
render({ model: model, el: el });
var position = find(el, "position")[0];
var value = find(el, "value")[0];
function seen() {
  return {
    min: position.min, max: position.max, slider: position.value,
    text: value.textContent, marked: value.classList.contains(OUTSIDE_CLASS),
    title: value.title,
  };
}
var out = { slider_title: position.title, inside: seen() };

// A flipped plane: only the depth changed, the slider's ends did not.
push("rows", [row({ normal: FLIPPED, position: -12.0, depth: 28.0 })]);
out.flipped = seen();

// Past the far face, then before the near one.
push("rows", [row({ position: 46.0, depth: 46.0, outside: true })]);
out.past = seen();
push("rows", [row({ position: -3.0, depth: -3.0, outside: true })]);
out.before = seen();
push("rows", [row()]);
out.back = seen();

// A drag sends the depth, and an older echo does not move a held thumb.
position.value = "7.25"; position.fire("input");
out.edit = JSON.parse(JSON.stringify(state.edit));
out.text_while_dragging = value.textContent;
push("rows", [row({ position: 6.5, depth: 6.5 })]);
out.slider_while_dragging = position.value;
position.fire("change");
push("rows", [row({ position: 7.25, depth: 7.25 })]);
out.slider_after_release = position.value;
JSON.stringify(out);
"""

CLIPPING = {
    "file": "clipping_planes.js",
    "outside_class": "cellier-clipping-planes-outside",
    "units": "Data",
    "state": {
        "title": "Clipping planes",
        "axis_names": ["z", "y", "x"],
        "error": "",
        "edit": {},
    },
    "base": {
        "id": "a",
        "enabled": True,
        "normal": [0, 0, 1],
        "position": 12.0,
        "facing": [2, 1],
        "depth": 12.0,
        "span": 40.0,
        "outside": False,
    },
    "flipped": [0, 0, -1],
}
RENDER = {
    "file": "render_planes.js",
    "outside_class": "cellier-render-planes-outside",
    "units": "World",
    "state": {
        "title": "Render planes",
        "error": "",
        "blocked": "",
        "can_add": True,
        "edit": {},
    },
    "base": {
        "id": "a",
        "enabled": True,
        "axes": ["z", "y", "x"],
        "normal": [0, 0, 1],
        "position": 12.0,
        "extent_0": [None, None],
        "extent_1": [-3, 8],
        "facing": [2, 1],
        "depth": 12.0,
        "span": 40.0,
        "outside": False,
    },
    "flipped": [0, 0, -1],
}


def _run(case: dict) -> dict:
    source = (STATIC / case["file"]).read_text()
    assert "export default { render };" in source
    source = source.replace("export default { render };", "")
    setup = (
        f"var BASE = {json.dumps(case['base'])};\n"
        f"var STATE = {json.dumps(case['state'])};\n"
        f"var FLIPPED = {json.dumps(case['flipped'])};\n"
        f"var OUTSIDE_CLASS = {json.dumps(case['outside_class'])};\n"
    )
    # Kept in a name: a value outlives its engine as ``undefined``.
    engine = QtQml.QJSEngine()
    result = engine.evaluate(FAKE_DOM + source + setup + SCENARIO)
    assert not result.isError(), (
        f"line {result.property('lineNumber').toInt()}: {result.toString()}"
    )
    return json.loads(result.toString())


@pytest.mark.parametrize("case", [CLIPPING, RENDER], ids=["clipping", "render"])
def test_the_slider_shows_and_sends_the_depth(case, qapp):
    from cellier.gui._clipping_planes import OUTSIDE_TOOLTIP, position_tooltip

    usual = position_tooltip(case["units"])
    out = _run(case)
    # The front end says what the Python side says.
    assert out["slider_title"] == usual

    def shown(slider, text, marked):
        return {
            "min": "0",
            "max": "40",
            "slider": slider,
            "text": text,
            "marked": marked,
            "title": f"{OUTSIDE_TOOLTIP} {usual}" if marked else usual,
        }

    assert out["inside"] == shown("12", "12.00", False)
    assert out["flipped"] == shown("28", "28.00", False)
    # Outside: pinned at the nearer end, the true depth, marked.
    assert out["past"] == shown("40", "46.00", True)
    assert out["before"] == shown("0", "-3.00", True)
    assert out["back"] == shown("12", "12.00", False)

    assert out["edit"]["action"] == "depth"
    assert (out["edit"]["index"], out["edit"]["value"]) == (0, 7.25)
    assert out["text_while_dragging"] == "7.25"
    assert out["slider_while_dragging"] == "7.25"
    assert out["slider_after_release"] == "7.25"
