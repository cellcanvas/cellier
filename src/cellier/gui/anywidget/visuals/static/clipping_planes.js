// Clipping planes ESM: one row per entry of `rows`.  The Python side syncs
// `rows` (what to draw) and `axis_names` (the data axes, in the order of a
// normal's entries); an action is reported by setting `edit`, and a refused
// one comes back as `error` with `rows` unchanged.
//
// A row is built once and then updated in place, so a slider is not
// destroyed while it is dragged.  Rows are only added or removed when the
// number of planes changes.

// The same words as ``position_tooltip`` and ``OUTSIDE_TOOLTIP`` in
// ``cellier/gui/_clipping_planes.py``.
const POSITION_TOOLTIP =
  "How far the plane is into the bounding box, along its normal. " +
  "0 is the face the normal points away from. Data units.";
const OUTSIDE_TOOLTIP = "The plane is outside the bounding box.";
const GIZMO_TOOLTIP = "Drag a gizmo in the 3D view to move and tilt this plane.";
// The same words as ``OUTLINE_TOOLTIP`` and ``OUTLINE_COLOR_TOOLTIP`` there.
const OUTLINE_TOOLTIP =
  "Draw a line around the plane where it crosses the data, in the 3D view.";
const OUTLINE_COLOR_TOOLTIP = "The outline's colour.";

// A normal's entry as the number input shows it: at most three decimals.
function shown(value) {
  return String(Number(Number(value).toFixed(3)));
}

function render({ model, el }) {
  el.classList.add("cellier-clipping-planes");
  let serial = 0;

  const heading = document.createElement("div");
  heading.className = "cellier-clipping-planes-title";
  el.appendChild(heading);

  const body = document.createElement("div");
  body.className = "cellier-clipping-planes-rows";
  el.appendChild(body);

  const add = document.createElement("button");
  add.className = "cellier-clipping-planes-add";
  add.textContent = "Add plane";
  add.title = "Add a plane across the last axis, through the middle.";
  add.dataset.role = "add";
  el.appendChild(add);

  const error = document.createElement("div");
  error.className = "cellier-clipping-planes-error";
  el.appendChild(error);

  function send(action, index, value) {
    serial += 1;
    model.set("edit", { action, index, value, serial });
    model.save_changes();
  }

  add.addEventListener("click", () => send("add", null, null));

  // One entry per built row: { root, apply(row) }.
  const built = [];

  function build(index) {
    const names = model.get("axis_names");
    const root = document.createElement("div");
    root.className = "cellier-clipping-planes-row";
    root.dataset.role = "row";
    root.dataset.index = String(index);

    // -- header: on or off, flip, remove
    const header = document.createElement("div");
    header.className = "cellier-clipping-planes-header";

    const enabledLabel = document.createElement("label");
    const enabled = document.createElement("input");
    enabled.type = "checkbox";
    enabled.title = "Whether this plane clips. It stays in the list.";
    enabled.dataset.role = "enabled";
    enabled.addEventListener("change", () => send("enabled", index, enabled.checked));
    enabledLabel.append(enabled, ` Plane ${index + 1}`);

    // Drawn from `rows`, not from the click: one canvas has one gizmo, so
    // switching one on switches another off.  Hidden when the row has no
    // `gizmo` entry (no 3D canvas to draw one in).
    const gizmo = document.createElement("button");
    gizmo.textContent = "Gizmo";
    gizmo.dataset.role = "gizmo";
    let gizmoOn = false;
    gizmo.addEventListener("click", () => send("gizmo", index, !gizmoOn));

    const flip = document.createElement("button");
    flip.textContent = "Flip";
    flip.title = "Keep the other side of the plane.";
    flip.dataset.role = "flip";
    flip.addEventListener("click", () => send("flip", index, null));

    const remove = document.createElement("button");
    remove.textContent = "Remove";
    remove.dataset.role = "remove";
    remove.addEventListener("click", () => send("remove", index, null));

    const spacer = document.createElement("span");
    spacer.className = "cellier-clipping-planes-spacer";
    header.append(enabledLabel, spacer, gizmo, flip, remove);

    // -- normal: one column per data axis, two buttons over an entry
    const normal = document.createElement("div");
    normal.className = "cellier-clipping-planes-normal";
    normal.style.gridTemplateColumns = `auto repeat(${names.length}, minmax(0, 1fr))`;

    const normalLabel = document.createElement("span");
    normalLabel.className = "cellier-clipping-planes-label";
    normalLabel.textContent = "Normal";
    normalLabel.title =
      "The normal, one entry per data axis. It points to the side that is kept. " +
      "Only its direction matters. A button faces the plane along its axis and " +
      "keeps that side.";
    normal.appendChild(normalLabel);

    // The normal last drawn, to refuse an all-zero one without a round trip.
    let current = names.map(() => 0);
    const facing = [];  // { axis, sign, button }
    const components = [];
    names.forEach((name, axis) => {
      const pair = document.createElement("div");
      pair.className = "cellier-clipping-planes-facing";
      pair.style.gridColumn = String(axis + 2);
      for (const [sign, text, side] of [[1, "+", "higher"], [-1, "-", "lower"]]) {
        const button = document.createElement("button");
        button.textContent = `${text}${name}`;
        button.title = `Cut across ${name} and keep the side toward ${side} ${name}.`;
        button.dataset.role = "facing";
        button.dataset.axis = String(axis);
        button.dataset.sign = String(sign);
        button.addEventListener("click", () => send("facing", index, [axis, sign]));
        pair.appendChild(button);
        facing.push({ axis, sign, button });
      }
      normal.appendChild(pair);
    });
    names.forEach((name, axis) => {
      const component = document.createElement("input");
      component.type = "number";
      component.step = "0.1";
      component.title = `The normal's entry on ${name}.`;
      component.dataset.role = "component";
      component.dataset.axis = String(axis);
      component.style.gridColumn = String(axis + 2);
      component.addEventListener("change", () => {
        const entry = Number(component.value);
        const next = current.map((v, i) => (i === axis ? entry : v));
        send("component", index, [axis, entry]);
        // An all-zero or unreadable normal is refused and `rows` does not
        // change, so nothing would draw the entry again: put it back here.
        if (!Number.isFinite(entry) || next.every((v) => v === 0)) {
          component.value = shown(current[axis]);
        }
      });
      normal.appendChild(component);
      components.push(component);
    });

    // -- position along the normal
    const along = document.createElement("div");
    along.className = "cellier-clipping-planes-along";

    const positionLabel = document.createElement("span");
    positionLabel.className = "cellier-clipping-planes-label";
    positionLabel.textContent = "Position";

    const position = document.createElement("input");
    position.type = "range";
    position.step = "any";
    position.className = "cellier-clipping-planes-position";
    // The slider and its number are the depth into the bounding box: 0 on
    // the face the normal points away from, row.span on the opposite one.
    position.title = POSITION_TOOLTIP;
    position.dataset.role = "position";
    let dragging = false;
    position.addEventListener("input", () => {
      dragging = true;
      value.textContent = Number(position.value).toFixed(2);
      send("depth", index, Number(position.value));
    });
    position.addEventListener("change", () => { dragging = false; });

    const value = document.createElement("span");
    value.className = "cellier-clipping-planes-value";
    value.dataset.role = "value";
    along.append(positionLabel, position, value);

    // -- outline: a box that draws it, and its colour
    const outline = document.createElement("div");
    outline.className = "cellier-clipping-planes-outline";
    const outlineLabel = document.createElement("label");
    const outlineOn = document.createElement("input");
    outlineOn.type = "checkbox";
    outlineOn.title = OUTLINE_TOOLTIP;
    outlineOn.dataset.role = "outline";
    outlineOn.addEventListener("change", () => send("outline", index, outlineOn.checked));
    outlineLabel.append(outlineOn, " Outline");
    const outlineColor = document.createElement("input");
    outlineColor.type = "color";
    outlineColor.title = OUTLINE_COLOR_TOOLTIP;
    outlineColor.dataset.role = "outline-color";
    // `change`, not `input`: one edit when the picker closes, not one per
    // colour the pointer passes over.
    outlineColor.addEventListener("change", () =>
      send("outline_color", index, outlineColor.value));
    outline.append(outlineLabel, outlineColor);

    root.append(header, normal, along, outline);

    function apply(row) {
      enabled.checked = Boolean(row.enabled);
      outlineOn.checked = Boolean(row.outline && row.outline.enabled);
      if (row.outline_hex && outlineColor.value !== row.outline_hex) {
        outlineColor.value = row.outline_hex;
      }
      gizmo.style.display = row.gizmo === undefined ? "none" : "";
      gizmoOn = Boolean(row.gizmo);
      gizmo.classList.toggle("cellier-clipping-planes-on", gizmoOn);
      gizmo.setAttribute("aria-pressed", String(gizmoOn));
      gizmo.disabled = Boolean(row.gizmo_blocked);
      gizmo.title = row.gizmo_blocked || GIZMO_TOOLTIP;
      current = row.normal.map(Number);
      for (const { axis, sign, button } of facing) {
        const on = Boolean(row.facing) && row.facing[0] === axis && row.facing[1] === sign;
        button.classList.toggle("cellier-clipping-planes-on", on);
        button.setAttribute("aria-pressed", String(on));
      }
      components.forEach((component, axis) => {
        // An entry being typed already reads as the value it will send;
        // only a different number is written over it.
        if (component.value === "" || shown(component.value) !== shown(current[axis])) {
          component.value = shown(current[axis]);
        }
      });
      const span = Number(row.span);
      const depth = Number(row.depth);
      position.min = "0";
      position.max = String(span);
      // While the thumb is held the element already shows the newest value;
      // writing an older echo back would make it jump.
      // Outside the box the thumb is pinned at the nearer end and the
      // number shows how far past it the plane is.
      if (!dragging) position.value = String(Math.min(Math.max(depth, 0), span));
      value.textContent = depth.toFixed(2);
      const outside = Boolean(row.outside);
      value.classList.toggle("cellier-clipping-planes-outside", outside);
      value.title = outside ? `${OUTSIDE_TOOLTIP} ${POSITION_TOOLTIP}` : POSITION_TOOLTIP;
    }
    return { root, apply };
  }

  function draw() {
    const rows = model.get("rows");
    while (built.length > rows.length) {
      built.pop().root.remove();
    }
    while (built.length < rows.length) {
      const entry = build(built.length);
      built.push(entry);
      body.appendChild(entry.root);
    }
    rows.forEach((row, index) => built[index].apply(row));
  }

  function drawTitle() {
    heading.textContent = model.get("title");
    heading.style.display = model.get("title") ? "" : "none";
  }

  function drawError() {
    error.textContent = model.get("error");
  }

  model.on("change:rows", draw);
  model.on("change:title", drawTitle);
  // A refused edit leaves `rows` as they were: draw them over what was typed.
  model.on("change:error", () => { drawError(); draw(); });
  drawTitle();
  draw();
  drawError();
}

export default { render };
