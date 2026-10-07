// Render planes ESM: one row per entry of `rows`.  The Python side syncs
// `rows` (what to draw), `blocked` (why the control is disabled, or "") and
// `can_add`; an action is reported by setting `edit`, and a refused one
// comes back as `error` with `rows` unchanged.
//
// A row is a clipping planes row with the extents added.  It is built once
// and then updated in place, so a slider is not destroyed while it is
// dragged.  Rows are only added or removed when the number of planes
// changes.  A plane has three axes, named by the row's `axes`.

const GIZMO_TOOLTIP =
  "Drag a gizmo in the 3D view to move, turn and resize this plane.";
const SIDES = ["min", "max"];

// A number as an input shows it: at most three decimals.
function shown(value) {
  return String(Number(Number(value).toFixed(3)));
}

function render({ model, el }) {
  el.classList.add("cellier-render-planes");
  let serial = 0;

  const heading = document.createElement("div");
  heading.className = "cellier-render-planes-title";
  el.appendChild(heading);

  const reason = document.createElement("div");
  reason.className = "cellier-render-planes-reason";
  reason.dataset.role = "reason";
  el.appendChild(reason);

  const body = document.createElement("div");
  body.className = "cellier-render-planes-rows";
  el.appendChild(body);

  const add = document.createElement("button");
  add.className = "cellier-render-planes-add";
  add.textContent = "Add plane";
  add.title =
    "Add a plane through the middle of the data, facing the first displayed " +
    "axis. A visual draws at most four.";
  add.dataset.role = "add";
  el.appendChild(add);

  const error = document.createElement("div");
  error.className = "cellier-render-planes-error";
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
    const root = document.createElement("div");
    root.className = "cellier-render-planes-row";
    root.dataset.role = "row";
    root.dataset.index = String(index);

    // -- header: on or off, gizmo, flip, remove
    const header = document.createElement("div");
    header.className = "cellier-render-planes-header";

    const enabledLabel = document.createElement("label");
    const enabled = document.createElement("input");
    enabled.type = "checkbox";
    enabled.title = "Whether this plane is drawn. It stays in the list.";
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
    flip.title =
      "Reverse the normal. A plane is drawn the same from both sides, so the " +
      "picture does not change.";
    flip.dataset.role = "flip";
    flip.addEventListener("click", () => send("flip", index, null));

    const remove = document.createElement("button");
    remove.textContent = "Remove";
    remove.dataset.role = "remove";
    remove.addEventListener("click", () => send("remove", index, null));

    const spacer = document.createElement("span");
    spacer.className = "cellier-render-planes-spacer";
    header.append(enabledLabel, spacer, gizmo, flip, remove);

    // -- normal: one column per axis of the plane, two buttons over an entry
    const normal = document.createElement("div");
    normal.className = "cellier-render-planes-normal";
    normal.style.gridTemplateColumns = "auto repeat(3, minmax(0, 1fr))";

    const normalLabel = document.createElement("span");
    normalLabel.className = "cellier-render-planes-label";
    normalLabel.textContent = "Normal";
    normalLabel.title =
      "The plane's normal, one entry per world axis of the plane. Only its " +
      "direction matters. A button faces the plane along its axis.";
    normal.appendChild(normalLabel);

    // The normal last drawn, to refuse an all-zero one without a round trip.
    let current = [0, 0, 0];
    const facing = [];  // { axis, sign, button }
    const components = [];
    for (let axis = 0; axis < 3; axis += 1) {
      const pair = document.createElement("div");
      pair.className = "cellier-render-planes-facing";
      pair.style.gridColumn = String(axis + 2);
      for (const sign of [1, -1]) {
        const button = document.createElement("button");
        button.dataset.role = "facing";
        button.dataset.axis = String(axis);
        button.dataset.sign = String(sign);
        button.addEventListener("click", () => send("facing", index, [axis, sign]));
        pair.appendChild(button);
        facing.push({ axis, sign, button });
      }
      normal.appendChild(pair);
    }
    for (let axis = 0; axis < 3; axis += 1) {
      const component = document.createElement("input");
      component.type = "number";
      component.step = "0.1";
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
    }

    // -- position along the normal
    const along = document.createElement("div");
    along.className = "cellier-render-planes-along";

    const positionLabel = document.createElement("span");
    positionLabel.className = "cellier-render-planes-label";
    positionLabel.textContent = "Position";

    const value = document.createElement("span");
    value.className = "cellier-render-planes-value";
    value.dataset.role = "value";

    const position = document.createElement("input");
    position.type = "range";
    position.step = "any";
    position.className = "cellier-render-planes-position";
    position.title = "Where the plane sits along its normal. World units.";
    position.dataset.role = "position";
    let dragging = false;
    position.addEventListener("input", () => {
      dragging = true;
      value.textContent = Number(position.value).toFixed(2);
      send("position", index, Number(position.value));
    });
    position.addEventListener("change", () => { dragging = false; });

    along.append(positionLabel, position, value);

    // -- extents: one line per in-plane axis, a min and a max, each with an
    // "unbounded" box
    const sides = [];  // { axis, side, input, box, last }
    const extents = [];
    for (let axis = 0; axis < 2; axis += 1) {
      const line = document.createElement("div");
      line.className = "cellier-render-planes-extent";
      const label = document.createElement("span");
      label.className = "cellier-render-planes-label";
      label.textContent = `Extent ${axis}`;
      label.title =
        `How far the plane reaches from its origin along in-plane axis ${axis}, ` +
        "in world units. An unbounded side ends at the data.";
      line.appendChild(label);
      for (let side = 0; side < 2; side += 1) {
        const group = document.createElement("span");
        group.className = "cellier-render-planes-side";
        const name = document.createElement("span");
        name.textContent = SIDES[side];
        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.title = `The ${SIDES[side]} along in-plane axis ${axis}.`;
        input.dataset.role = "side";
        input.dataset.axis = String(axis);
        input.dataset.side = String(side);
        const entry = { axis, side, input, box: null, last: null };
        input.addEventListener("change", () => {
          const amount = Number(input.value);
          if (input.value === "" || !Number.isFinite(amount)) {
            // Unreadable: nothing is sent, so put the last value back.
            input.value = entry.last === null ? "" : shown(entry.last);
            return;
          }
          send("side", index, [axis, side, amount]);
        });
        const boxLabel = document.createElement("label");
        const box = document.createElement("input");
        box.type = "checkbox";
        box.title =
          `No ${SIDES[side]}: the plane ends at the data on this side. ` +
          "Unticking fills in where the data ends now.";
        box.dataset.role = "unbounded";
        box.dataset.axis = String(axis);
        box.dataset.side = String(side);
        box.addEventListener("change", () =>
          send("bounded", index, [axis, side, !box.checked]));
        boxLabel.append(box, " unbounded");
        entry.box = box;
        group.append(name, input, boxLabel);
        line.appendChild(group);
        sides.push(entry);
      }
      extents.push(line);
    }

    root.append(header, normal, along, ...extents);

    function apply(row) {
      enabled.checked = Boolean(row.enabled);
      gizmo.style.display = row.gizmo === undefined ? "none" : "";
      gizmoOn = Boolean(row.gizmo);
      gizmo.classList.toggle("cellier-render-planes-on", gizmoOn);
      gizmo.setAttribute("aria-pressed", String(gizmoOn));
      gizmo.disabled = Boolean(row.gizmo_blocked);
      gizmo.title = row.gizmo_blocked || GIZMO_TOOLTIP;
      current = row.normal.map(Number);
      for (const { axis, sign, button } of facing) {
        const name = row.axes[axis];
        button.textContent = `${sign > 0 ? "+" : "-"}${name}`;
        button.title = `Face the plane along ${name}.`;
        const on = Boolean(row.facing) && row.facing[0] === axis && row.facing[1] === sign;
        button.classList.toggle("cellier-render-planes-on", on);
        button.setAttribute("aria-pressed", String(on));
      }
      components.forEach((component, axis) => {
        component.title = `The normal's entry on ${row.axes[axis]}.`;
        // An entry being typed already reads as the value it will send;
        // only a different number is written over it.
        if (component.value === "" || shown(component.value) !== shown(current[axis])) {
          component.value = shown(current[axis]);
        }
      });
      position.min = String(row.low);
      position.max = String(row.high);
      // While the thumb is held the element already shows the newest value;
      // writing an older echo back would make it jump.
      if (!dragging) position.value = String(row.position);
      value.textContent = Number(row.position).toFixed(2);
      for (const entry of sides) {
        const amount = row[`extent_${entry.axis}`][entry.side];
        const unbounded = amount === null || amount === undefined;
        entry.last = unbounded ? null : Number(amount);
        entry.box.checked = unbounded;
        entry.input.disabled = unbounded;
        if (unbounded) {
          entry.input.value = "";
        } else if (entry.input.value === "" || shown(entry.input.value) !== shown(amount)) {
          entry.input.value = shown(amount);
        }
      }
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

  function drawState() {
    const blocked = model.get("blocked");
    reason.textContent = blocked;
    reason.style.display = blocked ? "" : "none";
    body.classList.toggle("cellier-render-planes-blocked", Boolean(blocked));
    body.inert = Boolean(blocked);
    add.disabled = !model.get("can_add");
  }

  function drawError() {
    error.textContent = model.get("error");
  }

  model.on("change:rows", draw);
  model.on("change:title", drawTitle);
  model.on("change:blocked", drawState);
  model.on("change:can_add", drawState);
  // A refused edit leaves `rows` as they were: draw them over what was typed.
  model.on("change:error", () => { drawError(); draw(); });
  drawTitle();
  draw();
  drawState();
  drawError();
}

export default { render };
