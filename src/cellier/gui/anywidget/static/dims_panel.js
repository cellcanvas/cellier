// Dims anywidget ESM.  Renders one labeled range slider per slider axis
// that is not displayed -- continuous over [min, max], or stepping through a discrete axis's
// listed values -- plus an optional 2D/3D toggle button.  All the axis-slice logic
// extracted from panel.js so dims can live in a separate widget below the
// canvas while appearance controls stay on the left.


function render({ model, el }) {
  el.classList.add("cellier-dims-panel");

  // Read off the model rather than declared here, so this and the Qt front
  // end coalesce slider drags at the same rate (see
  // cellier.gui._constants.DIMS_SLIDER_THROTTLE_MS).
  // `?? 50` rather than a bare read: an undefined trait would make
  // `setTimeout(tick, undefined)` fire at 0 ms, silently removing the
  // throttle and sending every value of a drag to the slicer.
  const THROTTLE_MS = model.get("throttle_ms") ?? 50;

  // Signed step_size multiples a key moves a continuous slider.
  const PAGE_STEPS = 10;
  const KEY_STEPS = {
    ArrowRight: 1,
    ArrowUp: 1,
    ArrowLeft: -1,
    ArrowDown: -1,
    PageUp: PAGE_STEPS,
    PageDown: -PAGE_STEPS,
  };

  let guard = false;

  const dimsContainer = document.createElement("div");
  dimsContainer.className = "cellier-dims";
  el.appendChild(dimsContainer);

  const toggleButton = document.createElement("button");
  toggleButton.className = "cellier-dim-toggle";
  toggleButton.textContent = model.get("label");
  toggleButton.addEventListener("click", () => {
    model.set("_clicks", model.get("_clicks") + 1);
    model.save_changes();
  });
  el.appendChild(toggleButton);

  function updateToggle() {
    toggleButton.style.display = model.get("has_toggle") ? "" : "none";
    toggleButton.textContent = model.get("label");
  }

  let rows = {}; // axis(str) -> { row, input, readout, spec, dragging, thickness }

  // Datalist ids must be unique within the tree the input lives in (the page,
  // or marimo's shadow root), and two dims panels can share one.
  const panelKey = Math.random().toString(36).slice(2);

  // Leading + trailing throttle shared across axes (a user drags one at a
  // time).  `submit` writes immediately; `scheduleSubmit` rate-limits drags.
  let timer = null;
  let pending = null; // { axis, value } captured during the throttle window

  // World positions are floats, and a raw float64 readout is unreadable while
  // dragging.  A continuous axis shows its spec's "decimals" (the Qt panel's
  // precision too); a discrete value without a label keeps its bare form when
  // whole.
  function formatPosition(value, decimals) {
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value);
    if (decimals !== undefined) return number.toFixed(decimals);
    return Number.isInteger(number) ? String(number) : number.toFixed(3);
  }

  function submit(axis, value) {
    if (guard) return;
    const current = { ...(model.get("slice_indices") || {}) };
    current[axis] = Number(value);
    model.set("slice_indices", current);
    model.save_changes();
  }

  // A half-thickness box: world units, minimum 0, where 0 is a plane.  The
  // scene's thickness is the only thickness in the slicing path, so this is
  // how much depth every visual shows along the axis.  Submitted on "change"
  // (Enter, blur or a spinner step), not on every keystroke.
  function submitThickness(axis, value) {
    if (guard) return;
    const number = Math.max(0, Number(value));
    if (!Number.isFinite(number)) return;
    const current = { ...(model.get("thickness") || {}) };
    current[axis] = number;
    model.set("thickness", current);
    model.save_changes();
  }

  function buildThickness(axis, spec, entry, row) {
    const boxed = (model.get("thickness_axes") || []).map(String);
    if (!boxed.includes(String(axis))) return;
    const sign = document.createElement("span");
    sign.className = "cellier-dim-thickness-sign";
    sign.textContent = "+/-";
    const box = document.createElement("input");
    box.type = "number";
    box.className = "cellier-dim-thickness";
    box.min = 0;
    box.step = spec.kind === "discrete" ? 1 : spec.step_size;
    box.title =
      "Half-thickness of the slice along this axis, in world units. 0 is a plane.";
    const current = (model.get("thickness") || {})[axis];
    box.value = current !== undefined ? current : 0;
    box.addEventListener("change", () => {
      if (!(Number(box.value) >= 0)) box.value = 0;
      submitThickness(axis, box.value);
    });
    row.appendChild(sign);
    row.appendChild(box);
    entry.thickness = box;
  }

  // A dims interaction scope, held while a slider is pressed, so the scrub
  // ends on release instead of waiting for stillness.  One at a time: a user
  // presses one slider.
  //
  // The end message CARRIES the final position.  It cannot rely on the
  // slice_indices write that precedes it: a custom message can overtake a
  // traitlet sync (in Jupyter while the kernel is busy, in marimo always),
  // and Python would then end the scrub on a position the slider has left.
  let scope = null; // { axis, value: () => world value, entry }

  function beginScope(axis, entry, value) {
    if (scope !== null) endScope();
    scope = { axis, entry, value };
    model.send({ type: "interaction", phase: "begin" });
    // On the window, not the input: the pointer is often released off the
    // slider, and a press with no move fires no "change".
    window.addEventListener("pointerup", endScope, true);
    window.addEventListener("pointercancel", endScope, true);
  }

  function endScope() {
    if (scope === null) return;
    const { axis, entry, value } = scope;
    scope = null;
    window.removeEventListener("pointerup", endScope, true);
    window.removeEventListener("pointercancel", endScope, true);
    entry.dragging = false;
    // What the throttle still holds is superseded by the final position.
    pending = null;
    model.send({
      type: "interaction",
      phase: "end",
      slice_indices: { [axis]: Number(value()) },
    });
  }

  function scheduleSubmit(axis, value) {
    if (guard) return;
    if (timer === null) {
      submit(axis, value); // leading edge
      pending = null;
      timer = setTimeout(function tick() {
        if (pending !== null) {
          submit(pending.axis, pending.value);
          pending = null;
          timer = setTimeout(tick, THROTTLE_MS);
        } else {
          timer = null;
        }
      }, THROTTLE_MS);
    } else {
      pending = { axis, value }; // coalesce to the latest within the window
    }
  }

  // Readout for a discrete axis at slider position `position`.
  function discreteReadout(spec, position) {
    if (spec.labels) return spec.labels[position];
    return formatPosition(spec.values[position]);
  }

  function build() {
    dimsContainer.innerHTML = "";
    rows = {};
    const specs = model.get("axis_values") || {};
    const labels = model.get("axis_labels") || {};
    const slices = model.get("slice_indices") || {};

    for (const axis of Object.keys(specs)) {
      const spec = specs[axis];
      const discrete = spec.kind === "discrete";

      const row = document.createElement("div");
      row.className = "cellier-dim-row";

      const label = document.createElement("label");
      label.className = "cellier-dim-label";
      label.textContent = labels[axis] !== undefined ? labels[axis] : axis;

      const input = document.createElement("input");
      input.type = "range";
      const readout = document.createElement("span");
      readout.className = "cellier-dim-readout";
      const entry = { row, input, readout, spec, dragging: false };

      if (discrete) {
        // The slider steps through positions 0..N-1; what reaches the model
        // is always the world value at that position.
        input.min = 0;
        input.max = spec.values.length - 1;
        input.step = 1;
        // Only when the axis asks for them: one <option> per value is a DOM
        // node per sample, and the browser redraws every mark as the slider
        // moves.  See TICK_WARNING_LIMIT in cellier.gui._axis_values.
        if (spec.draw_ticks) {
          const ticks = document.createElement("datalist");
          ticks.id = `cellier-dim-ticks-${panelKey}-${axis}`;
          for (let position = 0; position < spec.values.length; position++) {
            const option = document.createElement("option");
            option.value = position;
            ticks.appendChild(option);
          }
          row.appendChild(ticks);
          input.setAttribute("list", ticks.id);
        }
        input.value = 0;
        readout.textContent = discreteReadout(spec, 0);

        // While the handle is held, positions echoed back from Python lag the
        // drag; applying them would pull the handle backwards mid-gesture.
        input.addEventListener("pointerdown", () => {
          entry.dragging = true;
          beginScope(axis, entry, () => spec.values[Number(input.value)]);
        });
        input.addEventListener("pointercancel", () => {
          entry.dragging = false;
        });
        input.addEventListener("input", () => {
          const position = Number(input.value);
          readout.textContent = discreteReadout(spec, position);
          scheduleSubmit(axis, spec.values[position]); // live, throttled
        });
        input.addEventListener("change", () => {
          entry.dragging = false;
          pending = null;
          endScope(); // if the release did not already
          submit(axis, spec.values[Number(input.value)]);
        });
      } else {
        input.min = spec.min;
        input.max = spec.max;
        // A slice position is a world coordinate, not a voxel index, so the
        // slider is continuous: on a 0.5 world-unit-per-voxel axis an integer
        // step cannot reach the odd-numbered planes.
        input.step = "any";
        input.value = slices[axis] !== undefined ? slices[axis] : spec.min;
        readout.textContent = formatPosition(input.value, spec.decimals);

        input.addEventListener("pointerdown", () => {
          beginScope(axis, entry, () => input.value);
        });
        input.addEventListener("input", () => {
          readout.textContent = formatPosition(input.value, spec.decimals);
          scheduleSubmit(axis, input.value); // live, throttled
        });
        // With step "any" the browser picks its own keyboard step, so the
        // arrow and page keys are handled here: one step_size per arrow,
        // PAGE_STEPS of them per page key (the Qt panel's steps too).  Home
        // and End keep the browser's jump to the ends.
        input.addEventListener("keydown", (event) => {
          const steps = KEY_STEPS[event.key];
          if (steps === undefined) return;
          event.preventDefault();
          const target = Number(input.value) + steps * spec.step_size;
          const clamped = Math.min(spec.max, Math.max(spec.min, target));
          if (clamped === Number(input.value)) return;
          input.value = clamped;
          readout.textContent = formatPosition(input.value, spec.decimals);
          scheduleSubmit(axis, input.value); // live, throttled
        });
        input.addEventListener("change", () => {
          // Final flush on release so the last position always lands even if
          // it arrived between throttle ticks.
          pending = null;
          endScope(); // if the release did not already
          submit(axis, input.value);
        });
      }

      row.appendChild(label);
      row.appendChild(input);
      row.appendChild(readout);
      buildThickness(axis, spec, entry, row);
      dimsContainer.appendChild(row);
      rows[axis] = entry;
    }
    updateVisibility();
    syncValues();
    syncDiscrete();
  }

  function updateVisibility() {
    const displayed = (model.get("displayed_axes") || []).map(String);
    // null means every axis gets a slider (a panel built without a scene).
    const sliderAxes = model.get("slider_axes");
    const sliders = sliderAxes == null ? null : sliderAxes.map(String);
    for (const axis of Object.keys(rows)) {
      const hidden =
        displayed.includes(axis) || (sliders !== null && !sliders.includes(axis));
      rows[axis].row.style.display = hidden ? "none" : "";
    }
  }

  // Continuous rows follow slice_indices directly.  Discrete rows follow
  // discrete_index instead, which Python derives from slice_indices: a
  // position between two listed values is resolved there, not here.
  function syncValues() {
    guard = true;
    try {
      const slices = model.get("slice_indices") || {};
      for (const axis of Object.keys(rows)) {
        if (rows[axis].spec.kind === "discrete") continue;
        if (Object.prototype.hasOwnProperty.call(slices, axis)) {
          rows[axis].input.value = slices[axis];
          rows[axis].readout.textContent = formatPosition(slices[axis], rows[axis].spec.decimals);
        }
      }
    } finally {
      guard = false;
    }
  }

  function syncThickness() {
    guard = true;
    try {
      const thickness = model.get("thickness") || {};
      for (const axis of Object.keys(rows)) {
        const box = rows[axis].thickness;
        if (!box || document.activeElement === box) continue;
        if (Object.prototype.hasOwnProperty.call(thickness, axis)) {
          box.value = thickness[axis];
        }
      }
    } finally {
      guard = false;
    }
  }

  function syncDiscrete() {
    guard = true;
    try {
      const positions = model.get("discrete_index") || {};
      for (const axis of Object.keys(rows)) {
        const entry = rows[axis];
        if (entry.spec.kind !== "discrete" || entry.dragging) continue;
        if (Object.prototype.hasOwnProperty.call(positions, axis)) {
          entry.input.value = positions[axis];
          entry.readout.textContent = discreteReadout(entry.spec, positions[axis]);
        }
      }
    } finally {
      guard = false;
    }
  }

  build();
  updateToggle();
  model.on("change:axis_values", build);
  model.on("change:axis_labels", build);
  model.on("change:slice_indices", syncValues);
  model.on("change:discrete_index", syncDiscrete);
  model.on("change:thickness", syncThickness);
  model.on("change:thickness_axes", build);
  model.on("change:displayed_axes", updateVisibility);
  model.on("change:slider_axes", updateVisibility);
  model.on("change:label", updateToggle);
  model.on("change:has_toggle", updateToggle);
}

export default { render };
