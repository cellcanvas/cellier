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

  let rows = {}; // axis(str) -> { row, input, readout, spec, dragging }

  // Datalist ids must be unique within the tree the input lives in (the page,
  // or marimo's shadow root), and two dims panels can share one.
  const panelKey = Math.random().toString(36).slice(2);

  // Leading + trailing throttle shared across axes (a user drags one at a
  // time).  `submit` writes immediately; `scheduleSubmit` rate-limits drags.
  let timer = null;
  let pending = null; // { axis, value } captured during the throttle window

  // World positions are floats now, and a raw float64 readout is unreadable
  // while dragging.  Whole numbers keep their bare form.
  function formatPosition(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value);
    return Number.isInteger(number) ? String(number) : number.toFixed(3);
  }

  function submit(axis, value) {
    if (guard) return;
    const current = { ...(model.get("slice_indices") || {}) };
    current[axis] = Number(value);
    model.set("slice_indices", current);
    model.save_changes();
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
        readout.textContent = formatPosition(input.value);

        input.addEventListener("input", () => {
          readout.textContent = formatPosition(input.value);
          scheduleSubmit(axis, input.value); // live, throttled
        });
        input.addEventListener("change", () => {
          // Final flush on release so the last position always lands even if
          // it arrived between throttle ticks.
          pending = null;
          submit(axis, input.value);
        });
      }

      row.appendChild(label);
      row.appendChild(input);
      row.appendChild(readout);
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
          rows[axis].readout.textContent = formatPosition(slices[axis]);
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
  model.on("change:displayed_axes", updateVisibility);
  model.on("change:slider_axes", updateVisibility);
  model.on("change:label", updateToggle);
  model.on("change:has_toggle", updateToggle);
}

export default { render };
