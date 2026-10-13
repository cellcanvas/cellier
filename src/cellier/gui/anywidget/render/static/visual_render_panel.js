// Per-visual render-settings ESM.  One file drives all four panels -- the
// outline slot picker, the labels variant with its per-label rows and its
// depth range, the occlusion menu and the picking checkbox -- from a
// `controls` description the Python side syncs.
//
// State is a single `values` dict keyed by the dotted field name, rather
// than a trait per field: these panels are small and fixed, and it avoids
// mangling `outline.slot` into a legal trait name.

function render({ model, el }) {
  el.classList.add("cellier-render-panel");

  // Set while pushing a model value into an input, so the input's own
  // change handler does not write it straight back.
  let guard = false;

  const heading = document.createElement("div");
  heading.className = "cellier-render-title";
  heading.textContent = model.get("title");
  el.appendChild(heading);

  const body = document.createElement("div");
  body.className = "cellier-render-body";
  el.appendChild(body);

  function values() {
    return model.get("values") || {};
  }

  function setValue(field, value) {
    if (guard) return;
    model.set("values", { ...values(), [field]: value });
    model.save_changes();
  }

  function makeRow(spec, control) {
    const row = document.createElement("div");
    row.className = "cellier-render-row";
    const label = document.createElement("label");
    label.className = "cellier-render-label";
    label.textContent = spec.label;
    if (spec.tooltip) label.title = spec.tooltip;
    row.appendChild(label);
    row.appendChild(control);
    return row;
  }

  function toHex(rgba) {
    const byte = (c) =>
      Math.round(Math.min(1, Math.max(0, c)) * 255).toString(16).padStart(2, "0");
    return "#" + byte(rgba[0]) + byte(rgba[1]) + byte(rgba[2]);
  }

  // --- controls ----------------------------------------------------------

  function buildBool(spec) {
    const input = document.createElement("input");
    input.type = "checkbox";
    // The labels outline toggle drives `outline.slot`, where 1 is "eligible"
    // and 0 is "not outlined" -- so a truthy slot is a checked box.
    const isSlot = spec.field === "outline.slot";
    input.checked = Boolean(values()[spec.field]);
    if (spec.tooltip) input.title = spec.tooltip;
    input.addEventListener("change", () =>
      setValue(spec.field, isSlot ? (input.checked ? 1 : 0) : input.checked)
    );

    const row = document.createElement("div");
    row.className = "cellier-render-row cellier-render-check";
    const label = document.createElement("label");
    label.className = "cellier-render-checklabel";
    if (spec.tooltip) label.title = spec.tooltip;
    label.appendChild(input);
    label.appendChild(document.createTextNode(" " + spec.label));
    row.appendChild(label);
    return { row, apply: (v) => { input.checked = Boolean(v); } };
  }

  function buildChoice(spec) {
    const select = document.createElement("select");
    (spec.options || []).forEach(([text], index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = text;
      select.appendChild(option);
    });
    if (spec.tooltip) select.title = spec.tooltip;

    // Matched by index, not by value: `null` (Auto) does not survive a
    // round trip through a select's string values.
    function indexOf(value) {
      const options = spec.options || [];
      for (let i = 0; i < options.length; i++) {
        if (options[i][1] === value) return i;
      }
      return 0;
    }
    select.value = String(indexOf(values()[spec.field]));
    select.addEventListener("change", () =>
      setValue(spec.field, spec.options[Number(select.value)][1])
    );
    return {
      row: makeRow(spec, select),
      apply: (v) => { select.value = String(indexOf(v)); },
    };
  }

  function buildSlot(spec) {
    // A row of swatches drawn from the live palette, plus Off.  Offering
    // only the slots that exist is what stops a user choosing one the
    // palette cannot colour.
    const wrap = document.createElement("div");
    wrap.className = "cellier-render-swatches";
    if (spec.tooltip) wrap.title = spec.tooltip;

    function draw() {
      wrap.textContent = "";
      const current = Number(values()[spec.field] || 0);

      const off = document.createElement("button");
      off.type = "button";
      off.className =
        "cellier-render-slot-off" + (current === 0 ? " cellier-render-chosen" : "");
      off.textContent = "Off";
      off.title = "Not outlined";
      off.addEventListener("click", () => setValue(spec.field, 0));
      wrap.appendChild(off);

      (model.get("palette") || []).forEach((rgba, index) => {
        const slot = index + 1;
        const button = document.createElement("button");
        button.type = "button";
        button.className =
          "cellier-render-swatch" + (current === slot ? " cellier-render-chosen" : "");
        button.style.backgroundColor = toHex(rgba);
        button.title = "Slot " + slot;
        button.addEventListener("click", () => setValue(spec.field, slot));
        wrap.appendChild(button);
      });
    }

    draw();
    return { row: makeRow(spec, wrap), apply: draw, redrawOnPalette: draw };
  }

  function buildRange(spec) {
    // Two number boxes for a pair [first, second], written as one value.
    // The second is never below the first: raising the first carries the
    // second up with it, and the second cannot be set under the first.
    const wrap = document.createElement("div");
    wrap.className = "cellier-render-range";
    if (spec.tooltip) wrap.title = spec.tooltip;

    const inputs = (spec.labels || ["", ""]).map((text) => {
      const label = document.createElement("label");
      label.className = "cellier-render-range-part";
      label.appendChild(document.createTextNode(text + " "));
      const input = document.createElement("input");
      input.type = "number";
      input.min = String(spec.minimum);
      input.step = String(spec.step);
      label.appendChild(input);
      wrap.appendChild(label);
      return input;
    });
    inputs[0].max = String(spec.first_maximum);
    inputs[1].max = String(spec.maximum);

    function show(pair) {
      if (!pair) return;
      inputs[0].value = Number(pair[0]).toFixed(spec.decimals);
      inputs[1].value = Number(pair[1]).toFixed(spec.decimals);
    }

    function commit() {
      const shown = values()[spec.field] || [spec.minimum, spec.minimum];
      // An emptied box is not a number (Number("") is 0): keep what it had.
      const read = (input) =>
        input.value.trim() === "" ? NaN : Number(input.value);
      let first = read(inputs[0]);
      let second = read(inputs[1]);
      if (!Number.isFinite(first)) first = Number(shown[0]);
      if (!Number.isFinite(second)) second = Number(shown[1]);
      first = Math.min(Math.max(first, spec.minimum), spec.first_maximum);
      second = Math.min(Math.max(second, first), spec.maximum);
      show([first, second]);
      setValue(spec.field, [first, second]);
    }

    inputs.forEach((input) => input.addEventListener("change", commit));
    show(values()[spec.field]);
    return { row: makeRow(spec, wrap), apply: show };
  }

  function buildLabelSelection(spec) {
    // One row per selected label: its value, the slot swatch it draws in,
    // and a remove button.
    const host = document.createElement("div");
    host.className = "cellier-render-label-rows";
    if (spec.tooltip) host.title = spec.tooltip;

    const rows = document.createElement("div");
    host.appendChild(rows);

    const addButton = document.createElement("button");
    addButton.type = "button";
    addButton.className = "cellier-render-action";
    addButton.textContent = "Add label";
    host.appendChild(addButton);

    function selection() {
      return { ...(values()[spec.field] || {}) };
    }

    function commit(next) {
      setValue(spec.field, next);
    }

    addButton.addEventListener("click", () => {
      const current = selection();
      const keys = Object.keys(current).map(Number);
      if (keys.length >= spec.max_rows) return;
      const next = keys.length ? Math.max(...keys) + 1 : 1;
      current[String(next)] = 1;
      commit(current);
    });

    function draw() {
      rows.textContent = "";
      const current = selection();
      const palette = model.get("palette") || [];
      Object.keys(current)
        .map(Number)
        .sort((a, b) => a - b)
        .forEach((labelValue) => {
          const slot = Number(current[String(labelValue)]);
          const row = document.createElement("div");
          row.className = "cellier-render-label-row";

          const value = document.createElement("input");
          value.type = "number";
          value.min = "0";
          value.value = String(labelValue);
          value.title = "Label value";
          value.addEventListener("change", () => {
            const renamed = selection();
            const next = String(Number(value.value));
            if (next in renamed) return;
            renamed[next] = renamed[String(labelValue)];
            delete renamed[String(labelValue)];
            commit(renamed);
          });
          row.appendChild(value);

          const swatch = document.createElement("button");
          swatch.type = "button";
          swatch.className = "cellier-render-swatch";
          swatch.style.backgroundColor =
            slot >= 1 && slot <= palette.length ? toHex(palette[slot - 1]) : "transparent";
          swatch.title = "Slot " + slot + " -- click to cycle";
          swatch.addEventListener("click", () => {
            if (!palette.length) return;
            const cycled = selection();
            cycled[String(labelValue)] = (slot % palette.length) + 1;
            commit(cycled);
          });
          row.appendChild(swatch);

          const remove = document.createElement("button");
          remove.type = "button";
          remove.className = "cellier-render-remove";
          remove.textContent = "x";
          remove.title = "Stop outlining this label";
          remove.addEventListener("click", () => {
            const shrunk = selection();
            delete shrunk[String(labelValue)];
            commit(shrunk);
          });
          row.appendChild(remove);

          rows.appendChild(row);
        });
      addButton.disabled = Object.keys(current).length >= spec.max_rows;
    }

    draw();
    return { row: makeRow(spec, host), apply: draw, redrawOnPalette: draw };
  }

  // --- assembly ----------------------------------------------------------

  let built = [];

  function build() {
    body.textContent = "";
    built = [];
    for (const spec of model.get("controls") || []) {
      let made;
      if (spec.kind === "slot") made = buildSlot(spec);
      else if (spec.kind === "choice") made = buildChoice(spec);
      else if (spec.kind === "label_selection") made = buildLabelSelection(spec);
      else if (spec.kind === "range") made = buildRange(spec);
      else made = buildBool(spec);
      built.push({ spec, ...made });
      body.appendChild(made.row);
    }
  }

  build();

  model.on("change:values", () => {
    guard = true;
    try {
      for (const item of built) item.apply(values()[item.spec.field]);
    } finally {
      guard = false;
    }
  });
  model.on("change:palette", () => {
    for (const item of built) {
      if (item.redrawOnPalette) item.redrawOnPalette();
    }
  });
  model.on("change:controls", build);
  model.on("change:title", () => {
    heading.textContent = model.get("title");
  });
}

export default { render };
