// Trail-window control ESM.  One block per offered data axis: whether the
// axis has a window, how far it reaches before and after the slice position,
// and whether it fades.
//
// State is one `windows` dict keyed by the axis index as a string (JSON object
// keys are strings), so an edit writes one entry and the Python side can tell
// which axis changed.  Labels and tooltips arrive in `text`, read from the
// same module the Qt control reads.

function render({ model, el }) {
  el.classList.add("cellier-trail-widget");

  // Set while pushing model values into inputs, so their change handlers do
  // not write them straight back.
  let guard = false;

  const heading = document.createElement("div");
  heading.className = "cellier-group-title";
  el.appendChild(heading);

  const body = document.createElement("div");
  body.className = "cellier-trail-body";
  el.appendChild(body);

  function applyTitle() {
    heading.textContent = model.get("title") || "Trail";
  }

  function windows() {
    return model.get("windows") || {};
  }

  function text(kind, field) {
    return ((model.get("text") || {})[kind] || {})[field] || "";
  }

  function write(key, field, value) {
    if (guard) return;
    const current = windows();
    model.set("windows", {
      ...current,
      [key]: { ...(current[key] || {}), [field]: value },
    });
    model.save_changes();
  }

  function checkRow(labelText, tooltip) {
    const row = document.createElement("div");
    row.className = "cellier-app-row cellier-trail-check";
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    if (tooltip) label.title = tooltip;
    label.appendChild(input);
    label.appendChild(document.createTextNode(" " + labelText));
    row.appendChild(label);
    return { row, input };
  }

  function numberRow(labelText, tooltip) {
    const row = document.createElement("div");
    row.className = "cellier-app-row";
    const label = document.createElement("label");
    label.className = "cellier-app-label";
    label.textContent = labelText;
    const input = document.createElement("input");
    input.type = "number";
    input.className = "cellier-trail-number";
    input.min = "0";
    input.step = "any";
    if (tooltip) {
      label.title = tooltip;
      input.title = tooltip;
    }
    row.appendChild(label);
    row.appendChild(input);
    return { row, input };
  }

  let appliers = [];

  function build() {
    body.textContent = "";
    appliers = [];
    for (const { axis, label } of model.get("axes") || []) {
      const key = String(axis);
      const block = document.createElement("div");
      block.className = "cellier-trail-axis";

      const enabled = checkRow(label, text("tooltips", "enabled"));
      const before = numberRow(text("labels", "before"), text("tooltips", "before"));
      const after = numberRow(text("labels", "after"), text("tooltips", "after"));
      const fade = checkRow(text("labels", "fade"), text("tooltips", "fade"));

      function apply() {
        const state = windows()[key] || {};
        const on = Boolean(state.enabled);
        enabled.input.checked = on;
        before.input.value = String(state.before ?? 0);
        after.input.value = String(state.after ?? 0);
        fade.input.checked = Boolean(state.fade);
        before.input.disabled = !on;
        after.input.disabled = !on;
        fade.input.disabled = !on;
      }

      function extent(input, field) {
        // `change` commits on Enter or blur, so a typed "4800" is one write.
        // An empty or negative entry is refused by putting the model back.
        const value = parseFloat(input.value);
        if (Number.isFinite(value) && value >= 0) write(key, field, value);
        else apply();
      }

      enabled.input.addEventListener("change", () =>
        write(key, "enabled", enabled.input.checked)
      );
      before.input.addEventListener("change", () => extent(before.input, "before"));
      after.input.addEventListener("change", () => extent(after.input, "after"));
      fade.input.addEventListener("change", () =>
        write(key, "fade", fade.input.checked)
      );

      block.appendChild(enabled.row);
      block.appendChild(before.row);
      block.appendChild(after.row);
      block.appendChild(fade.row);
      body.appendChild(block);

      apply();
      appliers.push(apply);
    }
  }

  applyTitle();
  build();

  model.on("change:title", applyTitle);
  model.on("change:axes", build);
  model.on("change:text", build);
  model.on("change:windows", () => {
    guard = true;
    try {
      for (const apply of appliers) apply();
    } finally {
      guard = false;
    }
  });
}

export default { render };
