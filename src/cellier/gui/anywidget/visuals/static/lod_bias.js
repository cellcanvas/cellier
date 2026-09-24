// LOD-bias slider ESM (split out of the former composite panel.js).
// Emits only on settled `change`, not on drag `input`, to avoid triggering a
// reslice on every tick.

function render({ model, el }) {
  el.classList.add("cellier-lod-bias");

  let guard = false;

  const row = document.createElement("div");
  row.className = "cellier-app-row";

  const label = document.createElement("label");
  label.className = "cellier-app-label";
  // The control names itself from the `title` trait -- one string, set
  // by whoever built the control, rather than one hard-coded here and
  // another in the layout that used to draw a group title around it.
  label.textContent = model.get("title") || "LOD bias";

  model.on("change:title", () => {
    label.textContent = model.get("title") || "LOD bias";
  });

  const initVal = model.get("lod_bias") ?? 1.0;
  const inp = document.createElement("input");
  inp.type = "range";
  inp.min = 0.001;
  inp.max = 5.0;
  inp.step = "any";
  inp.value = initVal;

  const readout = document.createElement("span");
  readout.className = "cellier-app-readout";
  readout.textContent = Number(initVal).toFixed(2);

  inp.addEventListener("change", () => {
    if (guard) return;
    readout.textContent = Number(inp.value).toFixed(2);
    model.set("lod_bias", parseFloat(inp.value));
    model.save_changes();
  });
  // Live readout while dragging, without emitting to the bus.
  inp.addEventListener("input", () => {
    readout.textContent = Number(inp.value).toFixed(2);
  });

  model.on("change:lod_bias", () => {
    guard = true;
    try {
      const v = model.get("lod_bias") ?? 1.0;
      inp.value = v;
      readout.textContent = Number(v).toFixed(2);
    } finally {
      guard = false;
    }
  });

  row.appendChild(label);
  row.appendChild(inp);
  row.appendChild(readout);
  el.appendChild(row);
}

export default { render };
