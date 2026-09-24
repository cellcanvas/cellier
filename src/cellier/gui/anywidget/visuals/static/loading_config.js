// Loading-settings ESM: one row per ProgressiveLoadingConfig field.  The
// Python side syncs `fields` (what to draw) and `config` (the values); an
// edit is reported by setting `edit`, and a refused edit comes back as
// `error` with `config` unchanged, which redraws the old values.

function render({ model, el }) {
  el.classList.add("cellier-loading-config");
  let guard = false;
  let serial = 0;

  const heading = document.createElement("div");
  heading.className = "cellier-loading-config-title";
  el.appendChild(heading);

  const body = document.createElement("div");
  body.className = "cellier-loading-config-body";
  el.appendChild(body);

  const error = document.createElement("div");
  error.className = "cellier-loading-config-error";
  el.appendChild(error);

  // field name -> function showing a value in its input
  const appliers = {};

  function send(field, value) {
    if (guard) return;
    serial += 1;
    model.set("edit", { field, value, serial });
    model.save_changes();
  }

  for (const spec of model.get("fields")) {
    const row = document.createElement("div");
    row.className = "cellier-app-row";
    const label = document.createElement("label");
    label.className = "cellier-app-label";
    label.textContent = spec.label;
    row.appendChild(label);

    let input;
    if (spec.kind === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.addEventListener("change", () => send(spec.name, input.checked));
      appliers[spec.name] = (v) => { input.checked = Boolean(v); };
    } else if (spec.kind === "choice") {
      input = document.createElement("select");
      for (const choice of spec.choices) {
        const option = document.createElement("option");
        option.value = choice;
        option.textContent = choice;
        input.appendChild(option);
      }
      input.addEventListener("change", () => send(spec.name, input.value));
      appliers[spec.name] = (v) => { input.value = v; };
    } else if (spec.kind === "level") {
      // 0 is "coarsest" (None); a select keeps that readable.
      input = document.createElement("select");
      for (let level = spec.minimum; level <= spec.maximum; level += 1) {
        const option = document.createElement("option");
        option.value = String(level);
        option.textContent = level === 0 ? model.get("coarsest_text") : String(level);
        input.appendChild(option);
      }
      input.addEventListener("change", () => send(spec.name, parseInt(input.value, 10)));
      appliers[spec.name] = (v) => { input.value = String(v); };
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.min = spec.minimum;
      input.max = spec.maximum;
      input.step = spec.step;
      input.addEventListener("change", () => send(spec.name, parseFloat(input.value)));
      appliers[spec.name] = (v) => { input.value = Number(v).toFixed(2); };
    }
    input.className = "cellier-loading-config-input";
    row.appendChild(input);
    body.appendChild(row);
  }

  function update() {
    guard = true;
    try {
      heading.textContent = model.get("title") || "Progressive loading";
      const config = model.get("config") || {};
      for (const [name, apply] of Object.entries(appliers)) {
        if (name in config) apply(config[name]);
      }
      const message = model.get("error") || "";
      error.textContent = message;
      error.style.display = message ? "" : "none";
    } finally {
      guard = false;
    }
  }

  model.on("change:config", update);
  model.on("change:error", update);
  model.on("change:title", update);
  update();
}

export default { render };
