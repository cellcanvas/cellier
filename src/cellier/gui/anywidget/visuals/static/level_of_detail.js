// "Level of detail" control ESM: a settled bias slider and one "coarsest
// while moving" checkbox.  The checkbox is bound to the 3D setting while the
// view displays three dimensions and to the 2D setting otherwise.
// The bias is sent on a settled `change`, not on drag `input`: it reslices.

function movingField(model) {
  return model.get("n_displayed_dimensions") === 3
    ? "coarsest_while_moving_3d"
    : "coarsest_while_moving_2d";
}

function render({ model, el }) {
  el.classList.add("cellier-level-of-detail");

  let guard = false;
  const labels = model.get("labels") || {};

  const title = document.createElement("div");
  title.className = "cellier-group-title";
  title.textContent = model.get("title") || "Level of detail";
  el.appendChild(title);
  model.on("change:title", () => {
    title.textContent = model.get("title") || "Level of detail";
  });

  // -- settled bias ---------------------------------------------------------
  const biasRow = document.createElement("div");
  biasRow.className = "cellier-app-row";
  const biasLabel = document.createElement("label");
  biasLabel.className = "cellier-app-label";
  biasLabel.textContent = labels.bias || "Settled bias";

  const range = model.get("bias_range") || [0.001, 5.0];
  const initVal = model.get("settled_lod_bias") ?? 1.0;
  const inp = document.createElement("input");
  inp.type = "range";
  inp.min = range[0];
  inp.max = range[1];
  inp.step = "any";
  inp.value = initVal;

  const readout = document.createElement("span");
  readout.className = "cellier-app-readout";
  readout.textContent = Number(initVal).toFixed(2);

  inp.addEventListener("change", () => {
    if (guard) return;
    readout.textContent = Number(inp.value).toFixed(2);
    model.set("settled_lod_bias", parseFloat(inp.value));
    model.save_changes();
  });
  // Live readout while dragging, without emitting to the bus.
  inp.addEventListener("input", () => {
    readout.textContent = Number(inp.value).toFixed(2);
  });
  model.on("change:settled_lod_bias", () => {
    guard = true;
    try {
      const v = model.get("settled_lod_bias") ?? 1.0;
      inp.value = v;
      readout.textContent = Number(v).toFixed(2);
    } finally {
      guard = false;
    }
  });

  biasRow.appendChild(biasLabel);
  biasRow.appendChild(inp);
  biasRow.appendChild(readout);
  el.appendChild(biasRow);

  // -- coarsest while moving ------------------------------------------------
  const movingRow = document.createElement("label");
  movingRow.className = "cellier-app-row cellier-lod-moving";
  const check = document.createElement("input");
  check.type = "checkbox";
  const movingText = document.createElement("span");
  movingText.textContent = labels.moving || "Coarsest while moving";
  movingRow.appendChild(check);
  movingRow.appendChild(movingText);
  el.appendChild(movingRow);

  const showMoving = () => {
    guard = true;
    try {
      check.checked = !!model.get(movingField(model));
    } finally {
      guard = false;
    }
  };
  showMoving();

  check.addEventListener("change", () => {
    if (guard) return;
    model.set(movingField(model), check.checked);
    model.save_changes();
  });
  model.on("change:coarsest_while_moving_3d", showMoving);
  model.on("change:coarsest_while_moving_2d", showMoving);
  model.on("change:n_displayed_dimensions", showMoving);
}

export default { render };
