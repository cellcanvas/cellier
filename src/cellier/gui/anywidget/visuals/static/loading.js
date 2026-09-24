// Loading indicator ESM: a bar of target chunks resident over needed, and a
// status line, under a group heading (the Qt QGroupBox title).  No label
// beside the bar.  Read-only; the Python side pushes every change.

function render({ model, el }) {
  el.classList.add("cellier-loading");

  const heading = document.createElement("div");
  heading.className = "cellier-group-title";

  const body = document.createElement("div");
  body.className = "cellier-loading-body";

  const bar = document.createElement("progress");
  bar.className = "cellier-loading-bar";

  const text = document.createElement("span");
  text.className = "cellier-loading-text";

  function update() {
    heading.textContent = model.get("title") || "Data fetch status";
    bar.max = Math.max(model.get("maximum") ?? 1, 1);
    bar.value = model.get("value") ?? 0;
    text.textContent = model.get("text") ?? "";
    el.classList.toggle("cellier-loading-busy", !!model.get("busy"));
  }

  for (const name of ["title", "maximum", "value", "text", "busy"]) {
    model.on(`change:${name}`, update);
  }
  update();

  body.appendChild(bar);
  body.appendChild(text);
  el.appendChild(heading);
  el.appendChild(body);
}

export default { render };
