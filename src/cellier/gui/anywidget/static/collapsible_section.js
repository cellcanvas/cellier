// Collapsible section: a titled <details> whose body mounts child anywidgets.
//
// A controls dock presented as collapsible sections shows one of these per
// configured visual.  Children are mounted through anywidget's composition
// API exactly as `slot.js` does, so a section works inside a live slot on both
// Jupyter and marimo.  `expanded` is two-way: a click is written back, so a
// section keeps its state when the dock re-renders around it.

async function render({ model, el, host, signal }) {
  const details = document.createElement("details");
  details.className = "cellier-section";

  const summary = document.createElement("summary");
  summary.className = "cellier-section-title";

  const body = document.createElement("div");
  body.className = "cellier-section-body";
  body.style.display = "flex";
  body.style.flexDirection = "column";

  details.appendChild(summary);
  details.appendChild(body);
  el.appendChild(details);

  let current = null;

  function applyChrome() {
    summary.textContent = model.get("title") || "";
    body.style.gap = `${model.get("gap")}px`;
  }

  function applyExpanded() {
    const expanded = Boolean(model.get("expanded"));
    if (details.open !== expanded) details.open = expanded;
  }

  // `toggle` also fires after `applyExpanded` sets `open`, so only a state
  // that differs from the model is a user's click worth writing back.
  details.addEventListener("toggle", () => {
    if (details.open === Boolean(model.get("expanded"))) return;
    model.set("expanded", details.open);
    model.save_changes();
  });

  async function mount() {
    if (current) current.abort();
    const controller = new AbortController();
    current = controller;
    const childSignal = signal
      ? AbortSignal.any([signal, controller.signal])
      : controller.signal;

    body.replaceChildren();
    try {
      for (const ref of model.get("children") || []) {
        const widget = await host.getWidget(ref);
        if (controller.signal.aborted) return;
        const childEl = document.createElement("div");
        body.appendChild(childEl);
        await widget.render({ el: childEl, signal: childSignal });
        if (controller.signal.aborted) return;
      }
    } catch (e) {
      if (controller.signal.aborted) return;
      console.error("[AnywidgetCollapsibleSection] mount failed", e);
      body.textContent = "AnywidgetCollapsibleSection error: " + (e && e.message);
    }
  }

  applyChrome();
  applyExpanded();
  model.on("change:title", applyChrome);
  model.on("change:gap", applyChrome);
  model.on("change:expanded", applyExpanded);
  model.on("change:children", mount);
  await mount();

  return () => {
    if (current) current.abort();
  };
}

export default { render };
