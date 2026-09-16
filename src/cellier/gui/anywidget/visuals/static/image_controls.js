// Unified image control ESM (self-contained; anywidget serves each _esm
// standalone).  Shared rows, a composite switch, and a page per mode.  Every
// control writes its whole dict trait back ("shared", "single" or
// "channels"); Python diffs it and emits one bus event per changed field.

const THROTTLE_MS = 50;

const LABELS = {
  visible: "Visible",
  transparency_mode: "Blending",
  interpolation: "Interpolation",
  attenuation: "Attenuation",
  color_map: "Colormap",
  clim: "Contrast",
  opacity: "Opacity",
  render_mode: "Render mode",
  iso_threshold: "Threshold",
};

const MODE_FIELDS = ["color_map", "clim", "opacity", "render_mode", "iso_threshold"];

function throttled(fn) {
  let timer = null;
  let pending = null;
  return function (value) {
    if (timer === null) {
      fn(value);
      timer = setTimeout(function tick() {
        if (pending !== null) {
          const v = pending;
          pending = null;
          fn(v);
          timer = setTimeout(tick, THROTTLE_MS);
        } else {
          timer = null;
        }
      }, THROTTLE_MS);
    } else {
      pending = value;
    }
  };
}

function makeCheckbox(initial, onChange) {
  const el = document.createElement("input");
  el.type = "checkbox";
  el.checked = Boolean(initial);
  el.addEventListener("change", () => onChange(el.checked));
  return { el, set: (v) => { el.checked = Boolean(v); } };
}

function makeSelect(options, initial, onChange) {
  const el = document.createElement("select");
  el.className = "cellier-app-select";
  let opts = (options || []).slice();
  if (initial && !opts.includes(initial)) opts = [initial, ...opts];
  for (const name of opts) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    el.appendChild(o);
  }
  el.value = initial;
  el.addEventListener("change", () => onChange(el.value));
  return {
    el,
    set(value) {
      const v = value || "";
      if (!Array.from(el.options).some((o) => o.value === v)) {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = v;
        el.prepend(o);
      }
      el.value = v;
    },
  };
}

function makeFloatSlider(min, max, initial, onChange) {
  const el = document.createElement("div");
  el.className = "cellier-app-row-inner";
  const inp = document.createElement("input");
  inp.type = "range";
  inp.min = min;
  inp.max = max;
  inp.step = "any";
  inp.value = initial;
  const readout = document.createElement("span");
  readout.className = "cellier-app-readout";
  readout.textContent = Number(initial).toFixed(3);
  const send = throttled(onChange);
  inp.addEventListener("input", () => {
    readout.textContent = Number(inp.value).toFixed(3);
    send(parseFloat(inp.value));
  });
  inp.addEventListener("change", () => onChange(parseFloat(inp.value)));
  el.appendChild(inp);
  el.appendChild(readout);
  return {
    el,
    set(v) {
      inp.value = v;
      readout.textContent = Number(v).toFixed(3);
    },
  };
}

function makeClimSlider(range, initial, onChange) {
  const el = document.createElement("div");
  el.className = "cellier-clim-track";
  const rail = document.createElement("div");
  rail.className = "cellier-clim-rail";
  const fill = document.createElement("div");
  fill.className = "cellier-clim-fill";
  const lo = document.createElement("input");
  const hi = document.createElement("input");
  for (const [inp, cls, v] of [[lo, "cellier-clim-lo", initial[0]], [hi, "cellier-clim-hi", initial[1]]]) {
    inp.type = "range";
    inp.className = cls;
    inp.min = range[0];
    inp.max = range[1];
    inp.step = "any";
    inp.value = v;
  }
  function updateFill() {
    const mn = parseFloat(lo.min);
    const span = parseFloat(lo.max) - mn || 1;
    fill.style.left = ((parseFloat(lo.value) - mn) / span) * 100 + "%";
    fill.style.right = ((parseFloat(lo.max) - parseFloat(hi.value)) / span) * 100 + "%";
  }
  const current = () => [parseFloat(lo.value), parseFloat(hi.value)];
  const send = throttled(onChange);
  lo.addEventListener("input", () => {
    if (parseFloat(lo.value) > parseFloat(hi.value)) lo.value = hi.value;
    updateFill();
    send(current());
  });
  hi.addEventListener("input", () => {
    if (parseFloat(hi.value) < parseFloat(lo.value)) hi.value = lo.value;
    updateFill();
    send(current());
  });
  lo.addEventListener("change", () => onChange(current()));
  hi.addEventListener("change", () => onChange(current()));
  el.appendChild(rail);
  el.appendChild(fill);
  el.appendChild(lo);
  el.appendChild(hi);
  updateFill();
  return {
    el,
    set(v) {
      lo.value = v[0];
      hi.value = v[1];
      updateFill();
    },
  };
}

function render({ model, el }) {
  el.classList.add("cellier-image-controls");
  let guard = false;
  // "page|channel|field" -> control
  let controls = {};

  function row(parent, field, control) {
    const r = document.createElement("div");
    r.className = "cellier-app-row";
    const label = document.createElement("label");
    label.className = "cellier-app-label";
    label.textContent = LABELS[field] || field;
    r.appendChild(label);
    r.appendChild(control.el);
    parent.appendChild(r);
  }

  // Write one field back into its dict trait.
  function write(page, channel, field, value) {
    if (guard) return;
    if (page === "channel") {
      const channels = JSON.parse(JSON.stringify(model.get("channels") || {}));
      channels[channel] = { ...(channels[channel] || {}), [field]: value };
      model.set("channels", channels);
    } else {
      model.set(page, { ...(model.get(page) || {}), [field]: value });
    }
    model.save_changes();
  }

  function modeControl(page, channel, field, values) {
    const fieldsOn = model.get("fields") || [];
    if (field !== "visible" && !fieldsOn.includes(field)) return null;
    const onChange = (v) => write(page, channel, field, v);
    const v = values[field];
    if (field === "visible") return makeCheckbox(v, onChange);
    if (field === "color_map") return makeSelect(model.get("colormap_names"), v, onChange);
    if (field === "clim") return makeClimSlider(model.get("clim_range") || [0, 1], v, onChange);
    if (field === "render_mode") return makeSelect(model.get("render_modes"), v, onChange);
    return makeFloatSlider(0.0, 1.0, v, onChange);
  }

  function build() {
    el.innerHTML = "";
    controls = {};
    const fieldsOn = model.get("fields") || [];

    const title = document.createElement("div");
    title.className = "cellier-group-title";
    title.textContent = model.get("title") || "Image";
    el.appendChild(title);

    const shared = model.get("shared") || {};
    const sharedBox = document.createElement("div");
    sharedBox.className = "cellier-image-shared";
    if (fieldsOn.includes("visible")) {
      const c = makeCheckbox(shared.visible, (v) => write("shared", null, "visible", v));
      controls["shared||visible"] = c;
      row(sharedBox, "visible", c);
    }
    for (const [field, options] of [
      ["transparency_mode", model.get("transparency_modes")],
      ["interpolation", model.get("interpolations")],
    ]) {
      const c = makeSelect(options, shared[field], (v) => write("shared", null, field, v));
      controls["shared||" + field] = c;
      row(sharedBox, field, c);
    }
    if ("attenuation" in shared && fieldsOn.includes("attenuation")) {
      const c = makeFloatSlider(0.0, 10.0, shared.attenuation, (v) =>
        write("shared", null, "attenuation", v),
      );
      controls["shared||attenuation"] = c;
      row(sharedBox, "attenuation", c);
    }
    el.appendChild(sharedBox);

    const switchRow = document.createElement("label");
    switchRow.className = "cellier-image-switch";
    const switchBox = document.createElement("input");
    switchBox.type = "checkbox";
    switchBox.checked = Boolean(model.get("composite"));
    switchBox.addEventListener("change", () => {
      if (guard) return;
      model.set("composite", switchBox.checked);
      model.save_changes();
    });
    switchRow.appendChild(switchBox);
    switchRow.appendChild(document.createTextNode(" Composite channels"));
    switchRow.style.display = model.get("has_channel_axis") ? "" : "none";
    controls["composite||composite"] = { set: (v) => { switchBox.checked = Boolean(v); } };
    el.appendChild(switchRow);

    const singlePage = document.createElement("div");
    singlePage.className = "cellier-image-page cellier-image-single";
    const single = model.get("single") || {};
    for (const field of MODE_FIELDS) {
      const c = modeControl("single", null, field, single);
      if (c === null) continue;
      controls["single||" + field] = c;
      row(singlePage, field, c);
    }
    el.appendChild(singlePage);

    const compositePage = document.createElement("div");
    compositePage.className = "cellier-image-page cellier-image-composite";
    const channels = model.get("channels") || {};
    const labels = model.get("channel_labels") || {};
    const keys = Object.keys(channels).sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
    if (keys.length === 0) {
      const empty = document.createElement("div");
      empty.className = "cellier-image-empty";
      empty.textContent = "No channels";
      compositePage.appendChild(empty);
    }
    for (const key of keys) {
      const group = document.createElement("div");
      group.className = "cellier-channel-row";
      const head = document.createElement("div");
      head.className = "cellier-section-title";
      head.textContent = labels[key] || `Channel ${key}`;
      group.appendChild(head);
      for (const field of ["visible", ...MODE_FIELDS]) {
        const c = modeControl("channel", key, field, channels[key]);
        if (c === null) continue;
        controls["channel|" + key + "|" + field] = c;
        row(group, field, c);
      }
      compositePage.appendChild(group);
    }
    el.appendChild(compositePage);
    applyPage();
  }

  function applyPage() {
    const composite = Boolean(model.get("composite"));
    for (const page of el.querySelectorAll(".cellier-image-page")) {
      const isComposite = page.classList.contains("cellier-image-composite");
      page.style.display = isComposite === composite ? "" : "none";
    }
  }

  function sync(page) {
    guard = true;
    try {
      if (page === "channels") {
        const channels = model.get("channels") || {};
        for (const [key, values] of Object.entries(channels)) {
          for (const [field, v] of Object.entries(values)) {
            const c = controls["channel|" + key + "|" + field];
            if (c) c.set(v);
          }
        }
      } else {
        for (const [field, v] of Object.entries(model.get(page) || {})) {
          const c = controls[page + "||" + field];
          if (c) c.set(v);
        }
      }
    } finally {
      guard = false;
    }
  }

  let channelKeys = Object.keys(model.get("channels") || {}).sort().join(",");
  build();
  model.on("change:shared", () => sync("shared"));
  model.on("change:single", () => sync("single"));
  model.on("change:channels", () => {
    const keys = Object.keys(model.get("channels") || {}).sort().join(",");
    if (keys !== channelKeys) {
      channelKeys = keys;
      build();
    } else {
      sync("channels");
    }
  });
  model.on("change:composite", () => {
    guard = true;
    try {
      controls["composite||composite"].set(model.get("composite"));
    } finally {
      guard = false;
    }
    applyPage();
  });
  model.on("change:fields", build);
  model.on("change:title", build);
}

export default { render };
