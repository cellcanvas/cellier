// Unified image control ESM (self-contained; anywidget serves each _esm
// standalone).  Shared rows, a composite switch, and a page per mode.  Every
// control writes its whole dict trait back ("shared", "single" or
// "channels"); Python diffs it and emits one bus event per changed field.
//
// Rows that only mean something for some render modes, or only in 3D, hide
// themselves: rowVisible mirrors cellier.gui._image_controls.row_visible,
// with the mode lists and n_displayed_dimensions read from the model.

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

// Decimal places for the fraction-like fields (opacity, attenuation); mirrors
// cellier.gui._image_controls.FRACTION_DECIMALS.  Data-unit fields (contrast
// limits, threshold) use the model's "decimals" trait instead.
const FRACTION_DECIMALS = 2;

// The shortest the contrast and threshold tracks may be, in pixels; mirrors
// cellier.gui._image_controls.MIN_TRACK_WIDTH_PX.  The dock grows to keep it.
const MIN_TRACK_WIDTH_PX = 120;

// Fields whose rows show only in 3D; mirrors
// cellier.gui._image_controls.THREE_D_FIELDS.
const THREE_D_FIELDS = ["render_mode", "iso_threshold", "attenuation"];

function rowVisible(field, modes, nDisplayed, thresholdModes, attenuationModes) {
  if (!THREE_D_FIELDS.includes(field)) return true;
  if (nDisplayed !== 3) return false;
  if (field === "iso_threshold") return modes.some((m) => thresholdModes.includes(m));
  if (field === "attenuation") return modes.some((m) => attenuationModes.includes(m));
  return true;
}

function formatNumber(value, decimals) {
  return Number(value).toFixed(decimals);
}

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

function makeFloatSlider(min, max, initial, onChange, decimals = FRACTION_DECIMALS) {
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
  readout.textContent = formatNumber(initial, decimals);
  const send = throttled(onChange);
  inp.addEventListener("input", () => {
    readout.textContent = formatNumber(inp.value, decimals);
    send(parseFloat(inp.value));
  });
  inp.addEventListener("change", () => onChange(parseFloat(inp.value)));
  el.appendChild(inp);
  el.appendChild(readout);
  return {
    el,
    set(v) {
      inp.value = v;
      readout.textContent = formatNumber(v, decimals);
    },
  };
}

function makeClimSlider(range, initial, onChange, decimals) {
  // Four numbers, as the Qt control shows: the current low and high on a line
  // above the track, and the range bounds at its two ends.
  const el = document.createElement("div");
  el.className = "cellier-clim";
  const values = document.createElement("div");
  values.className = "cellier-clim-values";
  const loReadout = document.createElement("span");
  loReadout.className = "cellier-clim-readout";
  const hiReadout = document.createElement("span");
  hiReadout.className = "cellier-clim-readout";
  values.appendChild(loReadout);
  values.appendChild(hiReadout);
  const trackRow = document.createElement("div");
  trackRow.className = "cellier-clim-row";
  const minBound = document.createElement("span");
  minBound.className = "cellier-clim-bound";
  minBound.textContent = formatNumber(range[0], decimals);
  const maxBound = document.createElement("span");
  maxBound.className = "cellier-clim-bound";
  maxBound.textContent = formatNumber(range[1], decimals);
  const track = document.createElement("div");
  track.className = "cellier-clim-track";
  track.style.minWidth = `${MIN_TRACK_WIDTH_PX}px`;
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
    loReadout.textContent = formatNumber(lo.value, decimals);
    hiReadout.textContent = formatNumber(hi.value, decimals);
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
  track.appendChild(rail);
  track.appendChild(fill);
  track.appendChild(lo);
  track.appendChild(hi);
  trackRow.appendChild(minBound);
  trackRow.appendChild(track);
  trackRow.appendChild(maxBound);
  el.appendChild(values);
  el.appendChild(trackRow);
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
  // Every row that can hide: { el, page, channel, field }.
  let hideable = [];

  function row(parent, field, control) {
    const r = document.createElement("div");
    r.className = "cellier-app-row";
    r.dataset.field = field;
    const label = document.createElement("label");
    label.className = "cellier-app-label";
    label.textContent = LABELS[field] || field;
    r.appendChild(label);
    r.appendChild(control.el);
    parent.appendChild(r);
    return r;
  }

  function hideableRow(parent, page, channel, field, control) {
    const r = row(parent, field, control);
    if (THREE_D_FIELDS.includes(field)) hideable.push({ el: r, page, channel, field });
    return r;
  }

  // The shared attenuation slider, on the single page (under the render
  // mode) and once on the composite page; both set the one shared field.
  function attenuationControl() {
    const shared = model.get("shared") || {};
    const fieldsOn = model.get("fields") || [];
    if (!("attenuation" in shared) || !fieldsOn.includes("attenuation")) return null;
    return makeFloatSlider(0.0, 10.0, shared.attenuation, (v) =>
      write("shared", null, "attenuation", v),
    );
  }

  // Show or hide every row that can hide, from the current state.  Run in
  // full on every trigger so the rule lives in one place.
  function applyVisibility() {
    const n = model.get("n_displayed_dimensions");
    const thresholdModes = model.get("threshold_modes") || [];
    const attenuationModes = model.get("attenuation_modes") || [];
    const single = model.get("single") || {};
    const channels = model.get("channels") || {};
    const channelModes = Object.values(channels).map((c) => c.render_mode);
    for (const { el: r, page, channel, field } of hideable) {
      let modes;
      if (page === "composite") modes = channelModes;
      else if (page === "channel") modes = [(channels[channel] || {}).render_mode];
      else modes = [single.render_mode];
      const visible = rowVisible(field, modes, n, thresholdModes, attenuationModes);
      r.style.display = visible ? "" : "none";
    }
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
    if (field === "render_mode") applyVisibility();
  }

  function modeControl(page, channel, field, values) {
    const fieldsOn = model.get("fields") || [];
    if (field !== "visible" && !fieldsOn.includes(field)) return null;
    const onChange = (v) => write(page, channel, field, v);
    const v = values[field];
    if (field === "visible") return makeCheckbox(v, onChange);
    if (field === "color_map") return makeSelect(model.get("colormap_names"), v, onChange);
    const decimals = model.get("decimals");
    if (field === "clim") {
      return makeClimSlider(model.get("clim_range") || [0, 1], v, onChange, decimals);
    }
    if (field === "render_mode") return makeSelect(model.get("render_modes"), v, onChange);
    if (field === "iso_threshold") {
      // A threshold is in data units, like the contrast limits.
      const [lo, hi] = model.get("clim_range") || [0, 1];
      const slider = makeFloatSlider(lo, hi, v, onChange, decimals);
      slider.el.querySelector("input").style.minWidth = `${MIN_TRACK_WIDTH_PX}px`;
      return slider;
    }
    return makeFloatSlider(0.0, 1.0, v, onChange, FRACTION_DECIMALS);
  }

  function build() {
    el.innerHTML = "";
    controls = {};
    hideable = [];
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
    const attenuation = [];
    for (const field of MODE_FIELDS) {
      const c = modeControl("single", null, field, single);
      if (c === null) continue;
      controls["single||" + field] = c;
      hideableRow(singlePage, "single", null, field, c);
      if (field === "render_mode") {
        const a = attenuationControl();
        if (a !== null) {
          attenuation.push(a);
          hideableRow(singlePage, "single", null, "attenuation", a);
        }
      }
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
        hideableRow(group, "channel", key, field, c);
      }
      compositePage.appendChild(group);
    }
    const compositeAttenuation = attenuationControl();
    if (compositeAttenuation !== null) {
      // One value for every channel, so one row below them all.
      attenuation.push(compositeAttenuation);
      hideableRow(compositePage, "composite", null, "attenuation", compositeAttenuation);
    }
    if (attenuation.length > 0) {
      controls["shared||attenuation"] = {
        set: (v) => {
          for (const a of attenuation) a.set(v);
        },
      };
    }
    el.appendChild(compositePage);
    applyPage();
    applyVisibility();
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
  model.on("change:single", () => {
    sync("single");
    applyVisibility();
  });
  model.on("change:channels", () => {
    const keys = Object.keys(model.get("channels") || {}).sort().join(",");
    if (keys !== channelKeys) {
      channelKeys = keys;
      build();
    } else {
      sync("channels");
      applyVisibility();
    }
  });
  for (const name of ["n_displayed_dimensions", "threshold_modes", "attenuation_modes"]) {
    model.on(`change:${name}`, applyVisibility);
  }
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
