/* Area-basemap CoG viewer — pixel aligned, segmented polylines only. */
(function () {
  "use strict";

  const state = {
    basemap: null,
    areas: new Map(), // slug -> meta
    catalog: null,
    paths: new Map(), // id -> loaded path JSON + visible
    activeArea: null,
    imageLayer: null,
    roomLayer: null,
    pathLayer: null,
    weight: 2,
    markers: true,
    scrubPoints: [],
    scrubMarker: null,
  };

  const map = L.map("map", {
    crs: L.CRS.Simple,
    minZoom: -3,
    maxZoom: 4,
    zoomSnap: 0.25,
    zoomDelta: 0.5,
  });
  map.attributionControl.setPrefix("retro_rl · area map CoG");

  function latlng(ax, ay) {
    return L.latLng(ay, ax);
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function scaleFor(slug) {
    const meta = state.areas.get(slug);
    return meta && meta.pixel_scale ? meta.pixel_scale : 1;
  }

  function areaSegments(pathData, areaSlug) {
    const scale = scaleFor(areaSlug);
    const segs = [];
    for (const seg of pathData.segments || []) {
      const sslug = seg.area_slug || slugify(seg.area);
      if (sslug !== areaSlug) continue;
      const pts = (seg.points || []).map((p) => ({
        f: p.f,
        r: p.r,
        ax: p.ax * scale,
        ay: p.ay * scale,
      }));
      if (pts.length >= 2) segs.push({ room_id: seg.room_id, points: pts });
    }
    return segs;
  }

  function areaMarkers(pathData, areaSlug) {
    const scale = scaleFor(areaSlug);
    const out = [];
    for (const m of pathData.markers || []) {
      const sslug = m.a || pathData.primary_area_slug;
      if (sslug && sslug !== areaSlug) continue;
      out.push({ f: m.f, r: m.r, ax: m.ax * scale, ay: m.ay * scale });
    }
    // Also expose segment endpoints as optional markers when enabled
    return out;
  }

  function slugify(area) {
    if (!area) return "";
    return String(area)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_|_$/g, "");
  }

  function drawActive() {
    if (state.pathLayer) {
      map.removeLayer(state.pathLayer);
      state.pathLayer = null;
    }
    state.scrubPoints = [];
    const group = L.layerGroup();
    const areaSlug = state.activeArea;
    if (!areaSlug) return;

    for (const [id, entry] of state.paths) {
      if (!entry.visible) continue;
      const data = entry.data;
      const color = entry.meta.color || data.color || "#00e5ff";
      const segs = areaSegments(data, areaSlug);
      for (const seg of segs) {
        const ll = seg.points.map((p) => latlng(p.ax, p.ay));
        L.polyline(ll, {
          color,
          weight: state.weight,
          opacity: 0.92,
          lineCap: "round",
          lineJoin: "round",
          smoothFactor: 0.5,
        })
          .bindPopup(
            `<strong>${escapeHtml(data.label || id)}</strong><br/>` +
              `room 0x${Number(seg.room_id).toString(16).toUpperCase()} · ${
                seg.points.length
              } pts`
          )
          .addTo(group);
        for (const p of seg.points) {
          state.scrubPoints.push({ ...p, pathId: id, color });
        }
      }

      if (state.markers) {
        for (const seg of segs) {
          const s = seg.points[0];
          const e = seg.points[seg.points.length - 1];
          L.circleMarker(latlng(s.ax, s.ay), {
            radius: 4,
            color: "#fff",
            weight: 1,
            fillColor: "#69f0ae",
            fillOpacity: 0.95,
          }).addTo(group);
          L.circleMarker(latlng(e.ax, e.ay), {
            radius: 4,
            color: "#fff",
            weight: 1,
            fillColor: "#ff5252",
            fillOpacity: 0.95,
          }).addTo(group);
        }
        for (const m of areaMarkers(data, areaSlug)) {
          L.circleMarker(latlng(m.ax, m.ay), {
            radius: 5,
            color: "#fff",
            weight: 1,
            fillColor: color,
            fillOpacity: 0.9,
          })
            .bindPopup(
              `<strong>${escapeHtml(data.label || id)}</strong> (marker)<br/>` +
                `frame ${m.f ?? "?"} room 0x${Number(m.r || 0)
                  .toString(16)
                  .toUpperCase()}`
            )
            .addTo(group);
        }
      }
    }

    state.pathLayer = group.addTo(map);
    setupScrubber();
  }

  async function setArea(slug) {
    const meta = state.areas.get(slug);
    if (!meta) return;
    state.activeArea = slug;

    if (state.imageLayer) map.removeLayer(state.imageLayer);
    if (state.roomLayer) {
      map.removeLayer(state.roomLayer);
      state.roomLayer = null;
    }

    const w = meta.display_width || meta.width_px;
    const h = meta.display_height || meta.height_px;
    const bounds = [
      [0, 0],
      [h, w],
    ];
    state.imageLayer = L.imageOverlay(meta.file, bounds, {
      opacity: 1,
      interactive: false,
    }).addTo(map);
    map.setMaxBounds(L.latLngBounds(bounds).pad(0.15));
    map.fitBounds(bounds);

    try {
      const res = await fetch(meta.rooms_file);
      if (res.ok) {
        const geo = await res.json();
        state.roomLayer = L.geoJSON(geo, {
          style: {
            color: "#80cbc4",
            weight: 1,
            fillColor: "#004d40",
            fillOpacity: 0.04,
          },
          onEachFeature: (feat, layer) => {
            const p = feat.properties || {};
            layer.bindTooltip(
              `${p.name || ""} (${p.room_id_hex || ""})`,
              { sticky: true, className: "room-tip" }
            );
          },
        });
        if (document.getElementById("chk-rooms").checked) {
          state.roomLayer.addTo(map);
        }
      }
    } catch (_) {
      /* optional */
    }

    drawActive();
  }

  async function loadPath(meta) {
    if (state.paths.has(meta.id)) return state.paths.get(meta.id);
    const res = await fetch("paths/" + meta.file);
    if (!res.ok) throw new Error("load failed " + meta.file);
    const data = await res.json();
    const entry = {
      meta,
      data,
      visible: true,
    };
    state.paths.set(meta.id, entry);
    return entry;
  }

  function renderPathList() {
    const root = document.getElementById("path-list");
    root.innerHTML = "";
    if (!state.catalog || !state.catalog.paths.length) {
      root.innerHTML =
        '<p class="muted">No paths. Run export-defaults / export-path.</p>';
      return;
    }
    for (const meta of state.catalog.paths) {
      const item = document.createElement("div");
      item.className = "path-item";
      const area = meta.primary_area || "?";
      item.innerHTML = `
        <div class="swatch" style="background:${meta.color || "#00e5ff"}"></div>
        <label>
          <input type="checkbox" data-id="${meta.id}" checked />
          <span>
            <strong>${escapeHtml(meta.label || meta.id)}</strong>
            <div class="meta">${escapeHtml(meta.kind)} · ${
        meta.segment_count ?? "?"
      } segs · ${escapeHtml(area)}</div>
          </span>
        </label>`;
      root.appendChild(item);
    }
    root.querySelectorAll("input[type=checkbox]").forEach((cb) => {
      cb.addEventListener("change", async () => {
        const id = cb.getAttribute("data-id");
        let entry = state.paths.get(id);
        if (!entry && cb.checked) {
          const meta = state.catalog.paths.find((p) => p.id === id);
          entry = await loadPath(meta);
        }
        if (entry) {
          entry.visible = cb.checked;
          // Switch area to primary of this path when enabling
          if (cb.checked && entry.meta.primary_area_slug) {
            document.getElementById("area-select").value =
              entry.meta.primary_area_slug;
            await setArea(entry.meta.primary_area_slug);
          } else {
            drawActive();
          }
        }
      });
    });
  }

  function fitPath() {
    const pts = [];
    for (const entry of state.paths.values()) {
      if (!entry.visible) continue;
      for (const seg of areaSegments(entry.data, state.activeArea)) {
        for (const p of seg.points) pts.push(latlng(p.ax, p.ay));
      }
      for (const m of areaMarkers(entry.data, state.activeArea)) {
        pts.push(latlng(m.ax, m.ay));
      }
    }
    if (pts.length) map.fitBounds(L.latLngBounds(pts).pad(0.12));
  }

  function setupScrubber() {
    const rng = document.getElementById("scrub");
    const info = document.getElementById("scrub-info");
    const pts = state.scrubPoints;
    if (!pts.length) {
      rng.disabled = true;
      rng.max = 0;
      info.textContent = "—";
      if (state.scrubMarker) {
        map.removeLayer(state.scrubMarker);
        state.scrubMarker = null;
      }
      return;
    }
    rng.disabled = false;
    rng.max = String(pts.length - 1);
    const update = (idx) => {
      const p = pts[idx];
      if (!p) return;
      const ll = latlng(p.ax, p.ay);
      if (!state.scrubMarker) {
        state.scrubMarker = L.circleMarker(ll, {
          radius: 7,
          color: "#fff",
          weight: 2,
          fillColor: "#ffea00",
          fillOpacity: 1,
        }).addTo(map);
      } else {
        state.scrubMarker.setLatLng(ll);
      }
      info.textContent = `i=${idx} frame=${p.f ?? "?"} room=0x${Number(p.r || 0)
        .toString(16)
        .toUpperCase()} (${p.ax.toFixed(1)}, ${p.ay.toFixed(1)})`;
    };
    rng.oninput = () => update(Number(rng.value));
    rng.value = "0";
    update(0);
  }

  async function init() {
    const basemapRes = await fetch("basemap.json");
    if (!basemapRes.ok) {
      document.getElementById("path-list").innerHTML =
        '<p class="muted">Missing basemap.json — run map_viewer prepare</p>';
      return;
    }
    state.basemap = await basemapRes.json();
    const sel = document.getElementById("area-select");
    sel.innerHTML = "";
    for (const a of state.basemap.areas || []) {
      state.areas.set(a.slug, a);
      const opt = document.createElement("option");
      opt.value = a.slug;
      opt.textContent = a.area;
      sel.appendChild(opt);
    }
    sel.onchange = () => setArea(sel.value);

    try {
      const catRes = await fetch("paths/index.json");
      state.catalog = catRes.ok ? await catRes.json() : { paths: [] };
    } catch (_) {
      state.catalog = { paths: [] };
    }
    renderPathList();

    for (const meta of state.catalog.paths || []) {
      try {
        await loadPath(meta);
      } catch (e) {
        console.warn(e);
      }
    }

    const defaultSlug =
      (state.catalog.paths[0] && state.catalog.paths[0].primary_area_slug) ||
      state.basemap.default_area ||
      (state.basemap.areas[0] && state.basemap.areas[0].slug);
    if (defaultSlug) {
      sel.value = defaultSlug;
      await setArea(defaultSlug);
      fitPath();
    }

    document.getElementById("btn-fit").onclick = fitPath;
    document.getElementById("chk-rooms").onchange = (ev) => {
      if (!state.roomLayer) return;
      if (ev.target.checked) state.roomLayer.addTo(map);
      else map.removeLayer(state.roomLayer);
    };
    document.getElementById("chk-markers").onchange = (ev) => {
      state.markers = ev.target.checked;
      drawActive();
    };
    document.getElementById("rng-weight").oninput = (ev) => {
      state.weight = Number(ev.target.value);
      drawActive();
    };
  }

  init().catch((err) => {
    console.error(err);
    document.getElementById("path-list").innerHTML =
      '<p class="muted">Init failed: ' + escapeHtml(err.message) + "</p>";
  });
})();
