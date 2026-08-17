"""Build the walkable Helsinki demo: a self-contained WebGL page with the mesh inlined.

    python tools/helsinki_demo.py --tile 672496        # measure and pick a subtile
    python tools/build_walk_demo.py                    # -> build/helsinki/helsinki_walk.html

One HTML file, no network at run time. The mesh is base64'd into the page because
the publishing target blocks every external host, so a viewer that fetched its
own geometry would render an empty street.

The byte budget is the design constraint. 16 MB for the page, base64 costs a
third on top, so the mesh has to land under about 11 MB: a 110 m crop at 384 px
textures is 7.4 MB and 206,042 triangles. That is why the crop is 110 m and not
the whole 250 m subtile, and why an inlined webfont was not affordable -- the
geometry is worth more than the typeface.

Rendering is unlit on purpose. Photogrammetric texture already contains the sun,
the sky and every self-shadow, so relighting it double-darkens exactly the
recesses that carry a facade's depth. The first attempt pushed exposure 1.55
through a filmic knee and blew every pale stucco wall to white.
"""
import base64
import pathlib

GLB = pathlib.Path("build/helsinki/demo.glb")
OUT = pathlib.Path("build/helsinki/helsinki_walk.html")
payload = base64.b64encode(GLB.read_bytes()).decode()

HEAD = r"""<title>Kamppi Walk</title>
<style>
  /* Palette taken off the mesh itself: Baltic slate ground, zinc panels, and
     the warm ochre that Helsinki's stucco actually is. Cool-biased neutrals,
     because Nordic render as cold next to that one warm accent.
     Single-theme on purpose - this is a night-lit 3D viewport, and a light
     variant would fight the render rather than support it. Every colour is
     still declared explicitly so the page holds on either host ground. */
  :root {
    --void:   #0d1116;
    --ground: #11151a;
    --panel:  rgba(20, 26, 33, 0.86);
    --edge:   rgba(180, 198, 216, 0.14);
    --text:   #e8eaec;
    --muted:  #8b95a1;
    --accent: #e4b363;
    --good:   #7fb069;
    --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
    --sans: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--ground);
    color: var(--text);
    font-family: var(--sans);
    overflow: hidden;
    -webkit-font-smoothing: antialiased;
  }
  #stage { position: fixed; inset: 0; background: var(--void); }
  canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }

  .overlay {
    position: fixed; inset: 0; pointer-events: none;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    grid-template-rows: auto 1fr auto;
    gap: 16px; padding: 16px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--edge);
    border-radius: 4px;
    backdrop-filter: blur(10px);
    padding: 14px 16px;
    pointer-events: auto;
  }

  /* Identity: the place, then what the thing is made of. */
  #ident { grid-column: 1; grid-row: 1; justify-self: start; max-width: 34ch; }
  #ident h1 {
    margin: 0; font-size: 15px; font-weight: 600; letter-spacing: 0.01em;
    text-wrap: balance;
  }
  #ident .where {
    margin: 2px 0 0; font-size: 12px; color: var(--muted);
  }
  #ident .prov {
    margin: 10px 0 0; font-family: var(--mono); font-size: 10.5px;
    line-height: 1.6; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.06em;
  }

  /* Measurements: the panel that makes this evidence and not a screenshot. */
  #facts { grid-column: 2; grid-row: 1 / span 2; justify-self: end; width: 268px; }
  #facts h2 {
    margin: 0 0 10px; font-size: 10.5px; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.1em;
  }
  .row {
    display: grid; grid-template-columns: 1fr auto; gap: 10px;
    padding: 5px 0; font-size: 12px; align-items: baseline;
  }
  .row + .row { border-top: 1px solid rgba(180, 198, 216, 0.07); }
  .row dt { color: var(--muted); }
  .row dd {
    margin: 0; font-family: var(--mono); font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
  }
  .row dd.win { color: var(--good); }
  .row dd.limit { color: var(--accent); }
  dl { margin: 0; }
  .note {
    margin: 12px 0 0; font-size: 11px; line-height: 1.55; color: var(--muted);
  }
  .note strong { color: var(--text); font-weight: 600; }

  /* Controls sit low-left, where a player's attention already is. */
  #controls { grid-column: 1; grid-row: 3; justify-self: start; }
  #controls .keys { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 11.5px; }
  #controls .k { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }
  kbd {
    font-family: var(--mono); font-size: 10px; line-height: 1;
    padding: 4px 6px; border-radius: 3px;
    background: rgba(180, 198, 216, 0.1);
    border: 1px solid var(--edge); color: var(--text);
  }
  #hud {
    grid-column: 2; grid-row: 3; justify-self: end; align-self: end;
    font-family: var(--mono); font-size: 11px; color: var(--muted);
    font-variant-numeric: tabular-nums;
    background: var(--panel); border: 1px solid var(--edge);
    border-radius: 4px; padding: 8px 10px;
  }

  /* Enter gate: pointer lock needs a gesture, so make the gesture the invite. */
  #gate {
    position: fixed; inset: 0; display: grid; place-items: center;
    background: radial-gradient(120% 90% at 50% 40%, rgba(17,21,26,0.72), rgba(9,12,16,0.94));
    pointer-events: auto; z-index: 5;
  }
  #gate.hidden { display: none; }
  #gate .inner { text-align: center; max-width: 40ch; padding: 0 24px; }
  #gate .eyebrow {
    font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--accent); margin: 0 0 14px;
  }
  #gate h2 { margin: 0; font-size: 26px; font-weight: 600; letter-spacing: -0.01em; }
  #gate p { margin: 12px 0 22px; font-size: 13px; line-height: 1.6; color: var(--muted); }
  #enter {
    font: inherit; font-size: 13px; font-weight: 600;
    color: var(--void); background: var(--accent);
    border: 0; border-radius: 3px; padding: 11px 26px; cursor: pointer;
    transition: transform 120ms ease, filter 120ms ease;
  }
  #enter:hover { filter: brightness(1.08); }
  #enter:active { transform: translateY(1px); }
  #enter:focus-visible { outline: 2px solid var(--text); outline-offset: 3px; }
  #loading { font-family: var(--mono); font-size: 12px; color: var(--muted); }

  @media (prefers-reduced-motion: reduce) {
    #enter { transition: none; }
  }
  @media (max-width: 820px) {
    #facts { display: none; }
    .overlay { padding: 10px; gap: 10px; }
  }
</style>
"""

BODY_TOP = r"""<div id="stage"><canvas id="gl"></canvas></div>

<div class="overlay">
  <section class="card" id="ident">
    <h1>Kamppi, Helsinki</h1>
    <p class="where">Historic core, 110 m of street</p>
    <p class="prov">
      Helsinki 3D+ reality mesh, 2017<br>
      206,042 triangles &middot; 60 textures<br>
      ETRS-GK25 &middot; CC BY 4.0
    </p>
  </section>

  <aside class="card" id="facts">
    <h2>Measured, not claimed</h2>
    <dl>
      <div class="row"><dt>Texture on a facade</dt><dd class="win">9.3 cm/texel</dd></div>
      <div class="row"><dt>Hamburg, same measure</dt><dd>24 cm/texel</dd></div>
      <div class="row"><dt>Facade relief RMS</dt><dd class="win">0.128 m</dd></div>
      <div class="row"><dt>Hamburg relief</dt><dd>0 &mdash; flat</dd></div>
      <div class="row"><dt>Detail at 0&ndash;3 m</dt><dd class="win">2.10 tri/m&sup2;</dd></div>
      <div class="row"><dt>Detail above 10 m</dt><dd>1.82 tri/m&sup2;</dd></div>
      <div class="row"><dt>Median triangle edge</dt><dd class="limit">77 cm</dd></div>
      <div class="row"><dt>Webbing removed</dt><dd>36,034 tri</dd></div>
    </dl>
    <p class="note">
      The mesh is <strong>densest at eye level</strong>, not at roofline &mdash; the
      inverse of every aerial-textured city model. That is why the ground floor
      exists here.
    </p>
    <p class="note">
      Walk up to a wall and it <strong>melts</strong>. That is the 77 cm triangle,
      reported rather than hidden.
    </p>
  </aside>

  <section class="card" id="controls">
    <div class="keys">
      <span class="k"><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move</span>
      <span class="k"><kbd>Shift</kbd> run</span>
      <span class="k"><kbd>Space</kbd>/<kbd>C</kbd> up &middot; down</span>
      <span class="k"><kbd>F</kbd> free fly</span>
      <span class="k"><kbd>Esc</kbd> release</span>
    </div>
  </section>

  <div id="hud">loading</div>
</div>

<div id="gate">
  <div class="inner">
    <p class="eyebrow">Spatial World Compiler</p>
    <h2>Walk a real Helsinki street</h2>
    <p>Photogrammetric mesh from the city's own survey. Mouse looks, WASD moves.
       Click to take the pointer.</p>
    <div id="loading">decoding mesh&hellip;</div>
    <button id="enter" hidden>Enter the street</button>
  </div>
</div>

<script id="glb" type="application/octet-stream-base64">"""

BODY_TAIL = r"""</script>
<script>
(() => {
  "use strict";

  // ---- spawn, measured from the mesh rather than guessed --------------------
  // Found by locating open ground (small vertical extent, near local minimum
  // surface) 5 m from a wall. A mesh's bounding-box floor is a basement or a
  // void triangle, so framing off it puts the camera underground.
  // Chosen by searching every open ground cell for the one with the LONGEST
  // unobstructed run, then facing along it: a 34 m corridor, 7.8 m from the
  // nearest wall. Picking a cell at a target standoff instead wedged the camera
  // against masonry, and picking the bounding-box centre put it inside a
  // building.
  const SPAWN = { x: -11.25, y: -11.74, z: 7.81, yaw: 90.0 * Math.PI / 180, pitch: 0.0 };
  const EYE_ABOVE_GROUND = 1.7;
  const GROUND_Y = -13.44;          // measured street surface at the spawn cell

  const canvas = document.getElementById("gl");
  const hud = document.getElementById("hud");
  const gate = document.getElementById("gate");
  const enter = document.getElementById("enter");
  const loading = document.getElementById("loading");

  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
  if (!gl) {
    loading.textContent = "WebGL2 is not available in this browser.";
    return;
  }

  // ---- GLB ------------------------------------------------------------------
  function decodeBase64(text) {
    const clean = text.replace(/\s+/g, "");
    const bin = atob(clean);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  function parseGlb(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, true) !== 0x46546c67) throw new Error("not a glb");
    let offset = 12, json = null, bin = null;
    while (offset < bytes.byteLength) {
      const length = view.getUint32(offset, true);
      const kind = view.getUint32(offset + 4, true);
      const chunk = bytes.subarray(offset + 8, offset + 8 + length);
      if (kind === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(chunk));
      else if (kind === 0x004e4942) bin = chunk;
      offset += 8 + length + ((length % 4) ? 4 - (length % 4) : 0);
    }
    return { json, bin };
  }

  const COMPONENT = {
    5120: Int8Array, 5121: Uint8Array, 5122: Int16Array,
    5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array,
  };
  const COUNT = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

  function readAccessor(doc, bin, index) {
    const spec = doc.accessors[index];
    const view = doc.bufferViews[spec.bufferView];
    const Type = COMPONENT[spec.componentType];
    const start = (view.byteOffset || 0) + (spec.byteOffset || 0);
    const n = spec.count * COUNT[spec.type];
    // Copy rather than subarray-view: the base offset is not guaranteed to be
    // aligned to the element size, and a misaligned typed-array view throws.
    return new Type(bin.buffer.slice(
      bin.byteOffset + start, bin.byteOffset + start + n * Type.BYTES_PER_ELEMENT));
  }

  // ---- shaders --------------------------------------------------------------
  // Unlit on purpose. Photogrammetric texture already contains the sun, the
  // sky and every self-shadow; lighting it again double-darkens exactly the
  // recesses that carry the facade's depth. A little distance haze and an
  // exposure lift is all the render adds.
  const VS = `#version 300 es
  layout(location=0) in vec3 aPos;
  layout(location=1) in vec2 aUv;
  uniform mat4 uProj, uView;
  out vec2 vUv;
  out float vDepth;
  void main() {
    vec4 eye = uView * vec4(aPos, 1.0);
    vDepth = -eye.z;
    vUv = aUv;
    gl_Position = uProj * eye;
  }`;

  const FS = `#version 300 es
  precision mediump float;
  in vec2 vUv;
  in float vDepth;
  uniform sampler2D uTex;
  uniform float uExposure;
  uniform vec3 uHaze;
  out vec4 outColor;
  void main() {
    // No exposure lift and no tone curve. The source is daylight
    // photogrammetry that is already correctly exposed; the first attempt
    // pushed 1.55 through a filmic knee and blew every pale stucco facade to
    // white. uExposure stays as a knob at 1.0 rather than a correction.
    vec3 c = texture(uTex, vUv).rgb * uExposure;
    float haze = 1.0 - exp(-vDepth * 0.0028);
    outColor = vec4(mix(c, uHaze, haze * 0.55), 1.0);
  }`;

  function compile(kind, source) {
    const s = gl.createShader(kind);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s));
    }
    return s;
  }
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, VS));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  const uProj = gl.getUniformLocation(program, "uProj");
  const uView = gl.getUniformLocation(program, "uView");
  const uTex = gl.getUniformLocation(program, "uTex");
  const uExposure = gl.getUniformLocation(program, "uExposure");
  const uHaze = gl.getUniformLocation(program, "uHaze");

  // ---- build buffers --------------------------------------------------------
  const parts = [];
  let triangles = 0;
  let placeholder = null;

  function makePlaceholder() {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA,
                  gl.UNSIGNED_BYTE, new Uint8Array([120, 124, 130, 255]));
    return tex;
  }

  async function build() {
    const { json: doc, bin } = parseGlb(decodeBase64(
      document.getElementById("glb").textContent));
    placeholder = makePlaceholder();

    const mesh = doc.meshes[0];
    // POSITION and TEXCOORD_0 are shared by every primitive in this export, so
    // upload each buffer once and let all 60 draws bind the same VBO.
    const shared = new Map();
    function vbo(accessorIndex) {
      if (shared.has(accessorIndex)) return shared.get(accessorIndex);
      const data = readAccessor(doc, bin, accessorIndex);
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      shared.set(accessorIndex, buffer);
      return buffer;
    }

    for (const primitive of mesh.primitives) {
      const positions = vbo(primitive.attributes.POSITION);
      const uvs = vbo(primitive.attributes.TEXCOORD_0);
      const indices = readAccessor(doc, bin, primitive.indices);
      const ibo = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, positions);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, uvs);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bindVertexArray(null);

      const part = { vao, count: indices.length, texture: placeholder };
      triangles += indices.length / 3;
      parts.push(part);

      const material = doc.materials[primitive.material];
      const slot = material?.pbrMetallicRoughness?.baseColorTexture?.index;
      if (slot === undefined) continue;
      const source = doc.textures[slot].source;
      const spec = doc.images[source];
      if (spec.bufferView === undefined) continue;
      const view = doc.bufferViews[spec.bufferView];
      const start = view.byteOffset || 0;
      const blob = new Blob(
        [bin.subarray(start, start + view.byteLength)],
        { type: spec.mimeType || "image/jpeg" });
      part.pending = blob;
    }

    // Decode textures off the main thread so the gate can open promptly, then
    // swap each in as it lands. A placeholder grey is visible for a moment
    // rather than a black hole.
    let decoded = 0;
    const total = parts.filter((p) => p.pending).length;
    await Promise.all(parts.map(async (part) => {
      if (!part.pending) return;
      try {
        const bitmap = await createImageBitmap(part.pending);
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, bitmap);
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        part.texture = tex;
        bitmap.close?.();
      } catch (err) {
        /* keep the placeholder; a missing texture must not blank the street */
      }
      part.pending = null;
      decoded += 1;
      loading.textContent = `decoding textures  ${decoded}/${total}`;
    }));
    return triangles;
  }

  // ---- camera ---------------------------------------------------------------
  const cam = { ...SPAWN, fly: false };
  const keys = new Set();
  let locked = false;

  function lookAtMatrix(out) {
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    // forward: yaw 0 looks down -z, positive yaw turns toward +x
    const fx = sy * cp, fy = sp, fz = -cy * cp;
    const rx = cy, ry = 0, rz = sy;
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    out[0] = rx; out[4] = ry; out[8] = rz;
    out[1] = ux; out[5] = uy; out[9] = uz;
    out[2] = -fx; out[6] = -fy; out[10] = -fz;
    out[3] = 0; out[7] = 0; out[11] = 0; out[15] = 1;
    out[12] = -(rx * cam.x + ry * cam.y + rz * cam.z);
    out[13] = -(ux * cam.x + uy * cam.y + uz * cam.z);
    out[14] = fx * cam.x + fy * cam.y + fz * cam.z;
    return { fx, fy, fz, rx, ry, rz };
  }

  function perspective(out, fovY, aspect, near, far) {
    const f = 1 / Math.tan(fovY / 2);
    out.fill(0);
    out[0] = f / aspect; out[5] = f;
    out[10] = (far + near) / (near - far); out[11] = -1;
    out[14] = (2 * far * near) / (near - far);
  }

  const view = new Float32Array(16);
  const proj = new Float32Array(16);

  canvas.addEventListener("click", () => {
    if (!locked) canvas.requestPointerLock();
  });
  document.addEventListener("pointerlockchange", () => {
    locked = document.pointerLockElement === canvas;
    gate.classList.toggle("hidden", locked || started);
  });
  document.addEventListener("mousemove", (event) => {
    if (!locked) return;
    cam.yaw += event.movementX * 0.0022;
    cam.pitch -= event.movementY * 0.0022;
    const limit = Math.PI / 2 - 0.02;
    cam.pitch = Math.max(-limit, Math.min(limit, cam.pitch));
  });
  addEventListener("keydown", (event) => {
    keys.add(event.code);
    if (event.code === "KeyF") cam.fly = !cam.fly;
    if (["Space", "KeyC"].includes(event.code)) event.preventDefault();
  });
  addEventListener("keyup", (event) => keys.delete(event.code));

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
    }
  }

  let started = false;
  let last = 0;
  let frames = 0, fps = 0, fpsClock = 0;

  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min((now - last) / 1000 || 0, 0.05);
    last = now;

    const basis = lookAtMatrix(view);
    const speed = (keys.has("ShiftLeft") || keys.has("ShiftRight") ? 11 : 3.6) * dt;
    let dx = 0, dy = 0, dz = 0;
    if (keys.has("KeyW")) { dx += basis.fx; dy += basis.fy; dz += basis.fz; }
    if (keys.has("KeyS")) { dx -= basis.fx; dy -= basis.fy; dz -= basis.fz; }
    if (keys.has("KeyD")) { dx += basis.rx; dz += basis.rz; }
    if (keys.has("KeyA")) { dx -= basis.rx; dz -= basis.rz; }
    if (!cam.fly) { dy = 0; }
    if (keys.has("Space")) dy += 1;
    if (keys.has("KeyC")) dy -= 1;
    const len = Math.hypot(dx, dy, dz);
    if (len > 0) {
      cam.x += (dx / len) * speed;
      cam.y += (dy / len) * speed;
      cam.z += (dz / len) * speed;
    }
    // Walking keeps the eye at a person's height; free fly releases it.
    if (!cam.fly && !keys.has("Space") && !keys.has("KeyC")) {
      cam.y += (GROUND_Y + EYE_ABOVE_GROUND - cam.y) * Math.min(1, dt * 8);
    }
    lookAtMatrix(view);

    resize();
    perspective(proj, 1.15, canvas.width / Math.max(canvas.height, 1), 0.15, 900);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.051, 0.066, 0.086, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);            // photogrammetric shells are not solid

    gl.useProgram(program);
    gl.uniformMatrix4fv(uProj, false, proj);
    gl.uniformMatrix4fv(uView, false, view);
    gl.uniform1f(uExposure, 1.0);
    gl.uniform3f(uHaze, 0.075, 0.094, 0.118);
    gl.uniform1i(uTex, 0);
    gl.activeTexture(gl.TEXTURE0);
    for (const part of parts) {
      gl.bindTexture(gl.TEXTURE_2D, part.texture);
      gl.bindVertexArray(part.vao);
      gl.drawElements(gl.TRIANGLES, part.count, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);

    frames += 1; fpsClock += dt;
    if (fpsClock >= 0.5) { fps = Math.round(frames / fpsClock); frames = 0; fpsClock = 0; }
    hud.textContent =
      `${fps} fps  ·  ${(triangles / 1000).toFixed(0)}k tri  ·  ` +
      `eye ${(cam.y - GROUND_Y).toFixed(1)} m  ·  ${cam.fly ? "free fly" : "walking"}`;
  }

  build().then(() => {
    loading.hidden = true;
    enter.hidden = false;
    enter.addEventListener("click", () => {
      started = true;
      gate.classList.add("hidden");
      canvas.requestPointerLock();
    });
    requestAnimationFrame(frame);
  }).catch((err) => {
    loading.textContent = "Could not load the mesh: " + err.message;
  });
})();
</script>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HEAD + BODY_TOP + payload + BODY_TAIL)
size = OUT.stat().st_size
print(f"{OUT}: {size/1e6:.2f} MB  (glb {GLB.stat().st_size/1e6:.2f} MB, "
      f"base64 {len(payload)/1e6:.2f} MB)")
assert size < 16_000_000, "over the 16 MB artifact limit"
print("under the 16 MB limit")
