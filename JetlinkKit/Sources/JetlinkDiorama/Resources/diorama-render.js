// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// Diorama's renderer: the engine's scene in WebGL2. Pure CGI, nothing from a
// camera is drawn. A virtual camera follows the ego car from behind while it
// cruises and pans up to a bird's-eye view when the engine's ViewChooser asks.
(function (root) {
  'use strict';
  const E = root.DioramaEngine;
  const { EGO_FRONT, EGO_REAR, X_IDXS, IDX_N, smoothstep, clamp } = E;

  const THEMES = {
    dark: {
      sky_top: [0.07, 0.075, 0.085], horizon: [0.17, 0.18, 0.20], ground: [0.135, 0.14, 0.15], road: [0.235, 0.24, 0.255],
      sidewalk: [0.18, 0.185, 0.20], curb: [0.50, 0.51, 0.54], paint: [0.86, 0.86, 0.87], edge: [1.0, 0.22, 0.22],
      path: [0.22, 0.52, 1.0], amb_sky: [0.58, 0.60, 0.65], amb_ground: [0.26, 0.26, 0.28], key: [0.62, 0.61, 0.60], rim: [0.30, 0.33, 0.40],
    },
    light: {
      sky_top: [0.95, 0.955, 0.96], horizon: [0.88, 0.885, 0.89], ground: [0.83, 0.835, 0.84], road: [0.955, 0.955, 0.96],
      sidewalk: [0.89, 0.89, 0.895], curb: [0.72, 0.72, 0.74], paint: [0.60, 0.60, 0.62], edge: [1.0, 0.30, 0.30],
      path: [0.22, 0.52, 1.0], amb_sky: [0.72, 0.73, 0.76], amb_ground: [0.48, 0.48, 0.50], key: [0.40, 0.40, 0.40], rim: [0.20, 0.20, 0.22],
    },
  };

  // the car's paint (diorama-assets.bin marks it): the lead's under its tint, and the ego car's
  const LEAD_PAINT = [0.62, 0.62, 0.64], EGO_PAINT = [0.66, 0.66, 0.68];

  const COMMON = `
uniform vec3 cam; uniform vec3 horizon; uniform vec2 fog;
vec3 apply_fog(vec3 c, vec3 w) {
  float f = clamp((length(w.xy - cam.xy) - fog.x) / (fog.y - fog.x), 0.0, 1.0);
  return mix(c, horizon, f * f * (3.0 - 2.0 * f));
}`;
  const SHADERS = {
    sky: [`#version 300 es
in vec2 pos; out float vy;
void main() { vy = pos.y; gl_Position = vec4(pos, 0.9999, 1.0); }`, `#version 300 es
precision highp float;
uniform vec3 sky_top; uniform vec3 horizon; in float vy; out vec4 frag;
void main() { frag = vec4(mix(horizon, sky_top, clamp(vy * 0.5 + 0.5, 0.0, 1.0)), 1.0); }`],
    ground: [`#version 300 es
uniform mat4 mvp; in vec3 pos; out vec3 vw;
void main() { vw = pos; gl_Position = mvp * vec4(pos, 1.0); }`, `#version 300 es
precision highp float;
${COMMON}
uniform sampler2D map_a; uniform sampler2D map_b; uniform sampler2D mask;
uniform vec3 ground; uniform vec3 sidewalk; uniform vec3 road; uniform vec3 curb; uniform vec3 paint;
uniform vec4 pose;    // the ego car in the map window: x, y (m from the window's corner), cos yaw, sin yaw
uniform float size;   // the window's side, m
uniform int has_map;
in vec3 vw; out vec4 frag;
vec2 map_uv(vec2 e) { vec2 w = pose.xy + vec2(pose.z * e.x - pose.w * e.y, pose.w * e.x + pose.z * e.y); return vec2(w.y, w.x) / size; }
float paint_at(vec2 e) { return texture(map_a, map_uv(e)).g; }
void main() {
  vec3 c = ground;
  vec2 muv = vec2((vw.y + 30.0) / 60.0, (90.0 - vw.x) / 120.0);
  vec4 m = texture(mask, muv) * ((muv.x > 0.0 && muv.x < 1.0 && muv.y > 0.0 && muv.y < 1.0) ? 1.0 : 0.0);
  // every texture read in uniform control flow (they are mipmapped), masked after
  vec2 uv = map_uv(vw.xy);
  float inside = (has_map == 1 && uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0) ? 1.0 : 0.0;
  vec4 a = texture(map_a, uv) * inside, b = texture(map_b, uv) * inside;
  float seen = b.g;
  // lane paint is thin: a painted blob wider than 0.9 m both ways is a misread (shadow, glare, hatching)
  float blob = step(0.5, a.g) * step(0.5, min(min(paint_at(vw.xy + vec2(0.45, 0.0)), paint_at(vw.xy - vec2(0.45, 0.0))),
                                              min(paint_at(vw.xy + vec2(0.0, 0.45)), paint_at(vw.xy - vec2(0.0, 0.45)))));
  c += (sidewalk - ground) * smoothstep(0.35, 0.6, b.r) * seen;
  c = mix(c, road, smoothstep(0.4, 0.6, max(min(a.r + a.g + a.b, 1.0), m.r)));
  c = mix(c, curb, smoothstep(0.3, 0.55, a.a) * seen);
  c = mix(c, paint, smoothstep(0.4, 0.6, a.g) * (1.0 - max(m.g, blob)) * seen);
  frag = vec4(apply_fog(c, vw), 1.0);
}`],
    mask: [`#version 300 es
in vec2 pos;
void main() { gl_Position = vec4(pos.y / 30.0, (30.0 - pos.x) / 60.0, 0.0, 1.0); }`, `#version 300 es
precision mediump float; out vec4 frag; void main() { frag = vec4(1.0); }`],
    decal: [`#version 300 es
uniform mat4 mvp; in vec3 pos; in vec4 rgba; out vec3 vw; out vec4 vc;
void main() { vw = pos; vc = rgba; gl_Position = mvp * vec4(pos, 1.0); }`, `#version 300 es
precision highp float;
${COMMON}
in vec3 vw; in vec4 vc; out vec4 frag;
void main() { frag = vec4(apply_fog(vc.rgb, vw), vc.a); }`],
    mesh: [`#version 300 es
uniform mat4 mvp; uniform mat4 model; uniform mat3 nmat; uniform float span;
in vec3 pos; in vec3 nrm; in vec3 col; in float emis;
out vec3 vw; out vec3 vn; out vec3 vc; out float ve;
void main() {
  vec4 p = vec4(pos * span, 1.0);  // stored as fractions of span metres
  vw = (model * p).xyz; vn = nmat * nrm; vc = col; ve = emis; gl_Position = mvp * p;
}`, `#version 300 es
precision highp float;
${COMMON}
uniform vec3 light_dir; uniform vec3 amb_sky; uniform vec3 amb_ground; uniform vec3 key; uniform vec3 rim_col;
uniform vec3 tint; uniform float tint_amt; uniform float spec_amt; uniform float brake; uniform vec3 paint;
in vec3 vw; in vec3 vn; in vec3 vc; in float ve; out vec4 frag;
void main() {
  vec3 N = normalize(vn), V = normalize(cam - vw);
  if (dot(N, V) < 0.0) N = -N;
  vec3 col = ve < 0.0 ? paint : vc;  // emission -1 marks the paint, coloured per car
  vec3 base = mix(col, col * 0.35 + tint * 0.75, tint_amt);
  vec3 amb = mix(amb_ground, amb_sky, 0.5 + 0.5 * dot(N, vec3(0.0, 0.0, -1.0)));
  float diff = clamp((dot(N, -light_dir) + 0.25) / 1.25, 0.0, 1.0);
  float spec = pow(max(dot(N, normalize(-light_dir + V)), 0.0), 48.0) * spec_amt;
  float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
  vec3 c = base * (amb + diff * key) + spec + rim * rim_col * 0.6;
  if (ve > 0.0) c += col * ve * (0.6 + 2.5 * brake);
  frag = vec4(apply_fog(c, vw), 1.0);
}`],
  };

  // ---------------------------------------------------------------- camera
  function rodrigues(r) {
    const th = Math.hypot(r[0], r[1], r[2]);
    if (th < 1e-9) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const k = r.map((v) => v / th), s = Math.sin(th), c = 1 - Math.cos(th);
    const K = [0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0], KK = E.m3mul(K, K);
    return K.map((v, i) => (i % 4 === 0 ? 1 : 0) + s * v + c * KK[i]);
  }
  function logRot(R) {
    const th = Math.acos(clamp((R[0] + R[4] + R[8] - 1) / 2, -1, 1));
    if (th < 1e-9) return [0, 0, 0];
    const f = th / (2 * Math.sin(th));
    return [f * (R[7] - R[5]), f * (R[2] - R[6]), f * (R[3] - R[1])];
  }
  function lookAt(C, target) {  // device_from_road: rows forward, right, down
    const f = [target[0] - C[0], target[1] - C[1], target[2] - C[2]], fn = Math.hypot(...f);
    const F = f.map((v) => v / fn);
    let y = [0 * F[2] - 1 * F[1], 1 * F[0] - 0 * F[2], 0];  // [0, 0, 1] x F
    const yn = Math.hypot(...y);
    y = y.map((v) => v / yn);
    const z = [F[1] * y[2] - F[2] * y[1], F[2] * y[0] - F[0] * y[2], F[0] * y[1] - F[1] * y[0]];
    return [...F, ...y, ...z];
  }
  const transpose3 = (A) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];

  // Each view's camera, which `python -m diorama camera` sets: zoom, the focal length over the image width; pitch,
  // the degrees the camera sits above the ego car, seen from the car's middle on the road; and forward and rise,
  // metres the eye is moved along the default view's forward and up; and shift, how far down the picture is
  // moved, in image widths. Pitch turns the whole camera, moved eye and all, about the car's middle, and zoom
  // (zoomView) scales the picture about the car by moving it, so neither moves the car on screen. The setup page
  // moves forward and rise together, so the camera slides along the road, level, at any pitch, and nothing else.
  // The defaults are the first views (an eye and the point on the road it looks at, as [x forward, height above
  // the road], and a zoom) moved BACK times as far from the car along the line to it, looking the same way, with
  // BACK times the zoom: the car the same size and in the same place, but seen from further off, so the
  // perspective stretches it less near the edge of the picture. Zoom keeps the car where the first views drew it.
  const PIVOT_X = (EGO_FRONT + EGO_REAR) / 2, BACK = 2.2;
  const VIEWS = {
    road: { first: { eye: [EGO_REAR - 6.5, 3.2], at: [18, 0], zoom: 0.62 }, cy: 0.42 },
    bird: { first: { eye: [-14, 30], at: [10, 0], zoom: 0.69 }, cy: 0.5 },
  };
  for (const v of Object.values(VIEWS)) {
    const { eye, at, zoom } = v.first, d = [(BACK - 1) * (eye[0] - PIVOT_X), (BACK - 1) * eye[1]];
    v.eye = [eye[0] + d[0], eye[1] + d[1]];
    v.at = [at[0] + d[0], at[1] + d[1]];
    v.zoom = BACK * zoom;
    v.pitch = Math.atan2(v.eye[1], PIVOT_X - v.eye[0]) * 180 / Math.PI;
    v.r = Math.hypot(PIVOT_X - v.eye[0], v.eye[1]);  // the camera's distance from the car, as pitch turns it
  }
  // no limits: any finite numbers, but a zoom above 0, a focal length
  const CAMERA_NUMBERS = ['zoom', 'pitch', 'forward', 'rise', 'shift'];
  // each view at its first zoom, zoomed to its own about the car: the car where the first view drew it
  const defaultCamera = () => Object.fromEntries(Object.entries(VIEWS).map(([k, v]) => {
    const cam = { [k]: { zoom: v.first.zoom, pitch: v.pitch, forward: 0, rise: 0, shift: 0 } };
    zoomView(cam, k, v.zoom / v.first.zoom);
    cam[k].zoom = v.zoom;  // exactly, not by a ratio
    return [k, cam[k]];
  }));
  // a camera file's views over the defaults, or why it is wrong
  function readCamera(file) {
    const cam = defaultCamera();
    if (!file || typeof file !== 'object' || Array.isArray(file)) throw new Error('it is not a JSON object');
    for (const [view, nums] of Object.entries(file)) {
      if (!(view in cam)) throw new Error(`"${view}" is not a view: road or bird`);
      if (!nums || typeof nums !== 'object' || Array.isArray(nums)) throw new Error(`${view} is not an object`);
      for (const [k, v] of Object.entries(nums)) {
        if (!CAMERA_NUMBERS.includes(k)) throw new Error(`${view}.${k} is not a camera number: zoom, pitch, forward, rise or shift`);
        if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`${view}.${k} is ${JSON.stringify(v)}, not a number`);
        if (k === 'zoom' && !(v > 0)) throw new Error(`${view}.zoom is ${v}, not above 0`);
        cam[view][k] = v;
      }
    }
    return cam;
  }
  const turn = (v, n) => (v.pitch - n.pitch) * Math.PI / 180;  // the car's pitch from the default's, about y
  // a view's eye and turn, for its numbers, on a road `hgt` below the device
  function placed(v, n, hgt) {
    const a = turn(v, n), c = Math.cos(a), s = Math.sin(a);
    const ex = v.eye[0] + n.forward - PIVOT_X, ez = -v.eye[1] - n.rise;  // the moved eye from the car, z down
    const C = [PIVOT_X + c * ex + s * ez, 0, hgt - s * ex + c * ez];  // turned about the car: Ry(a) e
    const M = lookAt([v.eye[0], 0, hgt - v.eye[1]], [v.at[0], 0, hgt - v.at[1]]);
    if (a === 0) return { C, M };
    return { C, M: E.m3mul(M, [c, 0, -s, 0, 1, 0, s, 0, c]) };  // the default's turn, then the car's pitch: M Ry(a)^T
  }

  // 4x4 row-major, and a column-major copy for GL
  const m4mul = (A, B) => { const C = new Float64Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += A[4 * i + k] * B[4 * k + j]; C[4 * i + j] = s; } return C; };
  const gl4 = (A) => Float32Array.from([0, 1, 2, 3].flatMap((j) => [A[j], A[4 + j], A[8 + j], A[12 + j]]));

  // ---------------------------------------------------------------- scene geometry
  function ribbon(out, pts, halfW, rgb, alphas, soft, lift = 0.02) {
    const n = pts.length;
    if (n < 2) return;
    const nrm = pts.map((_, i) => {
      const a = pts[Math.max(i - 1, 0)], b = pts[Math.min(i + 1, n - 1)];
      const dx = b[0] - a[0], dy = b[1] - a[1], l = Math.max(Math.hypot(dx, dy), 1e-9);
      return [-dy / l, dx / l];
    });
    const cols = soft ? [[-1, 0], [-0.35, 1], [0.35, 1], [1, 0]] : [[-1, 1], [1, 1]];
    const ring = (o, a) => pts.map((p, i) => [p[0] + nrm[i][0] * halfW * o, p[1] + nrm[i][1] * halfW * o, p[2] - lift, rgb[0], rgb[1], rgb[2], alphas[i] * a]);
    const rings = cols.map(([o, a]) => ring(o, a));
    for (let r = 0; r + 1 < rings.length; r++) for (let i = 0; i + 1 < n; i++) {
      const a = rings[r][i], b = rings[r][i + 1], c = rings[r + 1][i], d = rings[r + 1][i + 1];
      out.push(...a, ...b, ...c, ...b, ...d, ...c);
    }
  }
  function polygon(out, quad, z, rgb, alpha, lift = 0.025) {
    const v = quad.map(([x, y]) => [x, y, z - lift, rgb[0], rgb[1], rgb[2], alpha]);
    out.push(...v[0], ...v[1], ...v[2], ...v[0], ...v[2], ...v[3]);
  }
  function blobShadow(out, x, y, heading, L, W, z, alpha = 0.38) {
    const c = Math.cos(heading), s = Math.sin(heading);
    const rect = (l, w, a) => [[l, w], [l, -w], [-l, -w], [-l, w]].map(([px, py]) => [x + c * px / 2 - s * py / 2, y + s * px / 2 + c * py / 2, z - 0.015, 0, 0, 0, a]);
    const inner = rect(L * 0.85, W * 0.75, alpha), outer = rect(L + 0.6, W + 0.6, 0);
    out.push(...inner[0], ...inner[1], ...inner[2], ...inner[0], ...inner[2], ...inner[3]);
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      out.push(...inner[i], ...outer[i], ...outer[j], ...inner[i], ...outer[j], ...inner[j]);
    }
  }
  // a model line (sampled at X_IDXS) as 3D points, extended straight back behind the ego car
  function modelLine(ys, hgt, xMax = 80, xBack = -25) {
    const pts = [[xBack, ys[0], hgt]];
    for (let i = 0; i < IDX_N && X_IDXS[i] < xMax; i++) pts.push([X_IDXS[i], ys[i], hgt]);
    return pts;
  }

  // ---------------------------------------------------------------- the renderer
  class Renderer {
    constructor(canvas, assets, theme = 'dark') {
      const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, preserveDrawingBuffer: true });
      if (!gl) throw new Error('this browser has no WebGL2');
      this.gl = gl; this.canvas = canvas;
      this.t = THEMES[theme] || THEMES.dark;
      this.views = defaultCamera();  // each view's zoom, pitch and forward
      this.fixedView = null;  // 0 or 1 holds the road or the bird's-eye view, whatever the engine asks
      this.prog = {};
      for (const [name, [vs, fs]] of Object.entries(SHADERS)) this.prog[name] = this.program(vs, fs);
      this.aniso = gl.getExtension('EXT_texture_filter_anisotropic');
      this.loadAssets(assets);
      this.skyBuf = this.buffer(new Float32Array([-1, -1, 3, -1, -1, 3]));
      this.dynamic = gl.createBuffer();
      this.mapTex = null; this.mapSize = 0;
      // the model's road and lane lines, rasterised in the ego frame each frame (x -30..90 m, y -30..30 m, 0.1 m)
      this.maskTex = this.texture(600, 1200, null, false);
      this.maskFbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.maskFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.maskTex, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      const L = [0.35, -0.45, 0.82], ln = Math.hypot(...L);
      this.lightDir = L.map((v) => v / ln);  // travels forward-right and down
    }

    program(vs, fs) {
      const gl = this.gl, p = gl.createProgram();
      for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
        const s = gl.createShader(type);
        gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        gl.attachShader(p, s);
      }
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
      for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); u[info.name] = gl.getUniformLocation(p, info.name); }
      return { p, u, a: (name) => gl.getAttribLocation(p, name) };
    }
    buffer(data, target) {
      const gl = this.gl, b = gl.createBuffer();
      gl.bindBuffer(target || gl.ARRAY_BUFFER, b);
      gl.bufferData(target || gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      return b;
    }
    texture(w, h, data, mip) {
      const gl = this.gl, t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mip ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      if (mip && this.aniso) gl.texParameterf(gl.TEXTURE_2D, this.aniso.TEXTURE_MAX_ANISOTROPY_EXT, 8);
      return t;
    }

    // diorama-assets.bin (diorama/assets.py): u32 header length, JSON header, then 16-byte vertices (int16 position
    // as a fraction of the asset's span, int8 normal and emission, uint8 colour) and uint16 indices, in parts
    loadAssets(buf) {
      const gl = this.gl, dv = new DataView(buf), hl = dv.getUint32(0, true);
      const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
      const nv = Object.values(head).reduce((s, a) => s + a.parts.reduce((t, p) => t + p.vertices, 0), 0);
      const verts = new Uint8Array(buf, 4 + hl, nv * 16);
      const idx = new Uint16Array(buf.slice(4 + hl + nv * 16));
      const mp = this.prog.mesh;
      this.assets = {};
      for (const [name, a] of Object.entries(head)) {
        const parts = a.parts.map((m) => {
          const vao = gl.createVertexArray();
          gl.bindVertexArray(vao);
          this.buffer(verts.subarray(m.vertexOffset * 16, (m.vertexOffset + m.vertices) * 16));
          for (const [attr, size, type, off] of [['pos', 3, gl.SHORT, 0], ['nrm', 3, gl.BYTE, 6], ['emis', 1, gl.BYTE, 9], ['col', 3, gl.UNSIGNED_BYTE, 10]]) {
            const loc = mp.a(attr);
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, size, type, true, 16, off);
          }
          this.buffer(idx.subarray(m.indexOffset, m.indexOffset + m.indices), gl.ELEMENT_ARRAY_BUFFER);
          gl.bindVertexArray(null);
          return { vao, count: m.indices };
        });
        this.assets[name] = { parts, dims: a.dims, span: a.span };
      }
    }

    // the road map's two layers, uploaded where they changed
    syncMap(map) {
      const gl = this.gl;
      if (!map || map.origin === null) return false;
      if (!this.mapTex || this.mapSize !== map.N) {
        this.mapTex = [this.texture(map.N, map.N, map.a, true), this.texture(map.N, map.N, map.b, true)];
        this.mapSize = map.N;
        map.dirty = null;
      }
      const d = map.dirty;
      if (d) {
        const [r0, r1, c0, c1] = d === 'all' ? [0, map.N, 0, map.N] : d;
        gl.pixelStorei(gl.UNPACK_ROW_LENGTH, map.N);
        for (const [tex, data] of [[this.mapTex[0], map.a], [this.mapTex[1], map.b]]) {
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.texSubImage2D(gl.TEXTURE_2D, 0, c0, r0, c1 - c0, r1 - r0, gl.RGBA, gl.UNSIGNED_BYTE, data, (r0 * map.N + c0) * 4);
          gl.generateMipmap(gl.TEXTURE_2D);
        }
        gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
        map.dirty = null;
      }
      return true;
    }

    // the virtual camera for a view between the chase camera (0) and the bird's-eye view (1)
    camera(s, hgt, w, h) {
      const n = this.views, { C: Cc, M: Mc } = placed(VIEWS.road, n.road, hgt), { C: Cb, M: Mb } = placed(VIEWS.bird, n.bird, hgt);
      const rel = logRot(E.m3mul(Mb, transpose3(Mc)));
      const M = E.m3mul(rodrigues(rel.map((v) => v * s)), Mc);
      const C = [Cc[0] + (Cb[0] - Cc[0]) * s, Cc[1] + (Cb[1] - Cc[1]) * s, Cc[2] + (Cb[2] - Cc[2]) * s ** 0.7];  // rise a bit faster than it moves back
      const f = (n.road.zoom * (1 - s) + n.bird.zoom * s) * w;
      // each view's picture moved by its shift; between them, the car goes straight from where one draws it to where
      // the other does (unless a camera is past it: then the shift is blended)
      const c0 = (v) => VIEWS[v].cy * h + n[v].shift * w;
      let cy = c0('road') * (1 - s) + c0('bird') * s;
      if (s > 0 && s < 1) {
        const d = carSlope(C, M, hgt), dr = carSlope(Cc, Mc, hgt), db = carSlope(Cb, Mb, hgt);
        if (d !== null && dr !== null && db !== null) {
          const u = (v, dv) => n[v].zoom * w * dv + c0(v);
          cy = u('road', dr) * (1 - s) + u('bird', db) * s - f * d;
        }
      }
      return { f, cx: w / 2, cy, M, C };
    }

    draw(state, engine) {
      const gl = this.gl, th = this.t, canvas = this.canvas;
      const dpr = root.devicePixelRatio || 1;
      const w = Math.round(canvas.clientWidth * dpr) || canvas.width, h = Math.round(canvas.clientHeight * dpr) || canvas.height;
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      const hgt = engine.cam ? engine.cam.height : 1.22;
      this.hgt = hgt;

      // the model's road and lane lines into the ego-frame mask
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.maskFbo);
      gl.viewport(0, 0, 600, 1200);
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
      if (state) this.drawMask(state);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.colorMask(true, true, true, true);

      gl.viewport(0, 0, w, h);
      const camv = this.camera(this.fixedView ?? (state ? state.view : 0), hgt, w, h);
      const near = 0.3, far = 500;
      const proj = [2 * camv.f / w, 0, 2 * camv.cx / w - 1, 0, 0, -2 * camv.f / h, 1 - 2 * camv.cy / h, 0, 0, 0, (far + near) / (far - near), -2 * far * near / (far - near), 0, 0, 1, 0];
      const R = E.m3mul(E.VIEW_FROM_DEVICE, camv.M), C = camv.C;
      const t = E.m3vec(R, C);
      const view = [R[0], R[1], R[2], -t[0], R[3], R[4], R[5], -t[1], R[6], R[7], R[8], -t[2], 0, 0, 0, 1];
      this.vp = m4mul(proj, view);
      this.cam = C;

      gl.clearColor(...th.horizon, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      const common = (p) => {
        gl.uniform3fv(p.u.cam, C); gl.uniform3fv(p.u.horizon, th.horizon); gl.uniform2f(p.u.fog, 45, 190);
      };
      // sky
      const sky = this.prog.sky;
      gl.useProgram(sky.p);
      gl.uniform3fv(sky.u.sky_top, th.sky_top); gl.uniform3fv(sky.u.horizon, th.horizon);
      gl.bindVertexArray(null);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.skyBuf);
      gl.enableVertexAttribArray(sky.a('pos'));
      gl.vertexAttribPointer(sky.a('pos'), 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.disableVertexAttribArray(sky.a('pos'));

      // ground with the road map
      gl.enable(gl.DEPTH_TEST);
      const g = this.prog.ground, map = engine.roadmap;
      gl.useProgram(g.p);
      common(g);
      gl.uniformMatrix4fv(g.u.mvp, false, gl4(this.vp));
      for (const k of ['ground', 'sidewalk', 'road', 'curb', 'paint']) gl.uniform3fv(g.u[k], th[k]);
      const hasMap = state && this.syncMap(map);
      gl.uniform1i(g.u.has_map, hasMap ? 1 : 0);
      if (hasMap) {
        const p = state.pose;
        gl.uniform4f(g.u.pose, p[0] - map.origin[0], p[1] - map.origin[1], Math.cos(p[2]), Math.sin(p[2]));
        gl.uniform1f(g.u.size, map.N * map.RES);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.mapTex[0]);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.mapTex[1]);
      }
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
      gl.uniform1i(g.u.map_a, 0); gl.uniform1i(g.u.map_b, 1); gl.uniform1i(g.u.mask, 2);
      this.drawDynamic(g, [[-120, -300, hgt], [400, -300, hgt], [400, 300, hgt], [-120, -300, hgt], [400, 300, hgt], [-120, 300, hgt]].flat(), [['pos', 3]]);
      gl.activeTexture(gl.TEXTURE0);
      if (!state) return;

      const objs = this.objects(state);
      this.drawDecals(state, objs, hgt, common);
      this.drawMeshes(state, objs, hgt, common);
    }

    drawDynamic(p, data, layout, mode) {
      const gl = this.gl, stride = layout.reduce((s, [, n]) => s + n, 0) * 4;
      gl.bindVertexArray(null);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.dynamic);
      gl.bufferData(gl.ARRAY_BUFFER, data instanceof Float32Array ? data : new Float32Array(data), gl.DYNAMIC_DRAW);
      let off = 0;
      const locs = [];
      for (const [name, n] of layout) {
        const loc = p.a(name);
        if (loc >= 0) { gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, n, gl.FLOAT, false, stride, off); locs.push(loc); }
        off += n * 4;
      }
      gl.drawArrays(mode === undefined ? gl.TRIANGLES : mode, 0, data.length / (stride / 4));
      for (const loc of locs) gl.disableVertexAttribArray(loc);
    }

    drawMask(state) {
      const gl = this.gl, m = this.prog.mask, out = state.out;
      gl.useProgram(m.p);
      const edge = (i) => X_IDXS.map((_, j) => E.lane(out.road_edges, i, j, 0));
      const left = modelLine(edge(0), 0), right = modelLine(edge(1), 0);
      const tris = [];
      for (let i = 0; i + 1 < left.length; i++) {
        const a = left[i], b = left[i + 1], c = right[i], d = right[i + 1];
        tris.push(a[0], a[1], b[0], b[1], c[0], c[1], b[0], b[1], d[0], d[1], c[0], c[1]);
      }
      gl.colorMask(true, false, false, false);
      this.drawDynamic(m, tris, [['pos', 2]]);
      // paint next to a model lane line is drawn by that line
      const lines = [];
      for (let i = 0; i < 4; i++) {
        if (out.lane_lines_prob[2 * i + 1] <= 0.3) continue;
        const pts = modelLine(X_IDXS.map((_, j) => E.lane(out.lane_lines, i, j, 0)), 0);
        const rib = [];
        ribbon(rib, pts, 0.6, [0, 0, 0], pts.map(() => 1), false, 0);
        for (let v = 0; v < rib.length; v += 7) lines.push(rib[v], rib[v + 1]);
      }
      gl.colorMask(false, true, false, false);
      if (lines.length) this.drawDynamic(m, lines, [['pos', 2]]);
    }

    // only the lead, from this frame's prediction, as the one car: its rear at the model's distance, facing along its lane
    objects(state) {
      const l = state.lead, a = this.assets.car;
      if (!l || !a) return [];
      const [L, W] = a.dims;
      return [{ type: 'car', x: l.x + 0.5 * L * Math.cos(l.heading), y: l.y + 0.5 * L * Math.sin(l.heading), heading: l.heading, L, W, braking: l.braking, seg: !!l.seg }];
    }

    drawDecals(state, objs, hgt, common) {
      const gl = this.gl, th = this.t, out = state.out, tris = [];
      for (const o of objs) blobShadow(tris, o.x, o.y, o.heading, o.L, o.W, hgt);
      blobShadow(tris, (EGO_FRONT + EGO_REAR) / 2, 0, 0, EGO_FRONT - EGO_REAR, 1.85, hgt);
      // road edges: a soft red glow
      for (let i = 0; i < 2; i++) {
        const pts = modelLine(X_IDXS.map((_, j) => E.lane(out.road_edges, i, j, 0)), hgt);
        const a = clamp(1 - out.road_edges_stds[i * IDX_N * 2], 0.3, 1) * 0.5;
        ribbon(tris, pts, 0.4, th.edge, pts.map((p) => a * (1 - smoothstep(55, 85, p[0]))), true);
      }
      // lane lines, faded by probability and distance
      for (let i = 0; i < 4; i++) {
        const p = clamp(out.lane_lines_prob[2 * i + 1], 0, 1);
        if (p < 0.15) continue;
        const pts = modelLine(X_IDXS.map((_, j) => E.lane(out.lane_lines, i, j, 0)), hgt);
        ribbon(tris, pts, 0.065, th.paint, pts.map((q) => 0.95 * p * (1 - smoothstep(60, 90, q[0]))), false);
      }
      // crosswalk stripes (held in the world, drawn in this frame)
      const pose = state.pose;
      if (pose) {
        const c = Math.cos(pose[2]), s = Math.sin(pose[2]);
        for (const quad of state.stripes) {
          const q = quad.map(([wx, wy]) => { const dx = wx - pose[0], dy = wy - pose[1]; return [c * dx + s * dy, -s * dx + c * dy]; });
          polygon(tris, q, hgt, th.paint, 0.9);
        }
      }
      // the planned path, cut before the lead
      const px = state.px, maxD = state.lead ? Math.max(state.lead.x - (state.pathGap ?? 3), 0) : 80;
      const pts = [];
      for (let i = 0; i < IDX_N; i++) {
        if (px[i] >= maxD) {
          if (i > 0) {
            const f = (maxD - px[i - 1]) / Math.max(px[i] - px[i - 1], 1e-3);
            pts.push([maxD, state.py[i - 1] + f * (state.py[i] - state.py[i - 1]), hgt]);
          }
          break;
        }
        pts.push([px[i], state.py[i], hgt]);
      }
      if (pts.length >= 2) {
        // resample every 0.5 m along the arc, from just ahead of the bumper
        const arc = [0];
        for (let i = 1; i < pts.length; i++) arc.push(arc[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
        const total = arc[arc.length - 1];
        if (total >= 1) {
          const res = [];
          const at = [];
          for (let si = 0; si < total; si += 0.5) at.push(si);
          at.push(total);
          for (const si of at) {
            const p = [0, 1, 2].map((j) => E.interp(si, arc, pts.map((q) => q[j])));
            if (p[0] > EGO_FRONT - 0.5) res.push(p);
          }
          if (res.length >= 2) ribbon(tris, res, 0.85, th.path, res.map((p) => 0.8 * (1 - smoothstep(12, 70, p[0]))), true, 0.03);
        }
      }
      if (!tris.length) return;
      const d = this.prog.decal;
      gl.useProgram(d.p);
      common(d);
      gl.uniformMatrix4fv(d.u.mvp, false, gl4(this.vp));
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
      this.drawDynamic(d, tris, [['pos', 3], ['rgba', 4]]);
      gl.depthMask(true); gl.disable(gl.BLEND);
    }

    drawMeshes(state, objs, hgt, common) {
      const gl = this.gl, th = this.t, mp = this.prog.mesh;
      gl.useProgram(mp.p);
      common(mp);
      gl.uniform3fv(mp.u.light_dir, this.lightDir);
      gl.uniform3fv(mp.u.amb_sky, th.amb_sky); gl.uniform3fv(mp.u.amb_ground, th.amb_ground);
      gl.uniform3fv(mp.u.key, th.key); gl.uniform3fv(mp.u.rim_col, th.rim);
      const draw = (name, x, y, heading, paint, tint, tintAmt, spec, brake) => {
        const a = this.assets[name];
        const c = Math.cos(heading), s = Math.sin(heading);
        // local (x forward, y right, z up) -> road (z down)
        const model = [c, -s, 0, x, s, c, 0, y, 0, 0, -1, hgt, 0, 0, 0, 1];
        gl.uniformMatrix4fv(mp.u.mvp, false, gl4(m4mul(this.vp, model)));
        gl.uniformMatrix4fv(mp.u.model, false, gl4(model));
        // the normal matrix, (M^-1)^T for the rotation-and-flip part, column-major
        gl.uniformMatrix3fv(mp.u.nmat, false, new Float32Array([c, s, 0, -s, c, 0, 0, 0, -1]));
        gl.uniform3fv(mp.u.paint, paint);
        gl.uniform3fv(mp.u.tint, tint || [0, 0, 0]);
        gl.uniform1f(mp.u.tint_amt, tintAmt); gl.uniform1f(mp.u.spec_amt, spec); gl.uniform1f(mp.u.brake, brake);
        gl.uniform1f(mp.u.span, a.span);
        for (const part of a.parts) {
          gl.bindVertexArray(part.vao);
          gl.drawElements(gl.TRIANGLES, part.count, gl.UNSIGNED_SHORT, 0);
        }
      };
      // the model's lead under the path's tint; one the segmentation alone shows, plain
      for (const o of objs) draw(o.type, o.x, o.y, o.heading, LEAD_PAINT, th.path, o.seg ? 0 : 0.55, 0.06, o.braking ? 1 : 0);
      draw('car', (EGO_FRONT + EGO_REAR) / 2, 0, 0, EGO_PAINT, null, 0, 0.35, 0);
      gl.bindVertexArray(null);
    }
  }

  // the car's middle on the road, down over forward, as the camera at C turned M sees it; null when it is behind
  function carSlope(C, M, hgt = 0) {
    const [Z, , Y] = E.m3vec(M, [PIVOT_X - C[0], -C[1], hgt - C[2]]);
    return Z > 1e-6 ? Y / Z : null;
  }
  // `view` zoomed by `ratio` about the car: the picture moved so the car stays on its pixel
  function zoomView(cam, view, ratio) {
    const n = cam[view], z = n.zoom * ratio;
    if (!(Number.isFinite(z) && z > 0)) return;
    const { C, M } = placed(VIEWS[view], n, 0), d = carSlope(C, M);
    if (d !== null) n.shift += (n.zoom - z) * d;
    n.zoom = z;
  }

  // the camera setup's still scene of its own, whatever drives or not: the ego car, a lead 10 m ahead, the planned
  // path stopping 2 m short of it and the ego lane's two lines, all straight, and nothing from the segmentation
  function previewScene() {
    const lines = (...ys) => ys.flatMap((y) => X_IDXS.flatMap(() => [y, 0]));
    const out = {
      lane_lines: lines(-5.4, -1.8, 1.8, 5.4), lane_lines_prob: [0, 0, 0, 1, 0, 1, 0, 0],
      road_edges: lines(-5.6, 5.6), road_edges_stds: new Array(2 * IDX_N * 2).fill(0),
    };
    const state = {
      out, px: [...X_IDXS], py: X_IDXS.map(() => 0), lead: { x: EGO_FRONT + 10, y: 0, heading: 0, braking: false, seg: false },
      pathGap: 2, pose: null, stripes: [], view: 0, viewWhy: '', hud: { v: 0, a: 0, lead: 10 },
    };
    return { state, engine: { cam: null, roadmap: null } };
  }

  // `view`'s camera moved m metres along the road, level, whatever its pitch: forward and rise in the turned frame
  function moveForwardBy(cam, view, m) {
    const n = cam[view], a = turn(VIEWS[view], n);
    n.forward += m * Math.cos(a);
    n.rise -= m * Math.sin(a);
  }
  // a drag of `pixels`: 0.005 of the default eye's distance from the car a pixel
  const moveForward = (cam, view, pixels) => moveForwardBy(cam, view, pixels * 0.005 * VIEWS[view].r);
  // the metres forward that carry the road under (x, y) of the picture `c` (Renderer.camera, the road `hgt` below)
  // dy further down it, as a touch drag pulls the road along under the fingers; null where there is no road there
  // to follow (above the horizon, or past 400 m)
  function forwardFollowing(c, hgt, x, y, dy) {
    const a = (x - c.cx) / c.f, b = (y - c.cy) / c.f, M = c.M;
    const dir = [M[0] + a * M[3] + b * M[6], M[1] + a * M[4] + b * M[7], M[2] + a * M[5] + b * M[8]];  // M^T [1, a, b]
    if (!(dir[2] > 1e-9)) return null;
    const t = (hgt - c.C[2]) / dir[2];
    if (!(t > 0) || t * Math.hypot(...dir) > 400) return null;
    // the road point at q = t [1, a, b] in the camera; moving m along road x takes it to q - m R x
    const r = [M[0], M[3], M[6]], rate = c.f * (t * b * r[0] - r[2] * t) / (t * t);  // d(row)/dm
    return Math.abs(rate) > 1e-6 ? dy / rate : null;
  }
  root.DioramaRender = { Renderer, THEMES, CAMERA_NUMBERS, defaultCamera, readCamera, moveForward, moveForwardBy, forwardFollowing, zoomView, previewScene, CAMERA_VIEWS: VIEWS, CAMERA_BACK: BACK };
})(typeof globalThis !== 'undefined' ? globalThis : this);
