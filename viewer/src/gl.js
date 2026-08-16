// Thin WebGL2 helpers. No framework: the viewer draws three things (a mesh, a
// point cloud, some instances) and a dependency would outweigh all of it.

export function createContext(canvas) {
  const gl = canvas.getContext('webgl2', {
    antialias: true,
    alpha: false,
    powerPreference: 'high-performance',
  });
  if (!gl) throw new Error('WebGL2 is required and is not available in this browser.');
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.BACK);
  return gl;
}

function compile(gl, type, source, label) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`${label} shader failed to compile:\n${log}`);
  }
  return shader;
}

export function createProgram(gl, vertexSource, fragmentSource, label = 'program') {
  const program = gl.createProgram();
  const vs = compile(gl, gl.VERTEX_SHADER, vertexSource, `${label} vertex`);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fragmentSource, `${label} fragment`);
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`${label} failed to link:\n${gl.getProgramInfoLog(program)}`);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  // Cache every uniform and attribute location up front.
  const uniforms = {};
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const info = gl.getActiveUniform(program, i);
    const name = info.name.replace(/\[0\]$/, '');
    uniforms[name] = gl.getUniformLocation(program, name);
  }
  const attributes = {};
  const attrCount = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
  for (let i = 0; i < attrCount; i++) {
    const info = gl.getActiveAttrib(program, i);
    attributes[info.name] = gl.getAttribLocation(program, info.name);
  }
  return { program, uniforms, attributes };
}

export function createBuffer(gl, target, data, usage = gl.STATIC_DRAW) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(target, buffer);
  gl.bufferData(target, data, usage);
  return buffer;
}

/** Uploads a stack of equally sized RGBA images as a TEXTURE_2D_ARRAY. */
export function createTextureArray(gl, size, layers) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
  gl.texStorage3D(gl.TEXTURE_2D_ARRAY, mipLevels(size), gl.RGBA8, size, size, Math.max(layers, 1));
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  const aniso = gl.getExtension('EXT_texture_filter_anisotropic');
  if (aniso) {
    const max = gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
    gl.texParameterf(gl.TEXTURE_2D_ARRAY, aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, max));
  }
  return texture;
}

function mipLevels(size) {
  return Math.floor(Math.log2(size)) + 1;
}

export function uploadLayer(gl, texture, layer, source, size) {
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
  gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, layer, size, size, 1,
    gl.RGBA, gl.UNSIGNED_BYTE, source);
}

export function fillLayer(gl, texture, layer, rgba, size) {
  const pixels = new Uint8Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    pixels[i * 4] = rgba[0];
    pixels[i * 4 + 1] = rgba[1];
    pixels[i * 4 + 2] = rgba[2];
    pixels[i * 4 + 3] = 255;
  }
  uploadLayer(gl, texture, layer, pixels, size);
}

export function resizeCanvas(canvas, gl, maxDpr = 2) {
  const dpr = Math.min(window.devicePixelRatio || 1, maxDpr);
  const width = Math.floor(canvas.clientWidth * dpr);
  const height = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    gl.viewport(0, 0, width, height);
    return true;
  }
  return false;
}
