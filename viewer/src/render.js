// Draw passes: sky, opaque surfaces, instanced props, transparent surfaces,
// optional source point cloud.

import { createBuffer, createProgram } from './gl.js';
import {
  INSTANCE_FS, INSTANCE_VS, POINTS_FS, POINTS_VS, SKY_FS, SKY_VS,
  SURFACE_FS, SURFACE_VS,
} from './shaders.js';

const ATTR = { position: 0, normal: 1, uv: 2, context: 3, role: 4, node: 5, material: 6 };

export class Renderer {
  constructor(gl, world) {
    this.gl = gl;
    this.world = world;
    // Where the haze layer sits. The world's floor, not the camera's, so the
    // fog does not slide up and down as you fly -- LoDo spans 78 m of relief
    // and anchoring to anything mobile makes distant blocks breathe.
    this.fogBase = world.bounds ? world.bounds[0][2] : 0;
    this.surface = createProgram(gl, SURFACE_VS, SURFACE_FS, 'surface');
    this.pointsProgram = createProgram(gl, POINTS_VS, POINTS_FS, 'points');
    this.instanceProgram = createProgram(gl, INSTANCE_VS, INSTANCE_FS, 'instance');
    this.skyProgram = createProgram(gl, SKY_VS, SKY_FS, 'sky');

    this.stride = world.header.vertexStride;
    this._buildSurface();
    this._buildPoints();
    this._buildSky();
    this._buildInstances();

    this.materialTexture = null;
    this.uniforms = null;
    this.environment = null;
    this.debugMode = 0;
    this.contextMask = 0;
    this.highlightNode = 0xffffffff;
    this.showPoints = false;
    this.pointColorMode = 0;
    this.pointScale = 260;
    this.showInstances = true;
  }

  _buildSurface() {
    const gl = this.gl;
    const world = this.world;
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    this.vbo = createBuffer(gl, gl.ARRAY_BUFFER, world.vertexBytes);
    const stride = this.stride;
    gl.enableVertexAttribArray(ATTR.position);
    gl.vertexAttribPointer(ATTR.position, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(ATTR.normal);
    gl.vertexAttribPointer(ATTR.normal, 3, gl.FLOAT, false, stride, 12);
    gl.enableVertexAttribArray(ATTR.uv);
    gl.vertexAttribPointer(ATTR.uv, 2, gl.FLOAT, false, stride, 24);
    gl.enableVertexAttribArray(ATTR.context);
    gl.vertexAttribIPointer(ATTR.context, 1, gl.UNSIGNED_INT, stride, 32);
    gl.enableVertexAttribArray(ATTR.role);
    gl.vertexAttribIPointer(ATTR.role, 1, gl.UNSIGNED_INT, stride, 36);
    gl.enableVertexAttribArray(ATTR.node);
    gl.vertexAttribIPointer(ATTR.node, 1, gl.UNSIGNED_INT, stride, 40);

    // Theme-dependent, replaced on every theme switch.
    this.materialBuffer = createBuffer(gl, gl.ARRAY_BUFFER,
      new Uint32Array(world.header.mesh.vertexCount), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(ATTR.material);
    gl.vertexAttribIPointer(ATTR.material, 1, gl.UNSIGNED_INT, 0, 0);

    this.ibo = createBuffer(gl, gl.ELEMENT_ARRAY_BUFFER, world.indices);
    this.indexCount = world.indices.length;
    gl.bindVertexArray(null);
  }

  _buildPoints() {
    const gl = this.gl;
    const points = this.world.points;
    this.pointCount = points ? points.count : 0;
    if (!this.pointCount) return;
    this.pointVao = gl.createVertexArray();
    gl.bindVertexArray(this.pointVao);
    createBuffer(gl, gl.ARRAY_BUFFER, points.floats);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 16, 12);
    gl.bindVertexArray(null);
  }

  _buildSky() {
    const gl = this.gl;
    this.skyVao = gl.createVertexArray();
    gl.bindVertexArray(this.skyVao);
    createBuffer(gl, gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]));
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  _buildInstances() {
    const gl = this.gl;
    const groups = groupInstances(this.world.instances);
    this.instanceGroups = [];

    for (const group of groups) {
      if (!group.items.length) continue;
      const geometry = group.geometry;
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      createBuffer(gl, gl.ARRAY_BUFFER, geometry.vertices);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);

      const offsets = new Float32Array(group.items.length * 3);
      const scales = new Float32Array(group.items.length * 3);
      group.items.forEach((item, i) => {
        offsets.set(item.offset, i * 3);
        scales.set(item.scale, i * 3);
      });
      createBuffer(gl, gl.ARRAY_BUFFER, offsets);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(2, 1);
      createBuffer(gl, gl.ARRAY_BUFFER, scales);
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 3, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(3, 1);

      const tints = new Float32Array(group.items.length * 4);
      const tintBuffer = createBuffer(gl, gl.ARRAY_BUFFER, tints, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(4);
      gl.vertexAttribPointer(4, 4, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(4, 1);

      createBuffer(gl, gl.ELEMENT_ARRAY_BUFFER, geometry.indices);
      gl.bindVertexArray(null);
      this.instanceGroups.push({
        vao, tintBuffer, tints, count: geometry.indices.length,
        instances: group.items.length, role: group.role, items: group.items,
      });
    }
  }

  /** Rebind everything that depends on the active theme. */
  applyTheme(theme, materialIndices, texture, environment, uniforms) {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.materialBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, materialIndices, gl.DYNAMIC_DRAW);
    gl.bindVertexArray(null);

    this.theme = theme;
    this.materialTexture = texture;
    this.environment = environment;
    this.uniforms = uniforms;

    // Instance tints come from the same rule table as everything else.
    for (const group of this.instanceGroups) {
      group.items.forEach((item, i) => {
        const spec = theme.materials[item.materialIndex(theme)] || {};
        const rgb = spec.baseColor || [0.6, 0.6, 0.6];
        group.tints.set([rgb[0], rgb[1], rgb[2], 1], i * 4);
      });
      gl.bindBuffer(gl.ARRAY_BUFFER, group.tintBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, group.tints, gl.DYNAMIC_DRAW);
    }
  }

  render(camera) {
    const gl = this.gl;
    const env = this.environment;
    if (!env) return;

    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.depthMask(false);
    gl.useProgram(this.skyProgram.program);
    gl.bindVertexArray(this.skyVao);
    gl.uniform3fv(this.skyProgram.uniforms.uSky, env.sky);
    gl.uniform3fv(this.skyProgram.uniforms.uHorizon, env.horizon);
    gl.uniform1f(this.skyProgram.uniforms.uPitch, camera.pitch);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.depthMask(true);

    const p = this.surface;
    gl.useProgram(p.program);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.materialTexture);
    gl.uniform1i(p.uniforms.uAlbedo, 0);
    gl.uniformMatrix4fv(p.uniforms.uViewProjection, false, camera.viewProjection);
    gl.uniform3fv(p.uniforms.uCameraPosition, camera.position);
    gl.uniform4fv(p.uniforms.uMaterialColor, this.uniforms.color);
    gl.uniform4fv(p.uniforms.uMaterialParams, this.uniforms.params);
    gl.uniform4fv(p.uniforms.uMaterialEmissive, this.uniforms.emissive);
    gl.uniform3fv(p.uniforms.uSunDirection, env.sunDirection);
    gl.uniform3fv(p.uniforms.uSunColor, env.sunColor);
    gl.uniform3fv(p.uniforms.uAmbient, env.ambient);
    gl.uniform3fv(p.uniforms.uFogColor, env.fog);
    gl.uniform1f(p.uniforms.uFogDensity, env.fogDensity);
    gl.uniform1f(p.uniforms.uFogBase, this.fogBase);
    gl.uniform1f(p.uniforms.uFogHeight, env.fogHeight);
    gl.uniform1f(p.uniforms.uExposure, env.exposure);
    gl.uniform1ui(p.uniforms.uHighlightNode, this.highlightNode >>> 0);
    gl.uniform1ui(p.uniforms.uContextMask, this.contextMask >>> 0);
    gl.uniform1i(p.uniforms.uDebugMode, this.debugMode);
    gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_INT, 0);

    if (this.showInstances) this._renderInstances(camera, env);
    if (this.showPoints && this.pointCount) this._renderPoints(camera, env);
    gl.bindVertexArray(null);
  }

  _renderInstances(camera, env) {
    const gl = this.gl;
    const p = this.instanceProgram;
    gl.useProgram(p.program);
    gl.uniformMatrix4fv(p.uniforms.uViewProjection, false, camera.viewProjection);
    gl.uniform3fv(p.uniforms.uCameraPosition, camera.position);
    gl.uniform3fv(p.uniforms.uSunDirection, env.sunDirection);
    gl.uniform3fv(p.uniforms.uSunColor, env.sunColor);
    gl.uniform3fv(p.uniforms.uAmbient, env.ambient);
    gl.uniform3fv(p.uniforms.uFogColor, env.fog);
    gl.uniform1f(p.uniforms.uFogDensity, env.fogDensity);
    gl.uniform1f(p.uniforms.uFogBase, this.fogBase);
    gl.uniform1f(p.uniforms.uFogHeight, env.fogHeight);
    gl.uniform1f(p.uniforms.uExposure, env.exposure);
    for (const group of this.instanceGroups) {
      gl.bindVertexArray(group.vao);
      gl.drawElementsInstanced(gl.TRIANGLES, group.count, gl.UNSIGNED_INT, 0, group.instances);
    }
  }

  _renderPoints(camera, env) {
    const gl = this.gl;
    const p = this.pointsProgram;
    gl.useProgram(p.program);
    gl.bindVertexArray(this.pointVao);
    gl.uniformMatrix4fv(p.uniforms.uViewProjection, false, camera.viewProjection);
    gl.uniform3fv(p.uniforms.uCameraPosition, camera.position);
    gl.uniform1f(p.uniforms.uPointScale, this.pointScale);
    gl.uniform1i(p.uniforms.uColorMode, this.pointColorMode);
    gl.uniform3fv(p.uniforms.uFogColor, env.fog);
    gl.uniform1f(p.uniforms.uFogDensity, env.fogDensity * 0.5);
    gl.uniform1f(p.uniforms.uFogBase, this.fogBase);
    gl.uniform1f(p.uniforms.uFogHeight, env.fogHeight);
    gl.drawArrays(gl.POINTS, 0, this.pointCount);
  }
}

// --- instance proxy geometry ----------------------------------------------
// Vegetation, poles and vehicles are never meshed from points: the compiler
// emits position + size + role, and the runtime substitutes a proxy. Swapping
// these for authored assets is a theme concern, not a geometry one.

function cylinder(segments = 10, taper = 1.0) {
  const vertices = [];
  const indices = [];
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    const c = Math.cos(a), s = Math.sin(a);
    vertices.push(c, s, 0, c, s, 0);
    vertices.push(c * taper, s * taper, 1, c, s, 0.25);
  }
  for (let i = 0; i < segments; i++) {
    const b = i * 2;
    indices.push(b, b + 1, b + 3, b, b + 3, b + 2);
  }
  return { vertices: new Float32Array(vertices), indices: new Uint32Array(indices) };
}

function sphere(rings = 7, segments = 10) {
  const vertices = [];
  const indices = [];
  for (let r = 0; r <= rings; r++) {
    const phi = (r / rings) * Math.PI;
    for (let s = 0; s <= segments; s++) {
      const theta = (s / segments) * Math.PI * 2;
      const x = Math.sin(phi) * Math.cos(theta);
      const y = Math.sin(phi) * Math.sin(theta);
      const z = Math.cos(phi);
      vertices.push(x, y, z, x, y, z);
    }
  }
  for (let r = 0; r < rings; r++) {
    for (let s = 0; s < segments; s++) {
      const a = r * (segments + 1) + s;
      const b = a + segments + 1;
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return { vertices: new Float32Array(vertices), indices: new Uint32Array(indices) };
}

function box() {
  const v = [];
  const idx = [];
  const faces = [
    [[1, 0, 0], [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]]],
    [[-1, 0, 0], [[-1, 1, -1], [-1, -1, -1], [-1, -1, 1], [-1, 1, 1]]],
    [[0, 1, 0], [[1, 1, -1], [-1, 1, -1], [-1, 1, 1], [1, 1, 1]]],
    [[0, -1, 0], [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]]],
    [[0, 0, 1], [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]],
    [[0, 0, -1], [[-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]]],
  ];
  faces.forEach(([n, corners], f) => {
    corners.forEach((c) => v.push(c[0], c[1], c[2] * 0.5 + 0.5, n[0], n[1], n[2]));
    const b = f * 4;
    idx.push(b, b + 1, b + 2, b, b + 2, b + 3);
  });
  return { vertices: new Float32Array(v), indices: new Uint32Array(idx) };
}

const GEOMETRY = { trunk: cylinder(8, 0.65), crown: sphere(), pole: cylinder(6, 0.8), body: box() };

function groupInstances(instances) {
  const trunks = [];
  const crowns = [];
  const poles = [];
  const bodies = [];

  for (const item of instances) {
    const [x, y, z] = item.position;
    const [rx, ry, height] = item.size;
    const roleMaterial = (role) => (theme) => {
      for (const rule of theme.rules) {
        const pattern = rule.role;
        if (!pattern || pattern === '*' || role === pattern || role.startsWith(`${pattern}.`)) {
          if (rule.all || rule.any) continue;      // context rules cannot apply to proxies
          return rule.material;
        }
      }
      return theme.fallback ?? 0;
    };

    if (item.role.startsWith('volume.vegetation')) {
      const crownRadius = Math.max(item.attrs?.crown_radius || Math.max(rx, ry), 0.8);
      const trunkHeight = Math.max(height * 0.4, 0.8);
      trunks.push({ offset: [x, y, z], scale: [0.16, 0.16, trunkHeight],
                    materialIndex: roleMaterial('linear.pole') });
      crowns.push({ offset: [x, y, z + trunkHeight + (height - trunkHeight) * 0.45],
                    scale: [crownRadius, crownRadius, Math.max((height - trunkHeight) * 0.62, 0.8)],
                    materialIndex: roleMaterial(item.role) });
    } else if (item.role.startsWith('linear')) {
      poles.push({ offset: [x, y, z], scale: [Math.max(item.attrs?.radius || 0.1, 0.06),
                                              Math.max(item.attrs?.radius || 0.1, 0.06),
                                              Math.max(height, 1)],
                   materialIndex: roleMaterial(item.role) });
    } else if (item.role.startsWith('instance')) {
      bodies.push({ offset: [x, y, z], scale: [Math.max(rx, 0.6), Math.max(ry, 0.6),
                                               Math.max(height, 0.6)],
                    materialIndex: roleMaterial(item.role) });
    }
  }

  return [
    { role: 'trunk', geometry: GEOMETRY.trunk, items: trunks },
    { role: 'crown', geometry: GEOMETRY.crown, items: crowns },
    { role: 'pole', geometry: GEOMETRY.pole, items: poles },
    { role: 'body', geometry: GEOMETRY.body, items: bodies },
  ];
}
