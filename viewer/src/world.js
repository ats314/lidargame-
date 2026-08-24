// Loads a compiled world bundle and derives everything the runtime needs that
// is *not* theme-dependent: GPU buffers, a walkable height field, building
// blockers for collision, and a picking structure for the inspector.

export async function loadWorld(baseUrl) {
  const [header, blob] = await Promise.all([
    fetch(`${baseUrl}/world.json`).then((r) => {
      if (!r.ok) throw new Error(`world.json: ${r.status} ${r.statusText}`);
      return r.json();
    }),
    fetch(`${baseUrl}/world.bin`).then((r) => {
      if (!r.ok) throw new Error(`world.bin: ${r.status} ${r.statusText}`);
      return r.arrayBuffer();
    }),
  ]);

  const { mesh, points, vertexStride } = header;
  const floatStride = vertexStride / 4;

  const vertexBytes = blob.slice(mesh.vertexOffset, mesh.vertexOffset + mesh.vertexCount * vertexStride);
  const vertexFloats = new Float32Array(vertexBytes);
  const vertexUints = new Uint32Array(vertexBytes);
  const indices = new Uint32Array(blob.slice(mesh.indexOffset, mesh.indexOffset + mesh.indexCount * 4));

  // De-interleaved views for CPU work (rule evaluation, picking, heightfield).
  const count = mesh.vertexCount;
  const context = new Uint32Array(count);
  const role = new Uint32Array(count);
  const node = new Uint32Array(count);
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const f = i * floatStride;
    positions[i * 3] = vertexFloats[f];
    positions[i * 3 + 1] = vertexFloats[f + 1];
    positions[i * 3 + 2] = vertexFloats[f + 2];
    context[i] = vertexUints[f + 8];
    role[i] = vertexUints[f + 9];
    node[i] = vertexUints[f + 10];
  }

  let pointData = null;
  if (points && points.count > 0) {
    const bytes = blob.slice(points.offset, points.offset + points.count * points.stride);
    pointData = { floats: new Float32Array(bytes), uints: new Uint32Array(bytes), count: points.count };
  }

  const graph = new Map(header.graph.nodes.map((n) => [n.id, n]));
  const nodeNames = buildNodeNames(header, graph);

  return {
    header,
    vertexBytes,
    vertexFloats,
    indices,
    positions,
    context,
    role,
    node,
    nodeNames,
    points: pointData,
    graph,
    edges: header.graph.edges,
    instances: header.instances || [],
    roles: header.roles.map((r) => r.id),
    roleInfo: header.roles,
    contextFlags: header.contextFlags,
    bounds: header.bounds,
    heightfield: buildHeightfield(positions, role, header),
    blockers: buildBlockers(header, graph),
  };
}

// The mesh stores a small integer per vertex; the graph stores string ids. The
// exporter emits terrain as slot 0 and then one slot per surface node, in
// node order, so the mapping is just that traversal repeated here.
function buildNodeNames(header, graph) {
  const names = ['terrain'];
  for (const n of header.graph.nodes) {
    if (n.kind === 'surface' && n.geometry && n.geometry.kind === 'tiled_plane') names.push(n.id);
  }
  return names;
}

/** Coarse max-height grid over walkable roles, used for gravity in walk mode.
 *
 * Water is terrain and is not walkable. The compiler fills a surveyed canal at
 * an inferred level, which is what stops the ground ending at the quay -- but a
 * surface you can stroll across is a worse lie than a hole, so water cells are
 * flagged here and the walk controller refuses to enter them. Fly mode is
 * unaffected: looking down at a canal from above is the point of fly mode.
 */
function buildHeightfield(positions, role, header, cell = 1.0) {
  const [lo, hi] = header.bounds;
  const nx = Math.max(1, Math.ceil((hi[0] - lo[0]) / cell) + 1);
  const ny = Math.max(1, Math.ceil((hi[1] - lo[1]) / cell) + 1);
  const grid = new Float32Array(nx * ny).fill(-Infinity);
  const walkable = new Set();
  const wet = new Set();
  header.roles.forEach((r, i) => {
    if (r.id.startsWith('terrain')) walkable.add(i);
    if (r.id === 'terrain.water') wet.add(i);
  });
  const water = new Uint8Array(nx * ny);

  const count = positions.length / 3;
  for (let i = 0; i < count; i++) {
    if (!walkable.has(role[i])) continue;
    const gx = Math.floor((positions[i * 3] - lo[0]) / cell);
    const gy = Math.floor((positions[i * 3 + 1] - lo[1]) / cell);
    if (gx < 0 || gy < 0 || gx >= nx || gy >= ny) continue;
    const k = gy * nx + gx;
    const z = positions[i * 3 + 2];
    if (z > grid[k]) grid[k] = z;
    if (wet.has(role[i])) water[k] = 1;
  }
  // Fill unseen cells so the player never falls through a scan shadow.
  let fallback = 0;
  let seen = 0;
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] > -Infinity) { fallback += grid[i]; seen++; }
  }
  fallback = seen ? fallback / seen : 0;
  for (let i = 0; i < grid.length; i++) if (grid[i] === -Infinity) grid[i] = fallback;

  return {
    cell, nx, ny, origin: [lo[0], lo[1]], grid, water,
    isWater(x, y) {
      const ix = Math.min(this.nx - 1, Math.max(0, Math.floor((x - this.origin[0]) / this.cell)));
      const iy = Math.min(this.ny - 1, Math.max(0, Math.floor((y - this.origin[1]) / this.cell)));
      return this.water[iy * this.nx + ix] === 1;
    },
    sample(x, y) {
      const fx = (x - this.origin[0]) / this.cell;
      const fy = (y - this.origin[1]) / this.cell;
      const ix = Math.min(this.nx - 1, Math.max(0, Math.floor(fx)));
      const iy = Math.min(this.ny - 1, Math.max(0, Math.floor(fy)));
      return this.grid[iy * this.nx + ix];
    },
  };
}

/** Building footprints as AABBs, so walking does not clip through walls. */
function buildBlockers(header, graph) {
  const out = [];
  for (const n of header.graph.nodes) {
    if (n.role !== 'volume.building' || !n.attrs || !n.attrs.footprint) continue;
    const [lo, hi] = n.attrs.footprint;
    out.push({ id: n.id, minX: lo[0], minY: lo[1], maxX: hi[0], maxY: hi[1], height: n.attrs.height || 10 });
  }
  return out;
}

/**
 * Brute-force ray/triangle pick. ~70k triangles is well inside frame budget for
 * a click, and it avoids a second render target purely for selection.
 */
export function pickTriangle(world, origin, direction) {
  const { positions, indices } = world;
  let bestT = Infinity;
  let bestTri = -1;
  const [ox, oy, oz] = origin;
  const [dx, dy, dz] = direction;

  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;
    const ax = positions[a], ay = positions[a + 1], az = positions[a + 2];
    const e1x = positions[b] - ax, e1y = positions[b + 1] - ay, e1z = positions[b + 2] - az;
    const e2x = positions[c] - ax, e2y = positions[c + 1] - ay, e2z = positions[c + 2] - az;

    const px = dy * e2z - dz * e2y;
    const py = dz * e2x - dx * e2z;
    const pz = dx * e2y - dy * e2x;
    const det = e1x * px + e1y * py + e1z * pz;
    if (det > -1e-8 && det < 1e-8) continue;
    const inv = 1 / det;
    const tx = ox - ax, ty = oy - ay, tz = oz - az;
    const u = (tx * px + ty * py + tz * pz) * inv;
    if (u < 0 || u > 1) continue;
    const qx = ty * e1z - tz * e1y;
    const qy = tz * e1x - tx * e1z;
    const qz = tx * e1y - ty * e1x;
    const v = (dx * qx + dy * qy + dz * qz) * inv;
    if (v < 0 || u + v > 1) continue;
    const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
    if (t > 0.05 && t < bestT) { bestT = t; bestTri = i; }
  }
  if (bestTri < 0) return null;
  const vertex = world.indices[bestTri];
  return {
    distance: bestT,
    vertex,
    nodeSlot: world.node[vertex],
    nodeId: world.nodeNames[world.node[vertex]] ?? null,
    role: world.roles[world.role[vertex]],
    context: world.context[vertex],
    point: [ox + dx * bestT, oy + dy * bestT, oz + dz * bestT],
  };
}

export function decodeContext(flags, mask) {
  return Object.entries(flags).filter(([, bit]) => mask & bit).map(([name]) => name);
}
