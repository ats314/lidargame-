// Theme binding.
//
// This is the runtime half of the compiler's central claim: geometry is
// theme-independent. Switching theme runs `resolveMaterials` over the distinct
// (role, context) pairs in the world -- a few hundred, not a few hundred
// thousand -- uploads one small attribute buffer, and swaps a texture array.
// No vertex position is touched, nothing is re-meshed, nothing is re-fetched
// from the compiler.

import { createTextureArray, fillLayer, uploadLayer } from './gl.js';
import { MAX_MATERIALS } from './shaders.js';

export async function loadTheme(baseUrl, id) {
  const url = `${baseUrl}/themes/${id}/theme.json`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`theme ${id}: ${response.status} ${response.statusText}`);
  const theme = await response.json();
  theme.baseUrl = `${baseUrl}/themes/${id}`;
  return theme;
}

function roleMatches(roleId, pattern) {
  if (!pattern || pattern === '*') return true;
  return roleId === pattern || roleId.startsWith(`${pattern}.`);
}

/** Resolve one request. Rules arrive pre-sorted most-specific-first. */
export function resolveOne(theme, roleId, context) {
  for (const rule of theme.rules) {
    if (!roleMatches(roleId, rule.role)) continue;
    if (rule.all && (context & rule.all) !== rule.all) continue;
    if (rule.any && !(context & rule.any)) continue;
    if (rule.none && (context & rule.none)) continue;
    return { material: rule.material, rule };
  }
  return { material: theme.fallback ?? 0, rule: null };
}

/**
 * Per-vertex material index. Distinct (role, context) pairs are resolved once
 * and cached, which is why a theme swap is effectively free.
 */
export function resolveMaterials(world, theme) {
  const { role, context, roles } = world;
  const count = role.length;
  const out = new Uint32Array(count);
  const cache = new Map();
  for (let i = 0; i < count; i++) {
    const key = role[i] * 4294967296 + context[i];
    let material = cache.get(key);
    if (material === undefined) {
      material = resolveOne(theme, roles[role[i]] ?? 'unknown', context[i]).material;
      cache.set(key, material);
    }
    out[i] = material;
  }
  return { indices: out, distinctRequests: cache.size };
}

/** Flat uniform arrays for the surface shader. */
export function materialUniforms(theme) {
  const color = new Float32Array(MAX_MATERIALS * 4);
  const params = new Float32Array(MAX_MATERIALS * 4);
  const emissive = new Float32Array(MAX_MATERIALS * 4);
  theme.materials.slice(0, MAX_MATERIALS).forEach((m, i) => {
    const rgb = m.baseColor || [0.7, 0.7, 0.7];
    color.set([rgb[0], rgb[1], rgb[2], m.opacity ?? 1], i * 4);
    params.set([
      m.roughness ?? 0.85,
      m.metallic ?? 0,
      1 / Math.max(m.scale ?? 1, 0.01),
      m.textures && m.textures.albedo ? 1 : 0,
    ], i * 4);
    const e = m.emissive || [0, 0, 0];
    const strength = Math.max(e[0], e[1], e[2]) > 0 ? 1 : 0;
    emissive.set([e[0], e[1], e[2], strength], i * 4);
  });
  return { color, params, emissive };
}

export async function loadThemeTextures(gl, theme, size) {
  const layers = Math.min(theme.materials.length, MAX_MATERIALS);
  const texture = createTextureArray(gl, size, Math.max(layers, 1));

  await Promise.all(theme.materials.slice(0, MAX_MATERIALS).map(async (material, layer) => {
    const rgb = (material.baseColor || [0.7, 0.7, 0.7]).map((v) => Math.round(v * 255));
    if (!material.textures || !material.textures.albedo) {
      fillLayer(gl, texture, layer, rgb, size);
      return;
    }
    try {
      const response = await fetch(`${theme.baseUrl}/${material.textures.albedo}`);
      if (!response.ok) throw new Error(String(response.status));
      const bitmap = await createImageBitmap(await response.blob());
      uploadLayer(gl, texture, layer, bitmap, size);
      bitmap.close?.();
    } catch (error) {
      // A missing texture must not take the world down: fall back to base colour.
      console.warn(`texture ${material.id} unavailable (${error.message}); using base colour`);
      fillLayer(gl, texture, layer, rgb, size);
    }
  }));

  gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
  gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
  return texture;
}

export function environmentOf(theme) {
  const env = theme.environment || {};
  return {
    sky: env.sky || [0.5, 0.6, 0.72],
    horizon: env.horizon || [0.72, 0.76, 0.8],
    fog: env.fog || [0.66, 0.7, 0.75],
    fogDensity: env.fogDensity ?? 0.008,
    // Scale height of the haze layer, metres. Density at the ground is
    // fogDensity and falls off by 1/e every fogHeight above it, so a camera
    // that climbs out of the layer sees through it. 55 m is a little above
    // LoDo's roofline, which is what makes a rooftop view clear while a street
    // view keeps its depth.
    fogHeight: env.fogHeight ?? 55,
    sunDirection: env.sunDirection || [0.35, 0.55, 0.75],
    sunColor: env.sunColor || [1, 0.97, 0.9],
    ambient: env.ambient || [0.4, 0.42, 0.46],
    exposure: env.exposure ?? 1,
  };
}

/** Which material a specific surface got, and which rule decided it. */
export function explain(theme, roleId, context) {
  const { material, rule } = resolveOne(theme, roleId, context);
  const spec = theme.materials[material] || null;
  return { spec, rule, index: material };
}
