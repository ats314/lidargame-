// Entry point: load a compiled world, bind a theme, run the loop.

import { Camera } from './camera.js';
import { createContext, resizeCanvas } from './gl.js';
import { Hud } from './hud.js';
import { Renderer } from './render.js';
import {
  environmentOf, loadTheme, loadThemeTextures, materialUniforms, resolveMaterials,
  resolveOne,
} from './theme.js';
import { loadWorld, pickTriangle } from './world.js';

const params = new URLSearchParams(location.search);
const WORLD_URL = params.get('world') || './world';

async function boot() {
  const canvas = document.getElementById('view');
  const overlay = document.getElementById('overlay');
  const gl = createContext(canvas);

  let world;
  try {
    world = await loadWorld(WORLD_URL);
  } catch (error) {
    const hud = new Hud(overlay, {
      header: { name: '—', summary: { nodes: 0 }, themes: [], crs: '' },
      indices: [], contextFlags: {}, graph: new Map(), edges: [],
    }, {});
    hud.setError(error.message);
    console.error(error);
    return;
  }

  const renderer = new Renderer(gl, world);
  const camera = new Camera(world);
  camera.attach(canvas);

  const hud = new Hud(overlay, world, {
    onTheme: (id) => switchTheme(id),
    onDebugMode: (mode) => { renderer.debugMode = mode; },
    onPoints: (on) => { renderer.showPoints = on; },
    onInstances: (on) => { renderer.showInstances = on; },
    onFly: (on) => { camera.fly = on; },
    onContextMask: (mask) => { renderer.contextMask = mask; },
  });

  const themeCache = new Map();
  let activeTheme = null;

  async function switchTheme(id) {
    let entry = themeCache.get(id);
    if (!entry) {
      const theme = await loadTheme(WORLD_URL, id);
      const texture = await loadThemeTextures(gl, theme, theme.textureSize || 256);
      // Resolution is per distinct (role, context) pair, not per vertex.
      const resolved = resolveMaterials(world, theme);
      entry = { theme, texture, resolved, uniforms: materialUniforms(theme),
                environment: environmentOf(theme) };
      themeCache.set(id, entry);
    }
    activeTheme = entry;
    renderer.applyTheme(entry.theme, entry.resolved.indices, entry.texture,
                        entry.environment, entry.uniforms);
    hud.setTheme(entry.theme, entry.resolved);
    document.body.dataset.theme = id;
  }

  const themes = world.header.themes;
  if (!themes.length) {
    hud.setError('This world was compiled without any theme packs.');
    return;
  }
  await switchTheme(themes[0]);

  window.addEventListener('keydown', (event) => {
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') return;
    switch (event.code) {
      case 'KeyE': inspect(); break;
      case 'KeyF': camera.fly = !camera.fly; break;
      case 'KeyP': renderer.showPoints = !renderer.showPoints; break;
      case 'KeyR': camera.reset(); break;
      case 'Tab': {
        event.preventDefault();
        const index = themes.indexOf(activeTheme.theme.id);
        switchTheme(themes[(index + 1) % themes.length]);
        break;
      }
      default: break;
    }
  });
  canvas.addEventListener('mousedown', (event) => {
    if (document.pointerLockElement === canvas && event.button === 0) inspect();
  });

  function inspect() {
    const { origin, direction } = camera.ray();
    const pick = pickTriangle(world, origin, direction);
    renderer.highlightNode = pick ? pick.nodeSlot : 0xffffffff;
    hud.showPick(pick);
  }

  // Debug handle: inspect the loaded world, camera and active theme from the
  // console. Handy when a theme rule does not fire the way you expected.
  window.lidarworld = {
    world, camera, renderer,
    get theme() { return activeTheme.theme; },
    pick: (origin, direction) => pickTriangle(world, origin, direction),
    resolve: (role, context) => resolveOne(activeTheme.theme, role, context),
    switchTheme,
  };

  let last = performance.now();
  let frames = 0;
  let accumulated = 0;
  let fps = 0;

  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    frames++;
    accumulated += dt;
    if (accumulated > 0.5) {
      fps = frames / accumulated;
      frames = 0;
      accumulated = 0;
      const [x, y, z] = camera.position;
      hud.setStats(`${fps.toFixed(0)} fps · ${x.toFixed(1)}, ${y.toFixed(1)}, ${z.toFixed(1)} m`
        + ` · ${camera.fly ? 'fly' : 'walk'}`);
    }

    camera.update(dt);
    resizeCanvas(canvas, gl);
    camera.updateMatrices(canvas.width / Math.max(canvas.height, 1));
    renderer.render(camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

boot().catch((error) => {
  console.error(error);
  document.getElementById('overlay').innerHTML =
    `<section class="panel error"><h2>Startup failed</h2><p>${error.message}</p></section>`;
});
