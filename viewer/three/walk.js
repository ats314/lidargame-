/**
 * A walkable render of a compiled world, at eye height.
 *
 * The existing viewer is a deliberate dependency-free WebGL2 walkthrough and is
 * the right tool for inspecting what the compiler produced -- roles, context
 * masks, provenance. This is the other question: does the place look like a
 * place. That needs shadows, image-based lighting, tone mapping and normal
 * maps, which is a renderer rather than a viewer, so three.js does it.
 *
 * Nothing here is compiler logic. The glTF arrives already materialised by
 * `backends/gltf.py`; this only lights it. Anything that would change what the
 * world *is* belongs upstream of the backend boundary, not in a render page.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const params = new URLSearchParams(location.search);
const MODEL = params.get('model') || '../../build/walk/walk.gltf';
/**
 * Photogrammetric meshes are already lit.
 *
 * A reality mesh's texture is a photograph of a building under the sun that
 * was shining when the aircraft flew. Running it through a PBR shader with our
 * own sun and environment multiplies one lighting solution by another and the
 * result is mud -- which is exactly what the first Helsinki render was. For
 * that input the texture *is* the answer, so the shading model has to get out
 * of the way.
 *
 * Generated worlds are the opposite: their albedo is unlit by construction and
 * they need every bit of the lighting rig.
 */
const UNLIT = params.get('unlit') === '1';
const EYE_M = 1.7;                    // standing eye height, metres
const WALK_MS = 1.6;                  // metres per second
const RUN_MULTIPLIER = 3.4;

const stage = document.getElementById('stage');
const status = document.getElementById('status');

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
// Without tone mapping a sunlit facade clips to white and every material
// reads as the same overexposed grey. This is most of "it looks like a
// render" versus "it looks like a photograph".
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(62, 1, 0.1, 4000);

/**
 * Sky as an environment map, built here rather than fetched.
 *
 * A normal map and a roughness map do nothing under a single directional
 * light -- there is no environment for a rough surface to reflect, so every
 * facade returns the same lambert term and the relief we just started
 * exporting stays invisible. A gradient sky through PMREM is the cheapest
 * honest environment: sky above, bounce from the ground below, horizon
 * between. No HDRI to ship and no CDN to depend on.
 */
function skyEnvironment(sunDirection) {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size / 2;
  const context = canvas.getContext('2d');
  const gradient = context.createLinearGradient(0, 0, 0, size / 2);
  gradient.addColorStop(0.00, '#5d86bd');       // zenith
  gradient.addColorStop(0.48, '#b9cbdc');       // horizon haze
  gradient.addColorStop(0.52, '#6d6a63');       // ground bounce, warmer and darker
  gradient.addColorStop(1.00, '#3b3934');
  context.fillStyle = gradient;
  context.fillRect(0, 0, size, size / 2);

  // A bright spot where the sun is, so glancing highlights land somewhere
  // rather than being uniform across every surface.
  const u = (Math.atan2(sunDirection.x, -sunDirection.z) / (2 * Math.PI) + 0.5) * size;
  const v = Math.acos(THREE.MathUtils.clamp(sunDirection.y, -1, 1)) / Math.PI * (size / 2);
  const glow = context.createRadialGradient(u, v, 0, u, v, size * 0.12);
  glow.addColorStop(0, 'rgba(255, 248, 232, 0.95)');
  glow.addColorStop(1, 'rgba(255, 248, 232, 0)');
  context.fillStyle = glow;
  context.fillRect(0, 0, size, size / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  const pmrem = new THREE.PMREMGenerator(renderer);
  const target = pmrem.fromEquirectangular(texture);
  pmrem.dispose();
  texture.dispose();
  return target.texture;
}

/**
 * Ground height under a point.
 *
 * The *first* thing a downward ray meets is a roof, so taking hit[0] stands
 * the camera on the building instead of in the street -- the Helsinki street
 * shot came out on a rooftop next to a chimney. Aerial photogrammetry has no
 * interior, so the lowest surface at an x,z is the ground.
 */
function groundUnder(target, x, z, top) {
  const probe = new THREE.Raycaster();
  probe.set(new THREE.Vector3(x, top, z), new THREE.Vector3(0, -1, 0));
  const hits = probe.intersectObject(target, true);
  return hits.length ? hits[hits.length - 1].point.y : null;
}

const sun = new THREE.DirectionalLight(0xfff2dc, 3.1);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0006;
sun.shadow.normalBias = 0.05;
scene.add(sun, sun.target);

// Fills the gap the sun leaves on north faces. Kept low: the environment map
// is doing most of the ambient work, and stacking a bright hemisphere on top
// of it is how a scene ends up flat and milky.
scene.add(new THREE.HemisphereLight(0x9fb6d0, 0x4a453d, 0.35));

let radius = 200;
let sceneCentre = new THREE.Vector3();

function placeSun() {
  const elevation = THREE.MathUtils.degToRad(Number(document.getElementById('sun').value));
  const bearing = THREE.MathUtils.degToRad(Number(document.getElementById('bearing').value));
  const direction = new THREE.Vector3(
    Math.cos(elevation) * Math.sin(bearing),
    Math.sin(elevation),
    Math.cos(elevation) * Math.cos(bearing),
  );
  sun.position.copy(sceneCentre).addScaledVector(direction, radius * 2.2);
  sun.target.position.copy(sceneCentre);
  sun.target.updateMatrixWorld();

  // The shadow frustum has to wrap the whole model or shadows simply stop at
  // its edge, which reads as a lighting bug rather than a clipping one.
  const shadow = sun.shadow.camera;
  shadow.left = -radius; shadow.right = radius;
  shadow.top = radius; shadow.bottom = -radius;
  shadow.near = radius * 0.4; shadow.far = radius * 4.6;
  shadow.updateProjectionMatrix();

  // Low sun means warmer, dimmer light; midday is close to white.
  const warmth = THREE.MathUtils.smoothstep(Math.sin(elevation), 0.05, 0.7);
  sun.color.setHSL(0.09 - 0.02 * warmth, 0.55 - 0.45 * warmth, 0.5 + 0.08 * warmth);
  sun.intensity = 0.9 + 2.6 * warmth;

  scene.environment = skyEnvironment(direction);
  scene.background = scene.environment;
}

/** Walk controls: pointer lock for look, WASD for move, eye height held. */
const keys = new Set();
const euler = new THREE.Euler(0, 0, 0, 'YXZ');
let locked = false;

renderer.domElement.addEventListener('click', () => renderer.domElement.requestPointerLock());
document.addEventListener('pointerlockchange', () => {
  locked = document.pointerLockElement === renderer.domElement;
});
document.addEventListener('mousemove', (event) => {
  if (!locked) return;
  euler.setFromQuaternion(camera.quaternion);
  euler.y -= event.movementX * 0.0022;
  euler.x -= event.movementY * 0.0022;
  // Stop short of straight up so the horizon never flips.
  euler.x = THREE.MathUtils.clamp(euler.x, -Math.PI / 2 + 0.02, Math.PI / 2 - 0.02);
  camera.quaternion.setFromEuler(euler);
});
addEventListener('keydown', (event) => {
  keys.add(event.code);
  if (event.code === 'Space') event.preventDefault();
});
addEventListener('keyup', (event) => keys.delete(event.code));

const down = new THREE.Raycaster();
down.far = 500;
let world = null;
let groundHeld = true;
let frozen = false;

function step(dt) {
  const speed = WALK_MS * (keys.has('ShiftLeft') || keys.has('ShiftRight') ? RUN_MULTIPLIER : 1) * dt;
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.y = 0;
  if (forward.lengthSq() < 1e-9) forward.set(0, 0, -1);
  forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();

  const move = new THREE.Vector3();
  if (keys.has('KeyW')) move.add(forward);
  if (keys.has('KeyS')) move.sub(forward);
  if (keys.has('KeyD')) move.add(right);
  if (keys.has('KeyA')) move.sub(right);
  if (move.lengthSq() > 0) camera.position.addScaledVector(move.normalize(), speed);

  // A pose set for a screenshot must survive the frame loop. Without this the
  // ground-follow below immediately drags the camera back to eye height, so
  // every posed shot silently became a ground-level shot -- which is why
  // cameras kept ending up jammed against a facade.
  if (frozen) {
    if (keys.size) frozen = false;
    else return;
  }

  // Free-fly while space is held, otherwise stay a person's height above
  // whatever is underfoot. A fixed height puts the camera inside a hill.
  if (keys.has('Space')) {
    camera.position.y += speed;
    groundHeld = false;
  } else if (world) {
    down.set(new THREE.Vector3(camera.position.x, camera.position.y + 80, camera.position.z),
             new THREE.Vector3(0, -1, 0));
    const hits = down.intersectObject(world, true);
    if (hits.length) {
      const target = hits[hits.length - 1].point.y + EYE_M;
      camera.position.y += (target - camera.position.y) * (groundHeld ? Math.min(1, dt * 9) : Math.min(1, dt * 2.5));
      groundHeld = true;
    }
  }
}

function resize() {
  const width = stage.clientWidth, height = stage.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

for (const id of ['sun', 'bearing']) {
  document.getElementById(id).addEventListener('input', placeSun);
}
document.getElementById('exposure').addEventListener('input', (event) => {
  renderer.toneMappingExposure = Number(event.target.value);
});

const clock = new THREE.Clock();
let frames = 0, fpsAt = 0, fps = 0;

function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  step(dt);
  renderer.render(scene, camera);

  frames += 1;
  const now = performance.now();
  if (now - fpsAt > 500) {
    fps = Math.round(frames * 1000 / (now - fpsAt));
    frames = 0; fpsAt = now;
    const p = camera.position;
    status.textContent =
      `${fps} fps   x ${p.x.toFixed(1)}  y ${p.y.toFixed(1)}  z ${p.z.toFixed(1)}`;
  }
  requestAnimationFrame(frame);
}

new GLTFLoader().load(MODEL, (gltf) => {
  world = gltf.scene;
  world.traverse((node) => {
    if (!node.isMesh) return;
    node.castShadow = true;
    node.receiveShadow = true;
    // Facades are single-sided quads in places; back-face culling on a wall
    // whose winding went the other way renders a hole straight through the
    // building. Trust the geometry, not the material flag.
    if (!node.material) return;
    node.material.side = THREE.FrontSide;
    if (UNLIT) {
      const lit = node.material;
      node.material = new THREE.MeshBasicMaterial({
        map: lit.map, color: lit.color, side: THREE.FrontSide,
      });
      node.castShadow = false;
      node.receiveShadow = false;
    }
  });
  scene.add(world);

  const box = new THREE.Box3().setFromObject(world);
  box.getCenter(sceneCentre);
  radius = Math.max(30, box.getSize(new THREE.Vector3()).length() * 0.5);

  // Start on the ground at the edge, looking in -- a street-level arrival,
  // not an architect's axonometric.
  camera.position.set(sceneCentre.x, box.min.y + EYE_M, box.max.z - radius * 0.15);
  camera.lookAt(sceneCentre.x, box.min.y + EYE_M + 3, sceneCentre.z);

  // Distance haze only; the old viewer drowned a block at 63% opacity by
  // 100 m, so this starts well beyond the model and never closes in.
  scene.fog = new THREE.Fog(0xa8bccd, radius * 1.6, radius * 6.5);

  placeSun();
  resize();
  status.textContent = 'ready';
  frame();

  // Debug handle, same contract as the other viewer: a screenshot tool has to
  // be able to place the camera without pointer lock, and a black PNG with no
  // handle is indistinguishable from a black PNG with a broken one.
  window.walk = {
    THREE, scene, camera, renderer, world, sun,
    bounds: { min: box.min.toArray(), max: box.max.toArray() },
    centre: sceneCentre.toArray(),
    look(eye, target) {
      camera.position.set(eye[0], eye[1], eye[2]);
      camera.lookAt(target[0], target[1], target[2]);
      euler.setFromQuaternion(camera.quaternion);
      groundHeld = false;
      frozen = true;          // hold it until somebody actually walks
    },
    /**
     * A standing spot with room around it, and something to look at.
     *
     * Fixed camera offsets from the model centre are tuned to one block and
     * wrong for the next: the Denver poses put the lens flat against an
     * Amsterdam facade. A street is not at a known offset, but it is
     * findable -- it is the place with the most clearance around it.
     *
     * Done by rasterising rather than raycasting. The obvious implementation
     * casts a ring of rays from a grid of candidates, and that is thousands of
     * rays against a mesh with no BVH -- 1.6 million triangles times a few
     * thousand rays does not finish, which is how the first version of this
     * hung a render. Instead: project every vertex sitting in the band a
     * person occupies into a coarse 2D grid, distance-transform it, and take
     * the cell furthest from anything. One pass over the vertices, and the
     * answer is the same one the rays would have given.
     */
    findStreet({ cellM = 2.0, bandM = [1.0, 4.0],
                 minClearanceM = 3.0, maxClearanceM = 18.0 } = {}) {
      const box = new THREE.Box3().setFromObject(world);
      const floor = box.min.y;
      const nx = Math.max(8, Math.ceil((box.max.x - box.min.x) / cellM));
      const nz = Math.max(8, Math.ceil((box.max.z - box.min.z) / cellM));
      const blocked = new Uint8Array(nx * nz);

      // Terrain sits below the band and roofs above it, so both drop out
      // without either having to be identified.
      const low = floor + bandM[0], high = floor + bandM[1];
      const point = new THREE.Vector3();
      world.traverse((node) => {
        const position = node.isMesh && node.geometry && node.geometry.attributes.position;
        if (!position) return;
        for (let i = 0; i < position.count; i++) {
          point.fromBufferAttribute(position, i).applyMatrix4(node.matrixWorld);
          if (point.y < low || point.y > high) continue;
          const ix = Math.floor((point.x - box.min.x) / cellM);
          const iz = Math.floor((point.z - box.min.z) / cellM);
          if (ix >= 0 && ix < nx && iz >= 0 && iz < nz) blocked[iz * nx + ix] = 1;
        }
      });

      // Chamfer distance transform, forward then backward. Units are cells.
      const far = nx + nz;
      const distance = new Float32Array(nx * nz).fill(far);
      for (let i = 0; i < blocked.length; i++) if (blocked[i]) distance[i] = 0;
      const relax = (i, j, cost) => {
        if (distance[j] + cost < distance[i]) distance[i] = distance[j] + cost;
      };
      for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
        const i = z * nx + x;
        if (x > 0) relax(i, i - 1, 1);
        if (z > 0) relax(i, i - nx, 1);
        if (x > 0 && z > 0) relax(i, i - nx - 1, 1.41421);
      }
      for (let z = nz - 1; z >= 0; z--) for (let x = nx - 1; x >= 0; x--) {
        const i = z * nx + x;
        if (x < nx - 1) relax(i, i + 1, 1);
        if (z < nz - 1) relax(i, i + nx, 1);
        if (x < nx - 1 && z < nz - 1) relax(i, i + nx + 1, 1.41421);
      }

      // A street is open *and enclosed*. Maximising openness alone picks the
      // empty ground past the edge of the block and renders a photograph of a
      // field -- which is exactly what the first version did, reporting 118 m
      // of clearance and framing nothing.
      //
      // So: keep only cells whose clearance is street-sized, and among those
      // take the one deepest inside the built-up area, measured from the
      // centroid of everything that is blocked.
      let sumX = 0, sumZ = 0, count = 0;
      for (let i = 0; i < blocked.length; i++) {
        if (!blocked[i]) continue;
        sumX += i % nx; sumZ += Math.floor(i / nx); count++;
      }
      if (!count) return null;
      const coreX = sumX / count, coreZ = sumZ / count;

      const minCells = minClearanceM / cellM, maxCells = maxClearanceM / cellM;
      let bestIndex = -1, bestScore = Infinity;
      for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
        const i = z * nx + x;
        const d = distance[i];
        if (d < minCells || d > maxCells) continue;
        // Distance from the core, less a small bonus for standing in the
        // wider part of a street rather than pressed against a wall.
        const score = Math.hypot(x - coreX, z - coreZ) - d * 1.5;
        if (score < bestScore) { bestScore = score; bestIndex = i; }
      }
      if (bestIndex < 0) return null;
      const bestDistance = distance[bestIndex];

      const gx = bestIndex % nx, gz = Math.floor(bestIndex / nx);
      const x = box.min.x + (gx + 0.5) * cellM;
      const z = box.min.z + (gz + 0.5) * cellM;

      // Ground under the spot, raycast once. A fixed height puts the camera
      // underground on any block with relief.
      const ground = groundUnder(world, x, z, box.max.y + 5);
      const eyeY = (ground === null ? floor : ground) + EYE_M;

      // Aim along the gradient of the distance field, which runs down the
      // street rather than into the wall behind you.
      const at = (ix, iz) => distance[Math.min(nz - 1, Math.max(0, iz)) * nx
                                     + Math.min(nx - 1, Math.max(0, ix))];
      const along = new THREE.Vector3(at(gx + 1, gz) - at(gx - 1, gz), 0,
                                      at(gx, gz + 1) - at(gx, gz - 1));
      if (along.lengthSq() < 1e-6) along.set(1, 0, 0);
      along.normalize();

      const eye = new THREE.Vector3(x, eyeY, z);
      const target = eye.clone().addScaledVector(along, 60).setY(eyeY + 6);
      this.look(eye.toArray(), target.toArray());
      return { eye: eye.toArray(), clearanceM: bestDistance * cellM };
    },
    /**
     * Stand at x,z with the eye a real height above the ground there.
     *
     * The floor is not the model's lowest vertex. A reality mesh carries
     * below-ground geometry, so anchoring to bounds.min.y put a "street level"
     * camera fifteen metres under the pavement looking up at the underside of
     * the city -- which is what every street shot of Helsinki was.
     */
    standAt(x, z, height, target) {
      const box = new THREE.Box3().setFromObject(world);
      const ground = groundUnder(world, x, z, box.max.y + 10);
      if (ground === null) return null;
      const eyeY = ground + height;
      this.look([x, eyeY, z], [target[0], eyeY + target[1], target[2]]);
      return { ground, eye: eyeY };
    },
    setSun(elevation, bearing) {
      document.getElementById('sun').value = elevation;
      document.getElementById('bearing').value = bearing;
      placeSun();
    },
    ready: true,
  };
}, (event) => {
  if (event.total) status.textContent = `loading ${Math.round(event.loaded / event.total * 100)}%`;
}, (error) => {
  status.textContent = `failed to load ${MODEL}\n${error.message || error}`;
});
