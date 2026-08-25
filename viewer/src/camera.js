// First-person controls with two modes: walk (gravity, ground following, walls
// block you) and fly (free 6-DoF). Collision uses the height field and building
// footprints the loader derived from the world graph -- no physics engine, and
// no geometry the compiler did not already produce.

import { clamp, forwardFrom, mat4Multiply, mat4Perspective, mat4ViewZUp, rightFrom } from './math.js';

const EYE_HEIGHT = 1.68;
const RADIUS = 0.4;
const GRAVITY = 18.0;

export class Camera {
  constructor(world) {
    this.world = world;
    this.position = new Float32Array(3);
    this.yaw = 0.6;
    this.pitch = -0.05;
    this.verticalVelocity = 0;
    this.fly = false;
    this.speed = 6.0;
    this.onGround = false;
    this.view = new Float32Array(16);
    this.projection = new Float32Array(16);
    this.viewProjection = new Float32Array(16);
    this.keys = new Set();
    this.reset();
  }

  reset() {
    const [lo, hi] = this.world.bounds;
    const cx = (lo[0] + hi[0]) / 2;
    const cy = (lo[1] + hi[1]) / 2;
    this.position[0] = cx;
    this.position[1] = cy;
    this.position[2] = this.world.heightfield.sample(cx, cy) + EYE_HEIGHT;
    this.yaw = 0.6;
    this.pitch = -0.05;
    this.verticalVelocity = 0;
  }

  attach(canvas) {
    canvas.addEventListener('click', () => {
      if (!document.pointerLockElement) canvas.requestPointerLock();
    });
    document.addEventListener('mousemove', (event) => {
      if (document.pointerLockElement !== canvas) return;
      const sensitivity = 0.0022;
      this.yaw -= event.movementX * sensitivity;
      this.pitch = clamp(this.pitch - event.movementY * sensitivity, -1.5, 1.5);
    });
    window.addEventListener('keydown', (event) => {
      if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') return;
      this.keys.add(event.code);
      if (event.code === 'Space') event.preventDefault();
    });
    window.addEventListener('keyup', (event) => this.keys.delete(event.code));
    window.addEventListener('blur', () => this.keys.clear());
  }

  update(dt) {
    const forward = forwardFrom(this.yaw, this.fly ? this.pitch : 0);
    const right = rightFrom(this.yaw);

    let dx = 0;
    let dy = 0;
    let dz = 0;
    if (this.keys.has('KeyW')) { dx += forward[0]; dy += forward[1]; dz += forward[2]; }
    if (this.keys.has('KeyS')) { dx -= forward[0]; dy -= forward[1]; dz -= forward[2]; }
    if (this.keys.has('KeyD')) { dx += right[0]; dy += right[1]; }
    if (this.keys.has('KeyA')) { dx -= right[0]; dy -= right[1]; }

    const length = Math.hypot(dx, dy);
    if (length > 1e-4) { dx /= length; dy /= length; }

    const sprint = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 3.2 : 1;
    const speed = this.speed * sprint;

    if (this.fly) {
      let lift = dz;
      if (this.keys.has('Space')) lift += 1;
      if (this.keys.has('ControlLeft') || this.keys.has('KeyC')) lift -= 1;
      this.position[0] += dx * speed * dt;
      this.position[1] += dy * speed * dt;
      this.position[2] += lift * speed * dt;
      this.verticalVelocity = 0;
      return;
    }

    const [movedX, movedY] = this._resolve(
      this.position[0] + dx * speed * dt,
      this.position[1] + dy * speed * dt);
    this.position[0] = movedX;
    this.position[1] = movedY;

    if (this.keys.has('Space') && this.onGround) {
      this.verticalVelocity = 6.0;
      this.onGround = false;
    }
    this.verticalVelocity -= GRAVITY * dt;
    this.position[2] += this.verticalVelocity * dt;

    const target = this.world.heightfield.sample(this.position[0], this.position[1]) + EYE_HEIGHT;
    if (this.position[2] <= target) {
      this.position[2] = target;
      this.verticalVelocity = 0;
      this.onGround = true;
    } else {
      this.onGround = false;
    }
  }

  /** Push out of building footprints along the shallowest axis, so you slide.
   *  A step into water is refused outright rather than slid along: a canal is
   *  not an obstacle with a side to scrape past, it is the edge of the walkable
   *  world, and the quay is where you stop. */
  _resolve(x, y) {
    let outX = x;
    let outY = y;
    const field = this.world.heightfield;
    if (!this.fly && field.isWater && field.isWater(outX, outY)) {
      // Try each axis alone, so walking along a canal bank still slides.
      if (!field.isWater(outX, this.position[1])) outY = this.position[1];
      else if (!field.isWater(this.position[0], outY)) outX = this.position[0];
      else { outX = this.position[0]; outY = this.position[1]; }
    }
    for (const b of this.world.blockers) {
      if (!(outX > b.minX - RADIUS && outX < b.maxX + RADIUS
            && outY > b.minY - RADIUS && outY < b.maxY + RADIUS)) continue;
      if (this.position[2] > b.height + 0.6) continue;      // standing on the roof
      const toMinX = outX - (b.minX - RADIUS);
      const toMaxX = (b.maxX + RADIUS) - outX;
      const toMinY = outY - (b.minY - RADIUS);
      const toMaxY = (b.maxY + RADIUS) - outY;
      const smallest = Math.min(toMinX, toMaxX, toMinY, toMaxY);
      if (smallest === toMinX) outX = b.minX - RADIUS;
      else if (smallest === toMaxX) outX = b.maxX + RADIUS;
      else if (smallest === toMinY) outY = b.minY - RADIUS;
      else outY = b.maxY + RADIUS;
    }
    return [outX, outY];
  }

  updateMatrices(aspect) {
    mat4Perspective(this.projection, (68 * Math.PI) / 180, aspect, 0.08, 900);
    mat4ViewZUp(this.view, this.position, this.yaw, this.pitch);
    return mat4Multiply(this.viewProjection, this.projection, this.view);
  }

  /** Ray through the crosshair, in world space. */
  ray() {
    return {
      origin: [this.position[0], this.position[1], this.position[2]],
      direction: forwardFrom(this.yaw, this.pitch),
    };
  }
}
