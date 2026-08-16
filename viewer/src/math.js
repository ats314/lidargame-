// Small math helpers. Column-major 4x4 matrices, WebGL convention.
//
// Everything here works in the world's own frame: **Z-up**, because that is what
// survey and airborne LiDAR use and converting on ingest would mean converting
// back for every export. The view matrix handles the up-axis directly, so no
// stage of the viewer ever swizzles coordinates.

export const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
export const lerp = (a, b, t) => a + (b - a) * t;
export const DEG = Math.PI / 180;

export function mat4Perspective(out, fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[11] = -1;
  out[10] = (far + near) / (near - far);
  out[14] = (2 * far * near) / (near - far);
  return out;
}

/** Unit forward vector in a Z-up world. Yaw is measured from +X toward +Y. */
export function forwardFrom(yaw, pitch, out = [0, 0, 0]) {
  const cp = Math.cos(pitch);
  out[0] = Math.cos(yaw) * cp;
  out[1] = Math.sin(yaw) * cp;
  out[2] = Math.sin(pitch);
  return out;
}

/** Right vector on the horizontal plane, perpendicular to forward. */
export function rightFrom(yaw, out = [0, 0, 0]) {
  out[0] = Math.sin(yaw);
  out[1] = -Math.cos(yaw);
  out[2] = 0;
  return out;
}

/** View matrix for a Z-up world from eye position + yaw/pitch. */
export function mat4ViewZUp(out, eye, yaw, pitch) {
  const f = forwardFrom(yaw, pitch);
  const r = rightFrom(yaw);
  // up = right x forward
  const ux = r[1] * f[2] - r[2] * f[1];
  const uy = r[2] * f[0] - r[0] * f[2];
  const uz = r[0] * f[1] - r[1] * f[0];

  out[0] = r[0]; out[4] = r[1]; out[8] = r[2];
  out[1] = ux; out[5] = uy; out[9] = uz;
  out[2] = -f[0]; out[6] = -f[1]; out[10] = -f[2];
  out[3] = 0; out[7] = 0; out[11] = 0;
  out[12] = -(r[0] * eye[0] + r[1] * eye[1] + r[2] * eye[2]);
  out[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
  out[14] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
  out[15] = 1;
  return out;
}

export function mat4Multiply(out, a, b) {
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
    out[c * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
    out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
    out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
  return out;
}
