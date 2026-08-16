// GLSL ES 3.00 sources.
//
// The surface shader is where the architecture shows up at runtime: geometry
// arrives with a `role` and a `context` bitmask baked into the vertex stream,
// and the *only* theme-dependent input is aMaterial -- an index recomputed on
// the CPU whenever a theme is swapped. Nothing else is rebuilt.

export const MAX_MATERIALS = 32;

const LIGHTING = /* glsl */`
vec3 shade(vec3 albedo, vec3 normal, vec3 viewDir, float roughness, float metallic,
           vec3 sunDir, vec3 sunColor, vec3 ambient) {
  float ndl = max(dot(normal, sunDir), 0.0);
  // Wrapped diffuse keeps unlit faces readable instead of black.
  float wrapped = ndl * 0.85 + 0.15;
  vec3 halfway = normalize(sunDir + viewDir);
  float spec = pow(max(dot(normal, halfway), 0.0), mix(96.0, 6.0, roughness));
  spec *= (1.0 - roughness) * mix(0.25, 1.0, metallic);
  // Hemisphere ambient: sky above, bounced ground light below.
  float hemi = normal.z * 0.5 + 0.5;
  vec3 indirect = ambient * mix(0.55, 1.15, hemi);
  return albedo * (sunColor * wrapped + indirect) + sunColor * spec;
}

vec3 applyFog(vec3 color, vec3 fogColor, float dist, float density) {
  float f = 1.0 - exp(-dist * density);
  return mix(color, fogColor, clamp(f, 0.0, 1.0));
}
`;

export const SURFACE_VS = /* glsl */`#version 300 es
precision highp float;
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in uint aContext;
layout(location = 4) in uint aRole;
layout(location = 5) in uint aNode;
layout(location = 6) in uint aMaterial;

uniform mat4 uViewProjection;
uniform vec3 uCameraPosition;

out vec3 vWorld;
out vec3 vNormal;
out vec2 vUV;
flat out uint vContext;
flat out uint vRole;
flat out uint vNode;
flat out uint vMaterial;
out float vDistance;

void main() {
  vWorld = aPosition;
  vNormal = normalize(aNormal);
  vUV = aUV;
  vContext = aContext;
  vRole = aRole;
  vNode = aNode;
  vMaterial = aMaterial;
  vDistance = length(aPosition - uCameraPosition);
  gl_Position = uViewProjection * vec4(aPosition, 1.0);
}`;

export const SURFACE_FS = /* glsl */`#version 300 es
precision highp float;
precision highp sampler2DArray;

in vec3 vWorld;
in vec3 vNormal;
in vec2 vUV;
flat in uint vContext;
flat in uint vRole;
flat in uint vNode;
flat in uint vMaterial;
in float vDistance;

uniform sampler2DArray uAlbedo;
uniform vec4 uMaterialColor[${MAX_MATERIALS}];   // rgb, opacity
uniform vec4 uMaterialParams[${MAX_MATERIALS}];  // roughness, metallic, 1/scale, hasTexture
uniform vec4 uMaterialEmissive[${MAX_MATERIALS}];

uniform vec3 uSunDirection;
uniform vec3 uSunColor;
uniform vec3 uAmbient;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform vec3 uCameraPosition;
uniform float uExposure;
uniform uint uHighlightNode;
uniform uint uContextMask;     // when non-zero, flag-highlight mode is on
uniform int uDebugMode;        // 0 theme, 1 context, 2 role, 3 confidence-ish

out vec4 fragColor;

${LIGHTING}

vec3 roleColor(uint role) {
  float h = fract(float(role) * 0.6180339887);
  vec3 k = vec3(3.0, 2.0, 1.0);
  return 0.45 + 0.45 * cos(6.28318 * (h + k / 3.0));
}

void main() {
  int index = int(vMaterial);
  vec4 color = uMaterialColor[index];
  vec4 params = uMaterialParams[index];

  vec3 albedo = color.rgb;
  if (params.w > 0.5) {
    vec2 uv = vUV * params.z;
    albedo *= texture(uAlbedo, vec3(uv, float(index))).rgb;
  }

  if (uDebugMode == 2) albedo = roleColor(vRole);
  if (uDebugMode == 3) {
    // Show where geometry was inferred rather than measured.
    bool sparse = (vContext & 262144u) != 0u || (vContext & 524288u) != 0u;
    albedo = sparse ? vec3(0.85, 0.25, 0.35) : vec3(0.30, 0.65, 0.45);
  }

  vec3 normal = normalize(vNormal);
  if (!gl_FrontFacing) normal = -normal;
  vec3 viewDir = normalize(uCameraPosition - vWorld);
  vec3 lit = shade(albedo, normal, viewDir, params.x, params.y,
                   normalize(uSunDirection), uSunColor, uAmbient);
  lit += uMaterialEmissive[index].rgb * uMaterialEmissive[index].w;

  if (uContextMask != 0u && (vContext & uContextMask) != 0u) {
    lit = mix(lit, vec3(1.0, 0.85, 0.15), 0.65);
  }
  if (uDebugMode == 1) {
    // Paint the context flags that drive the theme rules.
    vec3 tint = vec3(0.22, 0.24, 0.30);
    if ((vContext & 32u) != 0u || (vContext & 64u) != 0u) tint = vec3(1.0, 0.78, 0.2);   // corner
    if ((vContext & 256u) != 0u) tint = vec3(0.95, 0.3, 0.4);                            // opening edge
    else if ((vContext & 128u) != 0u) tint = vec3(0.7, 0.35, 0.6);                       // near opening
    if ((vContext & 2048u) != 0u) tint = vec3(0.55, 0.45, 0.2);                          // ground contact
    if ((vContext & 65536u) != 0u) tint = mix(tint, vec3(0.2, 0.8, 0.9), 0.5);           // street facing
    lit = tint * (0.55 + 0.45 * max(dot(normal, normalize(uSunDirection)), 0.0));
  }

  if (vNode == uHighlightNode) lit = mix(lit, vec3(0.2, 1.0, 0.9), 0.45);

  lit = applyFog(lit, uFogColor, vDistance, uFogDensity);
  lit = 1.0 - exp(-lit * uExposure);
  fragColor = vec4(pow(max(lit, 0.0), vec3(1.0 / 2.2)), color.a);
}`;

export const POINTS_VS = /* glsl */`#version 300 es
precision highp float;
layout(location = 0) in vec3 aPosition;
layout(location = 1) in uint aPacked;   // semantic | role<<8 | confidence<<16

uniform mat4 uViewProjection;
uniform vec3 uCameraPosition;
uniform float uPointScale;
uniform int uColorMode;   // 0 semantic, 1 role, 2 confidence

out vec3 vColor;
out float vDistance;

vec3 palette(uint id) {
  float h = fract(float(id) * 0.6180339887 + 0.12);
  vec3 k = vec3(3.0, 2.0, 1.0);
  return 0.5 + 0.45 * cos(6.28318 * (h + k / 3.0));
}

void main() {
  uint semantic = aPacked & 255u;
  uint role = (aPacked >> 8) & 255u;
  float confidence = float((aPacked >> 16) & 255u) / 255.0;
  if (uColorMode == 0) vColor = palette(semantic);
  else if (uColorMode == 1) vColor = palette(role + 7u);
  else vColor = mix(vec3(0.9, 0.25, 0.25), vec3(0.25, 0.9, 0.55), confidence);

  vDistance = length(aPosition - uCameraPosition);
  gl_Position = uViewProjection * vec4(aPosition, 1.0);
  gl_PointSize = clamp(uPointScale / max(vDistance, 1.0), 1.0, 6.0);
}`;

export const POINTS_FS = /* glsl */`#version 300 es
precision highp float;
in vec3 vColor;
in float vDistance;
uniform vec3 uFogColor;
uniform float uFogDensity;
out vec4 fragColor;

void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  float f = 1.0 - exp(-vDistance * uFogDensity);
  fragColor = vec4(mix(vColor, uFogColor, clamp(f, 0.0, 1.0)), 1.0);
}`;

export const INSTANCE_VS = /* glsl */`#version 300 es
precision highp float;
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aOffset;
layout(location = 3) in vec3 aScale;
layout(location = 4) in vec4 aTint;

uniform mat4 uViewProjection;
uniform vec3 uCameraPosition;

out vec3 vNormal;
out vec4 vTint;
out float vDistance;

void main() {
  vec3 world = aPosition * aScale + aOffset;
  vNormal = normalize(aNormal / max(aScale, vec3(0.001)));
  vTint = aTint;
  vDistance = length(world - uCameraPosition);
  gl_Position = uViewProjection * vec4(world, 1.0);
}`;

export const INSTANCE_FS = /* glsl */`#version 300 es
precision highp float;
in vec3 vNormal;
in vec4 vTint;
in float vDistance;

uniform vec3 uSunDirection;
uniform vec3 uSunColor;
uniform vec3 uAmbient;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform float uExposure;
out vec4 fragColor;

void main() {
  vec3 normal = normalize(vNormal);
  float ndl = max(dot(normal, normalize(uSunDirection)), 0.0) * 0.8 + 0.2;
  vec3 lit = vTint.rgb * (uSunColor * ndl + uAmbient * (normal.z * 0.5 + 0.75));
  float f = 1.0 - exp(-vDistance * uFogDensity);
  lit = mix(lit, uFogColor, clamp(f, 0.0, 1.0));
  lit = 1.0 - exp(-lit * uExposure);
  fragColor = vec4(pow(max(lit, 0.0), vec3(1.0 / 2.2)), vTint.a);
}`;

export const SKY_VS = /* glsl */`#version 300 es
precision highp float;
layout(location = 0) in vec2 aPosition;
out vec2 vUV;
void main() {
  vUV = aPosition;
  gl_Position = vec4(aPosition, 0.999999, 1.0);
}`;

export const SKY_FS = /* glsl */`#version 300 es
precision highp float;
in vec2 vUV;
uniform vec3 uSky;
uniform vec3 uHorizon;
uniform float uPitch;
out vec4 fragColor;
void main() {
  float t = clamp(vUV.y * 0.5 + 0.5 + uPitch * 0.35, 0.0, 1.0);
  vec3 color = mix(uHorizon, uSky, pow(t, 0.8));
  fragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}`;
