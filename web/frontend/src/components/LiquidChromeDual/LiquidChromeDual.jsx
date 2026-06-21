import { useEffect, useMemo, useRef } from "react";
import { Renderer, Program, Mesh, Triangle } from "ogl";
import "./LiquidChromeDual.css";

function hexToUnitRgb(hex) {
  const clean = hex.replace("#", "");
  return [parseInt(clean.slice(0, 2), 16) / 255, parseInt(clean.slice(2, 4), 16) / 255, parseInt(clean.slice(4, 6), 16) / 255];
}

export default function LiquidChromeDual({ colorA = "#F2E700", colorB = "#355BFF", speed = 0.32, amplitude = 0.42, frequencyX = 3.4, frequencyY = 2.8, interactive = false }) {
  const containerRef = useRef(null);
  const rgbA = useMemo(() => hexToUnitRgb(colorA), [colorA]);
  const rgbB = useMemo(() => hexToUnitRgb(colorB), [colorB]);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const renderer = new Renderer({ antialias: true, alpha: true });
    const gl = renderer.gl; gl.clearColor(1, 1, 1, 1);

    const vertexShader = `attribute vec2 position; attribute vec2 uv; varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position, 0.0, 1.0); }`;
    const fragmentShader = `
      precision highp float;
      uniform float uTime; uniform vec3 uResolution; uniform vec3 uColorA; uniform vec3 uColorB;
      uniform float uAmplitude; uniform float uFrequencyX; uniform float uFrequencyY; uniform vec2 uMouse;
      varying vec2 vUv;
      float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
      float noise(vec2 p) { vec2 i = floor(p); vec2 f = fract(p); float a = hash(i); float b = hash(i + vec2(1.0, 0.0)); float c = hash(i + vec2(0.0, 1.0)); float d = hash(i + vec2(1.0, 1.0)); vec2 u = f * f * (3.0 - 2.0 * f); return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y; }
      vec2 liquidWarp(vec2 uv) { vec2 p = uv; for (float i = 1.0; i < 9.0; i++) { p.x += uAmplitude / i * cos(i * uFrequencyX * p.y + uTime * 1.12 + uMouse.x * 2.2); p.y += uAmplitude / i * sin(i * uFrequencyY * p.x + uTime * 0.92 + uMouse.y * 2.2); } return p; }
      vec4 renderImage(vec2 uvCoord) {
        vec2 fragCoord = uvCoord * uResolution.xy; vec2 uv = (2.0 * fragCoord - uResolution.xy) / min(uResolution.x, uResolution.y);
        vec2 p = liquidWarp(uv);
        vec2 diff = uvCoord - uMouse; float dist = length(diff); float falloff = exp(-dist * 18.0); float ripple = sin(12.0 * dist - uTime * 2.2) * 0.035;
        p += normalize(diff + 0.0001) * ripple * falloff;
        float angle = atan(p.y, p.x); float radius = length(p);
        float flowA = sin(p.x * 3.8 + p.y * 2.2 + uTime * 0.9); float flowB = cos(p.y * 4.6 - p.x * 2.4 - uTime * 0.75);
        float spiral = sin(angle * 2.2 + radius * 8.0 - uTime * 1.25); float n = noise(p * 1.8 + uTime * 0.08);
        float mixer = flowA * 0.42 + flowB * 0.32 + spiral * 0.34 + n * 0.34; mixer = smoothstep(-0.42, 0.52, mixer);
        vec3 color = mix(uColorA, uColorB, mixer);
        float whiteBandA = abs(sin(p.x * 4.8 + p.y * 3.6 + uTime * 0.72)); float whiteBandB = abs(cos(p.x * 2.4 - p.y * 5.2 - uTime * 0.55));
        float whiteMask = smoothstep(0.76, 0.98, whiteBandA * whiteBandB); color = mix(color, vec3(1.0), whiteMask * 0.82);
        float chrome = 0.72 + 0.32 * sin(p.x * 7.0 + p.y * 3.0 + uTime); chrome += 0.18 * cos(radius * 12.0 - uTime * 1.5); color *= chrome;
        float cyanMask = smoothstep(0.44, 0.54, mixer) * smoothstep(0.66, 0.52, mixer); color = mix(color, vec3(0.55, 0.95, 1.0), cyanMask * 0.28);
        color = pow(color, vec3(0.78)); color = clamp(color, 0.0, 1.0); return vec4(color, 1.0);
      }
      void main() { vec4 col = vec4(0.0); int samples = 0; for (int i = -1; i <= 1; i++) { for (int j = -1; j <= 1; j++) { vec2 offset = vec2(float(i), float(j)) * 0.75 / min(uResolution.x, uResolution.y); col += renderImage(vUv + offset); samples++; } } gl_FragColor = col / float(samples); }`;

    const geometry = new Triangle(gl);
    const program = new Program(gl, { vertex: vertexShader, fragment: fragmentShader, uniforms: { uTime: { value: 0 }, uResolution: { value: new Float32Array([gl.canvas.width, gl.canvas.height, gl.canvas.width / gl.canvas.height]) }, uColorA: { value: new Float32Array(rgbA) }, uColorB: { value: new Float32Array(rgbB) }, uAmplitude: { value: amplitude }, uFrequencyX: { value: frequencyX }, uFrequencyY: { value: frequencyY }, uMouse: { value: new Float32Array([0.5, 0.5]) } } });
    const mesh = new Mesh(gl, { geometry, program });

    function resize() { const dpr = Math.min(devicePixelRatio || 1, 2); renderer.setSize(Math.max(1, container.offsetWidth * dpr), Math.max(1, container.offsetHeight * dpr)); gl.canvas.style.width = "100%"; gl.canvas.style.height = "100%"; const r = program.uniforms.uResolution.value; r[0] = gl.canvas.width; r[1] = gl.canvas.height; r[2] = gl.canvas.width / gl.canvas.height; }
    resize(); window.addEventListener("resize", resize);

    let id; function update(t) { id = requestAnimationFrame(update); program.uniforms.uTime.value = t * 0.001 * speed; renderer.render({ scene: mesh }); }
    id = requestAnimationFrame(update); container.appendChild(gl.canvas);

    return () => { cancelAnimationFrame(id); window.removeEventListener("resize", resize); if (gl.canvas.parentElement) gl.canvas.parentElement.removeChild(gl.canvas); gl.getExtension("WEBGL_lose_context")?.loseContext(); };
  }, [rgbA, rgbB, speed, amplitude, frequencyX, frequencyY, interactive]);

  return <div ref={containerRef} className="liquidChromeDual" />;
}
