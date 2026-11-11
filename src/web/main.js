/**
 * MotionMind AI - Step 4 & 5: Hand-Controlled 3D Scene
 * Ingests live hand tracking telemetry via WebSocket and prepares 3D transform targets.
 */

import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

// --- 1. Three.js Scene Setup ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0c14);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 0, 5);

const canvas = document.getElementById('canvas-container');
const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

// Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0x00ffcc, 2.0);
keyLight.position.set(5, 5, 5);
scene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0xa855f7, 1.5);
rimLight.position.set(-5, -3, 2);
scene.add(rimLight);

// Spatial Floor Grid
const gridHelper = new THREE.GridHelper(12, 24, 0x00ffcc, 0x1e293b);
gridHelper.position.y = -2.2;
scene.add(gridHelper);

// 3D Interactive Cube Mesh
const geometry = new THREE.BoxGeometry(1.6, 1.6, 1.6);
const material = new THREE.MeshStandardMaterial({
  color: 0x0f172a,
  metalness: 0.8,
  roughness: 0.2,
});
const cube = new THREE.Mesh(geometry, material);
scene.add(cube);

// Neon Wireframe Edges
const edges = new THREE.EdgesGeometry(geometry);
const edgesMaterial = new THREE.LineBasicMaterial({ color: 0x00ffcc, linewidth: 2 });
const wireframe = new THREE.LineSegments(edges, edgesMaterial);
cube.add(wireframe);

// --- 2. Live Telemetry State ---
export const handState = {
  isTracking: false,
  targetPos: { x: 0, y: 0, z: 0 },
  targetRot: { x: 0, y: 0, z: 0 },
  targetScale: 1.0,
  streamFps: 0.0,
};

// UI Element References
const connDot = document.getElementById('conn-dot');
const connText = document.getElementById('conn-text');
const valStatus = document.getElementById('val-status');
const valPos = document.getElementById('val-pos');
const valRot = document.getElementById('val-rot');
const valScale = document.getElementById('val-scale');
const valFps = document.getElementById('val-fps');

// --- 3. WebSocket Ingestion Client ---
const WS_URL = 'ws://localhost:8765';
let socket = null;
let reconnectTimer = null;

function connectWebSocket() {
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    console.log('[WebSocket] Connected to Python tracking backend.');
    connDot.className = 'status-dot connected';
    connText.textContent = 'WebSocket: Connected';
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handState.streamFps = data.fps || 0;
      valFps.textContent = handState.streamFps.toFixed(1);

      if (data.is_tracking && data.transform) {
        handState.isTracking = true;
        const t = data.transform;

        // Position
        handState.targetPos.x = t.position.x;
        handState.targetPos.y = t.position.y;
        handState.targetPos.z = t.position.z;

        // Rotation
        handState.targetRot.x = t.rotation.x;
        handState.targetRot.y = t.rotation.y;
        handState.targetRot.z = t.rotation.z;

        // Scale
        handState.targetScale = t.scale;

        // Update UI HUD
        connDot.className = 'status-dot tracking';
        valStatus.textContent = 'Tracking Hand';
        valStatus.style.color = '#00ffcc';

        valPos.textContent = `(${t.position.x.toFixed(2)}, ${t.position.y.toFixed(2)}, ${t.position.z.toFixed(2)})`;
        valRot.textContent = `(${t.rotation.x.toFixed(2)}, ${t.rotation.y.toFixed(2)}, ${t.rotation.z.toFixed(2)})`;
        valScale.textContent = `${t.scale.toFixed(2)}x`;
      } else {
        handState.isTracking = false;
        connDot.className = 'status-dot connected';
        valStatus.textContent = 'Searching...';
        valStatus.style.color = '#f59e0b';
      }
    } catch (err) {
      console.error('[WebSocket] Failed to parse message:', err);
    }
  };

  socket.onclose = () => {
    connDot.className = 'status-dot';
    connText.textContent = 'WebSocket: Disconnected (Retrying...)';
    valStatus.textContent = 'Offline';
    valStatus.style.color = '#ef4444';
    handState.isTracking = false;

    // Retry connection every 2 seconds
    if (!reconnectTimer) {
      reconnectTimer = setTimeout(connectWebSocket, 2000);
    }
  };

  socket.onerror = () => {
    socket.close();
  };
}

// Start WebSocket connection
connectWebSocket();

// --- 4. Window Resize Handling ---
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

// --- 5. Render Loop ---
function animate() {
  requestAnimationFrame(animate);

  // If no hand is currently tracking, gently idle-rotate the cube
  if (!handState.isTracking) {
    cube.rotation.x += 0.005;
    cube.rotation.y += 0.008;
  }

  renderer.render(scene, camera);
}

animate();
