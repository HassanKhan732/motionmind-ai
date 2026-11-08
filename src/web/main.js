/**
 * MotionMind AI - Step 4: Basic 3D Scene using Three.js
 * Demonstrates 3D fundamentals: Scene, Perspective Camera, WebGL Renderer, Lights, and Mesh.
 */

import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

// 1. Scene: The 3D container holding all objects, cameras, and lights
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0c14);

// 2. Camera: Perspective camera (FOV, Aspect Ratio, Near clipping, Far clipping)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 1.5, 4);
camera.lookAt(0, 0, 0);

// 3. Renderer: WebGL renderer drawing the 3D scene onto HTML canvas
const canvas = document.getElementById('canvas-container');
const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

// 4. Lights: Ambient light for general visibility + Directional light for depth & shadows
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0x00ffcc, 1.8);
directionalLight.position.set(5, 8, 5);
scene.add(directionalLight);

const secondaryLight = new THREE.DirectionalLight(0x9d4edd, 1.2);
secondaryLight.position.set(-5, -3, 2);
scene.add(secondaryLight);

// 5. Grid Helper: Ground plane visualizer to perceive 3D space and orientation
const gridHelper = new THREE.GridHelper(10, 20, 0x00ffcc, 0x1e293b);
gridHelper.position.y = -1.2;
scene.add(gridHelper);

// 6. Mesh: Geometry + Material = 3D Object
// A BoxGeometry represents the 3D cube structure
const geometry = new THREE.BoxGeometry(1.6, 1.6, 1.6);

// MeshStandardMaterial reacts realistically to lights
const material = new THREE.MeshStandardMaterial({
  color: 0x0f172a,
  metalness: 0.7,
  roughness: 0.2,
});
const cube = new THREE.Mesh(geometry, material);
scene.add(cube);

// Adding glowing neon edges to make the 3D cube edges stand out
const edgesGeometry = new THREE.EdgesGeometry(geometry);
const edgesMaterial = new THREE.LineBasicMaterial({ color: 0x00ffcc, linewidth: 2 });
const wireframe = new THREE.LineSegments(edgesGeometry, edgesMaterial);
cube.add(wireframe);

// 7. Handle Window Resize: Keeps aspect ratio correct on screen changes
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

// 8. Animation Render Loop: Continuous 60 FPS rendering
function animate() {
  requestAnimationFrame(animate);

  // Subtle continuous rotation for testing 3D depth
  cube.rotation.x += 0.006;
  cube.rotation.y += 0.01;

  renderer.render(scene, camera);
}

// Start rendering
animate();
