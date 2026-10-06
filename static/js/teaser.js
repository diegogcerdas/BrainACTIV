import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const ROIS = ['FFA', 'EBA', 'VWFA', 'OPA', 'PPA', 'RSC'];
const IMAGE_DIR = 'static/images/teaser';
const LIGHT_PINK = new THREE.Color('#f093fb');
const DARK_PINK = new THREE.Color('#f5576c');

const canvas = document.getElementById('brain-canvas');
const output = document.getElementById('teaser-output');
const roiLabels = document.querySelectorAll('.teaser-roi-name');
const buttons = document.querySelectorAll('.roi-button');
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Preload the manipulated images so switching is instant
ROIS.forEach((roi) => { new Image().src = `${IMAGE_DIR}/${roi}.jpg`; });

// Scene
const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 1, 1000);
camera.position.set(92, 20, 104);
const controls = new OrbitControls(camera, canvas);
controls.enableZoom = false;
controls.enablePan = false;
controls.enableDamping = true;
controls.autoRotate = !reduceMotion;
controls.autoRotateSpeed = 1.2;
const brain = new THREE.Group();
scene.add(brain);

let positions = null;
let rois = null;
let roiPoints = null;
let roiFade = 1;

function resize() {
  const size = canvas.clientWidth;
  renderer.setSize(size, size, false);
  camera.aspect = 1;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);

function showRoi(roi) {
  if (roiPoints) {
    brain.remove(roiPoints);
    roiPoints.geometry.dispose();
  }
  const idx = rois[roi];
  const xyz = new Float32Array(idx.length * 3);
  const colors = new Float32Array(idx.length * 3);
  let yMin = Infinity, yMax = -Infinity;
  idx.forEach((v) => { yMin = Math.min(yMin, positions[3 * v + 1]); yMax = Math.max(yMax, positions[3 * v + 1]); });
  idx.forEach((v, i) => {
    xyz.set([positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]], 3 * i);
    // Light-to-dark pink gradient from the top to the bottom of the region
    const t = (yMax - positions[3 * v + 1]) / Math.max(yMax - yMin, 1);
    const c = LIGHT_PINK.clone().lerp(DARK_PINK, 0.25 + 0.75 * t);
    colors.set([c.r, c.g, c.b], 3 * i);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  roiPoints = new THREE.Points(geometry, new THREE.PointsMaterial({
    size: 3.6, vertexColors: true, transparent: true, opacity: 0, depthWrite: false, depthTest: false,
  }));
  roiPoints.renderOrder = 1;  // drawn over the translucent brain
  brain.add(roiPoints);
  roiFade = 0;
}

function selectRoi(roi) {
  buttons.forEach((b) => {
    const active = b.dataset.roi === roi;
    b.classList.toggle('is-active', active);
    b.setAttribute('aria-pressed', active);
  });
  roiLabels.forEach((el) => { el.textContent = roi; });
  output.style.opacity = 0;
  setTimeout(() => {
    output.src = `${IMAGE_DIR}/${roi}.jpg`;
    output.alt = `Reference image manipulated to enhance ${roi} activity`;
    output.style.opacity = 1;
  }, 200);
  if (rois) showRoi(roi);
}

buttons.forEach((b) => b.addEventListener('click', () => selectRoi(b.dataset.roi)));

// Open on a given region with ?roi=PPA (e.g. to share a specific example)
const initialRoi = new URLSearchParams(window.location.search).get('roi');
if (ROIS.includes(initialRoi)) selectRoi(initialRoi);

fetch('static/data/brain.json')
  .then((r) => r.json())
  .then((data) => {
    positions = data.positions;
    rois = data.rois;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    brain.add(new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 2.5, color: 0x9ca3af, transparent: true, opacity: 0.12, depthWrite: false,
    })));
    showRoi(document.querySelector('.roi-button.is-active').dataset.roi);
  });

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const t = clock.getElapsedTime();
  if (!reduceMotion) brain.position.y = Math.sin(t * 0.8) * 1.5;
  if (roiPoints && roiFade < 1) {
    roiFade = Math.min(1, roiFade + 0.04);
    roiPoints.material.opacity = roiFade;
  }
  controls.update();
  renderer.render(scene, camera);
});
