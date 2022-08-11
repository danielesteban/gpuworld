import './main.css';
import { vec2, vec3 } from 'gl-matrix';
import Camera from './render/camera.js';
import Input from './render/input.js';
import Renderer from './render/renderer.js';
import World from './compute/world.js';

const Main = ({ adapter, device }) => {
  const camera = new Camera({ device });
  const world = new World({ camera, device });
  const renderer = new Renderer({ adapter, camera, device });
  document.getElementById('renderer').appendChild(renderer.canvas);
  renderer.setSize(window.innerWidth, window.innerHeight);
  window.addEventListener('resize', () => (
    renderer.setSize(window.innerWidth, window.innerHeight)
  ), false);

  let clock = performance.now() / 1000;
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      clock = performance.now() / 1000;
    }
  }, false);

  const input = new Input({
    position: vec3.set(camera.position, 0, world.chunkSize.y * 0.5, 0),
    target: renderer.canvas,
  });

  const anchor = vec2.create();
  const chunk = vec2.fromValues(world.chunkSize.x, world.chunkSize.z);
  const current = vec2.fromValues(-1, -1);

  const animate = () => {
    requestAnimationFrame(animate);

    const time = performance.now() / 1000;
    const delta = Math.min(time - clock, 1);
    clock = time;

    input.update(delta);

    vec2.floor(anchor, vec2.divide(anchor, vec2.set(anchor, camera.position[0], camera.position[2]), chunk));
    if (!vec2.equals(current, anchor)) {
      vec2.copy(current, anchor);
      world.load(anchor, 4, 5);
    }

    vec3.add(camera.target, camera.position, input.vectors.forward);
    camera.updateView();

    const command = device.createCommandEncoder();
    world.compute(command);
    renderer.render(command, world.chunks.loaded);
    device.queue.submit([command.finish()]);
  };

  renderer.setClearColor(0.2, 0.6, 0.6);
  renderer.voxels.atlas.compute();
  renderer.voxels.atlas.setupDragAndDrop();
  requestAnimationFrame(animate);
};

const GPU = async () => {
  if (!navigator.gpu) {
    throw new Error('WebGPU support');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('WebGPU adapter');
  }
  const device = await adapter.requestDevice();
  const check = device.createShaderModule({
    code: `const checkConstSupport : f32 = 1;`,
  });
  const { messages } = await check.compilationInfo();
  if (messages.find(({ type }) => type === 'error')) {
    throw new Error('WGSL const support');
  }
  return { adapter, device };
};

GPU()
  .then(Main)
  .catch((e) => {
    console.error(e);
    document.getElementById('canary').classList.add('enabled');
  })
  .finally(() => document.getElementById('loading').classList.remove('enabled'));
