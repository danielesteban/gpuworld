import './main.css';
import { vec2, vec3 } from 'gl-matrix';
import Camera from './render/camera.js';
import Explosions from './render/explosions.js';
import Input from './render/input.js';
import Light from './render/light.js';
import Projectiles from './render/projectiles.js';
import Renderer from './render/renderer.js';
import SFX from './sfx/sfx.js';
import Voxels from './render/voxels.js';
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

  const voxels = new Voxels({
    camera,
    chunks: world.chunks.loaded,
    device,
    samples: renderer.samples,
    sunlight: renderer.sunlight,
  });
  voxels.atlas.compute();
  voxels.atlas.setupDragAndDrop();
  renderer.scene.push(voxels);

  const projectiles = new Projectiles({
    instances: world.simulation.projectiles.instances,
    camera,
    device,
    samples: renderer.samples,
    sunlight: renderer.sunlight,
  });
  renderer.scene.push(projectiles);

  const explosions = new Explosions({
    instances: world.simulation.explosions.instances,
    camera,
    device,
    geometry: projectiles.geometry,
    samples: renderer.samples,
    sunlight: renderer.sunlight,
  });
  renderer.scene.push(explosions);

  vec3.set(
    camera.position,
    world.chunkSize.x * -0.5 + 1.5,
    world.chunkSize.y * 0.2 - 1.5,
    world.chunkSize.z * 1.5 + 1.5
  );
  const input = new Input({
    camera,
    target: renderer.canvas,
    world,
  });
  const light = new Light(renderer);
  const sfx = new SFX();

  const anchor = vec2.create();
  const chunk = vec2.fromValues(world.chunkSize.x, world.chunkSize.z);
  const current = vec2.fromValues(-1000, -1000);
  const direction = vec3.create();

  let clock = performance.now() / 1000;
  let lastShot = clock;
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      clock = performance.now() / 1000;
    }
  }, false);

  const animate = () => {
    requestAnimationFrame(animate);

    const time = performance.now() / 1000;
    const delta = Math.min(time - clock, 1);
    clock = time;

    input.update(delta);
    light.update(delta);
    sfx.update(camera.position[1]);

    vec2.floor(anchor, vec2.divide(anchor, vec2.set(anchor, camera.position[0], camera.position[2]), chunk));
    if (!vec2.equals(current, anchor)) {
      vec2.copy(current, anchor);
      world.load(anchor, 4, 5);
    }

    if (input.buttons.primary && (time - lastShot) > 0.05) {
      lastShot = time;
      world.simulation.shoot(
        input.forward,
        vec3.add(direction, camera.position, input.forward),
      );
      sfx.play('shot');
    }

    const command = device.createCommandEncoder();
    world.compute(command, delta);
    renderer.render(command);
    device.queue.submit([command.finish()]);
  };

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
