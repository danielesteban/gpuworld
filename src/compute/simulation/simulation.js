import { vec3 } from 'gl-matrix';
import ExplosionsMesh from './explosions/mesh.js';
import ExplosionsStep from './explosions/step.js';
import ProjectilesCompute from './projectiles/compute.js';
import ProjectilesStep from './projectiles/step.js';
import SimulationSetup from './setup.js';

class Simulation {
  constructor({ count = 32, chunkSize, device }) {
    this.device = device;
    {
      const data = new Float32Array(1);
      this.delta = {
        buffer: device.createBuffer({
          size: data.byteLength,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
        }),
        data,
      };
    }
    {
      const data = new Float32Array(8);
      this.input = {
        buffer: device.createBuffer({
          size: data.byteLength,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        }),
        data: data.buffer,
        position: data.subarray(0, 3),
        direction: data.subarray(4, 7),
        enabled: new Uint32Array(data.buffer, 28, 1),
      };
    }
    {
      const instancesPerMesh = 64;
      const instances = device.createBuffer({
        mappedAtCreation: true,
        size: (
          5 * Uint32Array.BYTES_PER_ELEMENT
          + count * instancesPerMesh * 4 * Float32Array.BYTES_PER_ELEMENT
        ),
        usage: (
          GPUBufferUsage.INDIRECT
          | GPUBufferUsage.STORAGE
          | GPUBufferUsage.VERTEX
        ),
      });
      new Uint32Array(instances.getMappedRange(0, Uint32Array.BYTES_PER_ELEMENT))[0] = 36;
      instances.unmap();
      const meshes = device.createBuffer({
        size: count * 4 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE,
      });
      const state = device.createBuffer({
        size: count * 8 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE,
      });
      const workgroups = device.createBuffer({
        mappedAtCreation: true,
        size: 3 * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
      });
      new Uint32Array(workgroups.getMappedRange()).set([1, 0, 1]);
      workgroups.unmap();
      this.explosions = {
        instances,
        instancesPerMesh,
        meshes,
        state,
        workgroups,
      };
    }
    {
      const instances = device.createBuffer({
        mappedAtCreation: true,
        size: (
          5 * Uint32Array.BYTES_PER_ELEMENT
          + count * 6 * Float32Array.BYTES_PER_ELEMENT
        ),
        usage: (
          GPUBufferUsage.INDIRECT
          | GPUBufferUsage.STORAGE
          | GPUBufferUsage.VERTEX
        ),
      });
      new Uint32Array(instances.getMappedRange(0, Uint32Array.BYTES_PER_ELEMENT))[0] = 36;
      instances.unmap();
      const state = device.createBuffer({
        size: count * 12 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE,
      });
      this.projectiles = {
        instances,
        state,
      };
    }
    this.pipelines = {
      explosions: {
        mesh: new ExplosionsMesh({ count, device, explosions: this.explosions }),
        step: new ExplosionsStep({
          count,
          delta: this.delta,
          device,
          explosions: this.explosions,
        }),
      },
      projectiles: {
        compute: new ProjectilesCompute({ chunkSize, count, device, state: this.projectiles.state }),
        step: new ProjectilesStep({
          count,
          delta: this.delta,
          device,
          input: this.input,
          explosions: this.explosions,
          projectiles: this.projectiles,
        }),
      },
      setup: new SimulationSetup({ device, explosions: this.explosions, projectiles: this.projectiles }),
    };
  }

  compute(pass, chunk) {
    const { pipelines: { projectiles: { compute } } } = this;
    compute.compute(pass, chunk);
  }

  shoot(direction, origin) {
    const { device, input } = this;
    vec3.copy(input.position, origin);
    vec3.copy(input.direction, direction);
    input.enabled[0] = 1;
    device.queue.writeBuffer(input.buffer, 0, input.data);
  }

  step(delta, pass) {
    const { delta: { buffer, data }, device, pipelines: { explosions, projectiles, setup } } = this;
    data[0] = delta;
    device.queue.writeBuffer(buffer, 0, data);
    setup.compute(pass);
    explosions.step.compute(pass);
    explosions.mesh.compute(pass);
    projectiles.step.compute(pass);
  }
}

export default Simulation;
