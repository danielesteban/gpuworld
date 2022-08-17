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
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
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
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
      });
      this.projectiles = {
        instances,
        state,
      };
    }
    {
      const count = 8;
      const size = (
        4 * Uint32Array.BYTES_PER_ELEMENT
        + count * 4 * Float32Array.BYTES_PER_ELEMENT
      );
      this.sfx = {
        buffer: device.createBuffer({
          size,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
        }),
        output: device.createBuffer({
          size,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        data: new Float32Array(count * 4),
        count,
      };
    }
    this.pipelines = {
      explosions: {
        mesh: new ExplosionsMesh({
          count,
          device,
          explosions: this.explosions,
        }),
        step: new ExplosionsStep({
          count,
          delta: this.delta,
          device,
          explosions: this.explosions,
          projectiles: this.projectiles,
          sfx: this.sfx,
        }),
      },
      projectiles: {
        compute: new ProjectilesCompute({
          chunkSize,
          count,
          device,
          projectiles: this.projectiles,
        }),
        step: new ProjectilesStep({
          count,
          delta: this.delta,
          device,
          explosions: this.explosions,
          input: this.input,
          projectiles: this.projectiles,
          sfx: this.sfx,
        }),
      },
      setup: new SimulationSetup({
        device,
        explosions: this.explosions,
        projectiles: this.projectiles,
      }),
    };
  }

  compute(pass, chunk) {
    const { pipelines: { projectiles: { compute } } } = this;
    compute.compute(pass, chunk);
  }

  getQueuedSFX() {
    const { device, sfx } = this;
    if (sfx.isFetching) {
      return Promise.resolve({ count: 0 });
    }
    sfx.isFetching = true;
    const command = device.createCommandEncoder();
    command.copyBufferToBuffer(sfx.buffer, 0, sfx.output, 0, sfx.output.size);
    command.clearBuffer(sfx.buffer, 0, 4);
    device.queue.submit([command.finish()]);
    return sfx.output
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const count = Math.min(new Uint32Array(sfx.output.getMappedRange(0, Uint32Array.BYTES_PER_ELEMENT))[0], sfx.count);
        sfx.data.set(new Float32Array(sfx.output.getMappedRange(16, 16 + count * 4 * Float32Array.BYTES_PER_ELEMENT)));
        sfx.output.unmap();
        sfx.isFetching = false;
        return { count, data: sfx.data };
      });
  }

  shootProjectile(direction, origin) {
    const { device, input } = this;
    vec3.copy(input.position, origin);
    vec3.copy(input.direction, direction);
    input.enabled[0] = 1;
    device.queue.writeBuffer(input.buffer, 0, input.data);
  }

  step(pass, delta) {
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
