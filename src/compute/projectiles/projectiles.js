import { vec3 } from 'gl-matrix';
import ProjectilesCompute from './compute.js';
import ProjectilesStep from './step.js';

class Projectiles {
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
    this.instances = device.createBuffer({
      mappedAtCreation: true,
      size: (
        5 * Uint32Array.BYTES_PER_ELEMENT
        + count * 6 * Float32Array.BYTES_PER_ELEMENT
      ),
      usage: (
        GPUBufferUsage.COPY_DST
        | GPUBufferUsage.INDIRECT
        | GPUBufferUsage.STORAGE
        | GPUBufferUsage.VERTEX
      ),
    });
    new Uint32Array(this.instances.getMappedRange(0, Uint32Array.BYTES_PER_ELEMENT))[0] = 36;
    this.instances.unmap();
    this.state = device.createBuffer({
      size: count * 12 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    });
    this.passes = {
      compute: new ProjectilesCompute({ chunkSize, count, device, state: this.state }),
      step: new ProjectilesStep({
        count,
        delta: this.delta,
        device,
        input: this.input,
        instances: this.instances,
        state: this.state,
      }),
    };
  }

  compute(pass, chunk) {
    const { passes: { compute } } = this;
    compute.compute(pass, chunk);
  }

  setup(command, delta) {
    const { delta: { buffer, data }, device, instances } = this;
    data[0] = delta;
    device.queue.writeBuffer(buffer, 0, data);
    command.clearBuffer(instances, 4, 4);
  }

  shoot(direction, origin) {
    const { device, input } = this;
    vec3.copy(input.position, origin);
    vec3.copy(input.direction, direction);
    input.enabled[0] = 1;
    device.queue.writeBuffer(input.buffer, 0, input.data);
  }

  step(pass) {
    const { passes: { step } } = this;
    step.compute(pass);
  }
}

export default Projectiles;
