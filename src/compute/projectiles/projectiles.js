import ProjectilesCompute from './compute.js';
import ProjectilesStep from './step.js';

const _zero = new Uint32Array(1);

class Projectiles {
  constructor({ count = 32, chunkSize, device }) {
    this.device = device;
    this.delta = {
      buffer: device.createBuffer({
        size: Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
      }),
      data: new Float32Array(1),
    };
    this.input = {
      buffer: device.createBuffer({
        size: 8 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      }),
      data: new Float32Array(8),
    };
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

  shoot(direction, origin) {
    const { device, input } = this;
    input.data.set(origin);
    input.data.set(direction, 4);
    new Uint32Array(input.data.buffer, 28, 1)[0] = 1;
    device.queue.writeBuffer(input.buffer, 0, input.data.buffer);
  }

  step(pass, delta) {
    const { delta: { buffer, data }, device, instances, passes: { step } } = this;
    data[0] = delta;
    device.queue.writeBuffer(buffer, 0, data);
    device.queue.writeBuffer(instances, 4, _zero);
    step.compute(pass);
  }
}

export default Projectiles;
