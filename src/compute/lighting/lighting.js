import LightingCompute from './compute.js';
import LightingSetup from './setup.js';

class Lighting {
  constructor({ chunkSize, device }) {
    const uniforms = device.createBuffer({
      size: 2 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE,
    });
    const workgroups = device.createBuffer({
      size: 3 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
    });
    this.propagate = new LightingCompute({ chunkSize, device, uniforms, workgroups });
    this.setup = new LightingSetup({ chunkSize, device, uniforms, workgroups });
  }

  compute(pass, chunk) {
    const { propagate, setup } = this;
    setup.compute(pass, chunk);
    propagate.compute(pass, chunk);
  }
}

export default Lighting;
