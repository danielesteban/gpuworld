import MesherCompute from './compute.js';
import MesherSetup from './setup.js';

class Mesher {
  constructor({ chunkSize, device, frustum }) {
    const workgroups = device.createBuffer({
      size: 3 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
    });
    this.mesh = new MesherCompute({ chunkSize, device, workgroups });
    this.setup = new MesherSetup({ chunkSize, device, frustum, workgroups });
  }

  compute(pass, chunk) {
    const { mesh, setup } = this;
    setup.compute(pass, chunk);
    mesh.compute(pass, chunk);
  }
}

export default Mesher;
