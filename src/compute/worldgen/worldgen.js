import Grow from './grow.js';
import Populate from './populate.js';
import Terrain from './terrain.js';

class Worldgen {
  constructor({ chunkSize, device }) {
    this.trees = device.createBuffer({
      size: 25 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
    });
    this.grow = new Grow({ chunkSize, device, trees: this.trees });
    this.populate = new Populate({ chunkSize, device, trees: this.trees });
    this.terrain = new Terrain({ chunkSize, device, trees: this.trees });
  }

  compute(pass, chunk) {
    const { grow, populate, terrain } = this;
    terrain.compute(pass, chunk);
    populate.compute(pass, chunk);
    grow.compute(pass, chunk);
  }
}

export default Worldgen;
