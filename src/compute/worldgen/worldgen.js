import Grow from './grow.js';
import Populate from './populate.js';
import Terrain from './terrain.js';
import Setup from './setup.js';

class Worldgen {
  constructor({ chunkSize, device }) {
    {
      const count = 24;
      this.trees = {
        buffer: device.createBuffer({
          size: (1 + count) * Uint32Array.BYTES_PER_ELEMENT,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
        }),
        count,
      };
    }
    this.grow = new Grow({ chunkSize, device, trees: this.trees });
    this.populate = new Populate({ chunkSize, device, trees: this.trees });
    this.setup = new Setup({ device, trees: this.trees });
    this.terrain = new Terrain({ chunkSize, device });
  }

  compute(pass, chunk) {
    const { grow, populate, setup, terrain } = this;
    setup.compute(pass);
    terrain.compute(pass, chunk);
    populate.compute(pass, chunk);
    grow.compute(pass, chunk);
  }
}

export default Worldgen;
