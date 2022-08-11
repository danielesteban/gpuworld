import Grass from './grass.js';
import Terrain from './terrain.js';

class Worldgen {
  constructor({ chunkSize, device }) {
    this.grass = new Grass({ chunkSize, device });
    this.terrain = new Terrain({ chunkSize, device });
  }

  compute(pass, chunk) {
    const { grass, terrain } = this;
    terrain.compute(pass, chunk);
    grass.compute(pass, chunk);
  }
}

export default Worldgen;
