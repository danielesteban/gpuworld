import { vec2 } from 'gl-matrix';
import Chunk from './chunk.js';
import Frustum from './frustum.js';
import Lighting from './lighting/lighting.js';
import Mesher from './mesher/mesher.js';
import Worldgen from './worldgen/worldgen.js';

const _neighbors = [
  vec2.fromValues(1, 0),
  vec2.fromValues(-1, 0),
  vec2.fromValues(0, 1),
  vec2.fromValues(0, -1),
];
const _position = vec2.create();

class World {
  constructor({
    chunkSize = { x: 64, y: 64, z: 64 },
    camera,
    device,
  }) {
    this.chunkSize = chunkSize;
    this.device = device;
    this.chunks = { data: new Map(), loaded: [] };
    this.frustum = new Frustum({ device, camera });
    this.lighting = new Lighting({ chunkSize, device });
    this.mesher = new Mesher({ chunkSize, device, frustum: this.frustum });
    this.worldgen = new Worldgen({ chunkSize, device });
  }

  compute(command) {
    const { chunks, frustum, lighting, mesher, worldgen } = this;
    const pass = command.beginComputePass();
    frustum.compute(pass);
    chunks.loaded.forEach((chunk) => {
      if (!chunk.hasGenerated) {
        chunk.hasGenerated = true;
        worldgen.compute(pass, chunk);
      }
      if (!chunk.neighbors) {
        chunk.neighbors = _neighbors.map((offset) => {
          const neighbor = this.get(vec2.add(_position, chunk.position, offset));
          if (!neighbor.hasGenerated) {
            worldgen.compute(pass, neighbor);
          }
          return neighbor;
        });
      }
      lighting.compute(pass, chunk);
      mesher.compute(pass, chunk);
    });
    pass.end();
  }

  get(position) {
    const { chunks, chunkSize, device } = this;
    const key = `${position[0]}:${position[1]}`;
    let chunk = chunks.data.get(key);
    if (!chunk) {
      chunk = new Chunk({ chunkSize, device, position });
      chunks.data.set(key, chunk);
    }
    return chunk;
  }

  load(anchor, loadRadius, unloadRadius) {
    const { chunks } = this;
    for (let i = 0, l = chunks.loaded.length; i < l; i++) {
      const chunk = chunks.loaded[i];
      if (vec2.distance(anchor, chunk.position) > (unloadRadius - 0.5)) {
        chunks.loaded.splice(i, 1);
        i--;
        l--;
      }
    }
    World.getGrid(loadRadius).forEach((offset) => {
      const chunk = this.get(vec2.add(_position, anchor, offset));
      if (!chunks.loaded.includes(chunk)) {
        chunks.loaded.push(chunk);
      }
    });
  }

  static getGrid(radius) {
    let grid = World.grids.get(radius);
    if (!grid) {
      grid = [];
      for (let z = -radius; z <= radius; z++) {
        for (let x = -radius; x <= radius; x++) {
          vec2.set(_position, x, z);
          var distance = vec2.length(_position);
          if (distance <= (radius - 0.5)) {
            grid.push({ distance, position: vec2.clone(_position)});
          }
        }
      }
      grid.sort(({ distance: a }, { distance: b }) => a - b);
      grid = grid.map(({ position }) => position);
      World.grids.set(radius, grid);
    }
    return grid;
  }
}

World.grids = new Map();

export default World;
