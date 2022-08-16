import { vec2, vec3 } from 'gl-matrix';
import Chunk from './chunk.js';
import Frustum from './frustum.js';
import Lighting from './lighting/lighting.js';
import Mesher from './mesher/mesher.js';
import Query from './query.js';
import Simulation from './simulation/simulation.js';
import Worldgen from './worldgen/worldgen.js';

const _chunk = vec2.create();
const _neighbors = [
  vec2.fromValues(1, 0),
  vec2.fromValues(-1, 0),
  vec2.fromValues(0, 1),
  vec2.fromValues(0, -1),
];
const _voxel = vec3.create();

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
    this.query = new Query({ chunkSize, device });
    this.simulation = new Simulation({ chunkSize, device });
    this.worldgen = new Worldgen({ chunkSize, device });
  }

  compute(command, delta) {
    const { chunks, frustum, lighting, mesher, simulation, worldgen } = this;
    const pass = command.beginComputePass();
    simulation.step(delta, pass);
    chunks.loaded.forEach((chunk) => {
      if (!chunk.hasGenerated) {
        chunk.hasGenerated = true;
        worldgen.compute(pass, chunk);
      }
      if (!chunk.neighbors) {
        chunk.neighbors = _neighbors.map((offset) => {
          const neighbor = this.getChunk(vec2.add(_chunk, chunk.position, offset));
          if (!neighbor.hasGenerated) {
            neighbor.hasGenerated = true;
            worldgen.compute(pass, neighbor);
          }
          return neighbor;
        });
      }
      simulation.compute(pass, chunk);
    });
    chunks.loaded.forEach((chunk) => (
      lighting.compute(pass, chunk)
    ));
    frustum.compute(pass);
    chunks.loaded.forEach((chunk) => (
      mesher.compute(pass, chunk)
    ));
    pass.end();
  }

  getChunk(position) {
    const { chunks, chunkSize, device } = this;
    const key = `${position[0]}:${position[1]}`;
    let chunk = chunks.data.get(key);
    if (!chunk) {
      chunk = new Chunk({ chunkSize, device, position });
      chunks.data.set(key, chunk);
    }
    return chunk;
  }

  static getGrid(radius) {
    let grid = World.grids.get(radius);
    if (!grid) {
      grid = [];
      for (let z = -radius; z <= radius; z++) {
        for (let x = -radius; x <= radius; x++) {
          vec2.set(_chunk, x, z);
          const distance = vec2.length(_chunk);
          if (distance <= (radius - 0.5)) {
            grid.push({ distance, position: vec2.clone(_chunk)});
          }
        }
      }
      grid.sort(({ distance: a }, { distance: b }) => a - b);
      grid = grid.map(({ position }) => position);
      World.grids.set(radius, grid);
    }
    return grid;
  }

  getGround(position, height = 4) {
    const { chunkSize, chunks, query } = this;
    vec2.set(
      _chunk,
      Math.floor(position[0] / chunkSize.x),
      Math.floor(position[2] / chunkSize.z)
    );
    const key = `${_chunk[0]}:${_chunk[1]}`;
    const chunk = chunks.data.get(key);
    if (!chunk || !chunk.hasGenerated) {
      return Promise.resolve(-1);
    }
    vec3.copy(_voxel, position);
    _voxel[0] -= _chunk[0] * chunkSize.x;
    _voxel[1] = Math.min(_voxel[1], chunkSize.y - 1);
    _voxel[2] -= _chunk[1] * chunkSize.z;
    return query.getGround(chunk, _voxel, height);
  }

  load(anchor, loadRadius, unloadRadius) {
    const { chunks } = this;
    for (let i = 0, l = chunks.loaded.length; i < l; i++) {
      const chunk = chunks.loaded[i];
      if (vec2.distance(anchor, chunk.position) > (unloadRadius - 0.5)) {
        chunk.isLoaded = false;
        chunks.loaded.splice(i, 1);
        i--;
        l--;
      }
    }
    World.getGrid(loadRadius).forEach((offset) => {
      const chunk = this.getChunk(vec2.add(_chunk, anchor, offset));
      if (!chunk.isLoaded) {
        chunk.isLoaded = true;
        chunks.loaded.push(chunk);
      }
    });
  }
}

World.grids = new Map();

export default World;
