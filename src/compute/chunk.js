import { vec2 } from 'gl-matrix';

class Chunk {
  constructor({ chunkSize, device, position }) {
    this.bindings = {};
    this.position = vec2.clone(position);

    this.data = device.createBuffer({
      mappedAtCreation: true,
      size: (
        // bounds
        6
        // voxels
        + chunkSize.x * chunkSize.y * chunkSize.z * 2
        // remesh
        + 1
        // queue
        + 1
        // queues
        + (1 + chunkSize.x * chunkSize.z * 3) * 2
      ) * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    });
    new Uint32Array(this.data.getMappedRange(0, 3 * Uint32Array.BYTES_PER_ELEMENT)).set([
      chunkSize.x,
      chunkSize.y,
      chunkSize.z,
    ]);
    this.data.unmap();

    this.faces = device.createBuffer({
      size: (
        // indirect drawing buffer
        4 * Uint32Array.BYTES_PER_ELEMENT
        + (
          // worst-case scenario
          Math.ceil(chunkSize.x * chunkSize.y * chunkSize.z * 0.5)
        ) * 6 * 6 * Float32Array.BYTES_PER_ELEMENT
      ),
      usage: (
        GPUBufferUsage.INDIRECT
        | GPUBufferUsage.STORAGE
        | GPUBufferUsage.VERTEX
      ),
    });

    this.offset = device.createBuffer({
      mappedAtCreation: true,
      size: 3 * Int32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM,
    });
    new Int32Array(this.offset.getMappedRange()).set([
      position[0] * chunkSize.x,
      0,
      position[1] * chunkSize.z,
    ])
    this.offset.unmap();
  }
}

Chunk.compute = ({
  atomicBounds,
  atomicInstanceCount,
  atomicLight,
  atomicQueueCount,
  chunkSize,
}) => `
  const chunkSize : vec3<i32> = vec3<i32>(${chunkSize.x}, ${chunkSize.y}, ${chunkSize.z});
  const maxLight : u32 = 255;

  struct Bounds {
    min : array<${atomicBounds ? 'atomic<u32>' : 'u32'}, 3>,
    max : array<${atomicBounds ? 'atomic<u32>' : 'u32'}, 3>,
  }

  struct Faces {
    vertexCount : u32,
    instanceCount : ${atomicInstanceCount ? 'atomic<u32>' : 'u32'},
    firstVertex : u32,
    firstInstance : u32,
    data : array<f32>,
  }

  struct Queue {
    count : ${atomicQueueCount ? 'atomic<u32>' : 'u32'},
    data : array<u32, ${chunkSize.x * chunkSize.z * 3}>,
  }

  struct Voxel {
    value : u32,
    light : ${atomicLight ? 'atomic<u32>' : 'u32'},
  }

  struct Chunk {
    bounds : Bounds,
    voxels : array<Voxel, ${chunkSize.x * chunkSize.y * chunkSize.z}>,
    remesh : u32,
    queue : u32,
    queues : array<Queue, 2>,
  }

  fn getVoxel(pos : vec3<i32>) -> u32 {
    return u32(pos.z * chunkSize.x * chunkSize.y + pos.y * chunkSize.x + pos.x);
  }

  fn getPos(voxel : u32) -> vec3<i32> {
    let o : i32 = i32(voxel) % (chunkSize.x * chunkSize.y);
    return vec3<i32>(
      o % chunkSize.x,
      o / chunkSize.x,
      i32(voxel) / (chunkSize.x * chunkSize.y),
    );
  }
`;

export default Chunk;
