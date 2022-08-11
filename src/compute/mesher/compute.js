import Chunk from '../chunk.js';

const Compute = ({ chunkSize }) => `
${Chunk.compute({ atomicInstanceCount: true, chunkSize })}

@group(0) @binding(0) var<storage, read> chunk : Chunk;
@group(0) @binding(1) var<storage, read> chunk_east : Chunk;
@group(0) @binding(2) var<storage, read> chunk_west : Chunk;
@group(0) @binding(3) var<storage, read> chunk_north : Chunk;
@group(0) @binding(4) var<storage, read> chunk_south : Chunk;
@group(0) @binding(5) var<storage, read_write> faces : Faces;
@group(0) @binding(6) var<uniform> position : vec3<i32>;

fn isAir(pos : vec3<i32>) -> bool {
  if (
    pos.y < 0
    || (pos.x < 0 && pos.z < 0)
    || (pos.x >= chunkSize.x && pos.z < 0)
    || (pos.x < 0 && pos.z >= chunkSize.z)
    || (pos.x >= chunkSize.x && pos.z >= chunkSize.z)
  ) {
    return false;
  }
  if (pos.y >= chunkSize.y) {
    return true;
  }
  if (pos.x < 0) {
    return chunk_west.voxels[getVoxel(pos + vec3<i32>(chunkSize.x, 0, 0))] == 0;
  }
  if (pos.x >= chunkSize.x) {
    return chunk_east.voxels[getVoxel(pos - vec3<i32>(chunkSize.x, 0, 0))] == 0;
  }
  if (pos.z < 0) {
    return chunk_south.voxels[getVoxel(pos + vec3<i32>(0, 0, chunkSize.z))] == 0;
  }
  if (pos.z >= chunkSize.z) {
    return chunk_north.voxels[getVoxel(pos - vec3<i32>(0, 0, chunkSize.z))] == 0;
  }
  return chunk.voxels[getVoxel(pos)] == 0;
}

fn getLight(pos : vec3<i32>) -> u32 {
  if (pos.y < 0) {
    return 0;
  }
  if (pos.y >= chunkSize.y) {
    return maxLight;
  }
  if (pos.x < 0) {
    return chunk_west.light[getVoxel(pos + vec3<i32>(chunkSize.x, 0, 0))];
  }
  if (pos.x >= chunkSize.x) {
    return chunk_east.light[getVoxel(pos - vec3<i32>(chunkSize.x, 0, 0))];
  }
  if (pos.z < 0) {
    return chunk_south.light[getVoxel(pos + vec3<i32>(0, 0, chunkSize.z))];
  }
  if (pos.z >= chunkSize.z) {
    return chunk_north.light[getVoxel(pos - vec3<i32>(0, 0, chunkSize.z))];
  }
  return chunk.light[getVoxel(pos)]; 
}

struct Normals {
  f : vec3<i32>,
  u : vec3<i32>,
  v : vec3<i32>,
}

const faceNormals = array<Normals, 6>(
  Normals(vec3<i32>(0, 0, 1), vec3<i32>(0, 1, 0), vec3<i32>(1, 0, 0)),
  Normals(vec3<i32>(0, 1, 0), vec3<i32>(0, 0, -1), vec3<i32>(1, 0, 0)),
  Normals(vec3<i32>(0, -1, 0), vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 0)),
  Normals(vec3<i32>(-1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(0, 0, 1)),
  Normals(vec3<i32>(1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(0, 0, 1)),
  Normals(vec3<i32>(0, 0, -1), vec3<i32>(0, 1, 0), vec3<i32>(-1, 0, 0)),
);

const neighbors = array<vec2<i32>, 5>(
  vec2<i32>(0, 0),
  vec2<i32>(-1, 0),
  vec2<i32>(1, 0),
  vec2<i32>(0, -1),
  vec2<i32>(0, 1),
);

fn getLightAvg(pos : vec3<i32>, u : vec3<i32>, v : vec3<i32>) -> f32 {
  var level : f32;
  var count : i32;
  for (var n : i32 = 0; n < 5; n++) {
    let nuv = neighbors[n];
    let npos : vec3<i32> = pos + u * nuv.x + v * nuv.y;
    if (n == 0 || isAir(npos)) {
      let nlight : u32 = getLight(npos);
      if (nlight != 0) {
        level += f32(nlight);
        count++;
      }
    }
  }
  return level / f32(max(count, 1)) / f32(maxLight);
}

fn getTexture(face : i32, value : u32) -> i32 {
  if (value == 1 || face == 2) {
    return 0;
  }
  if (value == 2 && face == 1) {
    return 1;
  }
  return 2;
}

fn pushFace(pos : vec3<i32>, face : i32, texture : i32, light : f32) {
  if (light == 0) {
    return;
  }
  let offset : u32 = atomicAdd(&(faces.instanceCount), 1) * 6;
  faces.data[offset] = f32(pos.x) + 0.5;
  faces.data[offset + 1] = f32(pos.y) + 0.5;
  faces.data[offset + 2] = f32(pos.z) + 0.5;
  faces.data[offset + 3] = light;
  faces.data[offset + 4] = f32(face);
  faces.data[offset + 5] = f32(texture);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let pos : vec3<i32> = vec3<i32>(GlobalInvocationID.xyz);
  if (
    pos.x >= chunkSize.x || pos.y >= chunkSize.y || pos.z >= chunkSize.z
  ) {
    return;
  }
  let value : u32 = chunk.voxels[getVoxel(pos)];
  if (value != 0) {
    for (var face : i32 = 0; face < 6; face++) {
      let npos : vec3<i32> = pos + faceNormals[face].f;
      if (isAir(npos)) {
        pushFace(
          position + pos,
          face,
          getTexture(face, value),
          getLightAvg(npos, faceNormals[face].u, faceNormals[face].v)
        );
      }
    }
  }
}
`;

class MesherCompute {
  constructor({ chunkSize, device, workgroups }) {
    this.device = device;
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: Compute({ chunkSize }),
        }),
        entryPoint: 'main',
      },
    });
    this.workgroups = workgroups;
  }

  compute(pass, chunk) {
    const { device, pipeline, workgroups } = this;
    if (!chunk.bindings.mesherCompute) {
      chunk.bindings.mesherCompute = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [chunk, ...chunk.neighbors, { data: chunk.faces }, { data: chunk.offset }].map(({ data: buffer }, binding) => ({
          binding,
          resource: { buffer },
        })),
      });
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, chunk.bindings.mesherCompute);
    pass.dispatchWorkgroupsIndirect(workgroups, 0);
  }
}

export default MesherCompute;
