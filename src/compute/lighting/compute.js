import Chunk from '../chunk.js';

const Compute = ({ chunkSize }) => `
${Chunk.compute({ atomicQueueCount: true, atomicLight: true, chunkSize })}

struct Uniforms {
  count : u32,
  queue : u32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(1) @binding(0) var<storage, read_write> chunk : Chunk;
@group(1) @binding(1) var<storage, read_write> chunk_east : Chunk;
@group(1) @binding(2) var<storage, read_write> chunk_west : Chunk;
@group(1) @binding(3) var<storage, read_write> chunk_north : Chunk;
@group(1) @binding(4) var<storage, read_write> chunk_south : Chunk;

fn flood(pos : vec3<i32>, level : u32) {
  // I really tried to make this with pointers
  // and a single logic block but I couldn't manage to
  // pass a ptr<storage, Chunk, read_write> to a function
  if (pos.x == -1) {
    let voxel : u32 = getVoxel(vec3<i32>(chunkSize.x - 1, pos.y, pos.z));
    if (chunk_west.voxels[voxel].value != 0) {
      chunk_west.remesh = 1;
      return;
    }
    if (atomicMax(&chunk_west.voxels[voxel].light, level) < level) {
      chunk_west.queues[chunk_west.queue].data[atomicAdd(&chunk_west.queues[chunk_west.queue].count, 1)] = voxel;
    }
    return;
  }
  if (pos.x == chunkSize.x) {
    let voxel : u32 = getVoxel(vec3<i32>(0, pos.y, pos.z));
    if (chunk_east.voxels[voxel].value != 0) {
      chunk_east.remesh = 1;
      return;
    }
    if (atomicMax(&chunk_east.voxels[voxel].light, level) < level) {
      chunk_east.queues[chunk_east.queue].data[atomicAdd(&chunk_east.queues[chunk_east.queue].count, 1)] = voxel;
    }
    return;
  }
  if (pos.z == -1) {
    let voxel : u32 = getVoxel(vec3<i32>(pos.x, pos.y, chunkSize.z - 1));
    if (chunk_south.voxels[voxel].value != 0) {
      chunk_south.remesh = 1;
      return;
    }
    if (atomicMax(&chunk_south.voxels[voxel].light, level) < level) {
      chunk_south.queues[chunk_south.queue].data[atomicAdd(&chunk_south.queues[chunk_south.queue].count, 1)] = voxel;
    }
    return;
  }
  if (pos.z == chunkSize.z) {
    let voxel : u32 = getVoxel(vec3<i32>(pos.x, pos.y, 0));
    if (chunk_north.voxels[voxel].value != 0) {
      chunk_north.remesh = 1;
      return;
    }
    if (atomicMax(&chunk_north.voxels[voxel].light, level) < level) {
      chunk_north.queues[chunk_north.queue].data[atomicAdd(&chunk_north.queues[chunk_north.queue].count, 1)] = voxel;
    }
    return;
  }
  let voxel : u32 = getVoxel(pos);
  if (chunk.voxels[voxel].value != 0) {
    return;
  }
  if (atomicMax(&chunk.voxels[voxel].light, level) < level) {
    chunk.queues[chunk.queue].data[atomicAdd(&chunk.queues[chunk.queue].count, 1)] = voxel;
  }
}

const neighbors = array<vec3<i32>, 6>(
  vec3<i32>(0, -1, 0),
  vec3<i32>(1, 0, 0),
  vec3<i32>(-1, 0, 0),
  vec3<i32>(0, 0, 1),
  vec3<i32>(0, 0, -1),
  vec3<i32>(0, 1, 0),
);

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= uniforms.count) {
    return;
  }
  let voxel : u32 = chunk.queues[uniforms.queue].data[id.x];
  let light : u32 = atomicLoad(&chunk.voxels[voxel].light);
  let pos : vec3<i32> = getPos(voxel);
  for (var n : i32 = 0; n < 6; n++) {
    let npos = pos + neighbors[n];
    if (npos.y == -1 || npos.y == chunkSize.y) {
      continue;
    }
    var level : u32 = light;
    if (n != 0 || level < maxLight) {
      level -= 1;
    }
    flood(npos, level);
  }
}
`;

class LightingCompute {
  constructor({ chunkSize, device, uniforms, workgroups }) {
    this.device = device;
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Compute({ chunkSize }),
        }),
      },
    });
    this.bindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{
        binding: 0,
        resource: { buffer: uniforms },
      }],
    });
    this.workgroups = workgroups;
  }

  compute(pass, chunk) {
    const { bindings, device, pipeline, workgroups } = this;
    if (!chunk.bindings.lightingCompute) {
      chunk.bindings.lightingCompute = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [chunk, ...chunk.neighbors].map(({ data: buffer }, binding) => ({
          binding,
          resource: { buffer },
        })),
      });
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setBindGroup(1, chunk.bindings.lightingCompute);
    pass.dispatchWorkgroupsIndirect(workgroups, 0);
  }
}

export default LightingCompute;
