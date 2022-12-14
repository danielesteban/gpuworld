import Chunk from '../../chunk.js';

const Compute = ({ chunkSize, count }) => `
${Chunk.compute({ atomicQueueCount: true, atomicValue: true, chunkSize })}

struct Projectile {
  position: vec3<f32>,
  direction: vec3<f32>,
  iteration: u32,
  state: u32,
}

@group(0) @binding(0) var<storage, read_write> projectiles : array<Projectile, ${count}>;
@group(1) @binding(0) var<storage, read_write> chunk : Chunk;
@group(1) @binding(1) var<storage, read_write> chunk_east : Chunk;
@group(1) @binding(2) var<storage, read_write> chunk_west : Chunk;
@group(1) @binding(3) var<storage, read_write> chunk_north : Chunk;
@group(1) @binding(4) var<storage, read_write> chunk_south : Chunk;
@group(1) @binding(5) var<uniform> position : vec3<i32>;

fn flood(pos : vec3<i32>) {
  if (pos.y == -1 || pos.y == chunkSize.y) {
    return;
  }
  if (pos.x == -1) {
    let voxel : u32 = getVoxel(vec3<i32>(chunkSize.x - 1, pos.y, pos.z));
    if (atomicLoad(&chunk_west.voxels[voxel].value) == 0 && chunk_west.voxels[voxel].light != 0) {
      chunk_west.queues[chunk_west.queue].data[atomicAdd(&chunk_west.queues[chunk_west.queue].count, 1)] = voxel;
    }
    return;
  }
  if (pos.x == chunkSize.x) {
    let voxel : u32 = getVoxel(vec3<i32>(0, pos.y, pos.z));
    if (atomicLoad(&chunk_east.voxels[voxel].value) == 0 && chunk_east.voxels[voxel].light != 0) {
      chunk_east.queues[chunk_east.queue].data[atomicAdd(&chunk_east.queues[chunk_east.queue].count, 1)] = voxel;
    }
    return;
  }
  if (pos.z == -1) {
    let voxel : u32 = getVoxel(vec3<i32>(pos.x, pos.y, chunkSize.z - 1));
    if (atomicLoad(&chunk_south.voxels[voxel].value) == 0 && chunk_south.voxels[voxel].light != 0) {
      chunk_south.queues[chunk_south.queue].data[atomicAdd(&chunk_south.queues[chunk_south.queue].count, 1)] = voxel;
    }
    return;
  }
  if (pos.z == chunkSize.z) {
    let voxel : u32 = getVoxel(vec3<i32>(pos.x, pos.y, 0));
    if (atomicLoad(&chunk_north.voxels[voxel].value) == 0 && chunk_north.voxels[voxel].light != 0) {
      chunk_north.queues[chunk_north.queue].data[atomicAdd(&chunk_north.queues[chunk_north.queue].count, 1)] = voxel;
    }
    return;
  }
  let voxel : u32 = getVoxel(pos);
  if (atomicLoad(&chunk.voxels[voxel].value) == 0 && chunk.voxels[voxel].light != 0) {
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

fn collide(id : u32) {
  let pos : vec3<i32> = vec3<i32>(floor(projectiles[id].position)) - position;
  if (any(pos < vec3<i32>(0)) || any(pos >= chunkSize)) {
    return;
  }
  if (pos.y == 0) {
    projectiles[id].state = 2;
    return;
  }
  let voxel : u32 = getVoxel(pos);
  if (atomicMin(&chunk.voxels[voxel].value, 0) != 0) {
    projectiles[id].state = 2;
    for (var n : i32 = 0; n < 6; n++) {
      flood(pos + neighbors[n]);
    }
  }
}

fn detonate(id : u32) {
  let pos : vec3<i32> = vec3<i32>(floor(projectiles[id].position)) - position;
  let radius = i32(projectiles[id].state) - 2;
  for (var z : i32 = -radius; z <= radius; z++) {
    for (var y : i32 = -radius; y <= radius; y++) {
      for (var x : i32 = -radius; x <= radius; x++) {
        let npos : vec3<i32> = pos + vec3<i32>(x, y, z);
        if (
          any(npos < vec3<i32>(0, 1, 0))
          || any(npos >= chunkSize)
          || length(vec3<f32>(f32(x), f32(y), f32(z))) > min(f32(radius), 2.5)
        ) {
          continue;
        }
        let voxel : u32 = getVoxel(npos);
        if (atomicMin(&chunk.voxels[voxel].value, 0) != 0) {
          for (var n : i32 = 0; n < 6; n++) {
            flood(npos + neighbors[n]);
          }
        }
      }
    }
  }
}

@compute @workgroup_size(${Math.min(count, 256)})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= ${count}) {
    return;
  }
  switch (projectiles[id.x].state) {
    default {}
    case 1 {
      collide(id.x);
    }
    case 3, 4, 5 {
      detonate(id.x);
    }
  }
}
`;

class ProjectilesCompute {
  constructor({ chunkSize, count, device, projectiles }) {
    this.device = device;
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: Compute({ chunkSize, count }),
        }),
        entryPoint: 'main',
      },
    });
    this.bindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: projectiles.state },
        },
      ],
    });
    this.workgroups = Math.ceil(count / 256);
  }

  compute(pass, chunk) {
    const { bindings, device, pipeline, workgroups } = this;
    if (!chunk.bindings.projectiles) {
      chunk.bindings.projectiles = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [chunk, ...chunk.neighbors, { data: chunk.offset }].map(({ data: buffer }, binding) => ({
          binding,
          resource: { buffer },
        })),
      });
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setBindGroup(1, chunk.bindings.projectiles);
    pass.dispatchWorkgroups(workgroups);
  }
}

export default ProjectilesCompute;
