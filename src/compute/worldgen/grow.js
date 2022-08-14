import Chunk from '../chunk.js';
import Noise from './noise.js';

const Compute = ({ chunkSize, maxTrees }) => `
${Chunk.compute({ atomicBounds: true, chunkSize })}

${Noise}

struct Trees {
  count : u32,
  data : array<u32, ${maxTrees}>,
}

@group(0) @binding(0) var<storage, read_write> trees : Trees;
@group(1) @binding(0) var<storage, read_write> bounds : Bounds;
@group(1) @binding(1) var<storage, read_write> chunk : Chunk;
@group(1) @binding(2) var<uniform> position : vec3<i32>;

fn grow(pos : vec3<i32>, value : u32) -> bool {
  if (pos.y >= chunkSize.y - 1) {
    return false;
  }
  let voxel : u32 = getVoxel(pos);
  let v : u32 = chunk.voxels[voxel].value;
  if (v != 0 && v != 4 && v != 5) {
    return false;
  }
  chunk.voxels[voxel].value = value;
  atomicMin(&bounds.min[0], u32(pos.x));
  atomicMin(&bounds.min[1], u32(pos.y));
  atomicMin(&bounds.min[2], u32(pos.z));
  atomicMax(&bounds.max[0], u32(pos.x) + 1);
  atomicMax(&bounds.max[1], u32(pos.y) + 1);
  atomicMax(&bounds.max[2], u32(pos.z) + 1);
  return true;
}

@compute @workgroup_size(${Math.min(maxTrees, 256)})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= trees.count) {
    return;
  }
  let pos : vec3<i32> = getPos(trees.data[id.x]) + vec3<i32>(0, -1, 0);
  let height : i32 = 8 + i32(abs(noise3(vec3<f32>(position + pos) * 0.1)) * 8);
  for (var i : i32 = 0; i < height; i++) {
    if (!grow(pos + vec3<i32>(0, i, 0), 4) && i > 1) {
      return;
    }
  }
  let radius : f32 = 3 + abs(noise3(vec3<f32>(position + pos.yzx) * 0.05)) * 2;
  let iradius : i32 = i32(ceil(radius + 0.5));
  let offset : vec3<i32> = vec3<i32>(0, height, 0);
  for (var z : i32 = -iradius; z <= iradius; z++) {
    for (var y : i32 = -3; y <= iradius; y++) {
      for (var x : i32 = -iradius; x <= iradius; x++) {
        let npos : vec3<i32> = vec3<i32>(x, y, z);
        if (length(vec3<f32>(npos)) < radius) {
          grow(pos + offset + npos, 5);
        }
      }
    }
  }
}
`;

class Grow {
  constructor({ chunkSize, device, trees }) {
    const maxTrees = (trees.size / 4) - 1;
    this.device = device;
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Compute({ chunkSize, maxTrees }),
        }),
      },
    });
    this.bindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: trees },
        },
      ],
    });
    this.workgroups = Math.ceil(maxTrees / 256);
  }

  compute(pass, chunk) {
    const { bindings, device, pipeline, workgroups } = this;
    if (!chunk.bindings.grow) {
      chunk.bindings.grow = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: { buffer: chunk.bounds },
          },
          {
            binding: 1,
            resource: { buffer: chunk.data },
          },
          {
            binding: 2,
            resource: { buffer: chunk.offset },
          },
        ],
      });
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setBindGroup(1, chunk.bindings.grow);
    pass.dispatchWorkgroups(workgroups);
  }
}

export default Grow;
