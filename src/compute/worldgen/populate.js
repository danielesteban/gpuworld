import Chunk from '../chunk.js';
import Noise from './noise.js';

const Compute = ({ chunkSize, maxTrees }) => `
${Chunk.compute({ chunkSize })}

${Noise}

struct Trees {
  count : atomic<u32>,
  data : array<u32, ${maxTrees}>,
}

@group(0) @binding(0) var<storage, read_write> trees : Trees;
@group(1) @binding(0) var<storage, read_write> chunk : Chunk;
@group(1) @binding(1) var<uniform> position : vec3<i32>;

@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let pos : vec3<i32> = vec3<i32>(id);
  if (any(pos >= chunkSize)) {
    return;
  }
  let voxel = getVoxel(pos);
  if (
    chunk.voxels[voxel].value == 2
    && chunk.voxels[getVoxel(pos + vec3<i32>(0, 1, 0))].value == 0
  ) {
    chunk.voxels[voxel].value = 3;
    if (
      all(pos > vec3<i32>(5, 4, 5))
      && all(pos < (chunkSize - vec3<i32>(6, 20, 6)))
      && simplexNoise3(vec3<f32>(position + pos)) > 0.9
    ) {
      let tree = atomicAdd(&trees.count, 1);
      if (tree < ${maxTrees}) {
        trees.data[tree] = voxel;
      }
    }
  }
}
`;

class Populate {
  constructor({ chunkSize, device, trees }) {
    this.device = device;
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Compute({ chunkSize, maxTrees: trees.count }),
        }),
      },
    });
    this.bindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: trees.buffer },
        },
      ],
    });
    this.workgroups = {
      x: Math.ceil(chunkSize.x / 64),
      y: Math.ceil(chunkSize.y / 4),
      z: chunkSize.z,
    };
  }

  compute(pass, chunk) {
    const { bindings, device, pipeline, workgroups } = this;
    if (!chunk.bindings.populate) {
      chunk.bindings.populate = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: { buffer: chunk.data },
          },
          {
            binding: 1,
            resource: { buffer: chunk.offset },
          }
        ],
      });
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setBindGroup(1, chunk.bindings.populate);
    pass.dispatchWorkgroups(workgroups.x, workgroups.y, workgroups.z);
  }
}

export default Populate;
