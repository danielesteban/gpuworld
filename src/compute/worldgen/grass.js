import Chunk from '../chunk.js';

const Compute = ({ chunkSize }) => `
${Chunk.compute({ chunkSize })}

@group(0) @binding(0) var<storage, read_write> chunk : Chunk;
@group(0) @binding(1) var<uniform> position : vec3<i32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let pos : vec3<i32> = vec3<i32>(GlobalInvocationID.xyz);
  if (
    pos.x >= chunkSize.x || pos.y >= chunkSize.y || pos.z >= chunkSize.z
  ) {
    return;
  }
  let voxel = getVoxel(pos);
  if (chunk.voxels[voxel].value == 2 && chunk.voxels[getVoxel(pos + vec3<i32>(0, 1, 0))].value == 0) {
    chunk.voxels[voxel].value = 3;
  }
}
`;

class Grass {
  constructor({ chunkSize, device }) {
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
    this.workgroups = {
      x: Math.ceil(chunkSize.x / 4),
      y: Math.ceil(chunkSize.y / 4),
      z: Math.ceil(chunkSize.z / 4),
    };
  }

  compute(pass, chunk) {
    const { device, pipeline, workgroups } = this;
    if (!chunk.bindings.grass) {
      chunk.bindings.grass = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{
          binding: 0,
          resource: { buffer: chunk.data },
        }],
      });
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, chunk.bindings.grass);
    pass.dispatchWorkgroups(workgroups.x, workgroups.y, workgroups.z);
  }
}

export default Grass;
