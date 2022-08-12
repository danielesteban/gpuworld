import Chunk from '../chunk.js';
import Noise from './noise.js';

const Compute = ({ chunkSize }) => `
${Chunk.compute({ atomicBounds: true, atomicQueueCount: true, chunkSize })}

${Noise}

fn FBM(p : vec3<f32>) -> f32 {
  var value : f32;
  var amplitude : f32 = 0.5;
  var q : vec3<f32> = p;
  for (var i : i32 = 0; i < 3; i++) {
    value += simplexNoise3(q) * amplitude;
    q *= 2;
    amplitude *= 0.5;
  }
  return value;
}

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

  if (pos.y == chunkSize.y - 1) {
    let voxel = getVoxel(pos);
    chunk.voxels[voxel].light = maxLight;
    chunk.queues[chunk.queue].data[atomicAdd(&(chunk.queues[chunk.queue].count), 1)] = voxel;
    return;
  }

  let wpos = vec3<f32>(position + pos);
  if (wpos.y == 0 || wpos.y <= abs(FBM(wpos * 0.02) + 0.2) * f32(chunkSize.y) * 1.5) {
    var value : u32;
    if (abs(FBM(wpos.yzx * vec3<f32>(0.01, 0.04, 0.01))) > 0.4) {
      value = 1;
    } else {
      value = 2;
    }
    chunk.voxels[getVoxel(pos)].value = value;
    atomicMin(&chunk.bounds.min[0], u32(pos.x));
    atomicMin(&chunk.bounds.min[1], u32(pos.y));
    atomicMin(&chunk.bounds.min[2], u32(pos.z));
    atomicMax(&chunk.bounds.max[0], u32(pos.x) + 1);
    atomicMax(&chunk.bounds.max[1], u32(pos.y) + 1);
    atomicMax(&chunk.bounds.max[2], u32(pos.z) + 1);
  }
}
`;

class Terrain {
  constructor({ chunkSize, device }) {
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
    this.workgroups = {
      x: Math.ceil(chunkSize.x / 4),
      y: Math.ceil(chunkSize.y / 4),
      z: Math.ceil(chunkSize.z / 4),
    };
  }

  compute(pass, chunk) {
    const { device, pipeline, workgroups } = this;
    if (!chunk.bindings.terrain) {
      chunk.bindings.terrain = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: { buffer: chunk.data },
          },
          {
            binding: 1,
            resource: { buffer: chunk.offset },
          },
        ],
      });
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, chunk.bindings.terrain);
    pass.dispatchWorkgroups(workgroups.x, workgroups.y, workgroups.z);
  }
}

export default Terrain;
