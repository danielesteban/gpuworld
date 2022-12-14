import Chunk from '../chunk.js';
import Noise from './noise.js';

const Compute = ({ chunkSize }) => `
${Chunk.compute({ atomicQueueCount: true, chunkSize })}

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

@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let pos : vec3<i32> = vec3<i32>(id);
  if (any(pos >= chunkSize)) {
    return;
  }

  if (pos.y == chunkSize.y - 1) {
    let voxel = getVoxel(pos);
    chunk.voxels[voxel].light = maxLight;
    chunk.queues[chunk.queue].data[atomicAdd(&chunk.queues[chunk.queue].count, 1)] = voxel;
    return;
  }

  let wpos = vec3<f32>(position + pos);
  if (wpos.y == 0 || wpos.y <= abs(FBM(wpos * 0.015) + 0.3) * f32(chunkSize.y) * 1.2) {
    var value : u32;
    if (abs(FBM(wpos.yzx * vec3<f32>(0.06, 0.03, 0.03))) > 0.3) {
      value = 1;
    } else {
      value = 2;
    }
    chunk.voxels[getVoxel(pos)].value = value;
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
      x: Math.ceil(chunkSize.x / 64),
      y: Math.ceil(chunkSize.y / 4),
      z: chunkSize.z,
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
