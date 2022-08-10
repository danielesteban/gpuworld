import Chunk from '../chunk.js';

const Compute = ({ chunkSize }) => `
${Chunk.compute({ chunkSize })}

@group(0) @binding(0) var<storage, read_write> chunk : Chunk;
@group(0) @binding(1) var<uniform> position : vec3<i32>;

@compute @workgroup_size(1)
fn main() {
  var bmin : vec3<f32> = vec3<f32>(f32(chunk.bounds.min[0]), f32(chunk.bounds.min[1]), f32(chunk.bounds.min[2]));
  var bmax : vec3<f32> = vec3<f32>(f32(chunk.bounds.max[0]), f32(chunk.bounds.max[1]), f32(chunk.bounds.max[2]));
  chunk.bounds.center = vec3<f32>(position) + (bmin + bmax) * 0.5; 
  chunk.bounds.radius = length((bmax - bmin) * 0.5);
}
`;

class Bounds {
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
  }

  compute(pass, chunk) {
    const { device, pipeline } = this;
    if (!chunk.bindings.bounds) {
      chunk.bindings.bounds = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
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
    pass.setBindGroup(0, chunk.bindings.bounds);
    pass.dispatchWorkgroups(1);
  }
}

export default Bounds;
