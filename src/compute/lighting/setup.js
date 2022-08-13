import Chunk from '../chunk.js';

const Compute = ({ chunkSize }) => `
${Chunk.compute({ chunkSize })}

struct Uniforms {
  count : u32,
  queue : u32,
}

@group(0) @binding(0) var<storage, read_write> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read_write> workgroups : array<u32, 3>;
@group(1) @binding(0) var<storage, read_write> chunk : Chunk;

@compute @workgroup_size(1)
fn main() {
  let count : u32 = chunk.queues[chunk.queue].count;
  if (count == 0) {
    workgroups[0] = 0;
    return;
  }

  uniforms.count = count;
  uniforms.queue = chunk.queue;

  workgroups[0] = u32(ceil(f32(count) / 256));
  workgroups[1] = 1;
  workgroups[2] = 1;

  let next : u32 = (chunk.queue + 1) % 2;
  chunk.queue = next;
  chunk.queues[next].count = 0;

  chunk.remesh = 1;
}
`;

class LightingSetup {
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
      entries: [
        {
          binding: 0,
          resource: { buffer: uniforms },
        },
        {
          binding: 1,
          resource: { buffer: workgroups },
        },
      ],
    });
  }

  compute(pass, chunk) {
    const { bindings, device, pipeline } = this;
    if (!chunk.bindings.lightingSetup) {
      chunk.bindings.lightingSetup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [{
          binding: 0,
          resource: { buffer: chunk.data },
        }],
      });
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setBindGroup(1, chunk.bindings.lightingSetup);
    pass.dispatchWorkgroups(1);
  }
}

export default LightingSetup;
