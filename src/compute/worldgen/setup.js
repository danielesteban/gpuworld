const Compute = `
@group(0) @binding(0) var<storage, read_write> trees : u32;

@compute @workgroup_size(1)
fn main() {
  trees = 0;
}
`;

class WorldgenSetup {
  constructor({ device, trees }) {
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Compute,
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
  }

  compute(pass) {
    const { bindings, pipeline } = this;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(1);
  }
}

export default WorldgenSetup;
