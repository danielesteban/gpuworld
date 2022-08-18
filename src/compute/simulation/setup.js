const Compute = `
struct Instances {
  indexCount : u32,
  instanceCount : u32,
}

@group(0) @binding(0) var<storage, read_write> explosions : Instances;
@group(0) @binding(1) var<storage, read_write> projectiles : Instances;
@group(0) @binding(2) var<storage, read_write> workgroups : array<u32, 3>;

@compute @workgroup_size(1)
fn main() {
  explosions.instanceCount = 0;
  projectiles.instanceCount = 0;
  workgroups[1] = 0;
}
`;

class SimulationSetup {
  constructor({ device, explosions, projectiles }) {
    this.device = device;
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
          resource: { buffer: explosions.instances },
        },
        {
          binding: 1,
          resource: { buffer: projectiles.instances },
        },
        {
          binding: 2,
          resource: { buffer: explosions.workgroups },
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

export default SimulationSetup;
