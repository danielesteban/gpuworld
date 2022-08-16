const Compute = ({ count }) => `
struct Explosion {
  enabled: u32,
  position: vec3<f32>,
  step: f32,
}

@group(0) @binding(0) var<uniform> delta : f32;
@group(0) @binding(1) var<storage, read_write> meshes : array<vec4<f32>, ${count}>;
@group(0) @binding(2) var<storage, read_write> explosions : array<Explosion, ${count}>;
@group(0) @binding(3) var<storage, read_write> workgroups : array<atomic<u32>, 3>;

@compute @workgroup_size(${Math.min(count, 256)})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= ${count}) {
    return;
  }
  if (explosions[id.x].enabled != 1) {
    return;
  }
  explosions[id.x].step += delta * 4;
  if (explosions[id.x].step > 1) {
    explosions[id.x].enabled = 0;
    return;
  }
  var offset : u32 = atomicAdd(&workgroups[1], 1);
  meshes[offset] = vec4<f32>(explosions[id.x].position, explosions[id.x].step);
}
`;

class ExplosionsStep {
  constructor({ count, delta, device, explosions }) {
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Compute({ count }),
        }),
      },
    });
    this.bindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: delta.buffer },
        },
        {
          binding: 1,
          resource: { buffer: explosions.meshes },
        },
        {
          binding: 2,
          resource: { buffer: explosions.state },
        },
        {
          binding: 3,
          resource: { buffer: explosions.workgroups },
        },
      ],
    });
    this.workgroups = Math.ceil(count / 256);
  }

  compute(pass) {
    const { bindings, pipeline, workgroups } = this;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(workgroups);
  }
}

export default ExplosionsStep;
