const Compute = ({ count }) => `
struct Input {
  position: vec3<f32>,
  direction: vec3<f32>,
  enabled: atomic<u32>,
}

struct Instances {
  vertexCount : u32,
  instanceCount : atomic<u32>,
  firstVertex : u32,
  firstInstance : u32,
  data : array<f32>,
}

struct Projectile {
  position: vec3<f32>,
  direction: vec3<f32>,
  distance: u32,
  state: u32,
}

@group(0) @binding(0) var<uniform> delta : f32;
@group(0) @binding(1) var<storage, read_write> input : Input;
@group(0) @binding(2) var<storage, read_write> instances : Instances;
@group(0) @binding(3) var<storage, read_write> state : array<Projectile, ${count}>;

fn pushInstance(position : vec3<f32>, direction : vec3<f32>) {
  let offset : u32 = atomicAdd(&instances.instanceCount, 1) * 6;
  instances.data[offset] = position.x;
  instances.data[offset + 1] = position.y;
  instances.data[offset + 2] = position.z;
  instances.data[offset + 3] = direction.x;
  instances.data[offset + 4] = direction.y;
  instances.data[offset + 5] = direction.z;
}

@compute @workgroup_size(${Math.min(count, 256)})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= ${count}) {
    return;
  }
  if (state[id.x].state == 1) {
    state[id.x].distance++;
    if (state[id.x].distance > 256) {
      state[id.x].state = 0;
      return;
    }
    let direction : vec3<f32> = state[id.x].direction;
    state[id.x].position += direction * delta * 60;
    pushInstance(state[id.x].position, direction);
    return;
  }
  if (state[id.x].state == 2) {
    // @incomplete: Spawn explosion at state[id.x].position
    state[id.x].state = 0;
  }
  if (atomicMin(&input.enabled, 0) != 0) {
    state[id.x].position = input.position; 
    state[id.x].direction = input.direction; 
    state[id.x].distance = 0;
    state[id.x].state = 1;
  }
}
`;

class ProjectilesStep {
  constructor({ count, delta, device, input, instances, state }) {
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
          resource: { buffer: input.buffer },
        },
        {
          binding: 2,
          resource: { buffer: instances },
        },
        {
          binding: 3,
          resource: { buffer: state },
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

export default ProjectilesStep;
