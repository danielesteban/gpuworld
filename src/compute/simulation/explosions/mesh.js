import { vec3 } from 'gl-matrix';

const Compute = ({ count, instancesPerMesh }) => `
struct Instances {
  indexCount : u32,
  instanceCount : atomic<u32>,
  firstIndex : u32,
  baseVertex : u32,
  firstInstance : u32,
  data : array<f32, ${count * instancesPerMesh * 4}>,
}

@group(0) @binding(0) var<storage, read_write> instances : Instances;
@group(0) @binding(1) var<uniform> meshes : array<vec4<f32>, ${count}>;
@group(0) @binding(2) var<uniform> normals : array<vec3<f32>, ${instancesPerMesh}>;

@compute @workgroup_size(${instancesPerMesh})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let offset : u32 = (id.y * ${instancesPerMesh} + id.x) * 4;
  let step : f32 = meshes[id.y].w;
  let pos : vec3<f32> = meshes[id.y].xyz + normals[id.x] * step * 2;
  instances.data[offset] = pos.x;
  instances.data[offset + 1] = pos.y;
  instances.data[offset + 2] = pos.z;
  instances.data[offset + 3] = (1 - step) * 2;
  atomicAdd(&instances.instanceCount, 1);
}
`;

class ExplosionsMesh {
  constructor({ count, device, explosions }) {
    const normals = device.createBuffer({
      mappedAtCreation: true,
      size: (
        explosions.instancesPerMesh * 4 * Float32Array.BYTES_PER_ELEMENT
      ),
      usage: GPUBufferUsage.UNIFORM,
    });
    const data = new Float32Array(normals.getMappedRange());
    const normal = vec3.create();
    for (let i = 0, j = 0; i < explosions.instancesPerMesh; i++, j += 4) {
      vec3.set(normal, Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5);
      vec3.normalize(normal, normal);
      vec3.scale(normal, normal, 1.5 - Math.random());
      data.set(normal, j);
    }
    normals.unmap();
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Compute({ count, instancesPerMesh: explosions.instancesPerMesh }),
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
          resource: { buffer: explosions.meshes },
        },
        {
          binding: 2,
          resource: { buffer: normals },
        },
      ],
    });
    this.workgroups = explosions.workgroups;
  }

  compute(pass) {
    const { bindings, pipeline, workgroups } = this;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroupsIndirect(workgroups, 0);
  }
}

export default ExplosionsMesh;
