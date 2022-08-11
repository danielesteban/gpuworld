const Compute = `
struct Camera {
  projection : mat4x4<f32>,
  view : mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<storage, read_write> frustum : array<vec4<f32>, 6>;

fn update(index : i32, plane : vec4<f32>) {
  frustum[index] = plane * (1 / length(plane.xyz));
}

@compute @workgroup_size(1)
fn main() {
  let m : mat4x4<f32> = camera.projection * camera.view;
  update(0, vec4<f32>(m[0].w - m[0].x, m[1].w - m[1].x, m[2].w - m[2].x, m[3].w - m[3].x));
  update(1, vec4<f32>(m[0].w + m[0].x, m[1].w + m[1].x, m[2].w + m[2].x, m[3].w + m[3].x));
  update(2, vec4<f32>(m[0].w + m[0].y, m[1].w + m[1].y, m[2].w + m[2].y, m[3].w + m[3].y));
  update(3, vec4<f32>(m[0].w - m[0].y, m[1].w - m[1].y, m[2].w - m[2].y, m[3].w - m[3].y));
  update(4, vec4<f32>(m[0].w - m[0].z, m[1].w - m[1].z, m[2].w - m[2].z, m[3].w - m[3].z));
  update(5, vec4<f32>(m[0].w + m[0].z, m[1].w + m[1].z, m[2].w + m[2].z, m[3].w + m[3].z));
}
`;

class Frustum {
  constructor({ device, camera }) {
    this.buffer = device.createBuffer({
      size: 6 * 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE,
    });
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
          resource: { buffer: camera.buffer },
        },
        {
          binding: 1,
          resource: { buffer: this.buffer },
        }
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

export default Frustum;
