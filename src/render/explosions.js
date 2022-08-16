const Vertex = `
struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) origin : vec3<f32>,
  @location(3) scale : f32,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) normal : vec3<f32>,
  @location(1) depth : f32,
}

struct Camera {
  projection : mat4x4<f32>,
  view : mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera : Camera;

@vertex
fn main(explosion : VertexInput) -> VertexOutput {
  let position : vec3<f32> = explosion.position * explosion.scale + explosion.origin;
  let mvPosition : vec4<f32> = camera.view * vec4<f32>(position, 1);
  var out : VertexOutput;
  out.position = camera.projection * mvPosition;
  out.normal = explosion.normal;
  out.depth = -mvPosition.z;
  return out;
}
`;

const Fragment = `
struct FragmentInput {
  @location(0) normal : vec3<f32>,
  @location(1) depth : f32,
}

struct FragmentOutput {
  @location(0) color : vec4<f32>,
  @location(1) data : vec4<f32>,
}

@group(0) @binding(1) var<uniform> sunlight : vec3<f32>;

@fragment
fn main(explosion : FragmentInput) -> FragmentOutput {
  var output : FragmentOutput;
  output.color = vec4<f32>(sunlight * 1.2, 1);
  output.data = vec4<f32>(normalize(explosion.normal), explosion.depth);
  return output;
}
`;

class Explosions {
  constructor({ camera, device, geometry, instances, samples, sunlight }) {
    this.geometry = geometry;
    this.instances = instances;
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        buffers: [
          {
            arrayStride: 6 * Float32Array.BYTES_PER_ELEMENT,
            stepMode: 'vertex',
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: 'float32x3',
              },
              {
                shaderLocation: 1,
                offset: 3 * Float32Array.BYTES_PER_ELEMENT,
                format: 'float32x3',
              },
            ],
          },
          {
            arrayStride: 4 * Float32Array.BYTES_PER_ELEMENT,
            stepMode: 'instance',
            attributes: [
              {
                shaderLocation: 2,
                offset: 0,
                format: 'float32x3',
              },
              {
                shaderLocation: 3,
                offset: 3 * Float32Array.BYTES_PER_ELEMENT,
                format: 'float32',
              },
            ],
          }
        ],
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Vertex,
        }),
      },
      fragment: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Fragment,
        }),
        targets: [
          { format: 'rgba16float' },
          { format: 'rgba16float' },
        ],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
      },
      multisample: {
        count: samples,
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
          resource: { buffer: sunlight.buffer },
        },
      ],
    });
  }

  render(pass) {
    const { bindings, instances, geometry, pipeline } = this;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setIndexBuffer(geometry.index, 'uint16');
    pass.setVertexBuffer(0, geometry.vertex);
    pass.setVertexBuffer(1, instances, 20);
    pass.drawIndexedIndirect(instances, 0);
  }
}

export default Explosions;
