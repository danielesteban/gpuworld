const Vertex = `
struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) origin : vec3<f32>,
  @location(3) direction : vec3<f32>,
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

const worldUp : vec3<f32> = vec3<f32>(0, 1, 0);

fn getRotation(direction : vec3<f32>) -> mat3x3<f32> {
  var xaxis : vec3<f32> = normalize(cross(worldUp, direction));
  var yaxis : vec3<f32> = normalize(cross(direction, xaxis));
  return mat3x3<f32>(xaxis, yaxis, direction);
}

@group(0) @binding(0) var<uniform> camera : Camera;

@vertex
fn main(projectile : VertexInput) -> VertexOutput {
  var rotation : mat3x3<f32> = getRotation(projectile.direction);
  var position : vec3<f32> = rotation * projectile.position + projectile.origin;
  let mvPosition : vec4<f32> = camera.view * vec4<f32>(position, 1);
  var out : VertexOutput;
  out.position = camera.projection * mvPosition;
  out.normal = normalize(rotation * projectile.normal);
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

@fragment
fn main(projectile : FragmentInput) -> FragmentOutput {
  var output : FragmentOutput;
  output.color = vec4<f32>(1);
  output.data = vec4<f32>(normalize(projectile.normal), projectile.depth);
  return output;
}
`;

const Cube = (device) => {
  const buffer = device.createBuffer({
    size: 216 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set([
    -0.1, -0.1,  0.1,        0,  0,  1,
     0.1, -0.1,  0.1,        0,  0,  1,
     0.1,  0.1,  0.1,        0,  0,  1,
     0.1,  0.1,  0.1,        0,  0,  1,
    -0.1,  0.1,  0.1,        0,  0,  1,
    -0.1, -0.1,  0.1,        0,  0,  1,

     0.1, -0.1, -0.1,        0,  0,  -1,
    -0.1, -0.1, -0.1,        0,  0,  -1,
    -0.1,  0.1, -0.1,        0,  0,  -1,
    -0.1,  0.1, -0.1,        0,  0,  -1,
     0.1,  0.1, -0.1,        0,  0,  -1,
     0.1, -0.1, -0.1,        0,  0,  -1,

    -0.1,  0.1, -0.1,        0,  1,  0,
     0.1,  0.1, -0.1,        0,  1,  0,
     0.1,  0.1,  0.1,        0,  1,  0,
     0.1,  0.1,  0.1,        0,  1,  0,
    -0.1,  0.1,  0.1,        0,  1,  0,
    -0.1,  0.1, -0.1,        0,  1,  0,

    -0.1, -0.1,  0.1,        0, -1,  0,
     0.1, -0.1,  0.1,        0, -1,  0,
     0.1, -0.1, -0.1,        0, -1,  0,
     0.1, -0.1, -0.1,        0, -1,  0,
    -0.1, -0.1, -0.1,        0, -1,  0,
    -0.1, -0.1,  0.1,        0, -1,  0,

     0.1, -0.1, -0.1,        1,  0,  0,
     0.1, -0.1,  0.1,        1,  0,  0,
     0.1,  0.1,  0.1,        1,  0,  0,
     0.1,  0.1,  0.1,        1,  0,  0,
     0.1,  0.1, -0.1,        1,  0,  0,
     0.1, -0.1, -0.1,        1,  0,  0,

    -0.1, -0.1,  0.1,       -1,  0,  0,
    -0.1, -0.1, -0.1,       -1,  0,  0,
    -0.1,  0.1, -0.1,       -1,  0,  0,
    -0.1,  0.1, -0.1,       -1,  0,  0,
    -0.1,  0.1,  0.1,       -1,  0,  0,
    -0.1, -0.1,  0.1,       -1,  0,  0,
  ]);
  buffer.unmap();
  return buffer;
};

class Projectiles {
  constructor({ camera, device, instances, samples }) {
    this.geometry = Cube(device);
    this.instances = instances;
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        buffers: [
          {
            arrayStride: 6 * Float32Array.BYTES_PER_ELEMENT,
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
            arrayStride: 6 * Float32Array.BYTES_PER_ELEMENT,
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
                format: 'float32x3',
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
          { format: 'rgba8unorm' },
          { format: 'rgba16float' },
        ],
      },
      primitive: {
        topology: 'triangle-list',
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
      ],
    });
  }

  render(pass) {
    const { bindings, instances, geometry, pipeline } = this;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setVertexBuffer(0, geometry);
    pass.setVertexBuffer(1, instances, 16);
    pass.drawIndirect(instances, 0);
  }
}

export default Projectiles;
