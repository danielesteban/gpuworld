import Atlas from './atlas.js';

const Vertex = `
struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
  @location(2) origin : vec3<f32>,
  @location(3) light : f32,
  @location(4) face : f32,
  @location(5) texture : f32,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) normal : vec3<f32>,
  @location(1) uv : vec2<f32>,
  @location(2) depth : f32,
  @location(3) @interpolate(flat) light : vec3<f32>,
  @location(4) @interpolate(flat) texture : i32,
}

struct Camera {
  projection : mat4x4<f32>,
  view : mat4x4<f32>,
}

const faceNormal : vec3<f32> = vec3<f32>(0, 0, 1);
const PI : f32 = 3.141592653589793;

fn rotateX(rad : f32) -> mat3x3<f32> {
  let c : f32 = cos(rad);
  let s : f32 = sin(rad);
  return mat3x3<f32>(
    1, 0, 0,
    0, c, s,
    0, -s, c,
  );
}

fn rotateY(rad : f32) -> mat3x3<f32> {
  let c : f32 = cos(rad);
  let s : f32 = sin(rad);
  return mat3x3<f32>(
    c, 0, -s,
    0, 1, 0,
    s, 0, c,
  );
}

fn getRotation(face : i32) -> mat3x3<f32> {
  switch face {
    default {
      return mat3x3<f32>(
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
      );
    }
    case 1 {
      return rotateX(PI * -0.5);
    }
    case 2 {
      return rotateX(PI * 0.5);
    }
    case 3 {
      return rotateY(PI * -0.5);
    }
    case 4 {
      return rotateY(PI * 0.5);
    }
    case 5 {
      return rotateY(PI);
    }
  }
}

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<uniform> sunlight : vec3<f32>;

@vertex
fn main(voxel : VertexInput) -> VertexOutput {
  let rotation : mat3x3<f32> = getRotation(i32(voxel.face));
  let position : vec3<f32> = rotation * voxel.position + voxel.origin;
  let mvPosition : vec4<f32> = camera.view * vec4<f32>(position, 1);
  var out : VertexOutput;
  out.position = camera.projection * mvPosition;
  out.normal = normalize(rotation * faceNormal);
  out.uv = voxel.uv;
  out.depth = -mvPosition.z;
  out.light = sunlight * pow(voxel.light, 16);
  out.texture = i32(voxel.texture);
  return out;
}
`;

const Fragment = `
struct FragmentInput {
  @location(0) normal : vec3<f32>,
  @location(1) uv : vec2<f32>,
  @location(2) depth : f32,
  @location(3) @interpolate(flat) light : vec3<f32>,
  @location(4) @interpolate(flat) texture : i32,
}

struct FragmentOutput {
  @location(0) color : vec4<f32>,
  @location(1) data : vec4<f32>,
}

@group(0) @binding(2) var atlas : texture_2d_array<f32>;
@group(0) @binding(3) var atlasSampler : sampler;

@fragment
fn main(face : FragmentInput) -> FragmentOutput {
  var output : FragmentOutput;
  output.color = textureSample(atlas, atlasSampler, face.uv, face.texture);
  output.color *= vec4<f32>(face.light, 1);
  output.data = vec4<f32>(normalize(face.normal), face.depth);
  return output;
}
`;

const Face = (device) => {
  const buffer = device.createBuffer({
    mappedAtCreation: true,
    size: 30 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.VERTEX,
  });
  new Float32Array(buffer.getMappedRange()).set([
    -0.5, -0.5,  0.5,       0, 1,
     0.5, -0.5,  0.5,       1, 1,
     0.5,  0.5,  0.5,       1, 0,
     0.5,  0.5,  0.5,       1, 0,
    -0.5,  0.5,  0.5,       0, 0,
    -0.5, -0.5,  0.5,       0, 1,
  ]);
  buffer.unmap();
  return buffer;
};

class Voxels {
  constructor({ camera, chunks, device, samples, sunlight }) {
    this.atlas = new Atlas({ device });
    this.chunks = chunks;
    this.device = device;
    this.geometry = Face(device);
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        buffers: [
          {
            arrayStride: 5 * Float32Array.BYTES_PER_ELEMENT,
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: 'float32x3',
              },
              {
                shaderLocation: 1,
                offset: 3 * Float32Array.BYTES_PER_ELEMENT,
                format: 'float32x2',
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
                format: 'float32',
              },
              {
                shaderLocation: 4,
                offset: 4 * Float32Array.BYTES_PER_ELEMENT,
                format: 'float32',
              },
              {
                shaderLocation: 5,
                offset: 5 * Float32Array.BYTES_PER_ELEMENT,
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
        {
          binding: 2,
          resource: this.atlas.texture.createView(),
        },
        {
          binding: 3,
          resource: device.createSampler(),
        },
      ],
    });
  }

  render(pass) {
    const { bindings, chunks, geometry, pipeline } = this;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setVertexBuffer(0, geometry);
    chunks.forEach(({ faces }) => {
      pass.setVertexBuffer(1, faces, 40);
      pass.drawIndirect(faces, 0);
    });
  }
}

export default Voxels;
