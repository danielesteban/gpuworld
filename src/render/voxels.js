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
const sunlightColor : vec3<f32> = vec3<f32>(0.9, 0.8, 0.4);
const PI : f32 = 3.141592653589793;

fn rotateX(rad : f32) -> mat3x3<f32> {
  var c : f32 = cos(rad);
  var s : f32 = sin(rad);
  return mat3x3<f32>(
    1, 0, 0,
    0, c, s,
    0, -s, c,
  );
}

fn rotateY(rad : f32) -> mat3x3<f32> {
  var c : f32 = cos(rad);
  var s : f32 = sin(rad);
  return mat3x3<f32>(
    c, 0, -s,
    0, 1, 0,
    s, 0, c,
  );
}

@group(0) @binding(0) var<uniform> camera : Camera;

@vertex
fn main(voxel : VertexInput) -> VertexOutput {
  var rotation : mat3x3<f32>;
  switch (i32(voxel.face)) {
    default {
      rotation = mat3x3<f32>(
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
      );
    }
    case 1 {
      rotation = rotateX(PI * -0.5);
    }
    case 2 {
      rotation = rotateX(PI * 0.5);
    }
    case 3 {
      rotation = rotateY(PI * -0.5);
    }
    case 4 {
      rotation = rotateY(PI * 0.5);
    }
    case 5 {
      rotation = rotateY(PI);
    }
  }
  var mvPosition : vec4<f32> = camera.view * vec4<f32>(rotation * voxel.position + voxel.origin, 1);
  var out : VertexOutput;
  out.position = camera.projection * mvPosition;
  out.normal = normalize(rotation * faceNormal);
  out.uv = voxel.uv;
  out.depth = -mvPosition.z;
  out.light = sunlightColor * pow(voxel.light, 8);
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

@group(0) @binding(1) var atlas : texture_2d_array<f32>;
@group(0) @binding(2) var atlasSampler : sampler;

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
    size: 30 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set([
    -0.5, -0.5,  0.5,        0, 1,
     0.5, -0.5,  0.5,        1, 1,
     0.5,  0.5,  0.5,        1, 0,
     0.5,  0.5,  0.5,        1, 0,
    -0.5,  0.5,  0.5,        0, 0,
    -0.5, -0.5,  0.5,        0, 1,
  ]);
  buffer.unmap();
  return buffer;
};

class Voxels {
  constructor({ camera, device, samples }) {
    this.atlas = new Atlas({ device });
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
          { format: 'rgba8unorm' },
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
          resource: this.atlas.texture.createView(),
        },
        {
          binding: 2,
          resource: device.createSampler(),
        },
      ],
    });
  }

  render(pass, chunks) {
    const { bindings, geometry, pipeline } = this;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setVertexBuffer(0, geometry);
    chunks.forEach(({ faces }) => {
      pass.setVertexBuffer(1, faces, 16);
      pass.drawIndirect(faces, 0);
    });
  }
}

export default Voxels;
