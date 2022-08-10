const Vertex = `
@vertex
fn main(@location(0) position : vec4<f32>) -> @builtin(position) vec4<f32> {
  return position;
}
`;

const Fragment = `
struct Camera {
  projection : mat4x4<f32>,
  view : mat4x4<f32>,
  position : vec3<f32>,
  direction : vec3<f32>,
}

struct Effect {
  color : vec3<f32>,
  intensity : f32,
  depthScale : f32,
  normalScale : f32,
}

const effect : Effect = Effect(
  vec3<f32>(0, 0, 0),
  0.1,
  0.5,
  0.5,
);

const offset : vec3<i32> = vec3<i32>(1, 1, 0);

fn edgesDepth(pixel : vec2<i32>) -> f32 {
  var pixelCenter : f32 = textureLoad(positionTexture, pixel, 0).w;
  var pixelLeft : f32 = textureLoad(positionTexture, pixel - offset.xz, 0).w;
  var pixelRight : f32 = textureLoad(positionTexture, pixel + offset.xz, 0).w;
  var pixelUp : f32 = textureLoad(positionTexture, pixel + offset.zy, 0).w;
  var pixelDown : f32 = textureLoad(positionTexture, pixel - offset.zy, 0).w;
  return (
    abs(pixelLeft    - pixelCenter) 
    + abs(pixelRight - pixelCenter) 
    + abs(pixelUp    - pixelCenter) 
    + abs(pixelDown  - pixelCenter) 
  ) * effect.depthScale;
}

fn edgesNormal(pixel : vec2<i32>) -> f32 {
  var pixelCenter : vec3<f32> = textureLoad(normalTexture, pixel, 0).xyz;
  var pixelLeft : vec3<f32> = textureLoad(normalTexture, pixel - offset.xz, 0).xyz;
  var pixelRight : vec3<f32> = textureLoad(normalTexture, pixel + offset.xz, 0).xyz;
  var pixelUp : vec3<f32> = textureLoad(normalTexture, pixel + offset.zy, 0).xyz;
  var pixelDown : vec3<f32> = textureLoad(normalTexture, pixel - offset.zy, 0).xyz;
  var edge : vec3<f32> = (
    abs(pixelLeft    - pixelCenter)
    + abs(pixelRight - pixelCenter) 
    + abs(pixelUp    - pixelCenter) 
    + abs(pixelDown  - pixelCenter)
  );
  return (edge.x + edge.y + edge.z) * effect.normalScale;
}

@group(0) @binding(0) var colorTexture : texture_2d<f32>;
@group(0) @binding(1) var normalTexture : texture_2d<f32>;
@group(0) @binding(2) var positionTexture : texture_2d<f32>;

@fragment
fn main(@builtin(position) uv : vec4<f32>) -> @location(0) vec4<f32> {
  var pixel : vec2<i32> = vec2<i32>(floor(uv.xy));
  var color : vec3<f32> = textureLoad(colorTexture, pixel, 0).xyz;
  color = mix(color, effect.color, clamp(max(edgesDepth(pixel), edgesNormal(pixel)), 0, 1) * effect.intensity);
  return vec4<f32>(color, 1);
}
`;

const Screen = (device) => {
  const buffer = device.createBuffer({
    size: 18 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set([
    -1, -1,  1,
     1, -1,  1,
     1,  1,  1,
     1,  1,  1,
    -1,  1,  1,
    -1, -1,  1,
  ]);
  buffer.unmap();
  return buffer;
};

class Postprocessing {
  constructor({ device, format }) {
    this.device = device;
    this.descriptor = {
      colorAttachments: [{
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    };
    this.geometry = Screen(device);
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        buffers: [
          {
            arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT,
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: 'float32x3',
              },
            ],
          },
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
        targets: [{ format }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });
  }

  bindTextures([color, normal, position]) {
    const { device, pipeline } = this;
    this.bindings = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: color.resolveTarget,
        },
        {
          binding: 1,
          resource: normal.resolveTarget,
        },
        {
          binding: 2,
          resource: position.resolveTarget,
        },
      ],
    });
  }

  render(command, view) {
    const { bindings, descriptor, geometry, pipeline } = this;
    descriptor.colorAttachments[0].view = view;
    const pass = command.beginRenderPass(descriptor);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setVertexBuffer(0, geometry);
    pass.draw(6, 1, 0, 0);
    pass.end();
  }
}

export default Postprocessing;
