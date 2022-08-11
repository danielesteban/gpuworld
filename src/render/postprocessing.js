const Vertex = `
@vertex
fn main(@location(0) position : vec4<f32>) -> @builtin(position) vec4<f32> {
  return position;
}
`;

const Fragment = `
@group(0) @binding(0) var colorTexture : texture_2d<f32>;
@group(0) @binding(1) var dataTexture : texture_2d<f32>;

const edgeColor : vec3<f32> = vec3<f32>(0, 0, 0);
const edgeIntensity : f32 = 0.1;
const depthScale : f32 = 0.5;
const normalScale : f32 = 0.5;
const offset : vec3<i32> = vec3<i32>(1, 1, 0);

fn edge(pixel : vec2<i32>) -> f32 {
  let pixelCenter : vec4<f32> = textureLoad(dataTexture, pixel, 0);
  let pixelLeft : vec4<f32> = textureLoad(dataTexture, pixel - offset.xz, 0);
  let pixelRight : vec4<f32> = textureLoad(dataTexture, pixel + offset.xz, 0);
  let pixelUp : vec4<f32> = textureLoad(dataTexture, pixel + offset.zy, 0);
  let pixelDown : vec4<f32> = textureLoad(dataTexture, pixel - offset.zy, 0);
  let edge : vec4<f32> = (
    abs(pixelLeft    - pixelCenter)
    + abs(pixelRight - pixelCenter) 
    + abs(pixelUp    - pixelCenter) 
    + abs(pixelDown  - pixelCenter)
  );
  return clamp(max((edge.x + edge.y + edge.z) * normalScale, edge.w * depthScale), 0, 1);
}

@fragment
fn main(@builtin(position) uv : vec4<f32>) -> @location(0) vec4<f32> {
  let pixel : vec2<i32> = vec2<i32>(floor(uv.xy));
  var color : vec3<f32> = textureLoad(colorTexture, pixel, 0).xyz;
  color = mix(color, edgeColor, edge(pixel) * edgeIntensity);
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

  bindTextures([color, data]) {
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
          resource: data.resolveTarget,
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
