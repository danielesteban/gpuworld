import Noise from '../compute/worldgen/noise.js';

const Compute = ({ count, width, height, generator }) => `
struct Atlas {
  count : i32,
  width : i32,
  height : i32,
  stride : i32,
  length : i32,
}

const atlas : Atlas = Atlas(
  ${count},
  ${width},
  ${height},
  ${width * height},
  ${count * width * height},
);

@group(0) @binding(0) var texture : texture_storage_2d_array<rgba8unorm, write>;

${Noise}

${generator}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let id : i32 = i32(GlobalInvocationID.x);
  if (id >= atlas.length) {
    return;
  }
  let tex : i32 = id / atlas.stride;
  let index : i32 = id - tex * atlas.stride;
  let y : i32 = index / atlas.width;
  let pixel : vec2<i32> = vec2<i32>(index - y * atlas.width, y);
  textureStore(texture, pixel, tex, getColorAt(tex, pixel));
}
`;

const DefaultGenerator = `
fn getColorAt(texture : i32, pixel : vec2<i32>) -> vec4<f32> {
  if (texture == 0 || (texture == 2 && pixel.y > 4 - i32(4 * noise3(vec3<f32>(vec2<f32>(pixel), 0))))) {
    return vec4<f32>(1);
  }
  return vec4<f32>(0.6, 1.0, 0.6, 1.0);
}
`;

class Atlas {
  constructor({ device, count = 3, width = 16, height = 16 }) {
    this.device = device;
    this.count = count;
    this.width = width;
    this.height = height;
    this.texture = device.createTexture({
      dimension: '2d',
      size: [width, height, count],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.workgroups = Math.ceil((count * width * height) / 256);
  }

  compute(generator = DefaultGenerator) {
    const { device, count, width, height, texture, workgroups } = this;
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Compute({ count, width, height, generator }),
        }),
      },
    });
    const command = device.createCommandEncoder();
    const pass = command.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{
        binding: 0,
        resource: texture.createView(),
      }],
    }));
    pass.dispatchWorkgroups(workgroups);
    pass.end();
    device.queue.submit([command.finish()]);
  }
}

export default Atlas;
