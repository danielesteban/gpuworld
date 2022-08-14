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
const stone : vec3<f32> = vec3<f32>(0.6, 0.6, 0.8);
const dirt : vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);
const grass : vec4<f32> = vec4<f32>(0.6, 1.0, 0.6, 1.0);
fn getColorAt(texture : i32, pixel : vec2<i32>) -> vec4<f32> {
  switch texture {
    default {
      let n = (max(noise3(vec3<f32>(vec2<f32>(pixel) * vec2<f32>(0.75, 1), 0)) - 0.5, 0) - 0.5) * 0.1;
      return vec4<f32>(stone + n, 1.0);
    }
    case 1 {
      return dirt;
    }
    case 2 {
      let n = (max(noise3(vec3<f32>(vec2<f32>(pixel), 0)) - 0.5, 0) - 0.5) * 0.1;
      return vec4<f32>(grass.xyz + n, 1.0);
    }
    case 3 {
      if (pixel.y > 4 - i32(4 * noise3(vec3<f32>(vec2<f32>(pixel), 0)))) {
        return dirt;
      }
      return grass;
    }
  }
}
`;

class Atlas {
  constructor({ device, count = 4, width = 16, height = 16 }) {
    this.device = device;
    this.count = count;
    this.width = width;
    this.height = height;
    this.texture = device.createTexture({
      dimension: '2d',
      size: [width, height, count],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
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

  setupDragAndDrop() {
    window.addEventListener('dragenter', (e) => e.preventDefault(), false);
    window.addEventListener('dragover', (e) => e.preventDefault(), false);
    window.addEventListener('drop', (e) => {
      e.preventDefault();
      const [file] = e.dataTransfer.files;
      if (!file || file.type.indexOf('image/') !== 0) {
        return;
      }
      const image = new Image();
      image.addEventListener('load', () => {
        const { device, count, width, height, texture } = this;
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = width;
        canvas.height = height * count;
        ctx.drawImage(image, 0, 0);
        const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
        device.queue.writeTexture(
          { texture },
          pixels.data.buffer,
          { bytesPerRow: width * 4, rowsPerImage: height },
          [width, height, count]
        );
      }, false);
      image.src = URL.createObjectURL(file);
    }, false);
  }
}

export default Atlas;
