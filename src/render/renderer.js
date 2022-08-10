import Postprocessing from './postprocessing.js';
import Voxels from './voxels.js';

class Renderer {
  constructor({
    adapter,
    camera,
    device,
    samples = 4,
  }) {
    const format = navigator.gpu.getPreferredCanvasFormat(adapter);
    this.canvas = document.createElement('canvas');
    // I have no idea why but if I don't do this, sometimes it crashes with:
    // D3D12 reset command allocator failed with E_FAIL
    this.canvas.width = Math.floor(window.innerWidth * (window.devicePixelRatio || 1));
    this.canvas.height = Math.floor(window.innerHeight * (window.devicePixelRatio || 1));
    this.context = this.canvas.getContext('webgpu');
    this.context.configure({ alphaMode: 'opaque', device, format });
    this.camera = camera;
    this.device = device;
    this.samples = samples;
    this.textures = new Map();
    this.descriptor = {
      colorAttachments: [
        {
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
        {
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
        {
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
      depthStencilAttachment: {
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    };
    this.postprocessing = new Postprocessing({ device, format });
    this.voxels = new Voxels({ camera, device, samples });
  }

  render(command, chunks) {
    const {
      context,
      descriptor,
      postprocessing,
      voxels,
    } = this;
    const pass = command.beginRenderPass(descriptor);
    voxels.render(pass, chunks);
    pass.end();
    postprocessing.render(command, context.getCurrentTexture().createView());
  }

  setClearColor(r, g, b) {
    const { descriptor: { colorAttachments: [{ clearValue }] } } = this;
    clearValue.r = r;
    clearValue.g = g;
    clearValue.b = b;
  }

  setSize(width, height) {
    const {
      camera,
      canvas,
      descriptor,
      postprocessing,
    } = this;
    const pixelRatio = window.devicePixelRatio || 1;
    const size = [Math.floor(width * pixelRatio), Math.floor(height * pixelRatio)];
    canvas.width = size[0];
    canvas.height = size[1];
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    camera.aspect = width / height;
    camera.updateProjection();

    this.updateTexture(descriptor.colorAttachments[0], 'rgba8unorm', 'color', size);
    this.updateTexture(descriptor.colorAttachments[1], 'rgba16float', 'normal', size);
    this.updateTexture(descriptor.colorAttachments[2], 'rgba16float', 'position', size);
    this.updateTexture(descriptor.depthStencilAttachment, 'depth24plus', 'depth', size, false);
    postprocessing.bindTextures(descriptor.colorAttachments);
  }

  updateTexture(object, format, key, size, resolve = true) {
    const { device, samples, textures } = this;
    const current = textures.get(key);
    if (current) {
      current.forEach((texture) => texture.destroy());
    }
    textures.set(key, [samples, ...(resolve ? [1] : [])].map((sampleCount) => {
      const texture = device.createTexture({
        format,
        sampleCount,
        size,
        usage: (
          GPUTextureUsage.RENDER_ATTACHMENT
          | (sampleCount === 1 ? GPUTextureUsage.TEXTURE_BINDING : 0)
        ),
      });
      if (sampleCount === 1) {
        object.resolveTarget = texture.createView();
      } else {
        object.view = texture.createView();
      }
      return texture;
    }));
  }
}

export default Renderer;
