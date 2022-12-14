import Chunk from '../chunk.js';

const Compute = ({ chunkSize }) => `
${Chunk.compute({ chunkSize })}

@group(0) @binding(0) var<uniform> frustum : array<vec4<f32>, 6>;
@group(0) @binding(1) var<storage, read_write> workgroups : array<u32, 3>;
@group(1) @binding(0) var<storage, read_write> chunk : Chunk;
@group(1) @binding(1) var<storage, read_write> faces : Faces;
@group(1) @binding(2) var<uniform> position : vec3<i32>;

fn isInFrustum() -> bool {
  var maxPos : vec3<f32>;
  var minPos : vec3<f32>;
  if (chunk.remesh == 0) {
    maxPos = vec3<f32>(f32(faces.bounds.max[0]), f32(faces.bounds.max[1]), f32(faces.bounds.max[2]));
    minPos = vec3<f32>(f32(faces.bounds.min[0]), f32(faces.bounds.min[1]), f32(faces.bounds.min[2]));
  } else {
    maxPos = vec3<f32>(chunkSize);
    minPos = vec3<f32>(0);
  }
  let origin : vec3<f32> = vec3<f32>(position);
  for (var i : i32 = 0; i < 6; i++) {
    let plane : vec4<f32> = frustum[i];
    var corner : vec3<f32>;
    for (var j : i32 = 0; j < 3; j++) {
      if (plane[j] > 0) {
        corner[j] = maxPos[j];
      } else {
        corner[j] = minPos[j];
      }
    }
    if ((dot(plane.xyz, origin + corner) + plane.w) < 0) {
      return false;
    }
  }
  return true;
}

@compute @workgroup_size(1)
fn main() {
  if (isInFrustum()) {
    faces.vertexCount = 6;
  } else {
    faces.vertexCount = 0;
  }
  if (chunk.remesh == 0) {
    workgroups[0] = 0;
    return;
  }
  chunk.remesh = 0;
  faces.bounds.max[0] = 0;
  faces.bounds.max[1] = 0;
  faces.bounds.max[2] = 0;
  faces.bounds.min[0] = u32(chunkSize.x);
  faces.bounds.min[1] = u32(chunkSize.y);
  faces.bounds.min[2] = u32(chunkSize.z);
  faces.instanceCount = 0;
  workgroups[0] = u32(ceil(f32(chunkSize.x) / 64));
  workgroups[1] = u32(ceil(f32(chunkSize.y) / 4));
  workgroups[2] = u32(chunkSize.z);
}
`;

class MesherSetup {
  constructor({ chunkSize, device, frustum, workgroups }) {
    this.device = device;
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Compute({ chunkSize }),
        }),
      },
    });
    this.bindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: frustum.buffer },
        },
        {
          binding: 1,
          resource: { buffer: workgroups },
        }
      ],
    });
  }

  compute(pass, chunk) {
    const { bindings, device, pipeline } = this;
    if (!chunk.bindings.mesherSetup) {
      chunk.bindings.mesherSetup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: { buffer: chunk.data },
          },
          {
            binding: 1,
            resource: { buffer: chunk.faces },
          },
          {
            binding: 2,
            resource: { buffer: chunk.offset },
          },
        ],
      });
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setBindGroup(1, chunk.bindings.mesherSetup);
    pass.dispatchWorkgroups(1);
  }
}

export default MesherSetup;
