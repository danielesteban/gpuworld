import Chunk from './chunk.js';

const Compute = ({ chunkSize }) => `
${Chunk.compute({ chunkSize })}

struct Query {
  input : vec4<i32>,
  operation : i32,
  output : i32,
}

@group(0) @binding(0) var<storage, read_write> query : Query;
@group(1) @binding(0) var<storage, read> chunk : Chunk;

fn getGround() -> i32 {
  var pos : vec3<i32> = query.input.xyz;
  let height : i32 = query.input.w;
  if (
    chunk.voxels[getVoxel(pos)].value != 0
  ) {
    return -1;
  }
  pos.y--;
  for (; pos.y >= 0; pos.y--) {
    if (chunk.voxels[getVoxel(pos)].value == 0) {
      continue;
    }
    for (var h : i32 = 1; h <= height; h++) {
      let npos : vec3<i32> = pos + vec3<i32>(0, h, 0);
      if (npos.y >= chunkSize.y) {
        break;
      }
      if (chunk.voxels[getVoxel(npos)].value != 0) {
        return -1;
      }
    }
    return pos.y + 1;
  }
  return 0;
}

@compute @workgroup_size(1)
fn main() {
  switch query.operation {
    default {}
    case 1 {
      query.output = getGround();
    }
  }
}
`;

class Query {
  constructor({ chunkSize, device }) {
    this.device = device;
    this.outputBuffers = [];
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Compute({ chunkSize }),
        }),
      },
    });
    this.query = {
      buffer: device.createBuffer({
        size: 6 * Int32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
      }),
      input: new Int32Array(5),
    };
    this.bindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{
        binding: 0,
        resource: { buffer: this.query.buffer },
      }],
    });
  }

  compute(chunk) {
    const { bindings, device, outputBuffers, pipeline, query } = this;
    if (!chunk.bindings.query) {
      chunk.bindings.query = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [{
          binding: 0,
          resource: { buffer: chunk.data },
        }],
      });
    }
    const output = outputBuffers.pop() || device.createBuffer({
      size: Int32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const command = device.createCommandEncoder();
    const pass = command.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setBindGroup(1, chunk.bindings.query);
    pass.dispatchWorkgroups(1);
    pass.end();
    command.copyBufferToBuffer(query.buffer, 20, output, 0, 4);
    device.queue.submit([command.finish()]);
    return output
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const result = new Int32Array(output.getMappedRange())[0];
        output.unmap();
        outputBuffers.push(output);
        return result;
      });
  }

  getGround(chunk, voxel, height) {
    const { device, query } = this;
    query.input[0] = voxel[0];
    query.input[1] = voxel[1];
    query.input[2] = voxel[2];
    query.input[3] = height;
    query.input[4] = Query.operations.ground;
    device.queue.writeBuffer(query.buffer, 0, query.input);
    return this.compute(chunk);
  }
}

Query.operations = {
  ground: 1,
};

export default Query;
