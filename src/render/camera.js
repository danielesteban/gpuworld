import { glMatrix, mat4, vec3 } from 'gl-matrix';

const _worldUp = vec3.fromValues(0, 1, 0);

class Camera {
  constructor({ device, aspect = 1, fov = 75, near = 0.1, far = 1000 }) {
    this.device = device;
    this.buffer = device.createBuffer({
      size: 16 * 2 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    });
    this.aspect = aspect;
    this.fov = fov;
    this.near = near;
    this.far = far;

    this.projectionMatrix = mat4.create();
    this.viewMatrix = mat4.create();
    this.position = vec3.create();
    this.target = vec3.create();
  }

  updateProjection() {
    const { device, buffer, projectionMatrix, aspect, fov, near, far } = this;
    mat4.perspective(projectionMatrix, glMatrix.toRadian(fov), aspect, near, far);
    device.queue.writeBuffer(buffer, 0, projectionMatrix);
  }

  updateView() {
    const { device, buffer, viewMatrix, position, target } = this;
    mat4.lookAt(viewMatrix, position, target, _worldUp);
    device.queue.writeBuffer(buffer, 64, viewMatrix);
  }
}

export default Camera;
