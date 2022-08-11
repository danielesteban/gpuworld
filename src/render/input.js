import { vec2, vec3 } from 'gl-matrix';

const _direction = vec3.create();
const _look = vec2.create();
const _movement = vec3.create();

class Input {
  constructor({ position, target }) {
    this.target = target;
    this.gamepad = null;
    this.keyboard = vec3.create();
    this.pointer = {
      movement: vec2.create(),
      position: vec2.create(),
    };
    this.look = {
      state: vec2.fromValues(Math.PI * 0.5, 0),
      target: vec2.fromValues(Math.PI * 0.5, 0),
    };
    this.position = {
      state: position,
      target: vec3.clone(position),
    };
    this.speed = {
      state: 8,
      target: 8,
    };
    this.vectors = {
      forward: vec3.create(),
      right: vec3.create(),
      worldUp: vec3.fromValues(0, 1, 0),
    };
    target.addEventListener('contextmenu', this.onContextMenu.bind(this), false);
    window.addEventListener('gamepaddisconnected', this.onGamepadDisconnected.bind(this), false);
    window.addEventListener('gamepadconnected', this.onGamepadConnected.bind(this), false);
    window.addEventListener('keydown', this.onKeyDown.bind(this), false);
    window.addEventListener('keyup', this.onKeyUp.bind(this), false);
    target.addEventListener('mousedown', this.onMouseDown.bind(this), false);
    window.addEventListener('mousemove', this.onMouseMove.bind(this), false);
    window.addEventListener('mouseup', this.onMouseUp.bind(this), false);
    window.addEventListener('wheel', this.onMouseWheel.bind(this), { passive: false });
    document.addEventListener('pointerlockchange', this.onPointerLock.bind(this), false);
  }

  lock() {
    const { isLocked, target } = this;
    if (!isLocked) {
      target.requestPointerLock();
    }
  }

  unlock() {
    const { isLocked } = this;
    if (isLocked) {
      document.exitPointerLock();
    }
  }

  onContextMenu(e) {
    e.preventDefault();
  }

  onGamepadDisconnected({ gamepad: { index } }) {
    const { gamepad } = this;
    if (gamepad === index) {
      this.gamepad = null;
    }
  }

  onGamepadConnected({ gamepad: { index } }) {
    this.gamepad = index;
  }

  onKeyDown({ key, repeat, target }) {
    const { isLocked, keyboard } = this;
    if (!isLocked || repeat || target.tagName === 'INPUT') {
      return;
    }
    switch (key.toLowerCase()) {
      case 'w':
        keyboard[2] = 1;
        break;
      case 's':
        keyboard[2] = -1;
        break;
      case 'a':
        keyboard[0] = -1;
        break;
      case 'd':
        keyboard[0] = 1;
        break;
      case ' ':
        keyboard[1] = 1;
        break;
      case 'shift':
        keyboard[1] = -1;
        break;
      default:
        break;
    }
  }

  onKeyUp({ key }) {
    const { isLocked, keyboard } = this;
    if (!isLocked) {
      return;
    }
    switch (key.toLowerCase()) {
      case 'w':
        if (keyboard[2] > 0) keyboard[2] = 0;
        break;
      case 's':
        if (keyboard[2] < 0) keyboard[2] = 0;
        break;
      case 'a':
        if (keyboard[0] < 0) keyboard[0] = 0;
        break;
      case 'd':
        if (keyboard[0] > 0) keyboard[0] = 0;
        break;
      case ' ':
        if (keyboard[1] > 0) keyboard[1] = 0;
        break;
      case 'shift':
        if (keyboard[1] < 0) keyboard[1] = 0;
        break;
      default:
        break;
    }
  }

  onMouseDown({ button }) {
    const { isLocked, pointer } = this;
    if (!isLocked) {
      this.lock();
      return;
    }
    pointer.isDown = button === 0;
  }

  onMouseMove({ clientX, clientY, movementX, movementY }) {
    const { sensitivity } = Input;
    const { isLocked, pointer: { movement, position } } = this;
    if (!isLocked) {
      return;
    }
    movement[0] -= movementX * sensitivity.pointer;
    movement[1] += movementY * sensitivity.pointer;
    vec2.set(
      position,
      (clientX / window.innerWidth) * 2 - 1,
      -(clientY / window.innerHeight) * 2 + 1
    );
  }

  onMouseUp({ button }) {
    const { isLocked, pointer } = this;
    if (isLocked && button === 0) {
      pointer.isDown = false;
    }
  }

  onMouseWheel(e) {
    const { sensitivity, minSpeed, speedRange } = Input;
    const { isLocked, speed } = this;
    if (e.ctrlKey) {
      e.preventDefault();
    }
    if (!isLocked) {
      return;
    }
    const logSpeed = Math.min(
      Math.max(
        ((Math.log(speed.target) - minSpeed) / speedRange) - (e.deltaY * sensitivity.wheel),
        0
      ),
      1
    );
    speed.target = Math.exp(minSpeed + logSpeed * speedRange);
  }

  onPointerLock() {
    const { keyboard, pointer } = this;
    this.isLocked = !!document.pointerLockElement;
    if (!this.isLocked) {
      vec3.set(keyboard, 0, 0, 0);
      pointer.isDown = false;
    }
  }

  update(delta) {
    const { minPhi, maxPhi, sensitivity } = Input;
    const { isLocked, gamepad, keyboard, pointer, look, position, speed, vectors } = this;

    if (isLocked) {
      vec3.copy(_movement, keyboard);
      vec2.copy(_look, pointer.movement);
      if (gamepad !== null) {
        const { axes } = navigator.getGamepads()[gamepad];
        if (Math.max(Math.abs(axes[2]), Math.abs(axes[3])) > 0.1) {
          vec2.set(_look, -axes[2] * sensitivity.gamepad, -axes[3] * sensitivity.gamepad);
        }
        if (Math.max(Math.abs(axes[0]), Math.abs(axes[1])) > 0.1) {
          vec3.set(_movement, axes[0], 0, -axes[1]);
        }
      }
      look.target[0] = Math.min(Math.max(look.target[0] + _look[1], minPhi), maxPhi);
      look.target[1] += _look[0];
    }
    vec2.set(pointer.movement, 0, 0);

    const damp = 1 - Math.exp(-20 * delta);
    vec2.lerp(look.state, look.state, look.target, damp);
    speed.state = speed.state * (1 - damp) + speed.target * damp;

    vec3.set(
      vectors.forward,
      Math.sin(look.state[0]) * Math.sin(look.state[1]),
      Math.cos(look.state[0]),
      Math.sin(look.state[0]) * Math.cos(look.state[1])
    );
    vec3.normalize(vectors.right, vec3.cross(vectors.right, vectors.forward, vectors.worldUp));

    if (_movement[0] !== 0 || _movement[1] !== 0 || _movement[2] !== 0) {
      vec3.set(_direction, 0, 0, 0);
      vec3.scaleAndAdd(_direction, _direction, vectors.right, _movement[0]);
      vec3.scaleAndAdd(_direction, _direction, vectors.worldUp, _movement[1]);
      vec3.scaleAndAdd(_direction, _direction, vectors.forward, _movement[2]);
      vec3.set(_movement, 0, 0, 0);
      const length = vec3.length(_direction);
      if (length > 1) vec3.scale(_direction, _direction, 1 / length);
      vec3.scaleAndAdd(
        position.target,
        position.target,
        _direction,
        delta * speed.state
      );
    }
    vec3.lerp(position.state, position.state, position.target, 1 - Math.exp(-10 * delta));
  }
}

Input.sensitivity = {
  gamepad: 0.03,
  pointer: 0.003,
  wheel: 0.0003,
};
Input.minPhi = 0.01;
Input.maxPhi = Math.PI - 0.01;
Input.minSpeed = Math.log(4);
Input.maxSpeed = Math.log(32);
Input.speedRange = Input.maxSpeed - Input.minSpeed;

export default Input;
