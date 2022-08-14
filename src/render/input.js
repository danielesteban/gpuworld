import { vec2, vec3 } from 'gl-matrix';

const _direction = vec3.create();
const _look = vec2.create();
const _movement = vec3.create();
const _target = vec3.create();

const _forward = vec3.create();
const _right = vec3.create();
const _worldUp = vec3.fromValues(0, 1, 0);

class Input {
  constructor({ camera, target, world }) {
    this.camera = camera;
    this.target = target;
    this.world = world;
    this.isLocked = false;
    this.gamepad = null;
    this.keyboard = {
      buttons: { run: false },
      movement: vec3.create(),
    }
    this.pointer = {
      buttons: { primary: false },
      movement: vec2.create(),
      position: vec2.create(),
    };
    this.buttons = {
      primary: false,
    };
    this.forward = vec3.create();
    this.look = {
      state: vec2.fromValues(Math.PI * 0.5, Math.PI * -0.5),
      target: vec2.fromValues(Math.PI * 0.5, Math.PI * -0.5),
    };
    this.position = {
      target: vec3.clone(camera.position),
      targetY: camera.position[1],
      nextTarget: vec3.clone(camera.position),
    };
    this.speed = {
      state: 8,
      target: 8,
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
        keyboard.movement[2] = 1;
        break;
      case 's':
        keyboard.movement[2] = -1;
        break;
      case 'a':
        keyboard.movement[0] = -1;
        break;
      case 'd':
        keyboard.movement[0] = 1;
        break;
      case ' ':
        keyboard.movement[1] = 1;
        break;
      case 'shift':
        keyboard.buttons.run = true;
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
        if (keyboard.movement[2] > 0) keyboard.movement[2] = 0;
        break;
      case 's':
        if (keyboard.movement[2] < 0) keyboard.movement[2] = 0;
        break;
      case 'a':
        if (keyboard.movement[0] < 0) keyboard.movement[0] = 0;
        break;
      case 'd':
        if (keyboard.movement[0] > 0) keyboard.movement[0] = 0;
        break;
      case ' ':
        if (keyboard.movement[1] > 0) keyboard.movement[1] = 0;
        break;
      case 'shift':
        keyboard.buttons.run = false;
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
    pointer.buttons.primary = (button === 0 || button === 2);
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
    if (isLocked && (button === 0 || button === 2)) {
      pointer.buttons.primary = false;
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
    const { buttons, keyboard, pointer } = this;
    this.isLocked = !!document.pointerLockElement;
    document.body.classList[this.isLocked ? 'add' : 'remove']('pointerlock');
    if (!this.isLocked) {
      buttons.primary = false;
      vec3.set(keyboard.movement, 0, 0, 0);
      keyboard.buttons.run = false;
      pointer.buttons.primary = false;
    }
  }

  update(delta) {
    const { minPhi, maxPhi, sensitivity } = Input;
    const { isLocked, buttons, camera, forward, gamepad, keyboard, pointer, look, position, speed } = this;

    let isRunning = false;
    if (isLocked) {
      isRunning = keyboard.buttons.run;
      buttons.primary = pointer.buttons.primary;
      vec3.copy(_movement, keyboard.movement);
      vec2.copy(_look, pointer.movement);
      if (gamepad !== null) {
        const { axes, buttons: gamepadButtons } = navigator.getGamepads()[gamepad];
        if (
          (gamepadButtons[6] && gamepadButtons[6].pressed)
          || (gamepadButtons[7] && gamepadButtons[7].pressed)
        ) {
          buttons.primary = true;
        }
        if (Math.max(Math.abs(axes[2]), Math.abs(axes[3])) > 0.1) {
          vec2.set(_look, -axes[2] * sensitivity.gamepad, -axes[3] * sensitivity.gamepad);
        }
        if (Math.max(Math.abs(axes[0]), Math.abs(axes[1])) > 0.1) {
          vec3.set(_movement, axes[0], 0, -axes[1]);
        }
        if (
          (gamepadButtons[0] && gamepadButtons[0].pressed)
          || (gamepadButtons[4] && gamepadButtons[4].pressed)
          || (gamepadButtons[5] && gamepadButtons[5].pressed)
        ) {
          _movement[1] = 1;
        }
        if (gamepadButtons[10] && gamepadButtons[10].pressed) {
          isRunning = true;
        }
      }
      look.target[0] = Math.min(Math.max(look.target[0] + _look[1], minPhi), maxPhi);
      look.target[1] += _look[0];
    }
    vec2.set(pointer.movement, 0, 0);

    {
      const damp = 1 - Math.exp(-20 * delta);
      vec2.lerp(look.state, look.state, look.target, damp);
      speed.state = speed.state * (1 - damp) + speed.target * damp;
    }

    vec3.set(
      forward,
      Math.sin(look.state[0]) * Math.sin(look.state[1]),
      Math.cos(look.state[0]),
      Math.sin(look.state[0]) * Math.cos(look.state[1])
    );

    if (_movement[0] !== 0 || _movement[1] !== 0 || _movement[2] !== 0) {
      const isFlying = _movement[1] > 0;
      vec3.copy(_forward, forward);
      if (!isFlying) {
        _forward[1] = 0;
        vec3.normalize(_forward, _forward);
      }
      vec3.normalize(_right, vec3.cross(_right, _forward, _worldUp));
      vec3.set(_direction, 0, 0, 0);
      vec3.scaleAndAdd(_direction, _direction, _right, _movement[0]);
      vec3.scaleAndAdd(_direction, _direction, _worldUp, _movement[1]);
      vec3.scaleAndAdd(_direction, _direction, _forward, _movement[2]);
      vec3.set(_movement, 0, 0, 0);
      const length = vec3.length(_direction);
      if (length > 1) vec3.scale(_direction, _direction, 1 / length);
      position.nextTarget[1] = position.target[1];
      vec3.scaleAndAdd(
        position.nextTarget,
        position.nextTarget,
        _direction,
        delta * speed.state * (isRunning ? 2 : 1)
      );
      if (isFlying) {
        vec3.copy(position.target, position.nextTarget);
        position.targetY = position.target[1];
      } else {
        this.updatePositionTarget();
      }
    }

    {
      const damp = 1 - Math.exp(-10 * delta);
      position.target[1] = position.target[1] * (1 - damp) + position.targetY * damp;
      vec3.lerp(camera.position, camera.position, position.target, damp);
      vec3.add(camera.target, camera.position, forward);
      camera.updateView();
    }
  }

  updatePositionTarget() {
    const { position, world } = this;
    if (position.isUpdating) {
      position.needsUpdate = true;
      return;
    }
    position.isUpdating = true;
    vec3.copy(_target, position.nextTarget);
    world
      .getGround(_target)
      .then((height) => {
        position.isUpdating = false;
        if (height === -1) {
          vec3.copy(position.nextTarget, position.target);
          position.needsUpdate = false;
          return;
        }
        position.target[0] = _target[0];
        position.target[2] = _target[2];
        position.targetY = height + 3.2;
        if (position.needsUpdate) {
          position.needsUpdate = false;
          this.updatePositionTarget();
        }
      });
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
