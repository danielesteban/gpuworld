import { vec3 } from 'gl-matrix';

const _color = vec3.create();

class Light {
  constructor(renderer) {
    this.renderer = renderer;
    this.colors = {
      background: {
        day: vec3.fromValues(0.2, 0.6, 0.6),
        night: vec3.fromValues(0.05, 0.05, 0.1),
      },
      sunlight: {
        day: vec3.fromValues(1, 1, 0.6),
        night: vec3.fromValues(0.1, 0.1, 0.2),
      },
    };
    this.target = 1;
    this.state = 0;
    const toggle = document.getElementById('light');
    toggle.classList.add('enabled');
    const [day, night] = toggle.getElementsByTagName('svg');
    day.addEventListener('click', () => {
      toggle.classList.remove('night');
      toggle.classList.add('day');
      this.target = 1;
    }, false);
    night.addEventListener('click', () => {
      toggle.classList.remove('day');
      toggle.classList.add('night');
      this.target = 0;
    }, false);
  }

  update(delta) {
    if (Math.abs(this.state - this.target) < 0.001) {
      return;
    }
    const { colors, renderer, target } = this;
    const damp = 1 - Math.exp(-10 * delta);
    this.state = this.state * (1 - damp) + target * damp;
    vec3.lerp(_color, colors.background.night, colors.background.day, this.state);
    renderer.setBackground(_color[0], _color[1], _color[2]);
    vec3.lerp(_color, colors.sunlight.night, colors.sunlight.day, this.state);
    renderer.setSunlight(_color[0], _color[1], _color[2]);
  }
}

export default Light;
