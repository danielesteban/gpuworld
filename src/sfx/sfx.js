import Blast from './blast.ogg';
import Plains from './plains.ogg';
import Shot from './shot.ogg';
import Wind from './wind.ogg';

class SFX {
  constructor() {
    Promise.all([
      Promise.all([Blast, Plains, Shot, Wind].map((sound) => (
        fetch(sound).then((res) => res.arrayBuffer())
      ))),
      new Promise((resolve) => {
        const onFirstInteraction = () => {
          window.removeEventListener('keydown', onFirstInteraction);
          window.removeEventListener('mousedown', onFirstInteraction);
          resolve();
        };
        window.addEventListener('keydown', onFirstInteraction, false);
        window.addEventListener('mousedown', onFirstInteraction, false);
      })
    ])
    .then(([buffers]) => {
      this.context = new window.AudioContext();
      return Promise.all(buffers.map((buffer) => this.context.decodeAudioData(buffer)));
    })
    .then(([blast, plains, shot, wind]) => {
      const { context } = this;
      this.output = context.createGain();
      this.output.connect(context.destination);
      document.addEventListener('visibilitychange', () => {
        this.output.gain.value = document.visibilityState === 'visible' ? 1 : 0;
      }, false);
      {
        const getAmbientSource = (buffer) => {
          const source = context.createBufferSource();
          source.buffer = buffer;
          source.loop = true;
          source.start(0);
          const gain = context.createGain();
          gain.gain.value = 0;
          source.connect(gain);
          gain.connect(this.output);
          return gain;
        };
        this.ambient = {
          plains: getAmbientSource(plains),
          wind: getAmbientSource(wind),
        };
      }
      this.buffers = [blast, shot];
      this.panners = [];
      this.sfx = context.createGain();
      this.sfx.gain.value = 0.2;
      this.sfx.connect(this.output);
    });
  }

  playAt(id, position) {
    const { buffers, context, panners, sfx } = this;
    if (!buffers || !buffers[id]) {
      return;
    }
    let panner = panners.pop();
    if (!panner) {
      panner = context.createPanner();
      panner.panningModel = 'HRTF';
      panner.refDistance = 32;
      panner.connect(sfx);
    }
    panner.positionX.value = position[0];
    panner.positionY.value = position[1];
    panner.positionZ.value = position[2];
    const buffer = buffers[id];
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.detune.value = (Math.random() - 0.25) * 1000;
    source.onended = () => {
      source.disconnect(panner);
      panners.push(panner);
    };
    source.connect(panner);
    source.start(0);
  }

  update(position, orientation) {
    const { ambient, context } = this;
    if (ambient) {
      const { plains, wind } = ambient;
      const gain = Math.min(Math.max((position[1] - 24) / 64, 0), 1);
      plains.gain.value = Math.cos(gain * 0.5 * Math.PI) * 0.8;
      wind.gain.value = Math.cos((1.0 - gain) * 0.5 * Math.PI) * 0.4;
    }
    if (context) {
      const { listener } = context;
      listener.positionX.value = position[0];
      listener.positionY.value = position[1];
      listener.positionZ.value = position[2];
      listener.forwardX.value = orientation[0];
      listener.forwardY.value = orientation[1];
      listener.forwardZ.value = orientation[2];
    }
  }
}

export default SFX;
