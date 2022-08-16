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
          source.connect(gain);
          gain.connect(this.output);
          return gain;
        };
        this.ambient = {
          plains: getAmbientSource(plains),
          wind: getAmbientSource(wind),
        };
        this.update(0);
      }
      this.sfx = context.createGain();
      this.sfx.gain.value = 0.2;
      this.sfx.connect(this.output);
      this.buffers = [blast, shot];
    });
  }

  playAt(id/*, position */) {
    // @incomplete: setup panner node for positional audio
    const { buffers, context, sfx } = this;
    if (!buffers || !buffers[id]) {
      return;
    }
    const buffer = buffers[id];
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.detune.value = (Math.random() - 0.25) * 1000;
    source.connect(sfx);
    source.start(0);
  }

  update(altitude) {
    // @incomplete: update context listener for positional audio
    const { ambient } = this;
    if (!ambient) {
      return;
    }
    const { plains, wind } = ambient;
    const gain = Math.min(Math.max((altitude - 24) / 64, 0), 1);
    plains.gain.value = Math.cos(gain * 0.5 * Math.PI) * 0.8;
    wind.gain.value = Math.cos((1.0 - gain) * 0.5 * Math.PI) * 0.4;
  }
}

export default SFX;
