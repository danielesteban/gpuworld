import Ambient from './ambient.ogg';
import Shot from './shot.ogg';

class SFX {
  constructor() {
    Promise.all([
      Promise.all([Ambient, Shot].map((sound) => (
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
    .then(([ambient, shot]) => {
      const { context } = this;
      this.output = context.createGain();
      this.output.connect(context.destination);
      document.addEventListener('visibilitychange', () => {
        this.output.gain.value = document.visibilityState === 'visible' ? 1 : 0;
      }, false);
      {
        const source = context.createBufferSource();
        source.buffer = ambient;
        source.loop = true;
        source.start(0);
        const gain = context.createGain();
        gain.gain.value = 0.8;
        source.connect(gain);
        gain.connect(this.output);
      }
      this.sfx = context.createGain();
      this.sfx.gain.value = 0.2;
      this.sfx.connect(this.output);
      this.buffers = { shot };
    });
  }

  play(id) {
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
}

export default SFX;
