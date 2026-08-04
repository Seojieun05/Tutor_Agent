// Mic capture on the audio thread: hand every 128-sample block to the page.
//
// Framing to the VAD's 512-sample window and float32 → int16 conversion happen
// in index.html, where they are easier to read (and to change) than here.
class PCMCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0][0]; // mono float32, 128 samples
    if (channel) {
      // Copy: the engine reuses this buffer as soon as we return.
      this.port.postMessage(new Float32Array(channel));
    }
    return true; // keep the processor alive
  }
}

registerProcessor("pcm-capture", PCMCaptureProcessor);
