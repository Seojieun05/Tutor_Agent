// Mic capture on the audio thread: hand every 128-sample block to the page.
//
// Framing to the VAD's 512-sample window and float32 → int16 conversion happen
// in index.html, where they are easier to read (and to change) than here.
// [한글] 마이크 오디오 스레드 프로세서. 128 샘플 블록을 그대로 페이지로 넘긴다.
// VAD 창(512 샘플) 맞추기와 float32 → int16 변환은 index.html에서 한다.
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
