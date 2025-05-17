import * as deepspeech from 'deepspeech';

class DeepSpeechService {
  private model: deepspeech.Model | null = null;
  private modelPath = process.env.NEXT_PUBLIC_DEEPSPEECH_MODEL_PATH;

  constructor() {
    this.initializeModel();
  }

  private async initializeModel() {
    try {
      if (!this.modelPath) {
        throw new Error('DeepSpeech model path not configured');
      }

      this.model = new deepspeech.Model(this.modelPath);
      console.log('DeepSpeech model initialized successfully');
    } catch (error) {
      console.error('Failed to initialize DeepSpeech model:', error);
    }
  }

  public async processAudio(audioData: Float32Array): Promise<string> {
    if (!this.model) {
      throw new Error('DeepSpeech model not initialized');
    }

    try {
      const buffer = this.float32ToInt16(audioData);
      
      const stream = this.model.createStream();
      
      // Feed audio content to the stream
      stream.feedAudioContent(buffer);
      
      // Get the transcription
      const text = stream.finishStream();
      
      return text;
    } catch (error) {
      console.error('Error processing audio:', error);
      throw error;
    }
  }

  private float32ToInt16(float32Array: Float32Array): Int16Array {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16Array;
  }
}

export const deepSpeechService = new DeepSpeechService(); 