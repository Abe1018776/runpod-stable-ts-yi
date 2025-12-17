import faster_whisper
from pyannote.audio import Pipeline
from speechbrain.inference.speaker import EncoderClassifier
import os
from huggingface_hub import login

def download_models():
    # Authenticate if token is present
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("HF_TOKEN found, logging in...")
        login(token=hf_token)
    else:
        print("No HF_TOKEN found, proceeding without authentication...")

    print("Starting model downloads...")

    # 1. Whisper Models
    whisper_models = [
        "ivrit-ai/whisper-large-v3-turbo-ct2",
        "ivrit-ai/yi-whisper-large-v3-turbo-ct2",
        "large-v3-turbo"
    ]
    
    for model_id in whisper_models:
        print(f"Downloading Whisper model: {model_id}...")
        try:
            # We initialize to trigger the download
            faster_whisper.WhisperModel(model_id)
            print(f"Successfully downloaded {model_id}")
        except Exception as e:
            print(f"Error downloading {model_id}: {e}")

    # 2. Pyannote Pipeline
    print("Downloading Pyannote pipeline: ivrit-ai/pyannote-speaker-diarization-3.1...")
    try:
        # Note: This might require HF_TOKEN if gated, but we run it as requested
        Pipeline.from_pretrained("ivrit-ai/pyannote-speaker-diarization-3.1")
        print("Successfully downloaded Pyannote pipeline")
    except Exception as e:
        print(f"Error downloading Pyannote: {e}")

    # 3. Speechbrain Classifier
    print("Downloading Speechbrain classifier: speechbrain/spkrec-ecapa-voxceleb...")
    try:
        EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        print("Successfully downloaded Speechbrain classifier")
    except Exception as e:
        print(f"Error downloading Speechbrain: {e}")

    print("All downloads attempted.")

if __name__ == "__main__":
    download_models()
