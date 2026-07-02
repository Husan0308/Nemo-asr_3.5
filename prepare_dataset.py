from datasets import load_dataset
import soundfile as sf
import json
import os

TRAIN_SIZE = 100000
VAL_SIZE = 2000

os.makedirs("audio", exist_ok=True)
os.makedirs("manifests", exist_ok=True)

train_f = open("manifests/train.json", "w", encoding="utf-8")
val_f = open("manifests/val.json", "w", encoding="utf-8")

ds = load_dataset(
    "akmalsultanov/uzbooks_300hr",
    split="train",
    streaming=True
)

for idx, sample in enumerate(ds):

    if idx >= TRAIN_SIZE + VAL_SIZE:
        break

    audio = sample["audio"].get_all_samples()

    flac_path = f"audio/{idx:06d}.flac"

    sf.write(
        flac_path,
        audio.data.squeeze().numpy(),
        audio.sample_rate,
        format="FLAC"
    )

    duration = audio.data.shape[-1] / audio.sample_rate

    item = {
        "audio_filepath": os.path.abspath(flac_path),
        "duration": duration,
        "text": sample["aligned_sentence"],
        "target_lang": "uz-UZ"
    }

    line = json.dumps(item, ensure_ascii=False)

    if idx < TRAIN_SIZE:
        train_f.write(line + "\n")
    else:
        val_f.write(line + "\n")

    if idx % 10 == 0:
        print(f"Processed {idx}")

train_f.close()
val_f.close()

print("DONE")
