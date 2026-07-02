import json

for file in ["manifests/train.json", "manifests/val.json"]:
    fixed = []

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            item["language"] = "en"

            fixed.append(item)

    with open(file, "w", encoding="utf-8") as f:
        for item in fixed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Manifest fixed.")
