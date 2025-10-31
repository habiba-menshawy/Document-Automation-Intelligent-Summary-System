import os
import cv2
import json
import gc
from document_preperation_pipeline import DocumentProcessor

# Define document folders and entity types
documents = {
    "Scientific": "../archive/dataset/Scientific",
    "Email": "../archive/dataset/Email",
    "Report": "../archive/dataset/Report"
}

# Store results
scientific_results = []
email_results = []
report_results = []

results_map = {
    "Scientific": scientific_results,
    "Email": email_results,
    "Report": report_results
}

# Initialize processor once (avoid reloading heavy models)
processor = DocumentProcessor()

# Process up to 10 images per document type
for doc_type, folder in documents.items():
    print(f"Processing {doc_type} documents...")

    image_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ][:10]  # limit to 10 images

    for img_path in image_files:
        print(f"→ Running {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping {img_path} (failed to load)")
            continue

        try:
            result = processor.process(entity_type="EMAIL", image=img)

            # 🧹 Remove non-serializable image data before saving
            if "preprocessed_image" in result:
                del result["preprocessed_image"]

            results_map[doc_type].append({
                "image_path": img_path,
                "result": result
            })

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

        finally:
            # Free memory between iterations
            del img, result
            gc.collect()

print("\n✅ Summary:")
print(f"Scientific results: {len(scientific_results)}")
print(f"Email results: {len(email_results)}")
print(f"Report results: {len(report_results)}")

# ✅ Save as JSON files (now safely serializable)
with open("scientific_results.json", "w") as f:
    json.dump(scientific_results, f, indent=2)

with open("email_results.json", "w") as f:
    json.dump(email_results, f, indent=2)

with open("report_results.json", "w") as f:
    json.dump(report_results, f, indent=2)
