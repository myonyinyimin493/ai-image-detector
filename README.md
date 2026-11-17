# 🔍 AI Image Detector — Real or AI-Generated?

A deep learning project that estimates whether an image is a **real photograph** or **AI-generated** (e.g. Stable Diffusion, Midjourney, DALL·E), with explanations and honest uncertainty — designed with real-world use in mind, such as helping online-harm support workers triage suspicious images.

> ⚠️ **Important:** This tool gives a *probability*, not proof. AI image detection is an unsolved research problem, and no detector is 100% reliable. Results should support human judgement, never replace it.

---

## 📌 Table of Contents
- [Motivation](#-motivation)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Step-by-Step Roadmap](#-step-by-step-roadmap)
- [Evaluation Plan](#-evaluation-plan)
- [Results](#-results)
- [Limitations & Responsible Use](#-limitations--responsible-use)
- [References](#-references)

---

## 💡 Motivation
AI-generated images are now realistic enough to fool most people. They are used for misinformation, scams, fake evidence, and image-based abuse. Many existing detectors report very high accuracy on clean benchmark data but fail on real-world images (compressed, resized, screenshotted, or made by newer generators).

This project aims to:
1. Build a working real-vs-AI image classifier.
2. **Honestly measure where it fails** — on unseen generators and after common image edits.
3. Present results in a way that is **explainable and safe** for non-technical users.

---

## ⚙️ How It Works
The system combines three layers of checks:

```
            ┌──────────────────────┐
 Image ───► │ 1. Metadata check    │  C2PA / EXIF / generator tags (if present)
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │ 2. Classifier        │  CNN (ResNet/EfficientNet) or CLIP + linear probe
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │ 3. Explanation       │  Grad-CAM heatmap + confidence + caveats
            └──────────┬───────────┘
                       ▼
          "72% likely AI-generated — moderate confidence"
```

Two model approaches are compared:

| Approach | What is trained | Pros | Cons |
|---|---|---|---|
| **A. Fine-tuned CNN** (ResNet / EfficientNet) | Whole network | High accuracy on seen generators | Overfits to generator-specific traces |
| **B. Frozen CLIP + linear probe** | Only a small classifier | Fast, generalises better to unseen generators | Slightly lower in-distribution accuracy |

---

## ✨ Features
- [ ] Upload an image and get a real/AI probability
- [ ] Confidence levels with plain-language explanation
- [ ] Grad-CAM heatmap showing *where* the model looked
- [ ] Metadata / provenance check (EXIF, C2PA)
- [ ] Robustness report (JPEG, resize, blur)
- [ ] Web UI (FastAPI backend + simple frontend)

---

## 🛠 Tech Stack
| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep learning | PyTorch, torchvision, timm |
| CLIP | open_clip |
| Classic ML | scikit-learn |
| Explainability | pytorch-grad-cam |
| Image processing | Pillow, OpenCV |
| Backend | FastAPI |
| Frontend | HTML/JS or Streamlit (prototype) |
| Experiment tracking | Weights & Biases or TensorBoard (optional) |
| Deployment | Docker, Hugging Face Spaces (optional) |

---

## 📂 Datasets
| Dataset | Description | Use |
|---|---|---|
| **[CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)** | 120k small (32×32) images — real from CIFAR-10, fake from Stable Diffusion | Quick prototype / baseline |
| **GenImage** | Large benchmark with images from multiple generators (Midjourney, Stable Diffusion, ADM, GLIDE, BigGAN, etc.) | Main training + cross-generator testing |
| **Custom set** | Your own generated images + real photos (e.g. COCO) | Real-world testing |

> Datasets are **not** committed to this repo. See `data/README.md` for download instructions.

---

## 🗂 Project Structure
```
ai-image-detector/
├── data/
│   ├── raw/                # downloaded datasets (git-ignored)
│   ├── processed/          # resized / split data
│   └── README.md           # how to download data
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_baseline_cnn.ipynb
│   └── 03_clip_probe.ipynb
├── src/
│   ├── data.py             # Dataset & DataLoader
│   ├── models.py           # CNN and CLIP models
│   ├── train.py            # training loop
│   ├── evaluate.py         # metrics & robustness tests
│   ├── explain.py          # Grad-CAM
│   └── metadata.py         # EXIF / C2PA checks
├── app/
│   ├── main.py             # FastAPI app
│   └── static/             # frontend
├── models/                 # saved weights (git-ignored)
├── results/                # plots, tables, reports
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/ai-image-detector.git
cd ai-image-detector
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install torch torchvision timm open_clip_torch scikit-learn \
            pillow opencv-python matplotlib grad-cam fastapi uvicorn
pip freeze > requirements.txt
```

### 4. Download CIFAKE
```bash
pip install kaggle
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data/raw --unzip
```
(Requires a Kaggle account and API token — see Kaggle's docs.)

### 5. Train the baseline
```bash
python src/train.py --model resnet18 --data data/raw/cifake --epochs 5
```

### 6. Run the web app
```bash
uvicorn app.main:app --reload
# open http://localhost:8000
```

> 💡 No GPU? Use **Google Colab** or **Kaggle Notebooks** (both free with GPU) for training.

---

## 🧭 Step-by-Step Roadmap

### Phase 0 — Refresh the fundamentals (Week 1)
- [ ] Learn PyTorch basics: tensors, `Dataset`, `DataLoader`, training loop
- [ ] Review CNNs and transfer learning (fast.ai Lesson 1–2)
- [ ] Read the CLIP paper intro + figures
- [ ] Set up GitHub repo, virtual environment, and this README

**✅ Done when:** you can train a pretrained model on a small image dataset in Colab.

### Phase 1 — Baseline CNN (Week 2)
- [ ] Download CIFAKE and explore it (`01_explore_data.ipynb`): view samples, check class balance
- [ ] Build `Dataset` + train/val/test split
- [ ] Fine-tune **ResNet18** (pretrained on ImageNet)
- [ ] Track accuracy, precision, recall, F1, confusion matrix
- [ ] Save the best model weights

**✅ Done when:** you have a working baseline and a confusion matrix.

### Phase 2 — CLIP linear probe (Week 3)
- [ ] Load a pretrained CLIP model with `open_clip`
- [ ] Extract image embeddings for all images (frozen encoder)
- [ ] Train a **logistic regression** on the embeddings
- [ ] Compare against the CNN baseline

**✅ Done when:** you have a table comparing CNN vs CLIP.

### Phase 3 — Honest evaluation (Week 4) ⭐ *the most important phase*
- [ ] **Cross-generator test:** train on one generator (e.g. Stable Diffusion), test on others (e.g. Midjourney, BigGAN) using GenImage
- [ ] **Robustness test:** apply JPEG compression (quality 90/70/50), resizing, blur, and screenshot-like crops; measure accuracy drop
- [ ] **Calibration:** check whether "80% confident" really means right ~80% of the time
- [ ] Improve with data augmentation (train on compressed/resized images)

**✅ Done when:** you have a results table showing where each model succeeds and fails.

### Phase 4 — Explainability & metadata (Week 5)
- [ ] Add **Grad-CAM** heatmaps for the CNN
- [ ] Add EXIF / C2PA metadata check (`metadata.py`)
- [ ] Design the output message: probability + confidence band + caveats

**✅ Done when:** a single function takes an image and returns prediction + heatmap + metadata summary.

### Phase 5 — Web app & deployment (Week 6)
- [ ] Build FastAPI endpoint: `POST /predict` → JSON result
- [ ] Build a simple upload page that shows result, heatmap, and caveats
- [ ] Containerise with Docker
- [ ] (Optional) Deploy to Hugging Face Spaces
- [ ] Write up results in this README with plots

**✅ Done when:** someone else can upload an image via a link and understand the result.

### Stretch goals
- [ ] Ensemble CNN + CLIP
- [ ] Frequency-domain (FFT) features
- [ ] Detect partially edited images (inpainting)
- [ ] User testing with non-technical users for clarity of results

---

## 📊 Evaluation Plan
| Test | Question it answers |
|---|---|
| In-distribution accuracy | Does it work on the same kind of data it was trained on? |
| Cross-generator accuracy | Does it work on images from **new** AI models? |
| Robustness (JPEG / resize / blur) | Does it survive real-world image handling? |
| False positive rate | How often are **real** photos wrongly flagged as AI? |
| Calibration | Can we trust the confidence score? |

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, False Positive Rate.

---

## 📈 Results
*To be filled in as experiments are completed.*

| Model | Train data | Test: same generator | Test: unseen generator | Test: JPEG q=50 |
|---|---|---|---|---|
| ResNet18 | — | — | — | — |
| CLIP + linear probe | — | — | — | — |

---

## ⚖️ Limitations & Responsible Use
- **Not proof.** A high "AI" score is not evidence that an image is fake, and a low score does not prove it is real.
- **False positives cause harm.** Wrongly labelling a real photo as AI could discredit a genuine victim or witness. The UI always shows uncertainty.
- **Generators evolve.** New AI models may evade this detector; it should be re-evaluated regularly.
- **Privacy.** Uploaded images should not be stored by default. Sensitive images should be handled with care and consent.
- **Intended use:** education, research, and as *one input* to human review — not for automated decisions about people.

---

## 📚 References
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP).
- Ojha et al. (2023). *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (CLIP-based detection).
- Wang et al. (2020). *CNN-generated images are surprisingly easy to spot... for now*.
- Bird & Lotfi (2024). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*.
- Zhu et al. (2023). *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*.
- C2PA — Coalition for Content Provenance and Authenticity: https://c2pa.org

---

## 👤 Author
**Your Name** — [GitHub](https://github.com/<your-username>) · [LinkedIn](#)

## 📄 License
MIT
