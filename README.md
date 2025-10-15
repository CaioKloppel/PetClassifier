# 🐾 PetClassifier – Dog and Cat Classifier

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-91.2%25-success?style=for-the-badge)

Computer vision project in **PyTorch** for binary image classification between 🐱 **Cat** and 🐶 **Dog** using a 2D CNN with residual blocks.

</div>

---

# 📊 Dataset

This project uses the public dog and cat classification dataset available on Kaggle:

- **Provider:** [Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Creator:** [Bhavik Jikadara](https://www.kaggle.com/bhavikjikadara)

---

# 🧠 Model Architecture

### CNN Structure

The architecture implemented in `src/modelArchitecture/modelArchitecture.py` (class `CNN`) is a custom **Convolutional Neural Network** inspired by ResNet, specifically designed for binary image classification.

#### 🔧 Main Components

**1. Initial Layer**

- 2D Convolution (3→32 channels, 5×5 kernel)
- Batch Normalization
- ReLU Activation
- MaxPooling (3×3) for dimensional reduction

**2. Three Residual Blocks**

- **Block 1:** 32→64 channels (bottleneck: 32→16→16→64)
- **Block 2:** 64→128 channels (bottleneck: 64→32→32→128)
- **Block 3:** 128→256 channels (bottleneck: 128→64→64→256)
- Each block has residual connections (skip connections) to facilitate deep training

**3. Output Layer**

- Adaptive Average Pooling (2×2)
- Flatten
- Dense Layer: 1024→1 neuron
- Output: single logit (classification threshold: 0.5)

> 💡 **Why residual blocks?** Residual connections allow gradients to flow directly through the network, avoiding the vanishing gradient problem and enabling training of deeper networks with better performance.

### Technical Specifications

| Property             | Value                                                         |
| -------------------- | ------------------------------------------------------------- |
| Input                | RGB images of **any size** (automatically resized to 224×224) |
| Processing dimension | 224×224 pixels                                                |
| Output               | Single logit (threshold: 0.5)                                 |
| Loss Function        | BCEWithLogitsLoss                                             |
| Optimizer            | Adam (lr=0.001, weight_decay=1e-5)                            |
| Scheduler            | StepLR (step_size=50, gamma=0.1)                              |
| Epochs               | 150 for each training phase                                   |

**Trained weights:** `src/modelWeights/modelPet6.pth`

> 💡 **Input flexibility:** The model accepts images of any resolution. The prediction script automatically resizes to 224×224 before inference.

---

## 📈 Model Performance

### Achieved Accuracy

- **Validation Accuracy:** **91.2%** (0.912)
- **Best performance epoch:** Between epochs 50-150 (Val Accuracy stable between 0.910-0.913)
- **Training Accuracy:** 100% (final epochs)

> 📝 **Note:** Training documented in `src/modelTrain/modelPet.ipynb` and `src/modelTrain/modelPetTrain.ipynb` with **22,998 images** in total (11,499 cats + 11,499 dogs), split into 70% training and 30% validation

---

## 🚀 How to Run

### 📋 Requirements

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Or manually install the main dependencies:

```bash
pip install torch torchvision pillow
```

**Main dependencies:**

- `torch>=1.10.0` - Deep learning framework
- `torchvision>=0.11.0` - Computer vision utilities
- `Pillow>=9.0.0` - Image processing

**Training dependencies (optional):**

- `pandas`, `matplotlib`, `scikit-learn`, `jupyter`

### 🐍 Using Python Directly

Run the prediction script by passing the image path for analysis:

```bash
python src/petClassifier.py path/to/image.jpg
```

**Expected output:\*\***

```
==================================================
          ANALYZING IMAGE: catTest.jpg...
==================================================
                  IT'S A CAT
==================================================
```

> ⚠️ **Important:** The weights file `modelPet6.pth` must be in the current working directory.

### 📦 Creating and Using Executable (PyInstaller)

If you want to create a standalone executable, follow these steps:

**Step 1: Install PyInstaller**

```bash
pip install pyinstaller
```

**Step 2: Create the executable**

```bash
pyinstaller --onefile --add-data "src/modelWeights/modelPet6.pth;." src/petClassifier.py
```

This will create an executable in the `dist/` folder.

**Step 3: (Optional) Compress the executable**

```bash
# Windows (using PowerShell)
Compress-Archive -Path dist/petClassifier.exe -DestinationPath dist/petClassifier.zip
```

**Step 4: Run the executable**

```bash
./dist/petClassifier.exe path/to/image.jpg
```

> 💡 **Important:** Make sure `modelPet6.pth` is in the same directory where the execution command is being run (if not included via `--add-data` flag).

> ⏱️ **Performance note:** The executable may take a few seconds to initialize on first execution, as PyInstaller packages the entire PyTorch library into a single file (~300-500MB). This initialization time is normal and only occurs when loading the executable, not affecting the subsequent inference time.

---

## 📁 Project Structure

```
PetClassifier/
|
├── 📂 src/
│   ├── 📄 petClassifier.py  # Inference script
│   ├── 📂 modelArchitecture/
│   │   └── 📄 modelArchitecture.py  # CNN definition
│   ├── 📂 modelTrain/
│   │   ├── 📓 modelPet.ipynb        # Training notebook (1st training)
│   │   └── 📓 modelPetTrain.ipynb   # Training notebook (2nd training)
│   └── 📂 modelWeights/
│       └── 💾 modelPet6.pth         # Trained model weights
└── 📄 README.md             # This file
```

---

## ⚙️ Technical Details

### Image Preprocessing

#### During Training

- **Resizing:** 224×224 pixels (training base)
- **Data Augmentation:**
  - Random horizontal flip
  - Random rotation (±10°)
  - Brightness and contrast adjustment (±20%)
- **Normalization:** ToTensor

#### During Inference

- **Input:** Accepts images of **any size** (JPG, PNG, etc.)
- **Automatic resizing:** 224×224 pixels (maintains compatibility with trained model)
- **Conversion:** RGB (if the image is in another format)
- **Normalization:** ToTensor

### Hardware and Performance

- ✅ **GPU:** Automatic CUDA support (if available)
- ✅ **CPU:** Automatic fallback
- ⚡ **Inference time:** ~50-100ms per image (GPU) | ~200-400ms (CPU)

### Data Split

- **Total images:** 22,998 (11,499 cats + 11,499 dogs)
- **Training:** 70% (~16,099 images)
- **Validation:** 30% (~6,899 images)
- **Ratio:** 50% cats / 50% dogs

---

## 🎯 Results and Metrics

| Metric                    | Value     |
| ------------------------- | --------- |
| Validation Accuracy       | **91.2%** |
| Training Accuracy (final) | **100%**  |
| Validation Loss (final)   | ~0.37     |
| Convergence epoch         | ~50-60    |

---

## 📝 Important Notes

- 🔧 The model **automatically uses GPU** if available via `torch.cuda.is_available()`
- 📊 The reported accuracy refers to the **validation set** used in the notebook
- 🎲 Results may vary due to random seeds, data split, and environment
- 🖼️ **Image compatibility:** The model accepts images of any size and resolution - they are automatically resized to 224×224 during prediction
- ⚠️ Very corrupted images or unsupported formats may cause errors
- 💾 Make sure the `modelPet6.pth` file is accessible during execution

---

## 🤝 Credits

- **Dataset:** [Bhavik Jikadara](https://www.kaggle.com/bhavikjikadara) via [Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Framework:** PyTorch

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for more details.

**License summary:**

- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ No warranties

> ⚠️ **Note:** The dataset used is subject to [Kaggle licenses](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset).

---
