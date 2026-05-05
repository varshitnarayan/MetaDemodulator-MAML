# MetaDemodulator: MAML-Based Few-Shot Wireless Demodulation

> A meta-learning approach to adaptive wireless demodulation using Model-Agnostic Meta-Learning (MAML), with novel 4D feature engineering and ResNet architecture for fast few-shot adaptation at base stations.

---

## 📡 Overview

This project implements and extends the **MetaDemodulator** — a system that uses MAML (Model-Agnostic Meta-Learning) to enable a base station to rapidly adapt its demodulation model to new devices using only a handful of pilot symbols.

The core idea: instead of training a single fixed demodulator, we meta-train a neural network so that its initialization is **optimized for fast adaptation**. When a new device connects, the model adapts in just a few gradient steps using a small pilot sequence, achieving low symbol error rates (SER) even in nonlinear, fading channel conditions.

### Key Contributions

- **Result Reproduction**: Full PyTorch re-implementation of the original MAML demodulation paper results (toy 4-QAM and realistic 16-QAM scenarios).
- **Novel Architecture**: A proposed **4D ResNet** that enriches the input with magnitude and phase features alongside real/imaginary components, with a skip connection for improved gradient flow.
- **Online Learning**: A streaming scenario that simulates continuous device connections to a live base station, demonstrating real-time meta-learning adaptation.

---

## 🗂️ Project Structure

```
MLC Project/
├── Reproduction of Results Code/
│   ├── maml_toy_example.py          # Toy MAML: 4-QAM, AWGN, 1D channel
│   └── maml_realistic_scenario.py   # Baseline MAML: 16-QAM, complex fading, nonlinear PA
│
├── Novelty Code/
│   ├── maml_novelty_architecture.py # Proposed 4D ResNet vs. baseline comparison
│   └── maml_novelty_online.py       # Online/streaming MAML with replay buffer
│
├── figures/
│   ├── Result Reproduction Figures/
│   │   ├── Figure_4.png             # Toy example SER vs. pilots
│   │   ├── Figure_5.png             # Realistic scenario SER vs. pilots
│   │   └── Figure_6.png             # Additional reproduction result
│   └── Novelty Figures/
│       ├── Figure_Feature_Engineering.png   # 4D input feature visualization
│       ├── Figure_Novelty_Architecture.png  # ResNet vs. baseline SER comparison
│       ├── Figure_Novelty_Online.png        # Online learning curve
│       └── Figure_ResNet_Features.png       # ResNet feature analysis
│
└── Report and Paper/
    ├── Meta_Demodulator_Paper.pdf   # Reference paper
    └── MetaDemodulator.pptx         # Project presentation slides
```

---

## 🧠 Methods

### Channel Model

The system simulates a realistic wireless channel with:
- **Complex Rayleigh fading**: `h ~ CN(0, 1)`
- **Nonlinear Power Amplifier (PA)**: Saleh model with distortion parameter `β ~ U(0.05, 0.15)`
- **AWGN noise** at SNR = 21 dB
- **16-QAM modulation** (16 symbols)

### MAML Training

Standard MAML meta-training loop:
- **Inner loop**: 1-step gradient update on support set (pilot symbols) per device
- **Outer loop**: Adam optimizer updates the meta-initialization across K=20 devices
- **Meta-epochs**: 5000

### Proposed Novelty: 4D ResNet

| Feature | Baseline | Proposed |
|---|---|---|
| Input | 2D (Re, Im) | 4D (Re, Im, \|y\|, ∠y) |
| Architecture | 3-layer MLP, 10 neurons | 4-layer ResNet, 16 neurons |
| Skip connection | ✗ | ✓ (h1 → h3) |
| Adapt steps needed | 100 | 3 |

The 4D feature vector captures both Cartesian and polar representations of the received signal, giving the model richer geometry for fast adaptation.

### Online Learning

The online scenario simulates a base station receiving a continuous stream of new device connections. A **replay buffer** stores historical device channels, and the meta-model is updated online via mini-batches sampled from this buffer. This mirrors a realistic deployment where the base station must continuously adapt without forgetting.

---

## 📊 Results

### SER vs. Number of Pilots (Architecture Comparison)

The proposed 4D ResNet MAML significantly outperforms the 2D MLP baseline and traditional training approaches (Joint Training, Fixed Initialization) across all pilot counts.

![Architecture Comparison](figures/Novelty%20Figures/Figure_Novelty_Architecture.png)

### Online Adaptation Curve

The online MAML model achieves rapidly decreasing SER as more devices are seen, while fixed and joint training baselines stagnate.

![Online Learning](figures/Novelty%20Figures/Figure_Novelty_Online.png)

---

## ⚙️ Requirements

```
torch >= 1.12
numpy
matplotlib
```

Install with:

```bash
pip install torch numpy matplotlib
```

---

## 🚀 Usage

### 1. Reproduce Baseline Results

**Toy Example (4-QAM, simple AWGN channel):**
```bash
python "Reproduction of Results Code/maml_toy_example.py"
```

**Realistic Scenario (16-QAM, nonlinear fading channel):**
```bash
python "Reproduction of Results Code/maml_realistic_scenario.py"
```

### 2. Run Novelty Experiments

**Architecture Comparison (4D ResNet vs. 2D MLP):**
```bash
python "Novelty Code/maml_novelty_architecture.py"
```
> Trains both models for 5000 meta-epochs, then evaluates SER vs. number of pilots P ∈ {0, 2, 4, 8, 12, 14, 16}.

**Online Streaming Scenario:**
```bash
python "Novelty Code/maml_novelty_online.py"
```
> Simulates 3000 sequential device connections and evaluates SER every 100 time slots.

Generated figures are saved to a local `figures/` directory.

---

## 📄 Reference

This project is based on and extends:

> **"Meta-Learning for Fast Adaptive Wireless Demodulation"** — see `Report and Paper/Meta_Demodulator_Paper.pdf` for the full reference paper and `MetaDemodulator.pptx` for the project presentation.

---

## 📝 License

This project is for academic and educational purposes. Please cite the original paper if you build upon this work.
