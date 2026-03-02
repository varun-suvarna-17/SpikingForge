# ⚡ SpikingForge

Lightweight Spiking Neural Network (SNN) toolchain for Embedded AI.

SpikingForge is a CPU-only Python package for building, training, and exporting Spiking Neural Networks (SNNs) with a focus on simplicity, reproducibility, and embedded deployment.

---

## 🚀 Features

- 🧠 Vectorized Leaky Integrate-and-Fire (LIF) neurons
- 🔁 Trace-based STDP learning
- 📊 Linear readout layer with SGD
- 📄 YAML-driven experiment configuration
- 🖥 Command Line Interface (CLI)
- 🔌 Export trained weights to C header files for microcontrollers
- 💻 CPU-only (lightweight & dependency minimal)

---

## 📂 Project Structure

```
SpikingForge/
│
├── pyproject.toml
├── README.md
├── configs/
├── examples/
├── tests/
│
└── src/
    └── spikingforge/
        ├── core.py
        ├── learning.py
        ├── export.py
        ├── config.py
        ├── cli.py
        └── utils.py
```

---

## 🛠 Installation (Development Mode)

From the project root folder:

```bash
pip install -e .
```

This installs SpikingForge in editable mode.

---

## 🖥 CLI Usage

Train using a YAML configuration:

```bash
python -m spikingforge.cli train configs/example.yaml
```

(If your system PATH is configured correctly:)

```bash
spikingforge train configs/example.yaml
```

---

## 📄 Example YAML Configuration

```yaml
model:
  input_size: 100
  hidden_size: 50
  output_size: 10

training:
  epochs: 5
  lr: 0.01
```

---

## 🧪 Running Example Script

You can also test the package manually:

```bash
python examples/train_example.py
```

---

## 🔌 Exporting Weights (Planned)

Export trained model weights to C header format:

```bash
spikingforge export weights.npy weights.h
```

This enables deployment on microcontrollers (e.g., STM32).

---

## 🧠 Design Philosophy

SpikingForge is built to be:

- Lightweight
- Transparent
- Easy to understand
- Friendly for embedded experimentation
- Minimal dependency

No GPU. No heavy frameworks. Just clean NumPy-based SNN simulation.

---

## 👥 Team

Built by the SpikingForge Team.

---

## 📜 License

MIT License