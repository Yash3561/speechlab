<div align="center">

# 🎙️ SpeechLab

### Speech Model Training Infrastructure

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org)
[![Ray](https://img.shields.io/badge/Ray-2.9+-028cf0.svg)](https://ray.io)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A distributed ML pipeline for training and evaluating speech recognition models — built with Ray, PyTorch, and MLOps best practices.*

[**Live Demo**](#demo) · [**Documentation**](#documentation) · [**Quick Start**](#-quick-start)

</div>

---

## 🎯 What is SpeechLab?

SpeechLab is a **full-stack training and evaluation infrastructure** for speech models. It provides:

- 🚀 **Distributed Training** — Multi-GPU/multi-node training with Ray Train
- 📊 **Experiment Tracking** — Full reproducibility with MLflow
- 📈 **Real-Time Monitoring** — Live training dashboard with WebSocket updates
- 🎯 **Multi-Metric Evaluation** — WER, CER, RTF with regression detection
- ⚙️ **Config-Driven** — Change experiments via YAML, not code

---

## 🖼️ Dashboard Preview

<div align="center">
<img src="docs/dashboard-preview.png" alt="SpeechLab Dashboard" width="800"/>
</div>

> Real-time training metrics, experiment management, and system monitoring — all in one beautiful interface.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (optional, for services)

### Installation

```bash
# Clone the repository
git clone https://github.com/Yash3561/speechlab.git
cd speechlab

# Option 1: Run setup script (Windows)
.\setup.bat

# Option 2: Manual setup
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e ".[dev]"

cd frontend && npm install && cd ..
```

### Start the Application

```bash
# Terminal 1: Start backend API
.venv\Scripts\activate
uvicorn backend.api.main:app --reload --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev

# Open http://localhost:3000
```

### (Optional) Start Docker Services

```bash
docker-compose up -d
# PostgreSQL: localhost:5432
# Redis: localhost:6379
# MinIO: localhost:9000
# MLflow: localhost:5000
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       SpeechLab                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Frontend   │───▶│   FastAPI   │───▶│    Ray      │     │
│  │  (Next.js)  │◀───│   Backend   │◀───│   Cluster   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         │           ┌──────┴──────┐          │             │
│         │           ▼             ▼          ▼             │
│  ┌──────┴──────┐ ┌────────┐ ┌────────┐ ┌────────┐         │
│  │  WebSocket  │ │Postgres│ │ Redis  │ │ MLflow │         │
│  │  (Metrics)  │ │        │ │        │ │        │         │
│  └─────────────┘ └────────┘ └────────┘ └────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
speechlab/
├── backend/
│   ├── api/              # FastAPI endpoints
│   │   ├── main.py       # App entry point
│   │   └── endpoints/    # Route handlers
│   ├── core/             # Config, logging, utils
│   ├── data/             # Audio processing pipeline
│   │   ├── dataset.py    # Data loading
│   │   ├── features.py   # Feature extraction
│   │   └── augmentation.py
│   ├── training/         # Training infrastructure
│   │   ├── trainer.py    # Training loop
│   │   └── models.py     # Model registry
│   └── evaluation/       # Metrics & evaluation
├── frontend/
│   ├── app/              # Next.js pages
│   ├── components/       # React components
│   └── lib/              # Utilities
├── configs/              # Experiment configs (YAML)
├── scripts/              # CLI tools
├── tests/                # Unit tests
└── docker-compose.yml    # Infrastructure
```

---

## 🧪 Running Experiments

### Via CLI

```bash
# Activate environment
.venv\Scripts\activate

# Run training
python scripts/train.py --config configs/experiments/demo_whisper_tiny.yaml

# Dry run (validate config)
python scripts/train.py --config configs/experiments/demo_whisper_tiny.yaml --dry-run
```

### Example Config

```yaml
experiment:
  name: "whisper_tiny_demo"
  
model:
  architecture: "whisper"
  variant: "tiny"
  
training:
  max_epochs: 5
  batch_size: 8
  learning_rate: 0.0001
  mixed_precision: true
  gradient_accumulation_steps: 4
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Orchestration** | Ray 2.9+ (Train, Data, Tune) |
| **ML Framework** | PyTorch 2.1+ with TorchAudio |
| **API** | FastAPI (async, WebSocket) |
| **Experiment Tracking** | MLflow |
| **Frontend** | Next.js 14, Tailwind CSS |
| **Database** | PostgreSQL (Supabase) |
| **Cache/Queue** | Redis (Upstash) |
| **Storage** | S3-compatible (Cloudflare R2) |

---

## 🎓 Why This Architecture?

This project demonstrates **solid ML engineering patterns**:

1. **Separation of Concerns** — Data, training, evaluation are independent modules
2. **Scalability** — Ray enables distributed computing across GPUs/nodes
3. **Reproducibility** — Every experiment is tracked and versioned via MLflow
4. **Observability** — Real-time monitoring with WebSocket streaming
5. **Flexibility** — Config-driven, architecture-agnostic design

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for the ML community**

</div>
