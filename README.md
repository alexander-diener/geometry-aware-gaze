# Geometry-Aware Gaze Estimation (Research Prototype)

This repository provides a **minimal, reproducible baseline** for gaze estimation that combines:
1) a **geometry-inspired model** (head pose + relative eye angles), and  
2) a **small learned residual correction** (MLP) to compensate for systematic biases / domain effects.

The goal is to demonstrate **geometry-aware inductive bias** and a clean research workflow (setup → run → results).

## Why geometry (context)
Generic large vision foundation models are often optimized for semantic invariances and may underutilize subtle, geometry-sensitive cues.  
For gaze estimation, **small angular differences matter**; geometry-aware baselines can be strong and sample-efficient, and learning can be used to model residual errors.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.learning.train
