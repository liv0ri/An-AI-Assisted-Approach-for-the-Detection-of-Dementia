<h1 align="center">AI-Assisted Early Dementia Detection from Speech & Language</h1>

<p align="center">
 <a href="#">
   <img src="https://img.shields.io/badge/🎓%20Final%20Year%20Project-Department%20of%20AI%20%7C%202025-blue?style=for-the-badge&logo=graduation-cap&logoColor=white" alt="Final Year Project">
 </a>
</p>

<p align="center">
This repository contains the research, experiments, and implementation of an <strong>AI-assisted multi-modal system</strong> for the <strong>early detection of dementia</strong> using speech and language analysis.
</p>

<p align="center">
  <!-- Language & License -->
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  </a>
</p>

<p align="center">
  <!-- Datasets -->
  <a href="https://talkbank.org/dementia/access/English/Pitt.html">
    <img src="https://img.shields.io/badge/Dataset-Pitt%20Corpus-green?style=flat-square">
  </a>
  <a href="https://talkbank.org/dementia/ADReSSo-2021/index.html">
    <img src="https://img.shields.io/badge/Dataset-ADReSSo-orange?style=flat-square">
  </a>
</p>

---

## 📚 Table of Contents
<details>
<summary>Click-to-View</summary>

- [Project Overview](#-project-overview)
- [Motivation](#-motivation)
- [Key Features](#-key-features)
- [Datasets](#-datasets)
- [Methodology](#-methodology)
- [Model Architectures](#-model-architectures)
- [Evaluation Strategy](#-evaluation-strategy)
- [Ethics & Bias Considerations](#-ethics--bias-considerations)
- [Getting Started](#-getting-started)
- [Dissertation & Documentation](#-dissertation--documentation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#%EF%B8%8F-contact)

</details>

---

## 📌 Project Overview

Dementia is a progressive neurological disorder that significantly impacts cognitive function, communication, and daily living. Early detection is essential for timely intervention, improved care planning, and enhanced quality of life for both patients and caregivers.

This project investigates the use of **Artificial Intelligence (AI)** to support **early dementia detection** through **speech and language analysis**, leveraging recent advances in **deep learning**, **transformer models**, and **multi-modal learning**.

The system evaluates and compares several state-of-the-art AI architectures drawn from existing research, with the goal of identifying:
- the most effective modalities (text, audio, timing),
- the most suitable dataset,
- and the most efficient model architecture for real-world clinical support.

The final outcome is a **CNN-based multi-modal dementia classification model** designed to assist psychologists and clinicians as a **decision-support tool**, not a standalone diagnostic system.

---

## 💡 Motivation

- Dementia cases are projected to **nearly double by 2050**, both globally and within Malta.
- Early symptoms often resemble normal ageing, leading to **delayed diagnosis**.
- Traditional screening methods are time-consuming and resource-intensive.
- AI offers scalable, cost-effective tools for **early screening and risk assessment**.

This project aligns with **Malta’s National Dementia Strategy**, particularly:
- **Action Area 2**: Decreasing dementia risk  
- **Action Area 3**: Timely diagnosis  

By leveraging speech — a natural, non-invasive signal — this research explores how AI can complement clinical expertise.

---

## 🔎 Key Features

- **Multi-Modal Learning**
  - Combines **audio**, **textual**, and **temporal** speech features.
  - Converts audio into **Mel-spectrograms** for CNN/ViT processing.
  - Processes transcripts using **Transformer-based NLP models**.

- **Comparative Model Evaluation**
  - Replication and evaluation of two major research architectures:
    - Wav2Vec2 + Word2Vec + LSTM (Pitt Corpus)
    - ViT + RoBERTa with cross-attention (ADReSSo)

- **Transformer-Based Fusion**
  - Cross-attention mechanisms allow speech and text modalities to inform each other.
  - Enables richer representation of cognitive impairment signals.

- **Explainability & Clinical Awareness**
  - Emphasis on interpretable architectures.
  - Designed to support, not replace, psychologists and clinicians.

---

## 📊 Datasets

### 🗂️ Pitt Corpus (DementiaBank)
- Part of the **TalkBank** project.
- Contains speech recordings and transcripts from Alzheimer’s patients and healthy controls.
- Includes linguistic markers such as pauses, repetitions, and incomplete utterances.
- Widely used benchmark dataset for dementia research.

### 🗂️ ADReSSo Dataset
- Introduced in the **INTERSPEECH 2020 ADReSSo Challenge**.
- Focuses on dementia detection from spontaneous speech.
- Provides standardized train/test splits.
- Designed to reduce demographic bias across classes.

---

## 🧠 Methodology

1. **Literature Replication**
   - Implemented models from two recent peer-reviewed research papers.
   - Ensured reproducibility by using original datasets and preprocessing pipelines.

2. **Feature Extraction**
   - Audio → Mel-spectrograms
   - Text → Transformer embeddings (RoBERTa, Word2Vec)
   - Optional temporal features (timestamps, pacing)

3. **Model Selection**
   - Evaluated accuracy, precision, recall, F1-score, and AUC.
   - Selected top-performing architectures for further refinement.

4. **Model Refinement**
   - Removed poorly performing MMSE prediction components.
   - Optimized architecture for computational efficiency and stability.

---

## 🧩 Model Architectures

- **Wav2Vec2 + Word2Vec + LSTM**
  - Late fusion of acoustic and linguistic features.
  - Explores timing and pacing as cognitive indicators.

- **BertImage (ViT + RoBERTa)**
  - Vision Transformer processes spectrogram “images”.
  - RoBERTa encodes transcripts.
  - Cross-attention fusion for deep multi-modal interaction.

- **Final Chosen Architecture**
  - Based on the **BertImage** model.
  - Trained on the **Pitt Corpus**.
  - Focused solely on **binary dementia classification**.

---

## 📈 Evaluation Strategy

- **Metrics Used**
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC
  - Confusion Matrix

- **Validation**
  - Stratified cross-validation
  - Careful dataset splitting to avoid speaker leakage

- **Performance Considerations**
  - Computational cost
  - Model complexity
  - Suitability for real-world clinical deployment

---

## 🔐 Ethics & Bias Considerations

- Datasets exhibit **demographic and linguistic biases**.
- Risk of models learning **age-related speech patterns** rather than dementia itself.
- Models are **not diagnostic tools**.
- In line with **GDPR Article 22**, final decisions must involve qualified professionals.
- The system is designed as **decision support**, not automated diagnosis.

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9+
PyTorch / TensorFlow
CUDA-capable GPU (recommended)
