# Extracting Food Pantry Hours from Web Data using LLMs and RLHF

## 📌 Overview
Food pantry operating hours are often embedded in unstructured and noisy HTML content, making automated extraction challenging. This project develops a robust Large Language Model (LLM)-based pipeline to extract structured operating hours, with a specific focus on distinguishing **Food Pantry Hours** from other types of schedules.

We combine supervised fine-tuning (SFT), preference optimization (DPO), and reinforcement learning (PPO) to improve extraction accuracy and reduce hallucinations in real-world deployment settings.

---

## 🎯 Problem Statement
- Webpages contain **messy, inconsistent HTML text**
- Multiple hour types exist (e.g., *Food Hours*, *Office Hours*, *Donation Hours*)
- LLMs tend to **hallucinate or mix different hour types**
- Ground truth is often **unavailable during deployment**

This project addresses these challenges by combining structured evaluation and ensemble decision-making.

---

## 💡 Key Contributions
- 🔹 Developed a **multi-stage LLM pipeline** using SFT, DPO, and PPO
- 🔹 Designed a **structured JSON extraction framework** for evaluation
- 🔹 Built a **tuple-level accuracy metric** for comparing predicted vs true schedules
- 🔹 Implemented a **majority voting ensemble** for deployment without ground truth
- 🔹 Proposed a **monitoring strategy** for post-deployment performance tracking

---

## ⚙️ Methodology

### Pipeline
