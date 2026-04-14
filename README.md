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

**JSONLLM**

- Takes hours in free text format as input and produces JSON structures for better parsing
- Uses Llama 3.2 3B and 8B parameters
- Benchmarked against Prompt engineering with 2 examples in Prompt
- Supervised finetuning using LoRA 
- Cross-validated with hyperparameters rank and alpha of LoRA

  Example:
  Input:
<img width="135" height="39" alt="image" src="https://github.com/user-attachments/assets/f005cfef-09c0-4aa0-b8e3-cd0fd7447c50" />



  Output:

  
<img width="272" height="51" alt="image" src="https://github.com/user-attachments/assets/05b39b74-d926-4cf2-97cd-bf002421263a" />

Performance Metric: 
<img width="187" height="174" alt="image" src="https://github.com/user-attachments/assets/0f5340a8-37af-4091-94ce-2ce280a0f675" />


<img width="377" height="141" alt="image" src="https://github.com/user-attachments/assets/0d7833fc-a7e1-439a-8b61-58e6bf4a14fa" />

**Results**
**JSONLLM
Prompting Results**

<img width="536" height="136" alt="image" src="https://github.com/user-attachments/assets/27670d49-065c-4e42-b53e-c10d08cca2e9" />

- Lower temperature yields better accuracy.
Lower temperature ~ Lower variance

**Fine-tuning Results**
- The fine-tuned Qwen 500M model produces comparable results to Llama 8 B Prompting. 
- For Llama 8B, prompting performance is high enough, and room for improvement is small 
- Based on efficiency and performance, Llama 3B is the overall top performer




<img width="533" height="116" alt="image" src="https://github.com/user-attachments/assets/e1371bd5-2ab1-4055-87a1-a43f5c82173c" />

  

**HourLLM**
- Scrapper scrapes the contents of the food pantry webpages and collects HTML of the relevant pages
- Supervised Finetuned (SFT) LLM takes the HTML text as context and the actual hours as correct labels
- LoRA finetuning is appliedwith crossvalidation using hyperparameters ( alpha and rank)
- Reinforcement learning with Human Feedback (RLHF) and Direct Preference Optimization (DPO) was used to further finetune the SFT model

  
**SFT Model Performance for Llama 3B**


  <img width="327" height="260" alt="image" src="https://github.com/user-attachments/assets/dcd45e5f-431b-4df8-b289-f131dec6b3bb" />


