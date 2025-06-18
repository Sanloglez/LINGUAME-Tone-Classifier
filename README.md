---
title: Linguame
sdk: gradio
license: apache-2.0
tags:
- nlp
- communication
- tone-analysis
- text-classification
- email
- emotional-intelligence
- linguistics
- gradio
short_description: Emotional & Cognitive Analyzer for Professional Emails
sdk_version: 5.34.1
---

# 🧠 LINGUAME™  
**Professional Communication Style Classifier**

> Understand the emotional tone of your message — in seconds.

---

## 🔍 What is it?

**LINGUAME™** is a smart tool that analyzes the tone of professional texts and classifies them into **10 real-world communication styles**.  
It’s designed to support professionals, recruiters, coaches, and students in crafting more effective written communication.

---
# LINGUAME™ — Tone Classifier

Try it live on [🤗 Hugging Face Spaces](https://huggingface.co/spaces/tu-usuario/tu-space)  
[Demo](https://huggingface.co/spaces/tu-usuario/tu-space/badge.svg)
---

## 💡 What can it do?

- 🏷️ Detect the **communication style** of an English message
- 🧠 Provide an **interpretation** of the tone
- 💬 Suggest how to **improve** clarity, empathy or assertiveness
- 📊 Track usage and prediction statistics
- 📝 Collect real-time feedback from users
- ⬇️ Export user feedback for training and refinement

---

## 🧠 Style Labels Detected

- Professional Clear  
- Empathetic Mature  
- Assertive Critical  
- Passive Aggressive  
- Insecure Submissive  
- Controlling  
- Evasive  
- Veiled Reproach  
- Overjustification  
- Manipulative Guilt  

Each label includes:
- an **interpretation**
- and a **suggestion for improvement**

---

## 🛠️ How does it work?

- Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to embed the message  
- Feeds the embedding into a **multiclass logistic regression classifier**
- Returns the **most probable label + explanation + suggestion**

---

## ✨ Live Features

- 🔍 Instant tone classification
- 🧠 Interpretation & coaching tip
- 📊 Prediction stats per session
- 📝 Submit feedback on label accuracy
- ⬇️ Download feedback in `.csv` format

---

## 🚧 Coming Soon

- 🌐 Multilingual support
- 🔄 Fine-tuning with real user feedback
- 🧪 Private mode for organizations
- 📱 Mobile-optimized layout

---

## 👩‍💻 Created by

**Sandra López González**  
Data Scientist · Linguist · Content Creator 

[Portfolio](https://huggingface.co/sandylopg) · [LinkedIn](https://www.linkedin.com/in/sandralopezglez89/) · [GitHub](https://github.com/Sanloglez)

---

## 💬 License

All rights reserved. This code is for educational and demonstrative purposes only. 
Reproduction, commercial use, or redistribution is not permitted without explicit permission.


