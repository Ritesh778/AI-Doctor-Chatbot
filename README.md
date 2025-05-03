# 🤖🩺 AI Doctor Chatbot
 

This is an interactive, fine-tuned medical chatbot built using a LoRA-adapted large language model (LLM) and deployed with Streamlit. It provides informative responses to medical questions and demonstrates efficient training on resource-constrained environments.

---

## 🚀 Project Overview

- **Model Base**: Hugging Face model with LoRA (Low-Rank Adaptation)
- **Training**: Fine-tunes only adapter weights for efficient optimization
- **Interface**: Streamlit app for real-time input and response generation
- **Dataset**: `wiki_medical_terms_llam2_format` (Hugging Face)
- **Use Case**: Educational/Informational chatbot for medical topics

---

## 🧠 Key Features

- LoRA adapter fine-tuning (efficient, low-resource training)
- Interactive chatbot interface via Streamlit
- Handles medical symptom prompts with context-aware answers
- Runs on CPU (no GPU required)
- Tokenization with label alignment for causal LM training

---

## 📦 Installation

```bash
git clone https://github.com/Ritesh778/AI-Doctor-Chatbot.git
cd ai-doctor-chatbot
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the App

```bash
streamlit run app.py
```

If `streamlit` is not recognized, try:

```bash
python -m streamlit run app.py
```

---

## 📋 Example Prompts

- "Abdominal pain with nausea"
- "Persistent headache and dizziness"
- "Joint pain and swelling"

---

## 🛠 Technologies Used

- Python
- Hugging Face Transformers
- Datasets (🤗)
- PEFT (LoRA adapters)
- Streamlit

---

## 🧪 Results

- **Training Time**: Under 6 minutes (CPU)
- **Deployment**: Lightweight, real-time response via Streamlit
- **Accuracy**: Outputs medically coherent, symptom-based answers

---

## 🔐 Disclaimer

This chatbot is intended for educational purposes only and should **not** be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## 👤 Author

**Ritesh Janga**  
Role: Training & Deployment   
- Fine-tuned the model using LoRA  
- Evaluated chatbot output quality  
- Built the full inference and UI pipeline

---

## 📄 License

MIT License. See `LICENSE` file for details.
[Live Demo](https://mnscu-my.sharepoint.com/:f:/g/personal/tw9520gi_go_minnstate_edu/EtrAVhJe9uNCtJuaH2lsgugBGduLfhoeV7Zkyf76Hl_WBA?e=lWpSDE) 
