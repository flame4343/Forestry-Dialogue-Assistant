# 🌲 Forestry Dialogue Assistant

A lightweight, modular dialogue system designed for forestry-related conversational applications. Supports Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and real-time chat via a local web UI.

---

## 📸 Screenshot

![Forestry Dialogue Assistant UI](./1.png)

---

## 📁 Project Structure

```bash
├── 0_test.py                  # Initial test script  
├── 1_deepseek_data.py         # Data preparation  
├── 2_sft_qwen2.5.py           # Supervised fine-tuning  
├── 3_test_sft_qwen2.5.py      # SFT testing  
├── 4_DPO_qwen2.5.py           # DPO training  
├── 5_test_DPO_qwen2.5.py      # DPO testing  
├── chat_server.py             # Backend for dialogue  
├── web_ui.html                # Frontend UI  
├── requirements.txt           # Dependencies  
└── README.md                  # Project documentation
