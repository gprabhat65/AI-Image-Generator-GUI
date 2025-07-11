# 🎨 Stable Bud — AI Image Generator GUI

Stable Bud is a desktop application built with **Tkinter** and **CustomTkinter** that uses **Stable Diffusion** to generate stunning images from text prompts. Powered by Hugging Face’s `diffusers` library and PyTorch, this tool allows users to enter prompts and instantly generate high-quality AI images in a user-friendly interface.

---

## 📌 Features

- 💬 Enter any creative prompt
- 🖼️ View AI-generated images live in the app
- 💾 Save generated images as `GeneratedImage.png`
- ⚡ GPU-accelerated using CUDA for fast inference
- 🌙 Dark mode UI with CustomTkinter

---

## 🧰 Technologies Used

- Python 3.10+
- [Tkinter](https://docs.python.org/3/library/tkinter.html)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [PyTorch](https://pytorch.org/)
- PIL (Pillow)

---

## 🖼️ Sample Screenshot

> _You can include a screenshot of your app interface here_


---

## ⚙️ Installation & Setup

### ✅ Prerequisites

- Python 3.10 or later
- A CUDA-compatible GPU (for fast inference)
- Hugging Face Account and access to `CompVis/stable-diffusion-v1-4`

---

### 📦 Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/stable-bud.git
cd stable-bud


python -m venv venv
venv\Scripts\activate     # Windows
# OR
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

# authtoken.py
auth_token = "your-huggingface-access-token"

stable-bud/
├── app.py                  # Main application script
├── authtoken.py            # Hugging Face token file
├── requirements.txt        # All dependencies
├── GeneratedImage.png      # Output image
├── README.md               # Project documentation
