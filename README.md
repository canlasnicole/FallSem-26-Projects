---
title: CICD Gradio Assignment
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Automated CI/CD Pipeline with GitHub Actions & Hugging Face

This project sets up an automated CI/CD pipeline. Every time code is pushed to this GitHub repository, GitHub Actions automatically deploys the update to Hugging Face Spaces.

---

## How It Works

1. Local Setup: Code changes are made locally and managed with Git.
2. GitHub Push: Code is pushed to the main branch on GitHub.
3. GitHub Actions: .github/workflows/deploy.yml runs automatically when a push happens.
4. Hugging Face Deploy: GitHub Actions uses the HF_TOKEN secret to push changes directly to Hugging Face Spaces.

---

## Project Structure

```text
my-gradio-app/
├── .github/
│   └── workflows/
│       └── deploy.yml    # GitHub Actions workflow script
├── app.py                # Main Gradio application code
├── README.md             # Project documentation & HF config
└── requirements.txt      # Python dependencies
