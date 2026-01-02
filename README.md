# osrs-llm
A highly-performant, lightweight generative AI LLM trained on all OSRS content. Phi-3.5-mini-instruct model leveraged in this project. Phi-3.5-mini is a lightweight, state-of-the-art open model built upon datasets used for Phi-3 - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data. The model belongs to the Phi-3 model family and supports 128K token context length. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning, proximal policy optimization, and direct preference optimization to ensure precise instruction adherence and robust safety measures.

This repository comes with all raw and cleaned data, as well as binary arrow file with built-in training instructions already prepared. This data is up to date as of [1/2/2026].

### 🚀 Getting started

#### Installation
Install the necessary dependencies using the provided requirements file:
```bash
pip install -r requirements.txt
```

#### 📦 Getting the base model:
The models directory will contain the base Phi-3.5-mini-instruct model used for training. You can acquire the model using one of the following methods:

##### Option 1: Using hf
```bash
hf download microsoft/Phi-3.5-mini-instruct --local-dir models/phi-3.5-mini
```

##### Option 2: Using the utility script
```bash
python download_model.py
```

### 🛠 Training & Data Pipeline
This project includes a fully automated data pipeline. To conduct your own training—which includes a full wipe and reload of the data processing layers—use the Streamlit dashboard:

### 🔍 Utilities & Tools

env_check.py


To conduct your own training, which will do a full wipe and reload of the data pipelines, run "streamlit run pipelines/data.py" (screenshot). Ensure env_check.py has run and everything has passed before doing so!

inspect_arrow.py


Quick File Count
If you need to quickly check the number of raw data files available, you can use the following alias:

[!CAUTION]
Running the data pipeline will overwrite existing processed data. Ensure you have backed up any custom datasets before running.

```bash
#!/bin/bash
echo "Hello, world!"
# You can include multi-line scripts here.
```

count alias:
#!/bin/bash
cd data/raw
ls | wc -l

ADD SCREENSHOTS


### 📂 Project Structure
models/ — Contains the base and fine-tuned Phi-3.5 models.
data/ — Contains raw and processed OSRS content (as of [DATE]).
pipelines/ — Logic for data ingestion and transformation.
env_check.py — Hardware and dependency verification.
download_model.py — Automated model retrieval.
Acknowledgments
Base Model: Microsoft Phi-3.5-mini-instruct
Data Source: All OSRS-related content and community-driven datasets.