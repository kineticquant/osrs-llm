# OSRS LLM
A highly-performant, lightweight generative AI LLM trained on all OSRS content. Phi-3.5-mini-instruct model leveraged in this project. Phi-3.5-mini is a lightweight, state-of-the-art open model built upon datasets used for Phi-3 - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data. The model belongs to the Phi-3 model family and supports 128K token context length. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning, proximal policy optimization, and direct preference optimization to ensure precise instruction adherence and robust safety measures.

This project comes with a simple Streamlit UI configured for ease of training and progress tracking.

This repository also contains all raw and cleaned data, as well as binary arrow file with built-in training instructions already prepared. This data is up to date as of [1/2/2026].

## 🚀 Getting started

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

## 🛠 Training & Data Pipeline
This project includes a fully automated data pipeline. To conduct your own training, which includes a full wipe and reload of the data processing layers, run the following:
```bash
streamlit run pipelines/data.py
```
Once successful, run the actual training process:
```bash
python pipelines/train.py
```
[!CAUTION]
- Ensure your environment is adequately configured and everything was installed correctly before conducting training. To do so, run the env_check.py utility.
- Running the data pipeline will overwrite existing processed data. Ensure you have backed up any custom datasets before running.


## 🔍 Utilities & Tools
#### Environment Check
Run the environment check utility to ensure your environment is correctly configured to handle Torch and CUDA. Modify requirements.txt accordingly based on your GPU version if applicable.
```bash
python env_check.py
```

#### Inspecting Data
To verify the integrity of the processed .arrow file(s), use the inspection utility:
```bash
python inspect_arrow.py
```

#### Quick File Count
To quickly check the number of raw data files available while data extract is underway and in case the Streamlit UI is unresponsive, use the following:
```bash
./count
```

## 🖥️ The UI
![Pipeline UI]https://github.com/kineticquant/osrs-llm/blob/main/img/pipeline-in-progress.png?raw=true

## Acknowledgments
Base Model: [Microsoft Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
Data Source: All OSRS-related content and community-driven datasets, as well as the OSRS Wiki.