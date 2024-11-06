# Speech-Synthesis-Model-for-Malayalam-Language-Using-VITS

## Project Overview

This project aims to build a **speech synthesis model** for the Malayalam language using the **VITS (Variational Inference Text-to-Speech)** architecture. The goal is to generate high-quality, natural-sounding Malayalam speech from text using a custom-trained model on the **AI4Bharat Indic Voices** dataset. The trained model is capable of converting **WAV audio to FLAC format** via WebSocket for real-time applications.

---

## Table of Contents

1. [Project Objective](#project-objective)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training](#model-training)
7. [WebSocket Server Setup](#websocket-server-setup)
8. [Evaluation](#evaluation)
9. [Future Work](#future-work)
10. [License](#license)

---

## Project Objective

The objective of this project is to:
- **Train a speech synthesis model** for the Malayalam language.
- Use the **VITS model** to synthesize Malayalam speech from text.
- Stream the generated speech through a **WebSocket** connection, converting WAV audio to FLAC format.

This model leverages the **AI4Bharat Indic Voices** dataset, specifically designed for Indian languages, ensuring that it does not rely on news data but uses conversational speech data.

---

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- Libraries:
    - PyTorch
    - NumPy
    - SciPy
    - TensorFlow
    - Other dependencies listed in `requirements.txt`
- Hardware:
    - A machine with a GPU (recommended for faster training)
    - Sufficient storage (the dataset can take up significant space)


---

## Installation

1. **Clone the VITS repository:**

   Clone the official VITS repository or use your own version as needed:

   ```bash
   git clone https://github.com/your_vits_repo_url.git
2. **Create and activate the virtual environment:**

 ```bash
   python -m venv venv
source venv/bin/activate   # For Linux/MacOS
.\venv\Scripts\activate
```
3.**Install dependencies:**
  
   ```bash  
  pip install -r requirements.txt
```
## Dataset
Language Selection
For this project, the Malayalam language was chosen due to its underrepresentation in existing models. Malayalam is a Dravidian language spoken by millions of people, and having a TTS system for it opens up numerous applications.

---

## Data Acquisition
The dataset used in this project is from the AI4Bharat corpus. It focuses on non-news data to avoid biases often present in news datasets.

---

## Data Sources:
AI4Bharat Indic Voices: Audio data specifically curated for Indic languages.

---

## Data Preprocessing
The raw data files were pre-processed to ensure clean and normalized audio inputs, along with properly formatted transcripts. Pre-processing steps included:

 - Normalizing audio files to a uniform sample rate.
 - Converting audio to a consistent format (WAV).
 - Cleaning and formatting the transcript files to match the audio.
 - Splitting the dataset into training (80%), validation (10%), and testing (10%) sets.
 - Pre-processing scripts:
 - preprocess_data.py: For audio file pre-processing.
 - preprocess_transcripts.py: For cleaning and formatting transcript files.
 - split_data.py: For splitting the data into training, validation, and test sets.

---


## Model Training
**VITS Model Setup** 
-To configure the VITS model for Malayalam, the following steps were followed:

  1.Cloning and setting up the VITS repository.
  2.Modifying the configuration files in the configs directory to suit the Malayalam dataset.
    -Created a custom configuration for the Malayalam language in configs/malayalam_base.json.

---


## Training the Model
**Hyperparameters** : Adjusted learning rates, batch sizes, and other training configurations to fine-tune the model.
**Training Process**: Used a multi-stage training process, starting with basic training and moving to advanced stages with better fine-tuning for better synthesis quality.

 ---

 
**Training script:**
 -train.py: Main script for training the VITS model.
 -Training logs are saved to monitor the model's performance over epochs.

  ---

  
## Model Inference
Once trained, the model can be used for inference, converting Malayalam text to speech. The model uses the inference.ipynb script to generate the speech output.


 ---

 
## Deployment
**WebSocket Integration**
The model is integrated with a WebSocket server that allows for streaming WAV audio to FLAC. The deployment steps involve:

Setting up the server to accept audio requests.
Handling the conversion between WAV and FLAC formats.
Using WebSockets for real-time communication between the client and server.

 ---

 
## Future Improvements
**Model Optimization:** Improve the model's performance by experimenting with different architectures or optimizing hyperparameters.
**Multilingual Support:** Extend the system to support other Indian languages.
**Real-Time Synthesis:** Optimize the model for low-latency, real-time speech synthesis.

