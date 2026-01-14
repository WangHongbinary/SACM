## SACM: SEEG-Audio Contrastive Matching for Chinese Speech Decoding

## Overview

This repository contains the official implementation of the SACM framework for Chinese speech decoding, as described in our paper "SACM: SEEG-Audio Contrastive Matching for Chinese Speech Decoding".

## Dataset

The experiment uses a corpus of 48 Mandarin Chinese monosyllabic words. The word list can be found in `A_Preprocessing/labels.py`. The dataset is available upon request.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/WangHongbinary/SACM.git
cd SACM
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

Process SEEG and audio data:

```bash
python A_Preprocessing/preprocess_SEEG.py
```

Configuration parameters can be adjusted in `A_Preprocessing/preprocess_config.py`.

### Feature Extraction

Extract audio features:

```bash
python C_Decoding/dataset/wav_SEEG.py
```

### Model Training & Evaluation

#### Speech Detection

Run the following script to train and evaluate the speech detection model:

```bash
cd B_Detect/run
bash run_exp.sh
```

Results will be saved as CSV files in the log directory.

#### Speech Decoding

Execute the following to perform speech decoding:

```bash
cd C_Decoding/run
bash run_exp.sh
```

Results will be saved as CSV files in the log directory.

### Additional Analysis

The repository includes additional code for:

- Speech decoding on VocalMind: `D_Decoding_VocalMind/`
- Statistical analysis: `E_Stats_test/`
- Figure generation: `F_Figure_code/`

## Project Structure

```
.
├── A_Preprocessing/      # Data preprocessing scripts
├── B_Detect/             # Speech detection implementation on HUST-MIND
├── C_Decoding/           # Speech decoding implementation on HUST-MIND
├── D_Decoding_VocalMind/ # Speech decoding implementation on VocalMind
├── E_Stats_test/         # Statistical analysis
└── F_Figure_code/        # Figure generation scripts
```

## Citation

Cite our paper:

```
[Citation information will be added upon publication]
```
