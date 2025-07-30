# Quora Question Pairs

This repository contains code and resources for working with the [Quora Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs). The main goal is to identify whether a pair of questions asked on Quora are semantically similar (i.e., duplicates).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Overview

Duplicate questions can clutter question-and-answer platforms. Detecting semantic similarity helps improve user experience and search results. This project focuses on building models to classify whether two questions are duplicates.

## Dataset

The [Quora Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs/data) contains pairs of questions and a label indicating if they are duplicates.

- `question1`, `question2`: The question texts.
- `is_duplicate`: 1 if the questions are duplicates, 0 otherwise.

## Features

- Data preprocessing and cleaning
- Feature engineering (text similarity metrics, Vectorization, etc.)
- Model training (deep learning approaches)
- Evaluation scripts

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/vky6366/Quora-Question-Pairs.git
    cd Quora-Question-Pairs
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset:**
    - Download from [Kaggle Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs/data).
    - Place the dataset files in the appropriate directory (e.g., `data/`).

## Usage

Scripts and notebooks are provided for exploring the dataset, training models, and making predictions.

```bash
streamlit run main.py
```

Refer to the notebooks (`.ipynb`) for step-by-step tutorials.

## Project Structure

```
Quora-Question-Pairs/
├── data/                 
├── Preprocessing and training/                             
├── Utils/               
├── requirements.txt      
├── README.md    
├── main.py   

```

---

**Author:** [vky6366](https://github.com/vky6366)