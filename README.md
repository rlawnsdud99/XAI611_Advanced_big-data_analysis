### XAI611 Course Project Description

---

#### I. Project Title
**Optimizing Subject Independent Classification Performance in class imbalanced, EMG-Based Gesture Recognition**

---

#### II. Project Introduction

##### Objective
The primary goal is to enhance the classification accuracy of EMG signals for static hand gestures, specifically focusing on Class 1 to Class 7. We'll employ advanced neural network architectures and machine learning techniques for this. Class 0 can be used as a baseline or as deemed appropriate.

##### Motivation
Improving the accuracy of EMG-based gesture recognition has significant implications for BCI applications, including assistive technologies and human-computer interaction.

---

#### III. Dataset Description
The dataset contains raw EMG data from 36 subjects performing static hand gestures(Open data). Each subject executed two series of 6 or 7 basic gestures. Each gesture lasted for 3 seconds, with a 3-second pause between gestures. Data was collected using a MYO Thalmic bracelet equipped with eight sensors.

- **Columns:**
  1) Time - Time in ms
  2-9) Channel - Eight EMG channels from MYO Thalmic bracelet
  10) Class - Gesture labels:
    - 0: Unmarked data
    - 1: Hand at rest
    - 2: Hand clenched in a fist
    - 3: Wrist flexion
    - 4: Wrist extension
    - 5: Radial deviations
    - 6: Ulnar deviations
    - 7: Extended palm (not performed by all subjects)
    
- **Additional Column:**
  - Label: Refers to the subject who performed the experiment
  
---

#### Dataset Download
You can download the `.csv` files for the dataset from [this Google Drive link](https://drive.google.com/file/d/1xKA2EiEwc5dOqz3N_Ssdo1TAI2VI1ASJ/view?usp=sharing).

#### Requirement
`conda 23.3.0`, `Python 3.11.4`, `torch 2.0.1+cu117`

or use `jyk.yaml` the copy of my own env, and make conda env.

---

