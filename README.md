# Network Traffic Anomaly Analysis

This project analyzes network traffic anomalies using machine learning,
with a focus on understanding detected anomalies and reducing false-positive alerts.

---

## Project Objectives

* Learn normal network traffic behavior
* Detect anomalous traffic patterns
* Analyze feature deviations responsible for anomalies
* Reduce false alarms caused by normal traffic fluctuations
* Provide a simple monitoring and visualization interface

---

## Dataset

The project uses the **UNSW-NB15 dataset**, which contains network traffic flow records with multiple traffic features.

This dataset is widely used in network intrusion detection research and provides realistic traffic patterns for anomaly analysis.

Dataset source:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

---

## Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Streamlit (for visualization)

---

## Project Structure

```
network-traffic-anomaly-analysis/

│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding_and_windowing.ipynb
│   └── 02_baseline_anomaly_detection.ipynb
|
├── README.md
└── requirements.txt
```
---

## Project Status

Data pre-processing and window-based aggregation pipeline is finalized
The focus is on building a structured pipeline for anomaly detection and analysis in network traffic.