# Estimating CATE using the Meta-Learners
This repository contains the code and data for the CSE3000 project regarding "Personalizing Treatment for Intensive Care Unit Patients with Acute Respiratory Distress Syndrome" conducted by Juul Schnitzler.

## Overview
Mechanical ventilation is a vital supportive measure for patients with acute respiratory distress syndrome (ARDS) in the intensive care unit (ICU). This project investigates the personalization of positive end-expiratory pressure (PEEP) settings based on patient characteristics using three meta-learning algorithms (S-learner, T-learner, and X-learner) to estimate the conditional average treatment effect (CATE).

### Research Summary
- **Objective**: To determine which ICU patients suffering from ARDS benefit more from high PEEP compared to low PEEP based on patient characteristics.
- **Algorithms Used**: S-learner, T-learner, and X-learner.
- **Datasets**: Simulated data, MIMIC-IV dataset, and randomized trial data.

## Repository Structure
- `External Validation/`: Contains the setup for training and saving models for external validation.
- `MIMIC/`: Contains the setup for evaluating the performance of the meta-learners on the MIMIC-IV dataset, including pre-processing steps.
- `RCT/`:Contains the setup for visualizing the results gained from the external validation.
- `Simulation/`: Contains the different simulations as well as the implementations for the meta-learners (note: different implementations were shown due to the experimental approach of the research)

### Prerequisites
- Python 3.9
- Required Python libraries (specified in `requirements.txt`)

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/JuulSchnitzler/Estimating-CATE-using-the-meta-learners.git
   cd Estimating-CATE-using-the-meta-learners
