# Performance-Fatigue-monitoring-using-wearable-IMUs

Performance Fatigue Analysis in Manual Handling Tasks

Work-related musculoskeletal disorders (WMSDs) are often caused by prolonged physical exertion and poor movement coordination. This project investigates performance fatigue during manual handling tasks (MHT) using wearable inertial measurement units (IMUs). The study evaluates joint kinematics, coordination, and fatigue indicators throughout task repetitions.

## 📄 Publication

This project resulted in a peer-reviewed open-access paper:

> Bonakdar, A., et al. (2024). *Fatigue assessment in multi-activity manual handling tasks through joint angle monitoring with wearable sensors*. **Biomedical Signal Processing and Control**.  
> 🔗 [Read the full article here](https://doi.org/10.1016/j.bspc.2024.107398)

## Study Overview
Eight participants performed repetitive lifting, carrying, and lowering tasks while self-reporting fatigue using a Rating of Perceived Exertion (RPE) scale. Fatigue progression was analyzed using:
  - Joint angle patterns 
  - Peak excursion and variability 
  - Coordinative changes between adjacent joints via Mutual Information (MI) theory

Participants:

8 healthy adults (each performed 2 trials)

Task:
- Repetitive manual handling including lifting, carrying, and lowering
- Continued until self-reported fatigue reached high levels

Instrumentation:
- IMUs: Xsens MTw Awinda 
- Sampling Frequency: 100 Hz
- Sensor Placement:
      - Sternum
      - Sacrum
      - Right Shoulder
      - Right Elbow
      - Right Pelvis
      - Right Thigh
      - Right Shank


Signal Analysis:
  - Time-series joint angles analyzed in the sagittal plane
  - Metrics: Mean, peak joint excursions, and standard deviation
  - Coordinative analysis using Mutual Information (MI) between adjacent joints
