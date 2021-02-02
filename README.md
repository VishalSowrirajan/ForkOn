**ForkOn Coding Challenge:**
Author: Vishal Sowrirajan

Dataset Credits/Ownership: https://forkon.de/

**Problem Formulation:** 
Given the Segment, type, Time_interval_of_operation, Shock_level and Shock_Intensity, the ML model tries to predict the time of shock from the begin time.

**Preprocessing:**

- Data Parsing
- Null value handling
- Conversion of time stamp into hours elapsed
- Encoding categorical values to Numerical value


| Model | Accuracy  |  MSE_Error (in hours)
|-----|-----|---    |
| `Random Forest Regressor`| 46% |0.64 |
| `Linear Regressor`| 48.78% |0.61 |