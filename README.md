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

**Setup:**
The developed code was tested with Python 3.7.

To reproduce the code, clone the repository and run the following command:

- Download the required dependencies:
````
pip install -r requirement.txt
````

- To run the code:
````
python main.py
````

**Model Checkpoints:**
The weights of the model after training is stored in Checkpoints folder that can be used in future for further prediction. In short, it is done to avoid training of the model each time.

**Model Selection and Evaluation:** 

The accuracy and MSE error metric is set as baseline and 2 models are compared namely 'Random Forest Regressor' and 'Linear Regressor'


| Model | Accuracy  |  MSE_Error (in hours)
|-----|-----|---    |
| `Random Forest Regressor`| 46% |0.64 |
| `Linear Regressor`| 48.78% |0.61 |

**Usage of the provided feature:** 

Using this ML model, clients can analyse and anticipate the time at which the shock can occur for the given segments and type. This problem statement helps and supports the clients to have a better understanding over the Shock-interval of different machines.

**Future Scope:** 

- Implement data augmentation techniques to extract better features and compute feature importance functionality.
- Try different ML regression models to analyse the performance.
- Try Deep Nueral Networks like Multi Layer Perceptron (MLP) and LSTM's. 