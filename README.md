***ForkOn Coding Challenge:***
Author: Vishal Sowrirajan

Dataset Credits/Owner: https://forkon.de/

***Problem Statement:*** 
Given the Segment, type, Time_interval_of_shock, Time_interval_of_operation and Shock_Intensity, our ML model can classify the intensity level of the shock (1, 2 or 3)

***Preprocessing:***
- Data Parsing
- Null value handling using Pandas
- Timestamp conversion
- Encoding categorical values to Numerical value

***Model Selection:***
We use accuracy as our baseline and compare 2 models namely 'Decision Tree' and 'Random Forest'

***Model Evaluation:***
The model performance is tested against Accuracy as the metric precisely estimates our model's performance for the given dataset.

***Setup:***
The developed code was tested with Python 3.7.

To reproduce the code, run the following command:

- Download the required dependencies:
````
pip install -r requirement.txt
````

- To run the code:
````
python main.py
````

**Feature Importance**: We calculate the importance of different features that mainly contribute to our final classification score.

![Feature Importance](results/Feature_importance-level.png)

**Confusion Matrix**: Although the dataset is highly imbalanced, we can clearly see the our model is able to classify the Shock intensity with 95% accuracy.

![Confusion Matrix](results/Confusion_matrix.png)

**Pie chart**: 

![Pie chart](https://github.com/VishalSowrirajan/ForkOn/blob/main/results/Segment%20vs%20Shocks.png)

***Model Evaluation:***

| Model | Accuracy  |
|-----|-----|
| `Random Forest Classifier`| 95% |
| `Decision Tree Classifier`| 93% |