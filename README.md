Project 1 – Higgs Boson Challenge

Creator: Kaan Okumuş, Diana Zanoaga, Roxane Burri


### Repository description

- `src/` directory contains :
	- implementations.py : It is used as a library to provide functions about machine learning implementations. It also includes functions about getting statistics of the model such as accuracy and losses. It is called both by run.py and project1.pynb.
	- run.py : Main code that implements all necessary parts for machine learning project with the optimized solution.
	- helpers_data.py : It is used as a library to provide functions of data preprocessing and feature engineering. It is called both by run.py and project1.pynb.
	- proj1_helpers.py : It is used as a library to provide extra functions, which are load_csv_data, split_data, predict_labels, predict_labels_LogReg, create_csv_submission.
	- project1.ipynb : All the related studying for this project can be seen in this jupiter notebook. It includes parts about Exploratory Data Analysis, Feature Engineering, Data Preprocessing, Impelementation of All Machine Learning Models that we tried.
	
- `Data/` directory contains the data file `test.csv`, `train.csv` and `solution.csv` files.
- `report/` directory for report.


### Running
Add `test.csv` and `train.csv`  in `Data/`folder, because it take too much space

Execute the `run.py` file as :

```
python3 run.py
```
`test.csv`, `train.csv` must be in the `Data/` folder


### Output

The data result are in `solution.csv` file, in `Data/`folder.
