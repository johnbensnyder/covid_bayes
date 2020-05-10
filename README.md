# covid_bayes

Simple tool to display Bayesian time series model of observed versus expected mortality rates by state, or US as a whole.

### Setup

conda create -n prophet -c conda-forge python=3.7 ipykernel fbprophet matplotlib pandas numpy seaborn

### Run

conda activate prophet
python covid_plot.py --state Texas --output

State is optional. If not included, default is all United States.

Output flag will write plot to file, using name [state].png.
