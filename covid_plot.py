import argparse
import pandas as pd
from fbprophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

def get_data(data_source, state):
	df = pd.read_csv(data_source)
	# fix data column
	df['Week Ending Date'] = pd.to_datetime(df['Week Ending Date'])
	# just get nationwide totals
	df = df.loc[(df['State']==state) & \
            (df['Outcome']=='All causes') & \
            (df['Type']=='Predicted (weighted)')]
	return df

def gen_prediction(df):
	# fit time series curve based on previous data
	df_previous = df.loc[df['Week Ending Date']<"2020-03-21"][['Week Ending Date', 'Observed Number']]
	df_previous.columns = ['ds', 'y']
	m = Prophet()
	m.fit(df_previous)
	future = m.make_future_dataframe(periods=10, freq='W')
	forecast = m.predict(future)
	return forecast

def format_results(df, forecast, state):
	# format observations
	df_pred = df.loc[df['Week Ending Date']>="2020-01-01"][['Week Ending Date', 'Observed Number']]
	df_pred.columns = ['ds', 'actual']
	# drop last two weeks because data is incomplete
	df_pred = df_pred.iloc[0:df_pred.shape[0]-2]
	df_pred['ds'] = df_pred['ds'] + pd.to_timedelta(1, unit='d')
	df_merge = df_pred.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], how='inner', on='ds')

	end_date = list(df_merge['ds'])[-1]

	title = "Bayesian time series forecast for expected {0} weekly deaths vs. observed".format(state)
	caption = '\n'.join(["95% C.I. for observed deaths above predicted",
                  "for period from 3/15 to {0}/{1}: {2:.2f}-{3:.2f}".format(end_date.month, end_date.day,
                                                                            (df_merge['actual'] - df_merge['yhat_upper']).sum(),
                                                                                  (df_merge['actual'] - df_merge['yhat_lower']).sum()),
                  "With maximum a posteriori estimate {0:.2f}".format((df_merge['actual'] - df_merge['yhat']).sum())])
	return df_pred, df_merge, title, caption

def plot(forecast, df_pred, df_merge, title, caption, output, state):
	# Plot result

	fig, ax = plt.subplots(figsize=(15, 10))
	ax.plot(forecast['ds'], forecast['yhat'])
	ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], facecolor=(0,0,1,.4), edgecolor=(0,0,0,.5), label="Baseline")
	ax.plot(df_pred['ds'], df_pred['actual'], color=(1,.5,.0))
	ax.fill_between(df_merge['ds'], df_merge['yhat'], df_merge['actual'], facecolor=(1,.5,.0,.4), edgecolor=(0,0,0,.5), label="Observed")
	ax.legend()
	props = dict(boxstyle='round', alpha=0.5)
	# place a text box in upper left in axes coords
	ax.text(0.05, 0.9, caption, transform=ax.transAxes, fontsize=14,
        	verticalalignment='top', bbox=props)
	plt.title(title)
	if output:
		plt.savefig(state)
	plt.show()

def main(state, output):
	data_source = "https://data.cdc.gov/api/views/xkkf-xrst/rows.csv?accessType=DOWNLOAD&bom=true&format=true%20target="	
	df = get_data(data_source, state)
	forecast = gen_prediction(df)
	df_pred, df_merge, title, caption = format_results(df, forecast, state)
	plot(forecast, df_pred, df_merge, title, caption, output, state)

def add_cli_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--state', default='United States')
	parser.add_argument('-o', '--output', action='store_true')
	return parser

if __name__=='__main__':
	cmdline = add_cli_args()
	FLAGS, unknown_args = cmdline.parse_known_args()	
	main(FLAGS.state, FLAGS.output)
