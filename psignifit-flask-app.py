from flask import Flask, request
import json
import numpy as np
import psignifit as ps

app = Flask(__name__)

# then we need to process it
@app.route("/")
def fitting():

	# so we get data here
	data = np.array([[0.0010,   45.0000,   90.0000],
				 [0.15,   50.0000,   90.0000],
				 [0.20,   44.0000,   90.0000],
				 [0.25,   44.0000,   90.0000],
				 [0.30,   52.0000,   90.0000],
				 [0.35,   53.0000,   90.0000],
				 [0.40,   62.0000,   90.0000],
				 [0.45,   64.0000,   90.0000],
				 [0.50,   76.0000,   90.0000],
				 [0.60,   79.0000,   90.0000],
				 [0.70,   88.0000,   90.0000],
				 [0.80,   90.0000,   90.0000],
				 [0.1,   90.0000,   90.0000]]);

	options = dict();   # initialize as an empty dictionary
	options['sigmoidName'] = 'norm';   # choose a cumulative Gauss as the sigmoid  
	options['expType']     = '2AFC';   # choose 2-AFC as the experiment type  
									   # this sets the guessing rate to .5 (fixed) and  
									   # fits the rest of the parameters
	options['threshPC']    = 0.9;
	result_upper = ps.psignifit(data,options);
	options['threshPC']    = 0.7;
	result_lower = ps.psignifit(data,options);

	# example display all fit values as strings
	# fit_info = ''
	# for item in result['Fit']:
	#	fit_info += str(item) + '\n'
	# return fit_info

	upper_thresh = str(result_upper['Fit'][0]) 
	lower_thresh = str(result_lower['Fit'][0])

	return upper_thresh, lower_thresh 

if __name__ == "__main__":
	app.run()
