from flask import Flask, request, render_template, jsonify
#import json
import numpy as np
import psignifit as ps

app = Flask(__name__)

@app.route('/')
def output():
    return render_template('demo.html')

@app.route("/psignifit", methods=['POST'])
def calculate(): 
    if request.get_json() is None: # some sanitising here - only allow JSON
        # we should also check the the content length here
        abort(404)
    else:
        recieved_data = request.get_json()
        converted_data = recieved_data['faked_data']
        faked_data = np.array([[0.10,   45.0000,   90.0000],
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
                                 [0.90,   90.0000,   90.0000]]);

        options = dict();   # initialize as an empty dictionary
        options['sigmoidName'] = 'norm';   # choose a cumulative Gauss as the sigmoid  
        options['expType']     = '2AFC';   # choose 2-AFC as the experiment type  
                                                                           # this sets the guessing rate to .5 (fixed) and  
                                                                           # fits the rest of the parameters
        options['threshPC']    = 0.9;
        result_upper = ps.psignifit(converted_data,options);
        options['threshPC']    = 0.7;
        result_lower = ps.psignifit(converted_data,options);

        # example display all fit values as strings
        # fit_info = ''
        # for item in result['Fit']:
        #	fit_info += str(item) + '\n'
        # return fit_info
        
        thresholds = [result_upper['Fit'][0],result_lower['Fit'][0]]

        #upper_thresh = str(result_upper['Fit'][0]) 
        #lower_thresh = str(result_lower['Fit'][0])

        return jsonify(thresholds)

if __name__ == "__main__":
	app.run()
