from flask import Flask, request, render_template, jsonify
import numpy as np
import psignifit as ps

app = Flask(__name__, root_path='request_layer/') # make Flask look in the 'request_layer/' dir for templates etc, instead of in the root dir

@app.route('/')
def output(): # serve the demo script - can delete if not needed or point at your experiment file
    return render_template('demo.html')

@app.route("/psignifit", methods=['POST'])
def calculate():
    if request.get_json() is None: # abort if not JSON (sanitise input)
        abort(400)
    elif request.content_length is not None and request.content_length > 1024: # let's also limit the size of permitted payloads to 1 KB
        abort(413)
    else: # if JSON, continue
        recieved_data = request.get_json() # pull the data out of the POST request 
        converted_data = recieved_data['faked_data'] # format of the POST comes in as a dict, so just select the array

        # set up psignifit with some standard options
        options = dict();
        options['sigmoidName'] = 'norm';
        options['expType']     = '2AFC';
        # add any options here

        # then run psignifit
        result = ps.psignifit(converted_data,options);

        # example: pull threshold value and return it
        threshold = result['Fit'][0]
        response = jsonify(threshold) # we must first convert it to JSON
        return response # then return the JSON for the axios request to pick up
        # we could do the above in one line if we wanted: `return jsonify(result['Fit'][0])`

if __name__ == "__main__":
	app.run()
