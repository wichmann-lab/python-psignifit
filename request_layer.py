from flask import Flask, request, render_template, jsonify # import Flask and Flask dependencies
import argparse # for dealing with command line argument inputs
# import psignifit resources
import numpy as np
import psignifit as ps
import json

# create options to change host and port
parser = argparse.ArgumentParser()
parser.add_argument("-H", "--host", help="specify host")
parser.add_argument("-P", "--port", help="specify port")
args = parser.parse_args()
if args.host:
    print(" * changing host from default to {}".format(args.host))
else:
    print(" * running with default host 127.0.0.1")
if args.port:
    print(" * changing port from default to {}".format(args.port))
else:
    print(" * running with default port 5000")

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
        payload = request.get_json(); # pull the data out of the POST request 
        data = payload['data']; # format of the POST comes in as a dict, so just select the array
        options = payload['options'];

        # then run psignifit
        result = ps.psignifit(data,options);

        # then return the result as a JSON object
        return jsonify(result) 

if __name__ == "__main__":
	app.run(host=args.host if args.host else None, port=args.port if args.port else None)
