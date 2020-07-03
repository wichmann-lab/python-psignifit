# Readme for the Flask request layer

This request layer handles POST requests containing JSON data strings for use with psignifit-python using [Flask](https://flask.palletsprojects.com) from  e.g. JavaScript experiment libraries such as [jsPsych](https://www.jspsych.org).

The request layer can also be used to simultaneously serve the experiment and uses [axios](https://github.com/axios/axios) injected via jsDelivr CDN to make the POST request (which means one must be connected to the internet if you want to use the request layer to serve your experiment, even if running locally).

Demo is provided.

## Prerequisites

- psignifit toolbox is installed

## Installation

First install Flask:
`pip3 install Flask`

Ensure that the following files are in the root directory of the psignifit-python folder:
- request_layer.py
- request_layer/

Done!

## Usage

### Demo

To demo the default behaviour, run

`python3 request_layer.py`

in a terminal.

By default, the `request_layer.py` script launches a Flask server on `localhost:5000`. If you navigate to `localhost:5000`, the Flask template `/request_layer/templates/demo.html` will be run, containing a script that passes fake behavioural data to the psignifit toolbox, and then receiving the default threshold stimulus value that has been calculated. Note that the demo.html requires access to a CDN to make a POST request, so you need to be connected to the internet. Otherwise, you'll need to pull the axios library into the project.

To view the output, you must open the developer tools on your browser, and check the console log. In the console log, you can see the the response from the psignifit function, along with the data that was passed to the toolbox.

### Getting started

Flask is capable of running this request layer alongside an experiment. The file `/request_layer/templates/demo` would simply need to be altered to run your experiment (for example, by pulling in the jsPsych libraries and adjusting the script accordingly).

Alternatively, you can send POST requests from an external server to the Flask request layer by specifying the URL `localhost:5000/psignifit`, so long as your experiment script is configured to send POST requests and recieve responses as outlined in `/request_layer/templates/demo.html` and `request_layer.py`.

### Launching with the request layer

Flask provides a quickstart guide [here](https://flask.palletsprojects.com/en/1.1.x/quickstart/#) describing how to launch Flask across the network, as opposed to locally (i.e. localhost)). If you wish to launch the request layer to be accessible online, you should consult with your IT department or a specialist as without careful implementation you are vulnerable to cyber attacks.
