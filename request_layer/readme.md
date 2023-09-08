# Readme for the Flask request layer

This request layer allows one to recieve data for processing with the psignifit-python toolbox. It can be used to:
1) run an experiment and process the resulting data; or
2) recieve data from an external server and process the resulting data; and in either case,
3) processed data can be returned locally for further experimentation or sent to an external server

## Table of Contents

* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)
  * [Demo](#demo)
  * [Getting started](#getting-started)
  * [Deploying](#deploying)
	  
## Overview
The request layer handles POST requests containing JSON data strings for use with psignifit-python using [Flask](https://flask.palletsprojects.com). This means one can use an experiment library such as [jsPsych](https://www.jspsych.org) to generate data, and send it to this request layer for processing. So long as the data can be converted into a JSON object, it can be handled by this request layer.

The experiment can be served externally or it can be served locally by the request layer. A demo is provided.

Note:
The request layer uses [axios](https://github.com/axios/axios) injected via jsDelivr CDN to make the POST request. This means one must be connected to the internet if you want to use the request layer to serve your experiment, even if you are runnning locally.

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

### Quickstart

Run the command:
`python3 request_layer.py`

By default, this will launch a Flask server on `localhost:5000`.

To change the host use the flags `-H` or `--host`. To change the port use the flags `-P` or `--port`.

For example to open the server to the network, and change the port to 8000:
`python3 request_layer.py -H 0.0.0.0 -P 8000`

### Demo

To demo the default behaviour, run

`python3 request_layer.py`

in a terminal.

By default, the `request_layer.py` script launches a Flask server on `localhost:5000`. If you navigate to `localhost:5000` in your browser, the Flask template `/request_layer/templates/demo.html` will be run, containing a script that passes fake behavioural data to the psignifit toolbox, and then receiving the default threshold stimulus value that has been calculated. Note that the demo.html requires access to a CDN to make a POST request, so you need to be connected to the internet. Otherwise, you'll need to pull the [axios](https://github.com/axios/axios) library into the project.

To view the output, you must open the developer tools on your browser, and check the console log. In the console log, you can see the the response from the psignifit function, along with the data that was passed to the toolbox.

### Getting started

Flask is capable of running this request layer alongside an experiment. The file `/request_layer/templates/demo` would simply need to be altered to run your experiment (for example, by pulling in the jsPsych libraries and adjusting the script accordingly).

Alternatively, you can send POST requests from an external server to the Flask request layer by specifying the URL `localhost:5000/psignifit`, so long as your experiment script is configured to send POST requests and recieve responses as outlined in `/request_layer/templates/demo.html` and `request_layer.py`.

### Deploying 

Flask provides a quickstart guide [here](https://flask.palletsprojects.com/en/1.1.x/quickstart/#) describing how to launch Flask across the network, as opposed to locally (i.e. localhost)). If you wish to launch the request layer to be accessible online, you should view the [Flask deployment documentation](https://flask.palletsprojects.com/en/1.1.x/deploying/) as well as consult with your IT department or a specialist as without careful implementation you are vulnerable to cyber attacks.
