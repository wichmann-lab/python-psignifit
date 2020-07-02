from flask import Flask
app = Flask(__name__)
@app.route("/run-psignifit")
def output():
	return "test"
if __name__ == "__main__":
	app.run()
