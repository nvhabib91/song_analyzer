from flask import Flask, render_template

app = Flask(__name__)

@app.route("/results")
def results():
    return render_template("base.html", pg_title="Results")


@app.route("/")
def index():
    return render_template("index.html", result_dict=None)

if __name__ == "__main__":
    app.run(debug=True)
