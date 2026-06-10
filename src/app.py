from flask import Flask, render_template, request
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":
        amount = float(request.form.get("amount", 0))

        if amount > 2000:
            result = "🚨 FRAUD DETECTED"
        else:
            result = "✅ LEGITIMATE TRANSACTION"

    return render_template("predict.html", result=result)