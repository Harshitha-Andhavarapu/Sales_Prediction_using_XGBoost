import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from database import db, User  # Import database setup and User model

# Initialize Flask app (ONLY ONCE)
app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = "harshu"


# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite3"  # SQLite Database
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database, bcrypt, and login manager
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "signin"

# Load user by ID for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load trained XGBoost model
model_filename = "dmart_success.pkl"
model = pickle.load(open(model_filename, "rb"))

# Load dataset for visualization
df = pd.read_csv("final data set.csv")

# Compute average success rate per product type
avg_success_rate = df.groupby("Item_Type")["Success Rate (%)"].mean().reset_index()
data = avg_success_rate.to_dict(orient="records")

# Flask Routes
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/")
def signin():
    return render_template("SignUp_Login_Form.html")

@app.route("/register", methods=["POST"])
def register():
    username = request.form["username"]
    email = request.form["email"]
    password = request.form["password"]

    # Check if user already exists
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        flash("Email already registered. Please log in.", "error")
        return redirect(url_for("signin"))

    # Hash the password before storing it
    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

    # Create a new user object
    new_user = User(username=username, email=email, password=hashed_password)

    # Add to database
    db.session.add(new_user)
    db.session.commit()

    flash("Registration successful! You can now log in.", "success")
    return redirect(url_for("signin"))
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("signin"))


@app.route("/signup")
def signup():
    return render_template("SignUp_Login_Form.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/visualize")
def visualize():
    data = df.to_dict(orient="records")
    return render_template("visualize.html", data=data)

@app.route("/data")
def data():
    return render_template("data.html")
@app.route("/login", methods=["POST"])
def login():
    email = request.form["email"]
    password = request.form["password"]

    # Check if the user exists in the database
    user = User.query.filter_by(email=email).first()

    if user and bcrypt.check_password_hash(user.password, password):
        login_user(user)  # Log in the user
        flash("Login successful!", "success")
        return redirect(url_for("home"))  # Redirect to dashboard

    else:
        flash("Invalid email or password. Please try again.", "error")
        return redirect(url_for("signin"))  # Redirect back to login


@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Get input values from request
        try:
            features = [
                float(request.args.get("Inventory", 0) or 0),
                float(request.args.get("MRP", 0) or 0),
                float(request.args.get("Sales", 0) or 0),
                float(request.args.get("Weekly_Sales", 0) or 0),
            ]
        except ValueError as ve:
            return f"Invalid input: {ve}"

        print("\nReceived Input Features:", features)

        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        print("Model Prediction:", prediction)

        safe_prediction = str(round(prediction, 2)).encode("utf-8", "ignore").decode("utf-8")

        return render_template("output.html", prediction=safe_prediction)

    except Exception as e:
        return f"Error: {str(e)}"

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
