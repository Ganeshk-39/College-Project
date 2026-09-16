import os
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, session

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ======================================
# APP CONFIG
# ======================================
app = Flask(__name__)
app.secret_key = "secure_ocd_key"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "Dataset")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
PLOT_FOLDER = os.path.join(BASE_DIR, "static", "plots")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# ======================================
# GLOBAL STORAGE
# ======================================
df = None
accuracies = {}

users=[]
# ======================================
# HOME
# ======================================
@app.route("/")
def home():
    return render_template("home.html")

# ======================================
# ADMIN LOGIN
# ======================================
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "admin":
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))

    return render_template("admin_login.html")

# ======================================
# ADMIN DASHBOARD
# ======================================
@app.route("/admin_dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    return render_template("admin_dashboard.html", accuracies=accuracies)

# ======================================
# ADMIN LOGOUT
# ======================================
@app.route("/admin_logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("home"))

# ======================================
# DATASET UPLOAD (PERMISSION SAFE)
# ======================================
@app.route("/upload", methods=["GET", "POST"])
def upload():
    global df

    if "admin" not in session:
        return redirect(url_for("admin_login"))

    message = None
    table = None
    columns = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename.endswith(".csv"):

            # ✅ SAFE: Read CSV directly (NO SAVE = NO LOCK)
            try:
                df = pd.read_csv(file)

                message = "Dataset uploaded successfully!"
                columns = df.columns.tolist()
                table = df.head(10).to_dict(orient="records")

            except Exception as e:
                message = f"Error reading CSV: {str(e)}"

        else:
            message = "Please upload a valid CSV file."

    return render_template(
        "upload.html",
        message=message,
        table=table,
        columns=columns
    )


# ======================================
# ADMIN: PREPROCESS
# ======================================
@app.route("/preprocess")
def preprocess():
    global df, X_train, X_test, y_train, y_test

    if "admin" not in session:
        return redirect(url_for("admin_login"))

    encoder = LabelEncoder()
    for col in df.select_dtypes(include="object"):
        df[col] = encoder.fit_transform(df[col])

    df.dropna(inplace=True)

    X = df.drop(["Patient ID", "Obsession Type"], axis=1)
    y = df["Obsession Type"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return render_template(
        "preprocess.html",
        total=len(df),
        train=len(X_train),
        test=len(X_test)
    )

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
import pickle
@app.route("/train_models")
def train_models():
    global accuracies

    metrics = {}

    # ================================
    # 1. RANDOM FOREST
    # ================================
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    metrics["Random Forest"] = {
        "Accuracy": accuracy_score(y_test, rf_pred),
        "Precision": precision_score(y_test, rf_pred, average="macro"),
        "Recall": recall_score(y_test, rf_pred, average="macro"),
        "F1": f1_score(y_test, rf_pred, average="macro"),
        "Support": len(y_test)
    }
    pickle.dump(rf, open("models/rf.pkl", "wb"))

    # ================================
    # 2. DECISION TREE
    # ================================
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    metrics["Decision Tree"] = {
        "Accuracy": accuracy_score(y_test, dt_pred),
        "Precision": precision_score(y_test, dt_pred, average="macro"),
        "Recall": recall_score(y_test, dt_pred, average="macro"),
        "F1": f1_score(y_test, dt_pred, average="macro"),
        "Support": len(y_test)
    }
    pickle.dump(dt, open("models/dt.pkl", "wb"))

    # ================================
    # 3. SUPPORT VECTOR MACHINE
    # ================================
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    metrics["SVM"] = {
        "Accuracy": accuracy_score(y_test, svm_pred),
        "Precision": precision_score(y_test, svm_pred, average="macro"),
        "Recall": recall_score(y_test, svm_pred, average="macro"),
        "F1": f1_score(y_test, svm_pred, average="macro"),
        "Support": len(y_test)
    }
    pickle.dump(svm, open("models/svm.pkl", "wb"))

    # ================================
    # 4. K-NEAREST NEIGHBORS
    # ================================
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    metrics["KNN"] = {
        "Accuracy": accuracy_score(y_test, knn_pred),
        "Precision": precision_score(y_test, knn_pred, average="macro"),
        "Recall": recall_score(y_test, knn_pred, average="macro"),
        "F1": f1_score(y_test, knn_pred, average="macro"),
        "Support": len(y_test)
    }
    pickle.dump(knn, open("models/knn.pkl", "wb"))

    # ================================
    # 5. CNN (MULTI-CLASS – 5 OCD TYPES)
    # ================================
    Xp_train = pad_sequences(X_train, maxlen=50)
    Xp_train = Xp_train.reshape(-1, 50, 1)

    cnn = Sequential([
        Conv1D(32, 3, activation="relu", input_shape=(50, 1)),
        MaxPooling1D(2),
        Flatten(),
        Dense(5, activation="softmax")
    ])

    cnn.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cnn.fit(Xp_train, y_train, epochs=5, verbose=0)

    Xp_test = pad_sequences(X_test, maxlen=50)
    Xp_test = Xp_test.reshape(-1, 50, 1)

    cnn_pred = np.argmax(cnn.predict(Xp_test), axis=1)

    metrics["CNN"] = {
        "Accuracy": accuracy_score(y_test, cnn_pred),
        "Precision": precision_score(y_test, cnn_pred, average="macro"),
        "Recall": recall_score(y_test, cnn_pred, average="macro"),
        "F1": f1_score(y_test, cnn_pred, average="macro"),
        "Support": len(y_test)
    }
    cnn.save("models/cnn.h5")

    # ================================
    # SAVE METRICS
    # ================================
    accuracies = metrics

    # ================================
    # ONE COMBINED BAR GRAPH
    # ================================
    labels = []
    values = []

    for algo, vals in metrics.items():
        for m in ["Accuracy", "Precision", "Recall", "F1"]:
            labels.append(f"{algo}-{m}")
            values.append(vals[m])

    plt.figure(figsize=(18, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Comparison of ML & DL Algorithms")
    plt.tight_layout()
    plt.savefig("static/plots/all_algorithms_metrics.png")
    plt.close()

    # ================================
    # REDIRECT TO trainmodels.html
    # ================================
    return render_template("trainmodels.html", accuracies=accuracies)

def generate_comparison_graphs():
    if not accuracies:
        return

    # Extract algorithm names and accuracy values
    names = []
    acc_values = []

    for algo, metrics in accuracies.items():
        names.append(algo)
        acc_values.append(metrics["Accuracy"])

    # -------------------------
    # BAR GRAPH
    # -------------------------
    plt.figure(figsize=(8,5))
    plt.bar(names, acc_values, color="skyblue")
    plt.ylabel("Accuracy")
    plt.title("Algorithm Accuracy Comparison (Bar Chart)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/bar.png")
    plt.close()

    # -------------------------
    # LINE GRAPH
    # -------------------------
    plt.figure(figsize=(8,5))
    plt.plot(names, acc_values, marker="o", linestyle="-", color="green")
    plt.ylabel("Accuracy")
    plt.title("Algorithm Accuracy Comparison (Line Chart)")
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/line.png")
    plt.close()

    # -------------------------
    # PIE CHART
    # -------------------------
    plt.figure(figsize=(7,7))
    plt.pie(
        acc_values,
        labels=names,
        autopct="%1.1f%%",
        startangle=140
    )
    plt.title("Algorithm Accuracy Distribution (Pie Chart)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/pie.png")
    plt.close()
@app.route("/comparison")
def comparison():
    generate_comparison_graphs()
    return render_template("comparison.html", accuracies=accuracies)
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        users.append({
            "username": request.form["username"],
            "password": request.form["password"],
            "email": request.form["email"],
            "mobile": request.form["mobile"],
            "address": request.form["address"]
        })
        return redirect(url_for("user_login"))

    return render_template("register.html")

@app.route("/manage_users")
def manage_users():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    return render_template("manage_users.html", users=users)
@app.route("/delete_user/<username>")
def delete_user(username):
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    global users
    users = [u for u in users if u["username"] != username]

    return redirect(url_for("manage_users"))



@app.route("/user_login", methods=["GET", "POST"])
def user_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        for u in users:
            if u["username"] == username and u["password"] == password:
                session["user"] = u
                return redirect(url_for("user_home"))

        # Invalid credentials
        return render_template("user_login.html", error="Invalid Username or Password")

    return render_template("user_login.html")

@app.route("/user_home")
def user_home():
    if "user" not in session:
        return redirect(url_for("user_login"))
    return render_template("user_home.html")

@app.route("/user_logout")
def user_logout():
    session.pop("user", None)
    return redirect(url_for("home"))

# ======================================
@app.route("/analysis")
def analysis():

    

    plt.figure(figsize=(6,4))
    df["Obsession Type"].value_counts().plot(kind="bar", color="maroon")
    plt.title("Distribution of OCD Types")
    plt.xlabel("OCD Type")
    plt.ylabel("Count")
    plt.tight_layout()

    plot_path = os.path.join(PLOT_FOLDER, "ocd_types.png")
    plt.savefig(plot_path)
    plt.close()

    return render_template("analysis.html", plot="plots/ocd_types.png")


# ======================================
# USER: PROFILE
# ======================================
@app.route("/profile")
def profile():
    return render_template("profile.html", user=session.get("user"))
REVERSE_OCD_MAP = {
    0: "Harm-related",
    1: "Contamination",
    2: "Symmetry",
    3: "Hoarding",
    4: "Religious"
}

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        # ============================
        # TAKE PATIENT ID (DISPLAY ONLY)
        # ============================
        patient_id = request.form["Patient_ID"]

        # ============================
        # INPUT DATA FOR MODEL
        # (NO Patient ID, NO Obsession Type)
        # ============================
        input_data = {
            "Age": float(request.form["Age"]),
            "Gender": int(request.form["Gender"]),
            "Ethnicity": int(request.form["Ethnicity"]),
            "Marital Status": int(request.form["Marital_Status"]),
            "Education Level": int(request.form["Education_Level"]),
            "Duration of Symptoms (months)": float(request.form["Duration"]),
            "Previous Diagnoses": int(request.form["Previous_Diagnoses"]),
            "Family History of OCD": int(request.form["Family_History"]),
            "Compulsion Type": int(request.form["Compulsion_Type"]),
            "Y-BOCS Score (Obsessions)": float(request.form["YBOCS_Obs"]),
            "Y-BOCS Score (Compulsions)": float(request.form["YBOCS_Comp"]),
            "Depression Diagnosis": int(request.form["Depression"]),
            "Anxiety Diagnosis": int(request.form["Anxiety"]),
            "Medications": int(request.form["Medications"])
        }

        # ============================
        # CONVERT TO DATAFRAME
        # ============================
        input_df = pd.DataFrame([input_data])

        # ============================
        # LOAD TRAINED MODEL
        # ============================
        model = pickle.load(open("models/rf.pkl", "rb"))

        # ============================
        # PREDICT PROBABILITIES
        # ============================
        probs = model.predict_proba(input_df)[0]

        # ============================
        # TOP 3 OCD TYPES
        # ============================
        top_indices = np.argsort(probs)[::-1][:3]

        predictions = []
        for idx in top_indices:
            predictions.append({
                "ocd_category": REVERSE_OCD_MAP[idx],
                "confidence": round(probs[idx] * 100, 2)
            })

        # ============================
        # SEND RESULT TO UI
        # ============================
        return render_template(
            "result.html",
            patient_id=patient_id,
            predictions=predictions
        )

    return render_template("predict.html")

# ======================================
if __name__ == "__main__":
    app.run(debug=True)
