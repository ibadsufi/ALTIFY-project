from flask import Flask, render_template, request, session


from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)

app = Flask(__name__)
app.secret_key = "altify_secret_key"  # required for session


# ==============================
# LOAD PICKLE FILES
# ==============================

df = pickle.load(open("df.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ==============================
# RECOMMEND FUNCTION
# ==============================


def recommend(app_name):
    app_name = app_name.lower()

    # check if app exists
    if app_name not in df["App"].str.lower().values:
        return []

    # get index
    index = df[df["App"].str.lower() == app_name].index[0]

    # compute similarity
    similarity = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()

    # get top similar apps
    similar_indices = similarity.argsort()[::-1][1:15]

    recommendations = []

    for i in similar_indices:
        row = df.iloc[i]
        recommendations.append(
            {
                "name": row["App"],
                "category": row.get("Category", "N/A"),
                "Type": row.get("Type", "N/A"),
                "ContentRating": row.get("Content Rating", "N/A"),
                "app_link": row.get("app_link", "N/A"),
            }
        )

    return recommendations


# ==============================
# ROUTES
# ==============================


@app.route("/remove_history", methods=["POST"])
def remove_history():
    item = request.form.get("item")

    if "history" in session and item in session["history"]:
        session["history"].remove(item)

    return render_template("index.html", history=session.get("history", []))


@app.route("/clear_history", methods=["POST"])
def clear_history():
    session["history"] = []
    return render_template("index.html", history=[])


@app.route("/")
def home():
    history = session.get("history", [])
    return render_template("index.html", history=history)


@app.route("/recommend", methods=["POST"])
def get_recommendations():
    app_name = request.form.get("app_name")

    results = recommend(app_name)

    # initialize history
    if "history" not in session:
        session["history"] = []

    # add new search (avoid duplicates)
    if app_name not in session["history"]:
        session["history"].insert(0, app_name)

    # keep only last 5 searches
    session["history"] = session["history"][:5]

    return render_template(
        "index.html",
        recommendations=results,
        app_name=app_name,
        history=session["history"],
    )


from flask import jsonify


@app.route("/search")
def search():
    query = request.args.get("q")

    if not query:
        return jsonify([])

    results = df[df["App"].str.lower().str.contains(query.lower())]["App"].head(10)

    return jsonify(list(results))


# ==============================
# RUN APP
# ==============================


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
