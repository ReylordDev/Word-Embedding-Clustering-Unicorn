from flask import Flask, render_template, request, redirect, url_for
from main import main

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return "No file selected", 400
    if file:
        file.save(f"uploads/input{file.filename[-4:]}")
        return redirect(url_for("index"))
    return "Error", 500


@app.route("/api/run_clustering", methods=["POST"])
def run_clustering(
    col_delimiter=",",
    num_words_per_row=10,
    word_column_template="Stereotype%d",
    cluster_column_template="Stereotype%d_cluster",
    excluded_words=[],
    outlier_k=3,
    outlier_detection_threshold=3,
    automatic_k=False,
    max_num_clusters=10,
    seed=42,
    num_clusters=50,
    merge_threshold=0.8,
):
    main(
        col_delimiter=col_delimiter,
        num_words_per_row=num_words_per_row,
        word_column_template=word_column_template,
        cluster_column_template=cluster_column_template,
        excluded_words=excluded_words,
        outlier_k=outlier_k,
        outlier_detection_threshold=outlier_detection_threshold,
        automatic_k=automatic_k,
        max_num_clusters=max_num_clusters,
        seed=seed,
        K=num_clusters,
        merge_threshold=merge_threshold,
    )
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5100)
