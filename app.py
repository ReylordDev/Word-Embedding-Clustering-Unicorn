from flask import Flask, render_template, request, redirect, url_for
from main import main

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


# @app.route("/api/upload", methods=["POST"])
# def upload():
#     if "file" not in request.files:
#         return "No file uploaded", 400
#     file = request.files["file"]
#     if file.filename == "" or file.filename is None:
#         return "No file selected", 400
#     if file:
#         file.save(f"uploads/input{file.filename[-4:]}")
#         return redirect(url_for("index"))
#     return "Error", 500


def save_file(request):
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return "No file selected", 400
    if file:
        file.save(f"uploads/input{file.filename[-4:]}")


@app.route("/api/run_clustering", methods=["POST"])
def run_clustering():
    save_file(request)
    col_delimiter = request.form.get("col_delimiter", default=",")
    word_column_template = request.form.get("word_column", default="word%d").replace(
        "1", "%d"
    )
    cluster_column_template = request.form.get(
        "cluster_column", default="word%d_cluster"
    ).replace("1", "%d")
    num_words_per_row = request.form.get(
        "responses_per_participant", default=1, type=int
    )
    automatic_k = request.form.get("auto_number_of_clusters", default=False, type=bool)
    max_num_clusters = request.form.get("max_number_of_clusters", default=10, type=int)
    seed = request.form.get("seed", default=0, type=int)
    num_clusters = request.form.get("number_of_clusters", default=5, type=int)
    excluded_words = []
    outlier_k = 5
    outlier_detection_threshold = 1
    merge_threshold = 0.8

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
