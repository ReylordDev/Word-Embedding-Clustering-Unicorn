from flask import Flask, render_template, request, redirect, url_for
from main import main
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

PORT = int(os.getenv("PORT", 5100))


@app.route("/")
def index():
    return render_template("index.html")


def save_file(request):
    if "file" not in request.files:
        raise ValueError("No file part")
    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        raise ValueError("No file uploaded")
    if file:
        file.save(f"uploads/input{file.filename[-4:]}")
    return file.filename


@app.route("/api/run_clustering", methods=["POST"])
def run_clustering():
    file_name = ""
    try:
        file_name = save_file(request)
    except ValueError as e:
        return str(e)
    col_delimiter = request.form.get("col_delimiter", default=",")
    if col_delimiter == "":
        if file_name[-4:] == ".csv":
            col_delimiter = ","
        elif file_name[-4:] == ".tsv":
            col_delimiter = "\t"
        else:
            col_delimiter = ","
    word_column_template = request.form.get("word_column", default="word%d").replace(
        "1", "%d"
    )
    cluster_column_template = request.form.get(
        "cluster_column", default="word%d_cluster"
    ).replace("1", "%d")
    num_words_per_row = request.form.get(
        "responses_per_participant", default=1, type=int
    )
    automatic_k = request.form.get("auto_number_clusters", default=False, type=bool)
    max_num_clusters = request.form.get("max_number_of_clusters", default=10, type=int)
    seed = request.form.get("seed", default=0, type=int)
    num_clusters = request.form.get("number_of_clusters", default=5, type=int)
    excluded_words = []
    outlier_k = 5
    outlier_detection_threshold = 2
    merge_threshold = 0.9

    stats = main(
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
    return redirect(url_for("results", execution_time=stats.get("execution_time")))


@app.route("/results")
def results():
    execution_time = request.args.get("execution_time", default=0.0, type=float)
    return render_template("results.html", execution_time=execution_time)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
