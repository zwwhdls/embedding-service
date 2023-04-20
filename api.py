from flask import Flask, request, json, Response
from embeddings import HuggingFaceEmbedding, text2vect_embedding
import time

app = Flask(__name__)


@app.route("/embeddings/docs", methods=["POST"])
def embedding_docs():
    print("embedding begin. {}".format(time.asctime(time.localtime(time.time()))))
    data = request.json
    model_name = data["model"]
    texts = data["docs"]
    embedding_type = data.get("embed_type") or "huggingface"
    if embedding_type == "huggingface":
        if model_name == "text2vec-large-chinese" and text2vect_embedding is not None:
            res = text2vect_embedding.embedding_docs(texts)
            print("length of res: {}".format(len(res)))

    print("embedding end. {}".format(time.asctime(time.localtime(time.time()))))
    return Response(
        response=json.dumps({
            "result": res
        }),
        status=200,
        mimetype='application/json'
    )


@app.route("/embeddings/query", methods=["POST"])
def embedding_query():
    print("embedding begin. {}".format(time.asctime(time.localtime(time.time()))))
    data = request.json
    model_name = data["model"]
    text = data["text"]
    embedding_type = data.get("embed_type") or "huggingface"
    if embedding_type == "huggingface":
        if model_name == "text2vec-large-chinese" and text2vect_embedding is not None:
            res = text2vect_embedding.embedding_query(text)
            print("length of res: {}".format(len(res)))
    print("embedding end. {}".format(time.asctime(time.localtime(time.time()))))
    return Response(
        response=json.dumps({
            "result": res
        }),
        status=200,
        mimetype='application/json'
    )
