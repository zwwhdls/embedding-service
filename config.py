import os


class Config:
    text2vec_model_path = os.getenv("TEXT2VECTOR_PATH")


conf = Config()
