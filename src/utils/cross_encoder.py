from sentence_transformers import CrossEncoder

model_name = 'cross-encoder/stsb-mpnet-base-v2'

model = CrossEncoder(model_name)

def get_cosin_similarity(sent1, sent2, model):
    score = model.predict([(sent1, sent2)])
    return score