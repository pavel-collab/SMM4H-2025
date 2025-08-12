from sentence_transformers import CrossEncoder

'''
Здесь нужно будет написать все то же самое, что в скрипте cosin_similarity,
только функция get_similarity будет другой. И другая модель.
'''

model_name = 'cross-encoder/stsb-mpnet-base-v2'

model = CrossEncoder(model_name)

def get_similarity(sent1, sent2, model):
    score = model.predict([(sent1, sent2)])
    return score