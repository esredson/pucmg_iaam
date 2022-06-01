import spacy # python -m spacy download pt_core_news_sm
from spacy.lang.pt.examples import sentences 

titulos =  'Países áreabes iniciam ofensiva à Ucrânia'
nlp = spacy.load("pt_core_news_sm") #"tagger", "lemmatizer" "parser" "attribute_ruler" "tok2vec"
pos_tags_validos = ['NOUN', 'ADJ', 'PROPN']
importantes = [token.text for token in nlp(titulos) if token.pos_ in pos_tags_validos]
print(importantes)