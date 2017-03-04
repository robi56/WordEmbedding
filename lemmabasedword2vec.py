
import word2vec
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import tokenize
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
#Download Link: http://mattmahoney.net/dc/text8.zip
#Read data:
lines = []
with open('text8') as f:
    for line in f:
        lines.append(line)


#Lemmatize  File removing stopwords:
lemmas =[]
for line in lines:
    tknzr = TweetTokenizer()
    tokens=tknzr.tokenize(line)
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    for token in filtered_words:
        lemmas.append(lemmatizer.lemmatize(token))
#Save Output as Files in output directory or Single File:

file = open('result.txt', 'w')
lemmatizedString = ' '.join(str(e) for e in lemmas)
file.write(lemmatizedString)
file.close()

#Build and word2vec model:
word2vec.word2vec('result.txt', 'model.bin', size=100, verbose=True)

#Load Model:
model = word2vec.load('model.bin')
print model.vocab.size
print model.vectors.shape
vocabs = model.vocab
values = model.vectors

#The model information  can be saved in a form word<space>vector<newline>word<space><vector>...
file = open('words.txt','w')
for i in range(0, len(vocabs)):
    file.write(vocabs[i]+" "+' '.join(str(e) for e in values[i])
)
    file.write('\n')

file.close()
