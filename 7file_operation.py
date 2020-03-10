import nltk
from nltk.stem import WordNetLemmatizer

rfile = open("input.txt","r")


with open("tokenize.txt","w") as t:

    for sentence in rfile:

        w_tokens = nltk.word_tokenize(sentence)

        for w in w_tokens:
            t.write(str("\n"))
            t.write(str(w))


lemmatizer_output = WordNetLemmatizer()

with open("lemmatize.txt", "w") as l:

    wrd = open("tokenize.txt","r")
    for words in wrd:

        le = lemmatizer_output.lemmatize(words)
        l.write(str(le))

op = open("input.txt","r")
with open("trigram.txt","w") as tri_gram:
    for sentence in op:

            trigram = nltk.trigrams(sentence.split())


            for tri in trigram:
                tri_gram.write(str("\n"))
                tri_gram.write(str(tri))

fl = open("input.txt", "r")
fileread = fl.read()

tgm = []
word_tokens = nltk.word_tokenize(fileread)
for t in nltk.ngrams(word_tokens, 3):
    tgm.append(t)

wordFreq = nltk.FreqDist(tgm)
most_Common_trigram = wordFreq.most_common()


Top_10_trigrams = wordFreq.most_common(10)
print("Top 10 Trigrams:\n", Top_10_trigrams, "\n")


sent_tokens = nltk.sent_tokenize(fileread)
concat_result = []

for s in sent_tokens:

    for a, b, c in tgm:

        for ((p, q, r), length) in wordFreq.most_common(20): # Comparing the each with the top 10 trigrams
            if(a, b, c == p, q, r):
                concat_result.append(s)

print("Output of Concatenated Array: \n", concat_result)
print("Most repeated among Concatenated Array: \n ", max(concat_result))