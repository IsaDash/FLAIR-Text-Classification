from flair.models import TextClassifier
from flair.data import Sentence

classifier = TextClassifier.load('best-model.pt')
sentence = Sentence("i, i can't really decide what to buy between those two. I don't know whether the Cloud Alpha is worth buying since it costs more or less double of the price of the Cloud Stinger. May i hear your reviews about them? Thank you.")
classifier.predict(sentence)
print(sentence.labels)

