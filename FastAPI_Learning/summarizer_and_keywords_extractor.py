import re, nltk, spacy, string
import torch

from google_bing_trans import Google_translator
from summarizer import Summarizer
from nltk.corpus import stopwords
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from multi_rake import Rake

nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
stop_words = stopwords.words('english')
nltk.download('punkt')

################################################################################
#                                                                              #
#                           Text Contractions                                  #
#                                                                              #
################################################################################
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how does",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
" u ": " you ",
" ur ": " your ",
" n ": " and ",
"won't": "would not",
'dis': 'this',
'bak': 'back',
'brng': 'bring'}

def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x
################################################################################
#                                                                              #
#                        Cleaning Text                                         #
#                                                                              #
################################################################################
def clean_txt(docs, stopwords_lang):
  
  if stopwords_lang =='en':
    stop_words = set(stopwords.words('english'))
  elif stopwords_lang =='es':
    stop_words= set(stopwords.words('spanish')) 
    
  # split into words
  speech_words = nltk.word_tokenize(docs)
  # prepare regex for char filtering
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))
  # remove punctuation from each word
  stripped = [re_punc.sub('', w) for w in speech_words]
  # remove remaining tokens that are not alphabetic
  words = [word for word in stripped if word.isalpha()]
  # filter out stop words
  words = [w for w in words if not w in  list(stop_words)]
  # filter out short tokens
  words = [word for word in words if len(word) >= 2]
  #Stemm all the words in the sentence
  combined_text = ' '.join(words)
  return combined_text
################################################################################
#                                                                              #
#                        Keywords Extractor                                    #
#                                                                              #
################################################################################
def keywords_extractor(text, lang):
  rake = Rake(
    min_chars=3,
    max_words=3,
    min_freq=1,
    language_code=lang,
    lang_detect_threshold=100)
  
  keywords = rake.apply(text)

  return keywords
################################################################################
#                                                                              #
#                       Extractive Summarization                               #
#                                                                              #
################################################################################
def extractive_text_summarization_with_BERT(text):
    ## initializing BERT model
    model = Summarizer()
    result = model(body=text, ratio=0.2, min_length=50, num_sentences=5)
    return result
################################################################################
#                                                                              #
#                       Abstractive Summarization                              #
#                                                                              #
################################################################################
def abstractive_text_summarization_with_Pegasus(text):
    ## initializing Pegasus model
    model_name = 'tuner007/pegasus_paraphrase' 
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    src_text = [text]
################################################################################
#                                                                              #
#                         Text Translation                                     #
#                                                                              #
################################################################################
def translate_text(text, lang):
  translator = Google_translator()
  trans_text = translator.translate(text,lang)
  return trans_text

print(translate_text("Hello", 'es'))


def summarize_and_keywords_en(text):
    # Summarizer
    summary = extractive_text_summarization_with_BERT(text)
    # Keywords extractor
    text_extract = cont_to_exp(text)
    cleaned_text = clean_txt(text_extract, 'en')
    keywords = keywords_extractor(cleaned_text, 'en')
    return {"Summary": summary, 'Keywords': keywords}


def summarize_and_keywords_es(text_es):
    # Summarizer
    text_en = translate_text(text_es, 'en')
    summary = extractive_text_summarization_with_BERT(text_en)
    summary_es = translate_text(summary, 'es')
    # Keywords extractor
    cleaned_text_es = clean_txt(text_es, 'es')
    keywords = keywords_extractor(cleaned_text_es, 'es')
    return {"Summary": summary_es, 'Keywords': keywords}