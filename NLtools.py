import re
import nltk

__version__ = "0.1"
__name__ = "NLtools"

    
class word_Normalizer():
    def __init__(self):
        self.model = nltk.stem.WordNetLemmatizer()
    def transform(self, s):
        if s[0].isalpha():
            if s.lower()[:4] == "http":
                return("_http")
            else:
                return(self.model.lemmatize(self.model.lemmatize(re.sub(r'[^a-z]', '', s.lower())), "v"))
        else:
            return(s.lower())

        
class word_Tokenizer():
    def __init__(self, word_normalizer = word_Normalizer):
        self.regex_str = [
            r'<[^>]+>', # HTML tags
            r'(?:@[\w_]+)', # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
            r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
            r'(?:[\w_]+)', # other words
            r'(?:\S)' # anything else
        ]
        self.tokens_re = re.compile(r'('+'|'.join(self.regex_str)+')', re.VERBOSE | re.IGNORECASE)
        self.model = word_normalizer()
    def transform(self, s):
        op = [self.model.transform(t) for t in self.tokens_re.findall(s)]
        if not op:
            return(["_other"])
        return(op)
    
    
class Integer_Coder():
    def __init__(self):
        self.code_dict = {}
        self.count = 0
    def fit_transform(self, sL):
        op = []
        for i in sL:
            if i not in self.code_dict:
                self.count += 1
                self.code_dict[i] = self.count
                op.append(self.count)
            else:
                op.append(self.code_dict.get(i))
        return(op)
    def transform(self, sL):
        op = []
        for i in sL:
            j = self.code_dict.get(i)
            if j:
                op.append(j)
            else:
                op.append(0)
        return(op)