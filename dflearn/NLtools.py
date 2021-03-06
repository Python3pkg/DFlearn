import re
import nltk


stopwords = ["a", "an", "and", "are", "as", "be", "by", "for", "from", 
             "has", "he", "in", "is", "it", "its", "of", "on", 
             "that", "the", "to", "was", "were", "will", "with"]


class word_Normalizer():
    def __init__(self):
        self.model = nltk.stem.WordNetLemmatizer()
    def transform(self, s):
        if s[0].isalpha():
            if s.lower()[:4] == "http":
                return("<url>")
            else:
                return(self.model.lemmatize(self.model.lemmatize(re.sub(r'[^a-z]', '', s.lower())), "v"))
        elif s[0].isdigit():
            return(re.sub(",", "", s.lower()))
        else:
            return(s.lower())
        
        
class word_Tokenizer():
    def __init__(self):
        self.regex_str = [
            r'<[^>\s]+>', # HTML tags
            r'(?:@[\w_]+)', # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
            r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
            r'(?:[\w_]+)', # other words
            r'(?:\S)' # anything else
        ]
        self.tokens_re = re.compile(r'('+'|'.join(self.regex_str)+')', re.VERBOSE | re.IGNORECASE)
    def transform(self, s, f_word_norm = lambda x: x):
        op = [f_word_norm(t) for t in self.tokens_re.findall(s)]
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