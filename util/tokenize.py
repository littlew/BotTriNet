import pandas as pd
import numpy as np
import re


# 把多行语料分割成words
def split_texts_into_words(texts):
    sentences = [replace_tokenize(line).split() for line in texts]
    return sentences

# https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
def replace_tokenize(input_str):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    input_str = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', '<URL>', input_str)
    input_str = re.sub('/', ' / ', input_str) # Force splitting words appended with slashes (once we tokenized the URLs, of course)
    input_str = re.sub(r'@\w+', '<USER>', input_str)
    input_str = re.sub(r'%s%s[)d]+|[)d]+%s%s' % (eyes, nose, nose, eyes), '<SMILE>', input_str, flags=re.IGNORECASE)
    input_str = re.sub(r'%s%sp+' % (eyes, nose), '<LOLFACE>', input_str, flags=re.IGNORECASE)
    input_str = re.sub(r'%s%s\(+|\)+%s%s' % (eyes, nose, nose, eyes), '<SADFACE>', input_str)
    input_str = re.sub(r'%s%s[\/|l*]' % (eyes, nose), '<NEUTRALFACE>', input_str)
    input_str = re.sub('<3', '<HEART>', input_str)
    input_str = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', '<NUMBER>', input_str)
    #input_str = re.sub(r'#\S+', lambda match: ("<HASHTAG> " + match.group(1)[1:] if match.group(1).isupper() else " ".join(["<HASHTAG>"] + re.findall('[A-Z][^A-Z]*', match.group(1)))), input_str)
    input_str = re.sub(r'#\S+', "<HASHTAG>", input_str)
    input_str = re.sub(r'([!?.]){2,}', lambda match: match.group(1) + ' <REPEAT>', input_str)
    input_str = re.sub(r'\b(\S*?)(.)\2{2,}\b', lambda match: match.group(1) + match.group(2) + ' <ELONG>', input_str)
    input_str = re.sub(r'([^a-z0-9()<>\'`\-]){2,}', lambda match: match.group(1).lower() + ' <ALLCAPS>', input_str)

    return input_str

# Example usage:
print(replace_tokenize("Hello world! I'm feeling :-) today."))
