import numpy as np
import re
import torch

def downSampleData(nClasses, train, test):
    nClassesOrg = train.food.nunique()
    print(r"Number of original calsses = ", nClassesOrg)
    allFoods = list(set(train.food.unique()))
    mask = np.zeros(train.shape[0])
    for i in range(nClasses):
        indices = [t for t, x in enumerate(list(train.food)) if (x == allFoods[i])]
        mask[indices] = 1
    train = train[mask==1]
    mask = np.zeros(test.shape[0])
    for i in range(nClasses):
        indices = [t for t, x in enumerate(list(test.food)) if (x == allFoods[i])]
        mask[indices] = 1
    test = test[mask==1]
    nClasses = train.food.nunique()
    print(r"Number of reduced calsses = ", nClasses)
    return(train,test)

def classCounts(nClasses, train, test):
    allFoods = list(set(train.food.unique()))
    countsTrn = np.zeros(len(allFoods))
    countsTst = np.zeros(len(allFoods))
    for i in range(nClasses):
        indices = [t for t, x in enumerate(list(train.food)) if x == allFoods[i]]
        countsTrn[i] = countsTrn[i] + len(indices)

    for i in range(nClasses):
        indices = [t for t, x in enumerate(list(test.food)) if x == allFoods[i]]
        countsTst[i] = countsTst[i] + len(indices)

    return countsTrn,countsTst

def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    return sentence

def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

# Preprocessing of texts according to BERT

def get_masks(text, tokenizer, max_length):
    """Mask for padding"""
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]
    return np.asarray([1]*len(tokens) + [0] * (max_length - len(tokens)))

def get_segments(text, tokenizer, max_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]
    segments = []
    current_segment_id = 0
    with_tags = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return np.asarray(segments + [0] * (max_length - len(tokens)))

def get_ids(text, tokenizer, max_length):
    """Token ids from Tokenizer vocab"""
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = np.asarray(token_ids + [0] * (max_length-length))
    return input_ids


def prepare(text_array, tokenizer, max_length = 128):
    vec_get_masks = np.vectorize(get_masks, signature = '(),(),()->(n)')
    vec_get_segments = np.vectorize(get_segments, signature = '(),(),()->(n)')
    vec_get_ids = np.vectorize(get_ids, signature = '(),(),()->(n)')

    ids = vec_get_ids(text_array,
                      tokenizer, 
                      max_length).squeeze()
    masks = vec_get_masks(text_array,
                         tokenizer, 
                         max_length).squeeze()
    segments = vec_get_segments(text_array,
                                tokenizer, 
                                max_length).squeeze()

    return ids, segments, masks
    
def token2id(listData, maxLen, w2vModel):
    ids = torch.zeros( (len(listData),maxLen) )
    # set equal to id of 0/pad vector
    ids[:,:] = torch.tensor(w2vModel.wv.key_to_index[0])
    for i in range(len(listData)):
        for j in range(min([maxLen,len(listData[i])])):
            ids[i,j] = torch.tensor(w2vModel.wv.key_to_index[listData[i][j]])
    return ids