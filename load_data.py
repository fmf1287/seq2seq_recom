import jieba
import pandas as pd
import numpy as np
import re

def data_process(csv):
    tk_info = pd.read_csv(csv)

    r = re.compile("[^A-Za-z\+\-*/|\%^=()（）<>.\u4e00-\u9fa5\u0370-\u03ff]*")
    def textParse(sentence):
        sentenc = r.sub('', sentence)
        return list(jieba.cut(sentenc))

    q_content = [x for x in tk_info['content']]
    q_analysis = [x for x in tk_info['analysis']]

    def extract_character_vocab(data):
        '''
        构造映射表
        '''
        special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

        set_words = list(set(word for text in data for word in textParse(text)))
        # 这里要把四个特殊字符添加进词典
        int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

        return int_to_vocab, vocab_to_int

    # 构造映射表
    source_int_to_letter, source_letter_to_int = extract_character_vocab(q_content)
    target_int_to_letter, target_letter_to_int = extract_character_vocab(q_analysis)

    # 对字母进行转换
    source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
                   for letter in line] for line in q_content]
    target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
                   for letter in line] + [target_letter_to_int['<EOS>']] for line in q_analysis]

    return(tk_info,source_int_to_letter,source_letter_to_int,target_int_to_letter,target_letter_to_int,source_int,target_int)