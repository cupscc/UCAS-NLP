import  re
import  zhon.hanzi as zh
word_set = set()
with open("corpus.txt",'r',encoding='gbk')as f:
    text = f.read()#?
    rule_text = re.compile(r".*?(?=[\n])")
    sentences = re.findall(rule_text,text)
    rule_sentences = re.compile(r"[^\s].*?(?=[\s\s|\s\s\n])")
    #text_list = re.findall(rule,text)
    with open("new_corpus.txt",'w',encoding='gbk') as fout:
        for sentence in sentences:
            tokens = re.findall(rule_sentences,sentence)
            for token in tokens:
                if token == ('' or ' '):
                    pass
                elif '/' in token:
                    index = token.find('/')
                    if(len(token) > 10):
                        continue
                    newstring = token[0:index]
                    if newstring in zh.punctuation:
                        continue
                    if(len(word_set) < 2000 and newstring not in word_set):
                        word_set.add(newstring)
                    elif(len(word_set) >= 2000 and newstring not in word_set):
                        newstring = 'UNK'
                    fout.write(newstring+" ")
                else:
                    assert ("error")
            fout.write("\n")