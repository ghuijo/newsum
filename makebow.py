from konlpy.tag import Kkma
kkma = Kkma()
import re
import pandas as pd

def remove_unnecessary(document):
    document = re.sub(r'[\t\r\n\f\v]', ' ', str(document))
    document = re.sub('[^ ㄱ-ㅣ 가-힣 0-9 a-z A-Z]+', ' ', str(document))
    return document

def build_bag_of_words(total, document):
    document = document.replace('.', '')
    document = remove_unnecessary(document)
    tokenized_document = kkma.morphs(document)

    word_to_index = {}
    bows = []
    word_excluded = ["하는", "은", "는", "이", "가", "와", "과", "으로", "로", "을", "를", "고", "이나", "적", "에", "게", "께", "에게", "에서",
                    "등", "적극", "대신", "와의", "의", "했다", "별", "약", "이상", "이외", "이하", "미만", "초과", "포함", "모든", "모두",
                    "부터", "까지", "되나", "도", "이미", "된", "한", "된다", "될", "해야", "한다", "할", "다", "이다", "있다", "위해", "위하",
                    "여", "이번", "것", "명", "일", "하", "ㄴ", "아요", "요", "ㄴ다", "세", "ㄹ", "되", "들", "있", "며", "단", "었", "어",
                    "두", "년", "시", "원", "수", "도록", "기", "지", "면", "연", "만", "간", "더", "그동안", "겠", "아", "오", "으니", "니",]

    for word in tokenized_document:  
        if word not in word_to_index.keys():
            if word not in word_excluded:
                word_to_index[word] = len(word_to_index)  
                bows.insert(len(word_to_index) - 1, 1)
        else:
            index = word_to_index.get(word)
            bows[index] = bows[index] + 1

    words2 = []
    for word in word_to_index.keys():
        index = word_to_index.get(word)
        if bows[index] >= 2:
            words2.append(word)


    new_data = {
      "words_token" : [words2],
      "hashtags" : '0'
    }
    new_df = pd.DataFrame(new_data)
    total = pd.concat([total, new_df])
    total = pd.concat([total, new_df])
    total = pd.concat([total, new_df])
    
    return total