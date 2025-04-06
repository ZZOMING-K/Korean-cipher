import pandas as pd

# 초성 리스트
CHOSUNG_LIST = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

# 중성 리스트
JUNGSUNG_LIST = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ",
                 "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]

# 종성 리스트
JONGSUNG_LIST = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
                 "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

# 초성 변환 규칙
CHOSUNG_MAP = {
    "ㄱ": ["ㄱ","ㅋ", "ㄲ"],
    "ㄲ": ["ㄲ","ㄱ", "ㅋ"],
    "ㄷ": ["ㄷ" , "ㅌ", "ㄸ"],
    "ㄸ": ["ㄸ","ㄷ", "ㅌ"],
    "ㅂ": ["ㅂ","ㅍ", "ㅃ"],
    "ㅃ": ["ㅃ","ㅂ", "ㅍ"],
    "ㅅ": ["ㅅ","ㅆ"],
    "ㅆ": ["ㅆ","ㅅ"],
    "ㅈ": ["ㅈ","ㅊ", "ㅉ"],
    "ㅉ": ["ㅉ","ㅈ", "ㅊ"],
    "ㅊ": ["ㅊ","ㅉ"],
    "ㅋ": ["ㅋ","ㄲ"],
    "ㅌ": ["ㅌ","ㄸ"],
    "ㅍ": [ "ㅍ","ㅃ"],
}

# 중성 변환 규칙
JUNGSUNG_MAP = {
    "ㅏ": ["ㅏ","ㅑ"],
    "ㅑ": ["ㅑ","ㅏ"],
    "ㅓ": ["ㅓ","ㅕ"],
    "ㅕ": [ "ㅕ","ㅓ"],
    "ㅗ": ["ㅗ","ㅛ"],
    "ㅛ": ["ㅛ","ㅗ"],
    "ㅜ": ["ㅜ","ㅠ"],
    "ㅠ": ["ㅠ","ㅜ"],
    "ㅡ": ["ㅡ","ㅜ", "ㅢ"],
    "ㅢ": ["ㅢ","ㅣ", "ㅡ"],
    "ㅣ": ["ㅣ","ㅟ", "ㅢ"],
    "ㅐ": ["ㅐ","ㅔ", "ㅒ"],
    "ㅔ": ["ㅔ","ㅐ", "ㅖ"],
    "ㅒ": ["ㅒ","ㅐ", "ㅖ"],
    "ㅖ": ["ㅖ","ㅔ", "ㅒ"],
    "ㅚ": ["ㅚ","ㅙ", "ㅘ"],
    "ㅙ": ["ㅙ","ㅚ", "ㅞ"],
    "ㅘ": ["ㅘ","ㅙ", "ㅝ"],
    "ㅝ": ["ㅝ","ㅞ", "ㅘ"],
    "ㅞ": ["ㅞ","ㅝ", "ㅙ"]
}

# 종성 변환 규칙
JONGSUNG_MAP = {
    "ㄱ": ["ㄱ" , "ㄲ", "ㅋ"],
    "ㄲ": ["ㄲ","ㄱ", "ㅋ"],
    "ㅋ": ["ㅋ","ㄱ", "ㄲ"],
    "ㄴ": ["ㄴ","ㄶ"],
    "ㄶ": ["ㄶ","ㄴ"],
    "ㄹ": [ "ㄹ","ㄻ", "ㅀ"],
    "ㄻ": ["ㄻ","ㅁ"],
    "ㅀ": ["ㅀ","ㄹ"],
    "ㅁ": ["ㅁ","ㄻ"],
    "ㅂ": ["ㅂ","ㅄ"],
    "ㅄ": ["ㅄ","ㅂ"],
    "ㅅ": ["ㅅ","ㅆ"],
    "ㅆ": ["ㅆ","ㅅ"],
    "ㅇ": ["ㅇ","ㅎ"],
    "ㅎ": ["ㅎ","ㅇ"]
}

# 자모분리 : 초성,중성,종성으로 분리하는 함수
def split_syllable(syllable):
    code = ord(syllable) - 0xAC00
    cho = code // (21 * 28)
    jung = (code % (21 * 28)) // 28
    jong = code % 28
    return cho, jung, jong

# 자모 결합 : 초성,중성,종성을 합쳐서 한글 음절로 결합하는 함수
def combine_syllable(cho, jung, jong):
    if cho < 0 or cho >= len(CHOSUNG_LIST):
        return  
    if jung < 0 or jung >= len(JUNGSUNG_LIST):
        return  
    if jong < 0 or jong >= len(JONGSUNG_LIST):
        return 
    return chr(0xAC00 + cho * 21 * 28 + jung * 28 + jong)


import random

# 자모를 비슷한 발음으로 변환
def transform_hangul(text):
    result = []

    for char in text:

        if "가" <= char <= "힣":

            current_char = char
            cho, jung, jong = split_syllable(current_char)

            #초성변환
            chosung_char = CHOSUNG_LIST[cho]
            if chosung_char in CHOSUNG_MAP:
                chosung_char = random.choice(CHOSUNG_MAP[chosung_char])
            new_cho = CHOSUNG_LIST.index(chosung_char)

            # 중성 변환
            jungsung_char = JUNGSUNG_LIST[jung]
            if jungsung_char in JUNGSUNG_MAP:
                jungsung_char = random.choice(JUNGSUNG_MAP[jungsung_char])
            new_jung = JUNGSUNG_LIST.index(jungsung_char)

            # 종성 변환
            jongsung_char = JONGSUNG_LIST[jong]
            if jongsung_char in JONGSUNG_MAP:
                jongsung_char = random.choice(JONGSUNG_MAP[jongsung_char])
            new_jong = JONGSUNG_LIST.index(jongsung_char)

            result.append(combine_syllable(new_cho, new_jung, new_jong))

        else:
            result.append(char)  # 한글이 아니면 그대로 추가

    return ''.join(result)

# 의미 없는 받침 추가
def add_random_jongseong(word):

    result = []
    i = 0

    for char in word :

        if "가" <= char <= "힣":

            current_char = char
            cho, jung, jong = split_syllable(current_char)

            if jong == 0 : #종성이 없을 경우
                random_jong = random.choice(JONGSUNG_LIST)
                new_jong = JONGSUNG_LIST.index(random_jong)

                # 현재 글자의 종성 추가
                result.append(combine_syllable(cho, jung, new_jong))
                i += 1

            else:
                result.append(current_char)
                i += 1
        else:
            result.append(char)
            i += 1

    return ''.join(result)

# 연음법칙 적용
# 복합 종성(ㄵ, ㄶ, ㄺ, ㄻ, ㅄ 등)의 인덱스
COMPLEX_FINALS = {3, 5, 6, 9, 10, 11,12, 13,14, 15,18}

def apply_liaison(word):

    result = []
    i = 0

    while i < len(word):

        if i < len(word) - 1:

            current_char = word[i]

            if "가" <= current_char <= "힣":
                next_char = word[i + 1]

                cho, jung, jong = split_syllable(current_char)
                next_cho, next_jung, next_jong = split_syllable(next_char)

                if( jong != 0) and (next_cho == 11) and (jong not in COMPLEX_FINALS):   # 종성이 있고, 다음 글자가 'ㅇ'으로 시작할 때

                    # 종성을 다음 글자의 초성으로 이동
                    new_cho = JONGSUNG_LIST[jong]
                    next_cho = CHOSUNG_LIST.index(new_cho)

                    # 현재 글자의 종성 제거
                    result.append(combine_syllable(cho, jung, 0))

                    # 다음 글자의 초성 변경
                    result.append(combine_syllable(next_cho, next_jung, next_jong))
                    i += 2  # 다음 글자까지 처리했으므로 2칸 이동

                else:
                    result.append(current_char)
                    i += 1
            else:
                result.append(word[i])
                i += 1
        else :
            result.append(word[i])
            i+=1

    return ''.join(result)



# 초성을 종성으로 변환하는 매핑
CHO_TO_JONG = {
    0: 1, 1: 2, 2: 4, 3: 7, 5: 8, 6: 16, 9: 19, 10: 20, 11: 21, 12: 22, 14: 23, 15: 24, 16: 25, 17: 26
}

#뒤에 오는 자음을 받침으로 중복
def cho_to_jong(word):

    result = []

    for i in range(len(word)):

        syllable = word[i]

        if "가" <= syllable <= "힣":

            cho, jung, jong = split_syllable(syllable)

            if jong == 0 and i < len(word) - 1:
                next_syllable = word[i + 1]
                next_cho, next_jung, next_jong = split_syllable(next_syllable)

                # 초성을 종성으로 변환
                if next_cho in CHO_TO_JONG:
                    jong = CHO_TO_JONG[next_cho]
                    new_syllable = combine_syllable(cho, jung, jong)
                    result.append(new_syllable)

                else:
                    result.append(syllable)

            else:
                result.append(syllable)

        else :
            result.append(syllable)

    return ''.join(result)


import random

def obfuscate_korean(text, settings):
    methods_all = [
        ("apply_liaison", apply_liaison),  # 연음현상 적용
        ("cho_to_jong", cho_to_jong),  # 초성을 종성으로 이동
        ("transform_hangul", transform_hangul),  # 한글 자모 변형
        ("add_random_jongseong", add_random_jongseong),  # 종성 추가
    ]

    methods_short = [
        ("transform_hangul", transform_hangul),  # 한글 자모 변형
        ("add_random_jongseong", add_random_jongseong),  # 종성 추가
    ]

    obfuscated_words = []

    for word in text.split():
        methods = methods_short if len(word) == 1 else methods_all

        for name, method in methods:
            if random.random() < settings.get(name, 0):  # 확률에 따라 적용
                word = method(word)

        obfuscated_words.append(word)

    return ' '.join(obfuscated_words)


def aug_data(input_file, num_sample , settings) : 
    
    train = pd.read_csv(input_file) 
    
    train_output = train['output'].drop_duplicates().reset_index(drop = True) # 중복 제거

    augmented_data_list , origin_data_list = [] , []      

    for entry in train_output:
        for _ in range(num_sample):  # 샘플링 횟수
            origin_data_list.append(entry)  
            augmented_entry = obfuscate_korean(entry, settings)  
            augmented_data_list.append(augmented_entry)  

    augmented_data = {
        'input': augmented_data_list,  # 증강 데이터 
        'output': origin_data_list  # 원본 데이터
    }

    aug_train_df = pd.DataFrame(augmented_data)
    
    return train , aug_train_df 

def concat_data(train , aug_train_df) : 

    train = train[['input' , 'output']]
    aug_train = aug_train_df[['input' , 'output']]
    train_df = pd.concat([train , aug_train]) # 기존 train 데이터와 concat하기
    train_df = train_df.sample(frac=1).reset_index(drop=True) # 순서 섞기
    
    return train_df
    

def main() : 
    
    input_file = '../data/train.csv'
    output_file = '../data/aug_train.csv'
    #output_file = './data/aug_inference.csv'
    
    settings = {
        "transform_hangul": 0.6,
        "add_random_jongseong": 0.6,
        "apply_liaison": 0.5,
        "cho_to_jong": 0.4}  
    
    train , aug_train_df  = aug_data(input_file, 2 , settings) # 데이터 증강 
    train_aug = concat_data(train , aug_train_df) # 원본 데이터 + 증강 데이터 결합 
    train_aug.to_csv(output_file , index = False) # 증강 데이터 저장 
    
    #aug_train_df = aug_train_df.sample(frac=1).drop_duplicates().reset_index(drop = True) 
    #aug_train_df.to_csv(output_file , index = False) 
    
if __name__ == "__main__":
    main()
    