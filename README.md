# HangulRestore 

- 흔히 **에어비앤비체** 로 불리는 난독화된 한글 리뷰를 원본으로 복원하는 모델 개발
- 🫠 [난독화된 한글 리뷰 복원 모델 개발 회고](https://until.blog/@zzoming/-dacon--%EB%82%9C%EB%8F%85%ED%99%94%EB%90%9C-%ED%95%9C%EA%B8%80-%EB%A6%AC%EB%B7%B0-%EB%B3%B5%EC%9B%90-ai-%EB%AA%A8%EB%8D%B8-%EA%B0%9C%EB%B0%9C%EA%B8%B0)
<br>

![Image](https://github.com/user-attachments/assets/b6f6949b-f335-4093-99ca-b16f8cb3e2ba)

<br>

# 진행기간 및 성과

- 2025.01 ~ 2025.02 ( 약 1달간 진행 )
- 개인참가
- **상위 10% 내 등수 기록** (22등 / 291팀)
- F1 Score  : 0.44 → 0.85
  
# 데이터 셋 
Dacon : [난독화된 한글 리뷰 복원 AI 경진대회](https://dacon.io/competitions/official/236446/overview/description)
<br>

# 개발 프로세스 

### 1. Data Augmentation
- 한글 난독화 패턴을 반영하여 데이터 증강
- Train 데이터 1만 개 → 3만 개 확장
- **데이터 3만 개 (1 epoch) 학습 시 0.01 성능 향상 확인** 기록

### 2. BiLSTM Model Training
- 🙌 [BiLSTM Model Checkpoint](https://drive.google.com/drive/u/0/my-drive)
- 입출력의 글자 위치가 동일해야하는 대회 특성 상 **`Encoder`** 모델 및 음절단위 Tokenizer 활용
- Many-to-Many Classification 관점으로 학습 및 추론 진행
- 문맥상 자연스러운 변환에 한계 → 후처리(맞춤법 교정 등) 필요 

### 3. Gemma Model Training 
- 🙌 [LLM(Hugging Face)](https://huggingface.co/zzoming/hangul-restore-model)
- 문맥 및 맞춤법 교정 강화를 위해 SFT(Supervised Fine-Tuning) 진행 
- Quantization 및 PEFT(Parameter Efficient Fine-Tuning) 적용하여 경량화 학습
- [Fine-Tuning에 활용한 모델](https://huggingface.co/beomi/gemma-ko-7b)이 Chat Model이 아니므로 `Alpaca prompt` 활용 


### 프롬프트 엔지니어링
**instruction**

```
instruction 
당신은 한국어 맞춤법 및 문맥 교정 전문 도우미입니다.  
당신의 임무는 사용자로부터 입력받은 한국어 리뷰 문장을 읽고, 문맥에 맞지 않거나 맞춤법에 오류가 있는 글자를 찾아 다른 글자로 교체하는 것입니다. 
**띄어쓰기와 글자 수, 기호는 절대로 변경할 수 없으며, 한 글자를 다른 한 글자로 교체하는 방식으로만 수정되어야 합니다.** 
다음 조건을 반드시 지키세요. 

**조건**  
1. 띄어쓰기 수정은 절대 금지입니다. 띄어쓰기를 추가하거나 제거해서는 안 됩니다.
2. 글자를 추가하거나 삭제하지 마세요. 오직 기존 한 글자만 다른 글자로 교체해야 합니다.
3. 반드시 주어진 문장의 총 글자 수가 동일해야 합니다.
4. 기호, 숫자, 영어는 입력 그대로 출력해야 합니다.
5. 출력은 반드시 한국어로만 작성해야 합니다.

**예시**
### Input(입력):비가 올인 했으나 !! 너무 이뻐요 °ࡇ°  최고 뚀 올 거녜욤 ㅠㅠㅜㅜ 지권분들께서도 친졀하시고 너무 편해요 걈샤해요!!!!!
### Response(응답):비가 오긴 했으나 !! 너무 이뻐요 °ࡇ°  최고 또 올 거예요 ㅠㅠㅜㅜ 직원분들께서도 친절하시고 너무 편해요 감사해요!!!!!

이제 주어지는 문장을 위 조건에 따라 교정된 문장으로 출력하세요
```
**Alpaca Prompt**
```
"Below is an instruction that describes a task, paired with an input that provides further context.\n"
"아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
"Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
"### Instruction(명령어):{instruction}\n\n### Input(입력):{input}\n\n### Response(응답):{response}<eos>"
```
<br>

# 기술스택
| **Category**         | **Technologies**                                                                 |
|-----------------------|----------------------------------------------------------------------------------|
| **Programming**    | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) |
| **Data&AI**           | ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white) ![HuggingFace](https://img.shields.io/badge/-HuggingFace-FFD21E?logo=huggingface&logoColor=white)|
| **Web Interface**   | ![Gradio](https://img.shields.io/badge/-Gradio-F97316?logo=gradio&logoColor=white) |


<br>

# 데모
```bash
# 1. GitHub clone 
git clone https://github.com/ZZOMING-K/HangulRestore.git

# 2. 환경설정 
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. BiLSTM 모델 체크포인트 다운로드 후 지정 경로에 배치
(HangulRestore/BiLSTM/best_model_checkpoint.th)

# 4. 앱 실행
python app.py
```


