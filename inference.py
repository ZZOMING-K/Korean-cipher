import torch
import yaml
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from BiLSTM.model import BiLSTMModel
import pickle
import contextlib
import sys

@contextlib.contextmanager
def temporary_sys_path(path):
    
    original_sys_path = sys.path.copy()
    sys.path.append(path)
    
    try:
        yield
    finally:
        sys.path = original_sys_path


class KoreanLLMInference:
    
    def __init__(self, config_path='./config/inference.yaml'):
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        with temporary_sys_path('./BiLSTM'):
            
            with open('BiLSTM/tokenizer/input_tokenizer.pkl', 'rb') as f:
                input_tokenizer = pickle.load(f)
    

            with open('BiLSTM/tokenizer/output_tokenizer.pkl', 'rb') as f:
                output_tokenizer = pickle.load(f)

            
        # BiLSTM 모델 및 토크나이저 로드
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        
        self.bilstm_model = self._load_bilstm_model()

        # LLM 모델 및 토크나이저 로드

        self.llm_model = self._load_llm_model()
        print("LLM 모델 로딩 완료")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['base_model'])

        # 프롬프트 템플릿 정의
        self.prompt_input = (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
            "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "### Instruction(명령어):{instruction}\n\n### Input(입력):{input}\n\n### Response(응답):"
        )


    def _load_bilstm_model(self):

        model = BiLSTMModel(
            vocab_size=len(self.input_tokenizer.char2idx),
            embedding_dim=256,
            hidden_size=512,
            output_size=len(self.output_tokenizer.char2idx),
            num_layers=3,
            dropout=0.05,
            bidirectional=True
        )
        model.load_state_dict(torch.load(self.config['model']['bilstm_path'], map_location=torch.device('cuda')))
        model.eval()
        
        return model

    def _load_llm_model(self):

        model_id = self.config['model']['base_model']
        
        print(model_id)
        
        llm = LLM(
            model=model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization="bitsandbytes",
            max_model_len=4096,
            gpu_memory_utilization=0.8
        )
        
        return llm

    def bilstm_correct(self, input_texts):

        input_ids = [self.input_tokenizer.encode(text) for text in input_texts]
        input_tensors = [torch.LongTensor(ids).unsqueeze(0) for ids in input_ids]

        corrected_texts = []
        
        with torch.no_grad():
            
            for input_tensor in input_tensors:
                
                logits = self.bilstm_model(input_tensor)
                predicted_indices = torch.argmax(logits, dim=-1).squeeze(0).tolist()
                corrected_text = ''.join([self.output_tokenizer.idx2char[idx] for idx in predicted_indices if idx < len(self.output_tokenizer.idx2char)])
                corrected_texts.append(corrected_text)

        return corrected_texts

    def llm_generate(self, corrected_texts, instructions):

        generated_responses = []
        
        for corrected_text, instruction in zip(corrected_texts, instructions):
            
            print(corrected_text)
            
            prompt = self.prompt_input.replace('{instruction}', instruction).replace('{input}', corrected_text)
            
            output = self.llm_model.generate(
                prompt,
                sampling_params=SamplingParams(
                    temperature=self.config['inference']['temperature'],
                    top_p=self.config['inference']['top_p'],
                    top_k=self.config['inference']['top_k'],
                    seed=self.config['inference']['seed'],
                    max_tokens=len(corrected_text),
                    stop_token_ids=[self.tokenizer.eos_token_id]
                )
            )
            answer = output[0].outputs[0].text.strip()
            generated_responses.append(answer)
            print(answer)
        
        return generated_responses

    def inference(self, df):

        corrected_texts = self.bilstm_correct(df['input'].tolist())
        instructions = [ open('./data/instruction.txt').read() for _ in range(len(df)) ]
        
        generated_responses = self.llm_generate(corrected_texts, instructions)

        return generated_responses

    def save_results(self, output_path, results):
        
        output_df = pd.read_csv(output_path)
        output_df['output'] = results
        output_df.to_csv(output_path, index=False)


def main():
    
    df = pd.read_csv("./data/test.csv")

    inference_pipeline = KoreanLLMInference()

    results = inference_pipeline.inference(df)

    inference_pipeline.save_results(output_path = "./data/sample_submission.csv", results = results)
    print("Inference 완료.")


if __name__ == "__main__":
    main()