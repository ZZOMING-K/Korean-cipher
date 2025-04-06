import torch
import yaml
import pandas as pd
from transformers import AutoTokenizer, set_seed
from utils import create_test_datasets
from vllm import LLM, SamplingParams

set_seed(42)

class KoreanLLMInference: 
    
    def __init__(self, config_path: str = '../config/inference.yaml'):
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.test_dataset = self._load_dataset() # data load 
        self.model = self._load_model() # model load 
        self.tokenizer = self._load_eos_tokenizer() # tokenizer load 
        
    def _load_dataset(self):
        
        df = pd.read_csv(self.config['data']['path'])
        test_dataset = create_test_datasets(df)    
        
        return test_dataset 
    
    def _load_model(self): 
        
        model_id = self.config['model']['base_model']
        
        llm = LLM(model = model_id , 
                  dtype = torch.float16 , 
                  trust_remote_code = True , 
                  quantization = "bitsandbytes" , 
                  max_model_len = 4096, 
                 gpu_memory_utilization = 0.8) 

        return llm
    
    def _load_eos_tokenizer(self): 
        tokenizer = AutoTokenizer.from_pretrained(self.config['model']['base_model'])
        return tokenizer
    
    
    def inference(self): 
        
        restore_reviews = []
        
        llm = self.model 
        llm.llm_engine.scheduler_config.max_num_seqs = 64 
        
        sampling_params = SamplingParams(temperature = self.config['inference']['temperature'] , 
                                         top_p = self.config['inference']['top_p'], 
                                         top_k = self.config['inference']['top_k'], 
                                         seed = self.config['inference']['seed'] , 
                                         max_tokens = self.config['inference']['max_tokens'] , 
                                         stop_token_ids = [self.tokenizer.eos_token_id])

        outputs = llm.generate(self.test_dataset['text'], sampling_params)

        for output in outputs :
            generated_text = output.outputs[0].text
            print(generated_text)
            restore_reviews.append(generated_text)

        return restore_reviews
    
    def save_dataset(self, restore_reviews, output_path):
    
        df = pd.read_csv(self.config['data']['path'])
        df['restore_review'] = restore_reviews
        df.to_csv(output_path, index=False)
    
def main():
    inference_model = KoreanLLMInference()
    restore_reviews = inference_model.inference()
    inference_model.save_dataset(restore_reviews, '../data/gemma_inference.csv') 

if __name__ == "__main__":
    main()