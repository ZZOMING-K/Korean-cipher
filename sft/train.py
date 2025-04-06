import yaml
import torch
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
import wandb
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import pandas as pd
from sft.utils import create_train_datasets
from dotenv import load_dotenv
import os 
from huggingface_hub import login

def initialize_env() :
    
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    wandb.login(key=wandb_api_key) # wandb login
    login(hf_api_key) #huggingface login
    
class KoreanLLMTrainer:
    
    def __init__(self, config_path: str = '../config/train.yaml'):

        with open(config_path, 'r') as file: # 설정 파일 로드
            self.config = yaml.safe_load(file)
        
        # 데이터 로드
        self.train_dataset, self.eval_dataset = self._load_dataset()
        
        # 모델 및 토크나이저 로드
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        
        # LoRA 설정
        self.lora_config = self._configure_lora()
        
        # 학습 인자 설정
        self.training_args = self._configure_training_args()
    
    def _load_dataset(self): 
            
            df = pd.read_csv(self.config['data']['path'])
            
            df = df[df['restore_review'] != df['output']].reset_index(drop = True)
            
            train_dataset, valid_dataset = create_train_datasets(df)
            return train_dataset, valid_dataset
    
    def _load_model(self):

        bnb_config =  BitsAndBytesConfig(**self.config['quantization'])
        model = AutoModelForCausalLM.from_pretrained(
            self.config['training']['base_model'],
            torch_dtype=torch.bfloat16,
            device_map='auto',
            quantization_config = bnb_config
        )
        
        return model
    
    def _load_tokenizer(self):

        tokenizer = AutoTokenizer.from_pretrained(
            self.config['training']['base_model']
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        
        return tokenizer
    
    def _configure_lora(self) : 
     
        return LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['alpha'],
            lora_dropout=self.config['lora']['dropout'],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.config['lora']['target_modules']
        )
        
    
    def _configure_training_args(self) :
 
        train_config = self.config['training']
        
        return SFTConfig(
            output_dir=train_config['output_dir'],
            eval_strategy=train_config['eval_strategy'],
            eval_steps = train_config['eval_steps'],
            per_device_train_batch_size=train_config['batch_size'],
            per_device_eval_batch_size=train_config['batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            num_train_epochs=train_config['epochs'],
            lr_scheduler_type=train_config['lr_scheduler_type'],
            learning_rate= float(train_config['learning_rate']),
            warmup_ratio=train_config['warmup_ratio'],
            logging_strategy=train_config['logging_strategy'],
            logging_steps=train_config['logging_steps'],
            save_strategy=train_config['save_strategy'],
            seed = train_config['seed'],
            bf16=True,
            run_name=f"gemma-ko-{train_config['epochs']}-SFT",
            report_to='wandb',
            dataset_text_field='text',
            push_to_hub = True,
            optim = train_config['optimizer']
        )
        
    
    def train(self):

        try :
            trainer = SFTTrainer(
                model=self.model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                args=self.training_args,
                peft_config = self.lora_config)
            
            # 학습 시작
            trainer.train()
            
            adapter_dir = self.config['training']['adapter_dir']
            
            # 최종 모델 저장
            trainer.model.save_pretrained(adapter_dir)  
            self.tokenizer.save_pretrained(adapter_dir)
            
            logging.info(f"Training completed. Model saved to {self.config['training']['output_dir']}")
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise 
        

def main():
    initialize_env() # 환경 설정 초기화
    trainer = KoreanLLMTrainer()
    trainer.train()

if __name__ == "__main__":
    main()