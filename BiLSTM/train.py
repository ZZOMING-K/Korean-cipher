import torch
from utils import TextProcessor, CharTokenizer, CharDataset, char_collate_fn
from model import BiLSTMModel
from utils import set_seed, calculate_accuracy, evaluate
from torch.utils.data import DataLoader
import os
import torch
import pickle


class Config:
    
    # Data settings
    DATA_PATH = '../data/aug_train.csv'
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Model architecture
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3
    DROPOUT = 0.05
    BIDIRECTIONAL = True
    
    # Training settings
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 10
    
    # Paths
    CHECKPOINT_DIR = './checkpoints/'
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best_model_checkpoint.pth')
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
  
class Trainer : 
    
    def __init__(self, model , train_loader , test_loader , config , output_size=1614) :
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        
        self.output_size = output_size
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
    
        self.num_epochs = config.NUM_EPOCHS
        self.device = config.DEVICE
        self.checkpoint_path = config.CHECKPOINT_PATH
        
    
    def train(self) :
        
        best_val_accuracy = float('-inf')
       
        for epoch in range(self.num_epochs) :
            
            train_loss , train_correct , train_total = 0.0 , 0.0 , 0

            self.model.train()

            for batch_idx , (input_ids , output_ids) in enumerate(self.train_loader) :

                input_ids , output_ids  = input_ids.to(self.device) , output_ids.to(self.device)
                logits = self.model(input_ids)
                loss = self.criterion(logits.view(-1 , self.output_size) , output_ids.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += calculate_accuracy(logits.view(-1, self.output_size), output_ids.view(-1)) * output_ids.size(0)
                train_total += output_ids.size(0)

                # 100 배치마다 진행 상황 출력
                if (batch_idx + 1) % 100 == 0:
                    avg_loss = train_loss / (batch_idx + 1)
                    print(f"[Train] Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx+1}/{len(self.train_loader)}], Loss: {avg_loss:.4f}")

            train_accuracy = train_correct / train_total
            train_loss /= len(self.train_loader)

            # Validation
            val_loss, val_accuracy = evaluate(self.model, self.test_loader , self.criterion, self.device , self.output_size)

            print(f'Epoch {epoch+1}/{self.num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

            # 검증 손실이 최소일 때 체크포인트 저장
            if val_accuracy >= best_val_accuracy:
                print(f'Validation accuracy improved from {best_val_accuracy:.4f} to {val_accuracy:.4f}. 체크포인트를 저장합니다.')
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_model_checkpoint.pth')
            

def main():
    
    config = Config()
    
    set_seed(42)
    
    print("Loading and preprocessing data...")
    df = TextProcessor.preprocess_dataframe(config.DATA_PATH)
    train_data, test_data = TextProcessor.split_df(df, test_size=0.2, random_state=42)
    
    print("Creating tokenizers...")
    input_tokenizer = CharTokenizer('input', train_data)
    output_tokenizer = CharTokenizer('output', train_data)
    
    print(f"Input Character vocabulary size: {len(input_tokenizer.char2idx)}")
    print(f"Output Character vocabulary size: {len(output_tokenizer.char2idx)}")
    
    # tokenizer 저장
    with open('./tokenizer/input_tokenizer.pkl', 'wb') as f:
        pickle.dump(input_tokenizer, f)

    with open('./tokenizer/output_tokenizer.pkl', 'wb') as f:
        pickle.dump(output_tokenizer, f)
        
    print("Tokenizer 저장 완료 ✅ (input_tokenizer.pkl, output_tokenizer.pkl)")
    
    # Create datasets
    train_dataset = CharDataset(train_data, input_tokenizer, output_tokenizer)
    test_dataset = CharDataset(test_data, input_tokenizer, output_tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=char_collate_fn,
        drop_last=True,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=char_collate_fn,
        drop_last=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BiLSTMModel(
        
        vocab_size=len(input_tokenizer.char2idx),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_size=config.HIDDEN_SIZE,
        output_size=len(output_tokenizer.char2idx),
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        bidirectional=True
    
    ).to(device)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        output_size=len(output_tokenizer.char2idx)
    )
       
    print("Starting training...")
    trainer.train()
    
if __name__ == "__main__":
    main()

  
  
  
  