import pandas as pd
import torch
from utils import CharTokenizer
from model import BiLSTMModel
import re
import pickle

class BiLSTMInference:
    
    def __init__(self, model_path, input_tokenizer, output_tokenizer, device=None):
    
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.model = self._load_model(model_path) # Load the model

    def _load_model(self, model_path):
 
        model = BiLSTMModel(
            vocab_size=len(self.input_tokenizer.char2idx),
            embedding_dim=256,
            hidden_size=512,
            output_size=len(self.output_tokenizer.char2idx),
            num_layers=3,
            dropout=0.05,
            bidirectional=True
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model

    def predict(self, input_texts):

        input_ids = [self.input_tokenizer.encode(text) for text in input_texts]
        input_tensors = [torch.LongTensor(ids).unsqueeze(0).to(self.device) for ids in input_ids]

        result_sentences = []

        with torch.no_grad():
            for input_tensor in input_tensors:
                logits = self.model(input_tensor)
                predicted_indices = torch.argmax(logits, dim=-1).squeeze(0).tolist()
                answer = ''.join([self.output_tokenizer.idx2char[idx] for idx in predicted_indices if idx < len(self.output_tokenizer.idx2char)])
                result_sentences.append(answer)

        return result_sentences

def remove_extra_spaces(text):

    return re.sub(r'\s+', ' ', text).strip()


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    with open('./tokenizer/input_tokenizer.pkl', 'rb') as f:
        input_tokenizer = pickle.load(f)
    
    with open('./tokenizer/output_tokenizer.pkl', 'rb') as f:
        output_tokenizer = pickle.load(f)


    data = pd.read_csv('../data/aug_inference.csv')
    
    data['input'] = data['input'].apply(remove_extra_spaces)
    data['output'] = data['output'].apply(remove_extra_spaces)

    data = data[data['input'] != data['output']].drop_duplicates().reset_index(drop=True)

    model_path = 'best_model_checkpoint.pth'
    inference = BiLSTMInference(model_path, input_tokenizer, output_tokenizer, device)

    input_texts = data['input'].tolist()
    result_sentences = inference.predict(input_texts)
    print(result_sentences)

    data['restore_review'] = result_sentences
    data.to_csv('../data/aug_inference_result.csv', index=False)
    print("Inference 완료. 결과가 '../data/aug_inference_result.csv'에 저장되었습니다.")


if __name__ == "__main__":
    main()