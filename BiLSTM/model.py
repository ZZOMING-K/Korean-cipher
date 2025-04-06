import torch.nn as nn

class BiLSTMModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, 
                 num_layers=3, dropout=0.05, bidirectional=True):
 
        super(BiLSTMModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.layer_norm_input = nn.LayerNorm(embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_dim, output_size)
        
    def forward(self, x):

        # Embedding
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x = self.layer_norm_input(x)
        
        # LSTM
        outputs, _ = self.lstm(x)
        
        # Output layer
        logits = self.fc(outputs)
        
        return logits