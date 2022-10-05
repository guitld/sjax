import sjax
from sjax import nn

class Transformer(sjax.Module):
    def __init__(self, model_dimension, max_seq_len, voc_size_src, voc_size_tgt, n_heads, n_layers):
        super().__init__()

        self.encoder = Encoder(voc_size_src, 
                               max_seq_len,
                               model_dimension,
                               n_layers,
                               n_heads)

        self.decoder = Decoder(voc_size_tgt,
                               max_seq_len,
                               model_dimension,
                               n_layers,
                               n_heads)

    def __call__(self, seq_src, seq_tgt, mask_src, mask_tgt):
        seq_src = self.encoder(seq_src, mask_src)
        return self.decoder(seq_src, seq_tgt, mask_src, mask_tgt)

class Encoder(sjax.Module):
    def __init__(self, voc_size, max_seq_len, model_dimension, n_layers, n_heads=8):
        self.embedding = nn.Embedding(voc_size, model_dimension)
        self.pe = nn.PositionalEncoding(max_seq_len, model_dimension)
        self.layers = nn.Sequential([EncoderLayer(model_dimension, n_heads) for _ in range(n_layers)])

    def __call__(self, x, mask=None):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.layers(x, mask=mask)
        return x

class EncoderLayer(sjax.Module):
    def __init__(self, model_dimension, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self._model_dimension = model_dimension
        self.attention = nn.MultiHeadAttention(model_dimension, n_heads)
        self.ffn = nn.PointWiseFeedForward(model_dimension)
        self.norm = nn.LayerNorm()
        self.dropout = nn.Dropout()
    
    def __call__(self, x, mask=None):
        res = x
        x = self.attention(x, mask)
        x = self.norm(x + res)
        x = self.dropout(x)

        res = x
        x = self.ffn()
        x = self.norm(x + res)
        x = self.dropout(x)
        return x

class Decoder(sjax.Module):
    def __init__(self, voc_size, max_seq_len, model_dimension, n_layers, n_heads=8):
        self.embedding = nn.Embedding(voc_size, model_dimension)
        self.pe = nn.PositionalEncoding(max_seq_len, model_dimension)
        self.layers = nn.Sequential([DecoderLayer(model_dimension, n_heads) for _ in range(n_layers)])
        self.linear = nn.Linear(model_dimension, voc_size)
        self.softmax = nn.LogSoftmax()

    def __call__(self, x_src, x_tgt, mask_src, mask_tgt):
        x = self.embedding(x_tgt)
        x = self.pe(x)
        x = self.layers(x_src, x, mask_src, mask_tgt)
        x = self.linear(x)
        return self.softmax(x)

class DecoderLayer(sjax.Module):
    def __init__(self, model_dimension, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self._model_dimension = model_dimension
        self.attention = nn.MultiHeadAttention(model_dimension, n_heads)
        self.attention_enc = nn.MultiHeadAttention(model_dimension, n_heads)
        self.ffn = nn.PointWiseFeedForward(model_dimension)
        self.norm = nn.LayerNorm()
        self.dropout = nn.Dropout()

    def __call__(self, x_src, x_tgt, mask_src, mask_tgt):
        res = x_tgt
        x_tgt = self.attention(q=x_tgt, k=x_tgt, v=x_tgt, mask=mask_tgt)

        x = self.norm(x + res)
        x = self.dropout(x)

        if x_src:
            res = x
            x = self.attention_enc(q=x, k=x_src, v=x_src, mask=mask_src)

            x = self.norm(x + res)
            x = self.dropout()
        
        res = x
        x = self.ffn(x)

        x = self.norm(x + res)
        x = self.dropout(x)
        return x
