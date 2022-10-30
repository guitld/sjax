from ssl import ALERT_DESCRIPTION_UNSUPPORTED_EXTENSION
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import sjax


class Linear(sjax.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
    
    @sjax.module_method
    def __call__(self, x):
        if self.d_in:
            w = sjax.get_param('w', shape=(x.shape[-1], self.d_out))
        else:
            w = sjax.get_param('w', shape=(self.d_in, self.d_out))
        b = sjax.get_param('b', shape=(self.d_out,))
        return jnp.dot(x, w) + b

class Sequential(sjax.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = tuple(layers)

    @sjax.module_method
    def __call__(self, x):
        """Calls all layers sequentially."""
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return out

class Embedding(sjax.Module):
    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        self._vocab_size = vocab_size
        self._model_dimension = model_dimension

    @sjax.module_method
    def __call__(self, x):
        embedding_matrix = sjax.get_param('embedding_matrix', shape=(self._vocab_size, self._model_dimension))
        return jnp.take(embedding_matrix, x, axis=0)

class PositionalEncoding(sjax.Module):
    def __init__(self, max_seq_length, model_dimension):
        super().__init__()
        self._max_seq_length = max_seq_length
        self._model_dimension = model_dimension

    @sjax.module_method
    def __call__(self, x):
        p_id = np.expand_dims(np.arange(0, 5000), axis=1)
        freqs = np.power(10000., -np.arange(0, 512, 2) / 512)

        p_encodings = np.zeros(shape=(self._max_seq_length, self._model_dimension))
        p_encodings[:, 0::2] = np.sin(p_id * freqs)
        p_encodings[:, 1::2] = np.cos(p_id * freqs)
        p_encodings = jax.device_put(p_encodings)
        return x + p_encodings[:x.shape[0]]

class MaxPooling2D(sjax.Module):
    def __init__(self, pool_size, strides, padding='valid'):
        super().__init__()
        self._pool_size = pool_size
        self._padding = padding
        self._strides = strides

    @sjax.module_method
    def __call__(self, inputs):
        return jax.lax.reduce_window(inputs, -np.inf, jax.lax.max, self._pool_size, self._strides,self._padding)

class Sigmoid(sjax.Module):
    def __init__(self):
        super().__init__()

    @sjax.module_method
    def __call__(self, x):
        return jax.nn.sigmoid(x)

class LogSigmoid(sjax.Module):
    def __init__(self):
        super().__init__()

    @sjax.module_method
    def __call__(self, x):
        return jax.nn.log_sigmoid(x)

class Dropout(sjax.Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert (p > 0) and (p < 1), f"p must be in [0, 1]."
        self.p = p
    
    @sjax.module_method
    def __call__(self, x):
        keep_p = 1.0 - self.p
        keep = jax.random.bernoulli(sjax.current_rng(), keep_p, shape=x.shape)
        return keep * x / self.p

class ReLU(sjax.Module):
    def __init__(self):
        super().__init__()
    
    @sjax.module_method
    def __call__(self, x):
        return jax.nn.relu(x)

class LayerNorm(sjax.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    
    @sjax.module_method
    def __call__(self, x):
        return (x - x.mean()) / jnp.sqrt(x.var() + self.eps)

class MultiHeadAttention(sjax.Module):
    def __init__(self, model_dimension, n_heads=8):
        super().__init__()
        assert model_dimension % n_heads == 0, f'Model dimension must be divisible by the number of heads.'
        self._model_dimension = model_dimension
        self.n_heads = n_heads
        self.d_k = model_dimension // n_heads
        self.q = Linear(self._model_dimension, self._model_dimension)
        self.k = Linear(self._model_dimension, self._model_dimension)
        self.v = Linear(self._model_dimension, self._model_dimension)
        self.projection = Linear(self._model_dimension, self._model_dimension)
    
    def __call__(self, q, k, v, mask=None):
        out, attention = self.scaled_dot_product(q=q, k=k, v=v, mask=mask)
        out = out.reshape(self._model_dimension)
        out = self.projection(out)
        return out


    def scaled_dot_product(self, q, k, v, mask=None):
        q = self.q(q)
        k = self.k(k)
        v = self.k(v)

        q  = q.reshape((self.n_heads, self.d_k))
        k  = k.reshape((self.n_heads, self.d_k))
        v  = v.reshape((self.n_heads, self.d_k))

        score = (q @ k.T) / jnp.sqrt(self.d_k)

        if mask:
            score = jnp.where(mask == 0, score, -jnp.inf)

        score = jax.nn.softmax(score)

        v = score @ v

        return v, score

class PointWiseFeedForward(sjax.Module):
    def __init__(self, model_dimension, width_mult=4):
        super().__init__()
        self.linear1 = Linear(model_dimension, width_mult*model_dimension)
        self.linear2 = Linear(width_mult*model_dimension, model_dimension)

        self.dropout = Dropout()
        self.relu = ReLU()

    def __call__(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LogSoftmax(sjax.Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        return jax.nn.log_softmax(x)

class GeLU(sjax.Module):
    def __init__(self, approximate=False):
        super().__init__()
        self.approximate = approximate

    def __call__(self, x):
        return jax.nn.gelu(x, self.approximate)