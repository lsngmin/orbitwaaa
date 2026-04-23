"""
Numpy-only OrbitWarsActor inference. torch 의존성 없음.

config.yaml 값을 하드코딩해서 제출 환경에서 yaml 파싱 없이도 동작.
"""
import numpy as np

# ── Config (config.yaml에서 복사, 제출 환경 독립) ─────────────────────────────
EMBED_DIM              = 128
NUM_HEADS              = 8
HISTORY                = 20
MAX_PLANETS            = 40
MAX_FLEETS             = 100
PLANET_DIM             = 16
FLEET_DIM              = 7
ACTION_DIM             = MAX_PLANETS + 2   # 42
PLANET_TEMPORAL_LAYERS = 2
FLEET_TEMPORAL_LAYERS  = 1
FLEET_TEMPORAL         = True
LOCAL_LAYERS           = 2
GLOBAL_LAYERS          = 4


# ── Math primitives ───────────────────────────────────────────────────────────

def _sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def _layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias

def _linear(x, weight, bias):
    return x @ weight.T + bias


# ── Multi-head self-attention ─────────────────────────────────────────────────

def _mha(x, in_w, in_b, out_w, out_b):
    B, S, E = x.shape
    H, D = NUM_HEADS, E // NUM_HEADS

    qkv      = _linear(x, in_w, in_b)           # (B, S, 3E)
    q, k, v  = np.split(qkv, 3, axis=-1)

    q = q.reshape(B, S, H, D).transpose(0, 2, 1, 3)  # (B, H, S, D)
    k = k.reshape(B, S, H, D).transpose(0, 2, 1, 3)
    v = v.reshape(B, S, H, D).transpose(0, 2, 1, 3)

    attn = _softmax(q @ k.transpose(0, 1, 3, 2) / D**0.5, axis=-1)
    out  = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, E)

    return _linear(out, out_w, out_b)


# ── Transformer encoder layer (post-norm, ReLU) ───────────────────────────────

def _tf_layer(x, pfx, w):
    attn = _mha(
        x,
        w[f'{pfx}self_attn.in_proj_weight'],
        w[f'{pfx}self_attn.in_proj_bias'],
        w[f'{pfx}self_attn.out_proj.weight'],
        w[f'{pfx}self_attn.out_proj.bias'],
    )
    x = _layer_norm(x + attn, w[f'{pfx}norm1.weight'], w[f'{pfx}norm1.bias'])

    ffn = np.maximum(0, _linear(x, w[f'{pfx}linear1.weight'], w[f'{pfx}linear1.bias']))
    ffn = _linear(ffn, w[f'{pfx}linear2.weight'], w[f'{pfx}linear2.bias'])
    x   = _layer_norm(x + ffn, w[f'{pfx}norm2.weight'], w[f'{pfx}norm2.bias'])
    return x

def _tf_encoder(x, name, n_layers, w):
    for i in range(n_layers):
        x = _tf_layer(x, f'{name}.layers.{i}.', w)
    return x


# ── Forward pass ──────────────────────────────────────────────────────────────

def forward(obs_flat, w):
    """
    obs_flat : (1, HISTORY*(MAX_PLANETS*PLANET_DIM + MAX_FLEETS*FLEET_DIM)) float32
    returns  : action_logits (1, MAX_PLANETS, ACTION_DIM)
    """
    p_size = MAX_PLANETS * PLANET_DIM   # 520
    f_size = MAX_FLEETS  * FLEET_DIM    # 700

    obs   = obs_flat.reshape(1, HISTORY, p_size + f_size)
    p_raw = obs[:, :, :p_size].reshape(1, HISTORY, MAX_PLANETS, PLANET_DIM)
    f_raw = obs[:, :, p_size:].reshape(1, HISTORY, MAX_FLEETS,  FLEET_DIM)

    # 임베딩
    p_emb = _linear(p_raw, w['planet_embed.weight'], w['planet_embed.bias'])  # (1,H,P,E)
    f_emb = _linear(f_raw, w['fleet_embed.weight'],  w['fleet_embed.bias'])   # (1,H,F,E)

    # Temporal positional encoding
    t_idx = np.arange(HISTORY)
    p_pos = w['planet_temporal_pos.weight'][t_idx]   # (H, E)

    # Planet temporal attention: (1*P, H, E)
    p_t = p_emb.transpose(0, 2, 1, 3).reshape(MAX_PLANETS, HISTORY, EMBED_DIM)
    p_t = p_t + p_pos[np.newaxis]
    p_t = _tf_encoder(p_t, 'planet_temporal_attn', PLANET_TEMPORAL_LAYERS, w)
    p_t = p_t[:, -1, :].reshape(1, MAX_PLANETS, EMBED_DIM)

    # Fleet temporal attention: (1*F, H, E)
    if FLEET_TEMPORAL:
        f_pos = w['fleet_temporal_pos.weight'][t_idx]
        f_t   = f_emb.transpose(0, 2, 1, 3).reshape(MAX_FLEETS, HISTORY, EMBED_DIM)
        f_t   = f_t + f_pos[np.newaxis]
        f_t   = _tf_encoder(f_t, 'fleet_temporal_attn', FLEET_TEMPORAL_LAYERS, w)
        f_t   = f_t[:, -1, :].reshape(1, MAX_FLEETS, EMBED_DIM)
    else:
        f_t = f_emb[:, -1, :, :]   # (1, F, E)

    # Local attention: (1, P+F, E)
    local = np.concatenate([p_t, f_t], axis=1)
    local = _tf_encoder(local, 'local_attn', LOCAL_LAYERS, w)
    p_local = local[:, :MAX_PLANETS, :]

    # Global attention: (1, P, E)
    glob = _tf_encoder(p_local, 'global_attn', GLOBAL_LAYERS, w)

    # Actor head
    h = np.maximum(0, _linear(glob, w['actor.0.weight'], w['actor.0.bias']))
    return _linear(h, w['actor.2.weight'], w['actor.2.bias'])   # (1, P, ACTION_DIM)


# ── Action sampling (학습 분포와 동일) ────────────────────────────────────────

def _sample_categorical(logits):
    logits = logits - logits.max()
    probs  = np.exp(logits)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


def get_action(obs_flat, w):
    """
    obs_flat : (1, obs_dim) float32 numpy array
    returns  : (MAX_PLANETS, ACTION_DIM) float32 numpy array
    """
    logits = forward(obs_flat, w)[0]   # (P, ACTION_DIM)

    # Launch: Bernoulli
    launch = (np.random.random(MAX_PLANETS) < _sigmoid(logits[:, 0])).astype(np.float32)

    # Ships ratio: Normal(sigmoid(logit), 0.1)
    ships_ratio = np.clip(
        _sigmoid(logits[:, 1]) + 0.1 * np.random.standard_normal(MAX_PLANETS),
        0.0, 1.0,
    ).astype(np.float32)

    # Target: Categorical
    target = np.array([_sample_categorical(logits[i, 2:]) for i in range(MAX_PLANETS)])
    target_oh = np.zeros((MAX_PLANETS, MAX_PLANETS), dtype=np.float32)
    target_oh[np.arange(MAX_PLANETS), target] = 1.0

    return np.concatenate([launch[:, None], ships_ratio[:, None], target_oh], axis=-1)


# ── Weight loading ────────────────────────────────────────────────────────────

def load_weights(path):
    data = np.load(str(path))
    return dict(data)
