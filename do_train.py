import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim

# for tfrecords dataset
from torch.utils.data import DataLoader, IterableDataset
from tensorflow import string as tf_string
from tensorflow import reshape as tf_reshape
from tensorflow import float32 as tf_float32
import tensorflow.io as tf_io
import tensorflow.data as tf_data
import tensorflow as tf

# PyTorch/XLA imports
#import pytorch_lightning as pl
import lightning.pytorch as pl
import torch_xla.core.xla_model as xm
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor # For flexible checkpointing


# ---------------------------------------------------------------------------- #
#                                 Configuration                                #
# ---------------------------------------------------------------------------- #
TARGET_EPOCHS = 100
DO_EPOCHS = 100
CONTINUE_TRAINING = False # Set to True to attempt loading a checkpoint
SAVE_LAST_CHECKPOINT_ONLY = False # Only save the last checkpoint to GCS (saves space)

# GCS Checkpoint Path
GCS_CHECKPOINT_PATH = "gs://model_v6_ckpt"

# !DEBUG
train_dir = "gs://openfwi_tfrecord/*.tfrecord"
#train_dir = "gs://openfwi_valid_tfrecords/*.tfrecord"
valid_dir = "gs://openfwi_valid_tfrecords/*.tfrecord"
_TRAIN_SAMPLES = 250 * 1780
_VALID_SAMPLES = 250 * 100

# Ensure this bucket is accessible and writable by your TPU VM service account.

_BASE_BATCH_SIZE = 8
_BASE_LR_INIT = 1e-4

WARMUP_EPOCHS = 1
LR_END = 1e-6
WEIGHT_DECAY = 1e-1

CROP_X = 4
CROP_Y = 8

N_TRIAL = 5
EMB_DIM = 384
HC_CHANNEL = 2
N_HEADS = 12

ENCODER_BLOCK = 4
N_BLOCK = 8

_NUM_DIVICES_CONCEPTUAL = 4
_UNIT_RAM_GB_CONCEPTUAL = 32

BATCH_SIZE = int( _BASE_BATCH_SIZE * _UNIT_RAM_GB_CONCEPTUAL / 16)
BATCH_SIZE_GLOBAL = BATCH_SIZE* _NUM_DIVICES_CONCEPTUAL
LR_INIT = _BASE_LR_INIT * BATCH_SIZE_GLOBAL / 32

GLOBAL_TRAIN_STEPS_PER_EPOCH = math.ceil(_TRAIN_SAMPLES / BATCH_SIZE_GLOBAL)
TOTAL_TRAIN_STEPS = GLOBAL_TRAIN_STEPS_PER_EPOCH * TARGET_EPOCHS
WARMUP_STEPS = GLOBAL_TRAIN_STEPS_PER_EPOCH * WARMUP_EPOCHS

# ---------------------------------------------------------------------------- #
#                                  Data Modules                                #
# ---------------------------------------------------------------------------- #

def parse_tfrecord_fn(example):
    feature_description = {
        'seis': tf_io.FixedLenFeature([], tf_string),
        'velo': tf_io.FixedLenFeature([], tf_string),
        "style": tf_io.FixedLenFeature([], tf_string)
    }
    example = tf_io.parse_single_example(example, feature_description)
    seis = tf_io.decode_raw(example['seis'], tf_float32)
    seis = tf_reshape(seis, [5, 1000, 70]) 

    velo = tf_io.decode_raw(example['velo'], tf_float32)
    velo = tf_reshape(velo, [70, 70])
    return seis, velo


def preprocessing(s, v):#, sample_weights
    # binding random flip
    
    if tf.random.uniform(shape = [1], maxval = 2, dtype = tf.dtypes.int32) == 1 :
        s = tf.reverse(s, axis = [-1])
        #s = tf.reverse(s, axis = [0])
        v = tf.reverse(v, axis = [-1])

    # binding random crop.
    # 1024x72 and 72x72 are more easy to compute than 1000x70 and 70x70.
    s = tf.pad(s, ((0, 0),(0, CROP_Y+ 12), (CROP_X+ 1, CROP_X+ 1)), "REFLECT")
    s = tf.pad(s, ((0, 0),(CROP_Y+ 12, 0), (0, 0)))
    idx_x = tf.random.uniform(shape = (), minval = 0, maxval = CROP_X*2+ 1, dtype = tf.dtypes.int32)
    #s = tf.map_fn(crop_s(idx_x), s)
    #s = tf.slice(s, [0, 0, idx_x], [5, 1024, 72])
    idx_y = tf.random.uniform(shape = (), minval = 0, maxval = CROP_Y*2 +1, dtype = tf.dtypes.int32)
    s = tf.slice(s, [0, idx_y, idx_x], [5, 1024, 72])
    v = tf.pad(v, ((1, 1), (CROP_X+ 1, CROP_X+ 1)), "REFLECT")
    v = tf.slice(v, [0, idx_x], [72, 72])


    # feature engineering to avoid precision damage
    s_abs = tf.math.abs(s)
    s_log = tf.math.log(s_abs+ np.exp(-50).astype(np.float32)) * (np.sqrt(2)/25).astype(np.float32) + np.sqrt(2).astype(np.float32)
    s_sqrt = tf.math.sign(s)* tf.math.sqrt(s_abs) * 2
    s = tf.stack([s_log, s_sqrt], axis = -1) # 5, 1024, 72, 2

    # v scaling augments
    scaling_size = tf.random.uniform(shape = (), minval = 205, maxval = 320, dtype = tf.dtypes.int32)
    scaling_factor = tf.cast(256 / scaling_size, tf.float32)

    s = tf.image.resize(s, [scaling_size, 72], method = "area")

    def pad_button():
        return tf.pad(s, [[0, 0], [0, 256- scaling_size], [0, 0], [0, 0]])

    def crop_button():
        return tf.slice(s, [0,0,0,0], [5, 256, 72, 2])
    
    s = tf.cond(scaling_size < 256,
                pad_button,
                crop_button
            )

    # normalize
    v = v/1000* scaling_factor - 3
    
    return s, v

class TFRecordLightningDataset(IterableDataset):
    def __init__(self, tfrecord_path):
        self.path = tfrecord_path

    def __iter__(self):
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        files = tf_io.gfile.glob(self.path)
        np.random.shuffle(files)
        files = files[rank::world_size]  # shard files

        dataset = tf_data.Dataset.from_tensor_slices(files)
        dataset = dataset.interleave(
            lambda f: tf_data.TFRecordDataset(f, compression_type="GZIP")
            ).map(parse_tfrecord_fn, num_parallel_calls=16)
        dataset = dataset.map(preprocessing, num_parallel_calls=16).prefetch(BATCH_SIZE*2)

        for raw in dataset:
            seis, velo = raw
            zelo = np.zeros_like(velo, dtype=np.float32)
            yield (torch.tensor(seis.numpy()), torch.tensor(zelo)), torch.tensor(velo.numpy())

# ---------------------------------------------------------------------------- #
#                                  Model Layers                                #
# ---------------------------------------------------------------------------- #
class _Conv2D_KerasCompatible(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x) # channels_first operation
        x = x.permute(0, 2, 3, 1)
        return x

class _AveragePooling2DKerasCompatible(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.pool(x) # channels_first operation
        x = x.permute(0, 2, 3, 1)
        return x

class IdentityVectorInitializer(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def __call__(self, shape, dtype=torch.float32):
        tmp = torch.randn(shape, dtype=dtype) * 1e-3
        vec = torch.eye(shape[-1], dtype=dtype)[self.idx].reshape(shape) /4
        return vec + tmp
    
class bSiLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.beta = nn.Parameter(torch.full([channels], 1.702))

    def forward(self, inputs):

        beta_tensor = self.beta.view(*([1] * (inputs.ndim - 1)), -1,)
        #beta_tensor = beta_tensor.to(inputs.device).type_as(inputs)

        return inputs * torch.sigmoid(beta_tensor * inputs)

class ResConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = _Conv2D_KerasCompatible(in_channels=channels, out_channels=channels // 2, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = _Conv2D_KerasCompatible(in_channels=channels // 2, out_channels=channels // 2, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        self.proj = nn.Linear(channels // 2, channels)

    def forward(self, inputs):
        x = self.relu1(self.conv1(inputs))
        x = self.relu2(self.conv2(x))
        return inputs + self.proj(x)

class SEISemb(nn.Module):
    def __init__(self):
        super().__init__()
        self.siren1 = nn.Linear(2, 16)
        self.siren2 = nn.Linear(16, 32)
        self.linear_conv = _Conv2D_KerasCompatible(in_channels=2, out_channels=32, kernel_size=3, padding='same')
        
        self.to_patch = nn.Sequential(
            nn.Linear(32 + 32, 128),
            ResConvBlock(128),
            ResConvBlock(128),
            _Conv2D_KerasCompatible(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0),
            ResConvBlock(256),
            ResConvBlock(256),
            nn.Linear(256, 384),
            _AveragePooling2DKerasCompatible(kernel_size=(2, 1)),
            ResConvBlock(384),
            ResConvBlock(384),
            ResConvBlock(384),
            _Conv2D_KerasCompatible(in_channels=384, out_channels=EMB_DIM, kernel_size=(2, 3), stride=(2, 3), padding=0),
            ResConvBlock(EMB_DIM),
            ResConvBlock(EMB_DIM),
            ResConvBlock(EMB_DIM),
        )

    def forward(self, x):
        siren1_out = torch.sin(self.siren1(x))
        siren2_out = torch.sin(self.siren2(siren1_out))
        linear_conv_out = self.linear_conv(x)
        
        x = torch.cat([siren2_out, linear_conv_out], dim=-1)
        
        patch_output = self.to_patch(x)
        
        B, H_prime, W_prime, C = patch_output.shape
        return patch_output.reshape(B, H_prime * W_prime, C)

class VELOemb(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.AvgPool2d(kernel_size=(6, 6), stride=(6, 6)),
            nn.Conv2d(1, EMB_DIM, 1, stride=1, padding=0),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedder(x)
        return x.reshape(x.shape[0], -1, EMB_DIM)

class PoseEmbBase(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.proj1 = nn.Linear(1, 128)
        
        idx = 2.5 * (torch.arange(steps, dtype=torch.float32).reshape(1, steps, 1) / steps - 0.5)
        self.register_buffer('idx', idx)

    def forward(self, x_dummy=None):
        return torch.sin(self.proj1(self.idx))

class PoseEmb2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_x = nn.Linear(128, EMB_DIM, bias=False)
        self.proj_y = nn.Linear(128, EMB_DIM, bias=False)

    def forward(self, base_x, base_y):
        proj_x_out = self.proj_x(base_x).unsqueeze(1)
        proj_y_out = self.proj_y(base_y).unsqueeze(2)
        
        pos_emb_grid = proj_x_out + proj_y_out
        return pos_emb_grid.reshape(1, -1, EMB_DIM)

class PreHC(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.proj = nn.Linear(EMB_DIM, n_channels * EMB_DIM)
        self.n_channels = n_channels

    def forward(self, inputs):
        x = self.proj(inputs)
        return x.reshape(x.shape[0], x.shape[1], EMB_DIM, self.n_channels)

class HCdown(nn.Module):
    def __init__(self, output_norm=True):
        super().__init__()
        self.output_norm = output_norm
        self.proj_in = nn.Parameter(torch.full((HC_CHANNEL,), 1.0 / HC_CHANNEL))
        
        if self.output_norm:
            self.norm = nn.LayerNorm(EMB_DIM, eps= 1e-3)

    def forward(self, inputs):
        f = torch.einsum('bshc,c->bsh', inputs, self.proj_in)
        if self.output_norm:
            f = self.norm(f)
        return f

class HCup(nn.Module):
    def __init__(self, layer_idx): # layer_idx is crucial for IdentityVectorInitializer
        super().__init__()
        self.proj_out_initializer = IdentityVectorInitializer(layer_idx % HC_CHANNEL) 
        self.proj_out_scalar = nn.Parameter(self.proj_out_initializer(shape=(1, 1, 1, HC_CHANNEL)))

    def forward(self, inputs):
        return inputs.unsqueeze(-1) * self.proj_out_scalar

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(EMB_DIM, d_model)
        self.wk = nn.Linear(EMB_DIM, d_model)
        self.wv = nn.Linear(EMB_DIM, d_model)

        #self.rope_q = RotaryEmbedding(self.depth)
        #self.rope_k = RotaryEmbedding(self.depth)

        self.dense = nn.Linear(d_model, EMB_DIM)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = q.view(batch_size, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)

        #q = self.rope_q.apply_rotary(q)
        #k = self.rope_k.apply_rotary(k)

        # Using F.scaled_dot_product_attention from PyTorch 2.0+
        qk = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.depth)
        qk = qk.to(torch.float32)
        attn = F.softmax(qk, dim=-1).to(q.dtype)
        
        output = torch.matmul(attn, v)

        concat_attention = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)
        return output
    
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, im_shape=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.im_shape = im_shape
        self.act = bSiLU(hidden_dim)

        self.f = nn.Linear(EMB_DIM, self.hidden_dim * 2)
        self.proj_out = nn.Linear(self.hidden_dim, EMB_DIM)

        if self.im_shape is not None:
            self.gate = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                                  kernel_size=3, padding='same', groups=self.hidden_dim)

    def forward(self, inputs):
        g, h = torch.split(self.f(inputs), self.hidden_dim, dim=-1)

        if self.im_shape is not None:
            B, L, H_dim = g.shape
            H, W = self.im_shape
            assert H * W == L, f"im_shape {self.im_shape} does not match sequence length {L}"
            
            g_reshaped = g.permute(0, 2, 1).reshape(B, H_dim, H, W)
            g_gated = self.gate(g_reshaped)
            g = g_gated.reshape(B, H_dim, H * W).permute(0, 2, 1)

        return self.proj_out(self.act(g) * h)

class flat_trial(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(-1, *x.shape[2:])

class stack_trial(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B_fused = x.shape[0]
        remaining_dims = x.shape[1:]
        
        x_reshaped = x.reshape(B_fused // N_TRIAL, N_TRIAL, *remaining_dims)
        return torch.mean(x_reshaped, dim=1)

class pooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = (EMB_DIM // 2 // 128 + 1) * 128 
        self.proj_in = nn.Linear(EMB_DIM, self.hidden)
        self.proj_out = nn.Linear(self.hidden, EMB_DIM)
        
    def forward(self, inputs):
        B_fused, L, D = inputs.shape
        
        x = self.proj_in(inputs)
        x_reshaped = x.view(B_fused // N_TRIAL, N_TRIAL, L, self.hidden)
        x_pooled = x_reshaped.mean(dim=1, keepdim=True)
        x_projected = self.proj_out(x_pooled)
        x_final = x_projected.expand(-1, N_TRIAL, -1, -1).reshape(B_fused, L, D)
        return x_final

# ---------------------------------------------------------------------------- #
#                                 Main Model Class                             #
# ---------------------------------------------------------------------------- #
class OpenFWI_Model(nn.Module):
    def __init__(self):
        super().__init__()
        #self.save_hyperparameters() # Saves all init args as self.hparams

        self.velo_emb = VELOemb()
        self.flat_trial = flat_trial()
        self.seis_emb = SEISemb()

        self.pose_emb_t_base = PoseEmbBase(32)
        self.pose_emb_y_base = PoseEmbBase(12)
        self.pose_emb_x_base = PoseEmbBase(12)
        self.pose_emb_2d_seis = PoseEmb2D()
        self.pose_emb_2d_velo = PoseEmb2D()

        self.s_pre_hc = PreHC(HC_CHANNEL)
        self.v_pre_hc = PreHC(HC_CHANNEL)

        # --- Encoder Block (Seismic) layers ---
        self.encoder_hc_downs_1 = nn.ModuleList([HCdown() for _ in range(ENCODER_BLOCK)])
        self.encoder_mha = nn.ModuleList([MultiHeadAttention(N_HEADS * 64, N_HEADS) for _ in range(ENCODER_BLOCK)])
        self.encoder_pooler = nn.ModuleList([pooler() for _ in range(ENCODER_BLOCK)])
        self.encoder_hc_ups_1 = nn.ModuleList([HCup(layer_idx=j) for j in range(ENCODER_BLOCK)])

        self.encoder_hc_downs_2 = nn.ModuleList([HCdown() for _ in range(ENCODER_BLOCK)])
        self.encoder_swiglu_s = nn.ModuleList([SwiGLU((EMB_DIM // 2 // 128 + 1) * 128, im_shape=(32, 12)) for _ in range(ENCODER_BLOCK)])
        self.encoder_hc_ups_2 = nn.ModuleList([HCup(layer_idx=j) for j in range(ENCODER_BLOCK)])

        # --- Main Block (Velo-Seismic Interaction) layers ---
        self.main_hc_downs_v11 = nn.ModuleList([HCdown() for _ in range(N_BLOCK)])
        self.main_hc_downs_s11 = nn.ModuleList([HCdown() for _ in range(N_BLOCK)])
        self.main_mha_v_cross = nn.ModuleList([MultiHeadAttention(N_HEADS * 64, N_HEADS) for _ in range(N_BLOCK)])
        self.main_mha_s_cross = nn.ModuleList([MultiHeadAttention(N_HEADS * 64, N_HEADS) for _ in range(N_BLOCK)])
        self.main_hc_ups_v11 = nn.ModuleList([HCup(layer_idx=j) for j in range(N_BLOCK)])
        self.main_hc_ups_s11 = nn.ModuleList([HCup(layer_idx=j) for j in range(N_BLOCK-1)])

        self.main_hc_downs_v12 = nn.ModuleList([HCdown() for _ in range(N_BLOCK)])
        self.main_hc_downs_s12 = nn.ModuleList([HCdown() for _ in range(N_BLOCK-1)])
        self.main_swiglu_v1 = nn.ModuleList([SwiGLU((EMB_DIM // 2 // 128 + 1) * 128, im_shape=(12, 12)) for _ in range(N_BLOCK)])
        self.main_swiglu_s1 = nn.ModuleList([SwiGLU((EMB_DIM // 2 // 128 + 1) * 128, im_shape=(32, 12)) for _ in range(N_BLOCK)])
        self.main_hc_ups_v12 = nn.ModuleList([HCup(layer_idx=j) for j in range(N_BLOCK)])
        self.main_hc_ups_s12 = nn.ModuleList([HCup(layer_idx=j) for j in range(N_BLOCK-1)])

        
        self.main_hc_downs_v21 = nn.ModuleList([HCdown() for _ in range(N_BLOCK)])
        self.main_hc_downs_s21 = nn.ModuleList([HCdown() for _ in range(N_BLOCK-1)])
        self.main_mha_v_self = nn.ModuleList([MultiHeadAttention(N_HEADS * 64, N_HEADS) for _ in range(N_BLOCK)])
        self.main_mha_s_self = nn.ModuleList([MultiHeadAttention(N_HEADS * 64, N_HEADS) for _ in range(N_BLOCK)])
        self.main_hc_ups_v21 = nn.ModuleList([HCup(layer_idx=j) for j in range(N_BLOCK)])
        self.main_hc_ups_s21 = nn.ModuleList([HCup(layer_idx=j) for j in range(N_BLOCK-1)])

        self.main_hc_downs_v22 = nn.ModuleList([HCdown() for _ in range(N_BLOCK)])
        self.main_hc_downs_s22 = nn.ModuleList([HCdown() for _ in range(N_BLOCK-1)])
        self.main_swiglu_v2 = nn.ModuleList([SwiGLU((EMB_DIM // 2 // 128 + 1) * 128, im_shape=(12, 12)) for _ in range(N_BLOCK)])
        self.main_swiglu_s2 = nn.ModuleList([SwiGLU((EMB_DIM // 2 // 128 + 1) * 128, im_shape=(32, 12)) for _ in range(N_BLOCK)])
        self.main_hc_ups_v22 = nn.ModuleList([HCup(layer_idx=j) for j in range(N_BLOCK)])
        self.main_hc_ups_s22 = nn.ModuleList([HCup(layer_idx=j) for j in range(N_BLOCK-1)])

        self.stack_trial = stack_trial()
        self.final_down = HCdown(output_norm=False)
        self.final_dense = nn.Linear(EMB_DIM, 6 * 6)

    def forward(self, seis_input, velo_input):
        v = self.velo_emb(velo_input)
        
        s = self.flat_trial(seis_input)
        s = self.seis_emb(s)

        pose_base_t_out = self.pose_emb_t_base()
        pose_base_y_out = self.pose_emb_y_base()
        pose_base_x_out = self.pose_emb_x_base()
        
        pos_s_flat = self.pose_emb_2d_seis(pose_base_x_out, pose_base_t_out).expand(s.shape[0], -1, -1)
        s = s + pos_s_flat

        pos_v_flat = self.pose_emb_2d_velo(pose_base_x_out, pose_base_y_out).expand(v.shape[0], -1, -1)
        v = v + pos_v_flat

        s_hc = self.s_pre_hc(s)
        v_hc = self.v_pre_hc(v)

        #print("Before encoder s.shape: ", s_hc.shape)
        for i in range(ENCODER_BLOCK):
            s_down = self.encoder_hc_downs_1[i](s_hc)
            s_att = self.encoder_mha[i](s_down, s_down, s_down)
            s_pool = self.encoder_pooler[i](s_down)
            s_hc = s_hc + self.encoder_hc_ups_1[i](s_pool + s_att)
            
            s_down = self.encoder_hc_downs_2[i](s_hc)
            s_ffn = self.encoder_swiglu_s[i](s_down)
            s_hc = s_hc + self.encoder_hc_ups_2[i](s_ffn)

        #print("After encoder s.shape: ", s_hc.shape)

        s_hc = self.stack_trial(s_hc)

        #print("Before decoder s.shape: ", s_hc.shape)
        for i in range(N_BLOCK):
            #print("decode ", i)
            
            v_down_cross = self.main_hc_downs_v11[i](v_hc)
            s_down_cross = self.main_hc_downs_s11[i](s_hc)
            #print("    v cross:  {}  |  s cross  {}".format(v_down_cross.shape, s_down_cross.shape))
            att_v = self.main_mha_v_cross[i](v_down_cross, s_down_cross, s_down_cross)
            v_hc = v_hc + self.main_hc_ups_v11[i](att_v)
            if i < N_BLOCK - 1:
                att_s = self.main_mha_s_cross[i](s_down_cross, v_down_cross, v_down_cross)
                s_hc = s_hc + self.main_hc_ups_s11[i](att_s)
            
            
            v_down_ffn1 = self.main_hc_downs_v12[i](v_hc)
            v_ffn1 = self.main_swiglu_v1[i](v_down_ffn1)
            v_hc = v_hc + self.main_hc_ups_v12[i](v_ffn1)

            v_down_self = self.main_hc_downs_v21[i](v_hc)
            att_v_self = self.main_mha_v_self[i](v_down_self, v_down_self, v_down_self)
            v_hc = v_hc + self.main_hc_ups_v21[i](att_v_self)

            v_down_ffn2 = self.main_hc_downs_v22[i](v_hc)
            v_ffn2 = self.main_swiglu_v2[i](v_down_ffn2)
            v_hc = v_hc + self.main_hc_ups_v22[i](v_ffn2)

            if i < N_BLOCK - 1:
                s_down_ffn1 = self.main_hc_downs_s12[i](s_hc)
                s_ffn1 = self.main_swiglu_s1[i](s_down_ffn1)
                s_hc = s_hc + self.main_hc_ups_s12[i](s_ffn1)
                
                s_down_self = self.main_hc_downs_s21[i](s_hc)
                att_s_self = self.main_mha_s_self[i](s_down_self, s_down_self, s_down_self)
                s_hc = s_hc + self.main_hc_ups_s21[i](att_s_self)
                
                s_down_ffn2 = self.main_hc_downs_s22[i](s_hc)
                s_ffn2 = self.main_swiglu_s2[i](s_down_ffn2)
                s_hc = s_hc + self.main_hc_ups_s22[i](s_ffn2)

        v_final_down = self.final_down(v_hc)
        v_dense_out = self.final_dense(v_final_down)
        v_reshaped = v_dense_out.reshape(v_dense_out.shape[0], 12, 12, 6, 6)
        v_permuted = v_reshaped.permute(0, 1, 3, 2, 4)
        v_final = v_permuted.reshape(v_permuted.shape[0], 72, 72)

        return v_final

# ---------------------------------------------------------------------------- #
#                                Loss Functions                                #
# ---------------------------------------------------------------------------- #
class EMA_MAE_Loss(nn.Module):
    def __init__(self, alpha=0.98, beta=2.0, epsilon=1e-3):
        super().__init__()
        self.alpha = alpha
        self.delta = 1-alpha
        self.beta = beta
        self.epsilon = epsilon

        self.register_buffer("ema_mae", torch.tensor(1.0))

        Scharr_filter_np = np.array([
            [[3, 3], [0, 10], [-3, 3]], [[10, 0], [0, 0], [-10, 0]], [[3, -3], [0, -10], [-3, -3]]
        ]).reshape(3, 3, 1, 2)
        Laplacian_filter_np = np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]]).reshape(3, 3, 1, 1)

        filters = np.concatenate([Laplacian_filter_np, Scharr_filter_np], axis=-1)
        filters = filters / filters.std(axis=(0, 1), keepdims=True)

        filters = torch.tensor(filters, dtype=torch.bfloat16).permute(3, 2, 0, 1)
        self.register_buffer("physic_filter", filters, persistent=False)

    def forward(self, y_pred, y_true):
        mae = torch.mean(torch.abs(y_true - y_pred))
        mse = torch.mean(torch.square(y_true - y_pred))

        with torch.no_grad():
            self.ema_mae.mul_(self.alpha).add_(self.delta * mae)

        loss_main = mse / (self.ema_mae.detach() + self.epsilon)

        true_filtered = F.conv2d(y_true.unsqueeze(1), self.physic_filter, stride=1, padding=1)
        pred_filtered = F.conv2d(y_pred.unsqueeze(1), self.physic_filter, stride=1, padding=1)

        physic_loss = torch.mean(torch.abs(true_filtered - pred_filtered))

        return self.beta * loss_main + physic_loss

# ---------------------------------------------------------------------------- #
#                              Distributed Training Loop                       #
# ---------------------------------------------------------------------------- #

class OpenFWI_LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr_init: float, lr_end: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.model = model
        self.model.apply(self._init_weights)

        self.loss_fn = EMA_MAE_Loss()
        self.lr_init = lr_init
        self.lr_end = lr_end
        print("LR: init {}  |  end {}".format(self.lr_init, self.lr_end))

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        print("Steps : warmup {}  |  total {}".format(self.warmup_steps , self.total_steps))

        self.save_hyperparameters(ignore=["model"])
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, seis, velo):
        return self.model(seis, velo)

    def training_step(self, batch, batch_idx):
        (seis, zelo), velo = batch

        preds = self(seis, zelo)
        loss = self.loss_fn(preds, velo)
        self.log("train_loss", loss.detach(), sync_dist=False)
        print(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        (seis, zelo), velo = batch

        preds = self(seis, zelo)
        loss = self.loss_fn(preds, velo)
        self.log("val_loss", loss.detach(), sync_dist=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr_init, betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY)
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return ((self.lr_init - self.lr_end) * 0.5 * (1 + np.cos(np.pi * progress)) + self.lr_end) / self.lr_init

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # call scheduler.step() after each optimizer step
                'frequency': 1,
            }
        }


class OpenFWIDataModule(pl.LightningDataModule):
    def __init__(self, train_path, valid_path, batch_size):
        super().__init__()
        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        #print("Setting up OpenFWI datasets...")
        self.train_dataset = TFRecordLightningDataset(self.train_path)
        self.valid_dataset = TFRecordLightningDataset(self.valid_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            persistent_workers=False,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            persistent_workers=False,
            pin_memory=False
        )
# ---------------------------------------------------------------------------- #
#                                   Main Entry                                 #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    lightning_model = OpenFWI_LitModel(
        model=OpenFWI_Model(),
        lr_init=LR_INIT,
        lr_end=LR_END,
        warmup_steps=WARMUP_STEPS,
        total_steps=TOTAL_TRAIN_STEPS,
    )

    datamodule = OpenFWIDataModule(
        train_path=train_dir,
        valid_path=valid_dir,
        batch_size=BATCH_SIZE
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=GCS_CHECKPOINT_PATH,
        monitor = "val_loss",
        save_top_k=1 if SAVE_LAST_CHECKPOINT_ONLY else 3,
        save_last=True,
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        accelerator="tpu",
        devices=4,
        strategy="xla",
        precision="bf16-true",
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        log_every_n_steps = 100,
        max_epochs=2,
        gradient_clip_val=1.5,
    )
    trainer.fit(lightning_model, datamodule=datamodule)