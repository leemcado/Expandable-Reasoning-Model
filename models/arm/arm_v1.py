from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear

# --- 설정 클래스 (Pydantic 모델) ---
class FrontalModuleConfig(BaseModel):
    num_layers: int
    hidden_size: int
    num_heads: int
    expansion: float

class ThalamusModuleConfig(BaseModel):
    hidden_size: int
    num_layers: int

class ReasoningModuleConfig(BaseModel):
    num_layers: int
    hidden_size: int
    num_heads: int
    expansion: float
    inner_loops: int # 더 이상 사용되지 않음

class ARMConfig(BaseModel):
    frontal_module: FrontalModuleConfig
    thalamus_module: ThalamusModuleConfig
    reasoning_module: ReasoningModuleConfig
    
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    pos_encodings: str
    initial_modules: int
    max_modules: int
    
    halt_max_steps: int
    halt_exploration_prob: float

    # HRM 호환 파라미터
    H_cycles: int = 2
    L_cycles: int = 2

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"

# --- 모듈별 클래스 정의 ---
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, expansion: float, norm_eps: float):
        super().__init__()
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)
        self.norm_eps = norm_eps

    def forward(self, hidden_states: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states

class FrontalModule(nn.Module):
    def __init__(self, config: FrontalModuleConfig, main_config: ARMConfig):
        super().__init__()
        self.reasoning_proj = CastedLinear(
            main_config.reasoning_module.hidden_size, 
            config.hidden_size, 
            bias=False
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config.hidden_size, config.num_heads, config.expansion, main_config.rms_norm_eps) 
             for _ in range(config.num_layers)]
        )

    def forward(self, z_f: torch.Tensor, z_r_active: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        x = z_f + self.reasoning_proj(z_r_active)
        for layer in self.layers:
            x = layer(x, cos_sin)
        return x

class ThalamusModule(nn.Module):
    def __init__(self, config: ThalamusModuleConfig, main_config: ARMConfig):
        super().__init__()
        in_dim = main_config.frontal_module.hidden_size + main_config.reasoning_module.hidden_size
        layers = [CastedLinear(in_dim, config.hidden_size, bias=True), nn.ReLU()]
        for _ in range(config.num_layers - 1):
            layers.extend([CastedLinear(config.hidden_size, config.hidden_size, bias=True), nn.ReLU()])
        layers.append(CastedLinear(config.hidden_size, main_config.max_modules, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z_f: torch.Tensor, input_embedding: torch.Tensor) -> torch.Tensor:
        x = torch.cat((z_f[:, 0, :], input_embedding[:, 0, :]), dim=-1)
        return self.net(x)

class ReasoningModule(nn.Module):
    def __init__(self, config: ReasoningModuleConfig, main_config: ARMConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [TransformerBlock(config.hidden_size, config.num_heads, config.expansion, main_config.rms_norm_eps) 
             for _ in range(config.num_layers)]
        )

    def forward(self, z_r: torch.Tensor, z_f: torch.Tensor, input_embedding: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        x = z_r + z_f + input_embedding
        for layer in self.layers:
            x = layer(x, cos_sin)
        return x

# --- Carry 객체 및 메인 모델 ---
@dataclass
class ARMInnerCarry:
    z_f: torch.Tensor
    z_r_states: List[torch.Tensor]

@dataclass
class ARMCarry:
    inner_carry: ARMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, Any]

class ARM_Inner(nn.Module):
    def __init__(self, config: ARMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        self.embed_scale  = math.sqrt(config.reasoning_module.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.reasoning_module.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(config.frontal_module.hidden_size, config.vocab_size, bias=False)
        self.q_head       = CastedLinear(config.frontal_module.hidden_size, 2, bias=True)

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=config.reasoning_module.hidden_size // config.reasoning_module.num_heads,
                                              max_position_embeddings=config.seq_len,
                                              base=config.rope_theta)
        
        self.frontal_module = FrontalModule(config.frontal_module, config)
        self.thalamus_module = ThalamusModule(config.thalamus_module, config)
        self.reasoning_modules = nn.ModuleList(
            [ReasoningModule(config.reasoning_module, config) for _ in range(config.initial_modules)]
        )

    def add_new_reasoning_module(self):
        if len(self.reasoning_modules) < self.config.max_modules:
            device = next(self.parameters()).device
            new_module = ReasoningModule(self.config.reasoning_module, self.config).to(device)
            self.reasoning_modules.append(new_module)
            print(f"New reasoning module added. Total modules: {len(self.reasoning_modules)}")
            return True
        return False

    def _input_embeddings(self, input_ids: torch.Tensor):
        embedding = self.embed_tokens(input_ids.to(torch.int32))
        return self.embed_scale * embedding

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ARM_Inner.forward is not meant to be called directly. Logic is in pretrain.py")

class ARM(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ARMConfig(**config_dict)
        self.inner = ARM_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> ARMCarry:
        batch_size = batch["inputs"].shape[0]
        
        z_f = torch.zeros(batch_size, self.config.seq_len, self.config.frontal_module.hidden_size, 
                          dtype=getattr(torch, self.config.forward_dtype), device="cuda")
        
        num_current_modules = len(self.inner.reasoning_modules)
        z_r_states = [torch.zeros(batch_size, self.config.seq_len, self.config.reasoning_module.hidden_size, 
                                  dtype=getattr(torch, self.config.forward_dtype), device="cuda")
                      for _ in range(num_current_modules)]
        
        inner_carry = ARMInnerCarry(z_f=z_f, z_r_states=z_r_states)

        return ARMCarry(
            inner_carry=inner_carry,
            steps=torch.zeros((batch_size,), dtype=torch.int32, device="cuda"),
            halted=torch.ones((batch_size,), dtype=torch.bool, device="cuda"),
            current_data={k: v.clone() for k, v in batch.items()}
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ARM.forward logic is handled by the custom training loop in pretrain.py")