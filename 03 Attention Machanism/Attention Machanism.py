import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """位置编码 - Transformer的核心组件之一"""
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # 计算位置编码：PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_seq_length, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [seq_length, batch_size, d_model]
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多头自注意力机制 - Transformer的核心"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层：Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力机制"""
        # Q, K, V shape: [batch_size, num_heads, seq_length, d_k]
        d_k = Q.size(-1)
        
        # 计算注意力分数：Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码（用于生成任务，防止看到未来信息）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重到V
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性变换得到Q, K, V
        Q = self.W_q(query)  # [batch_size, seq_length, d_model]
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. 重塑为多头形式
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # shape: [batch_size, num_heads, seq_length, d_k]
        
        # 3. 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask)
        
        # 4. 连接多头结果
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # 5. 最终线性变换
        output = self.W_o(attention_output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    """前馈神经网络 - Transformer Block的第二个组件"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 1. 多头自注意力 + 残差连接 + 层归一化
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class SimpleTransformerLM(nn.Module):
    """简单的Transformer语言模型（生成式）"""
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer块堆叠
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 参数初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_length):
        """创建因果掩码，防止模型看到未来的token"""
        mask = torch.tril(torch.ones(seq_length, seq_length))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]
    
    def forward(self, input_ids, targets=None):
        batch_size, seq_length = input_ids.shape
        
        # 1. 词嵌入 + 位置编码
        # 注意：嵌入需要缩放
        token_embeddings = self.embedding(input_ids) * math.sqrt(self.d_model)
        token_embeddings = token_embeddings.transpose(0, 1)  # [seq_length, batch_size, d_model]
        
        # 添加位置编码
        x = self.pos_encoding(token_embeddings)
        x = x.transpose(0, 1)  # [batch_size, seq_length, d_model]
        x = self.dropout(x)
        
        # 2. 创建因果掩码
        causal_mask = self.create_causal_mask(seq_length).to(input_ids.device)
        
        # 3. 通过Transformer块
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, causal_mask)
            attention_weights.append(attn_weights)
        
        # 4. 最终层归一化 + 输出投影
        x = self.ln_f(x)
        logits = self.head(x)  # [batch_size, seq_length, vocab_size]
        
        loss = None
        if targets is not None:
            # 计算交叉熵损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-100
            )
        
        return logits, loss, attention_weights
    
    def generate(self, input_ids, max_new_tokens=10, temperature=1.0, top_k=None):
        """文本生成函数"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # 获取当前序列的logits
            with torch.no_grad():
                logits, _, _ = self(input_ids)
                
                # 只关注最后一个位置的logits
                logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]
                
                # Top-k采样
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # 采样下一个token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 拼接到输入序列
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

def demo_transformer():
    """演示Transformer模型"""
    print("🚀 Transformer语言模型演示")
    print("=" * 50)
    
    # 模型参数
    vocab_size = 1000      # 词汇表大小
    d_model = 128          # 模型维度
    num_heads = 8          # 注意力头数
    num_layers = 4         # Transformer层数
    d_ff = 512            # 前馈网络维度
    max_seq_length = 100   # 最大序列长度
    
    # 创建模型
    model = SimpleTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length
    )
    
    print(f"📊 模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024**2:.1f} MB")
    
    # 模拟输入数据
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"\n📝 输入数据:")
    print(f"  批次大小: {batch_size}")
    print(f"  序列长度: {seq_length}")
    print(f"  输入形状: {input_ids.shape}")
    
    # 前向传播
    print(f"\n⚡ 前向传播:")
    logits, loss, attention_weights = model(input_ids, targets)
    
    print(f"  输出logits形状: {logits.shape}")
    print(f"  损失值: {loss.item():.4f}")
    print(f"  注意力权重层数: {len(attention_weights)}")
    print(f"  每层注意力形状: {attention_weights[0].shape}")
    
    # 文本生成演示
    print(f"\n🎯 文本生成演示:")
    seed_sequence = torch.randint(0, vocab_size, (1, 5))  # 种子序列
    print(f"  种子序列: {seed_sequence.tolist()[0]}")
    
    # 生成文本
    generated = model.generate(
        seed_sequence, 
        max_new_tokens=8, 
        temperature=0.8, 
        top_k=50
    )
    
    print(f"  生成序列: {generated.tolist()[0]}")
    print(f"  新生成的tokens: {generated.tolist()[0][5:]}")
    
    # 注意力权重分析
    print(f"\n🔍 注意力权重分析 (第一层, 第一个头):")
    first_layer_attn = attention_weights[0][0, 0]  # [seq_length, seq_length]
    print(f"  注意力矩阵形状: {first_layer_attn.shape}")
    print(f"  第一行注意力权重 (前5个): {first_layer_attn[0][:5].tolist()}")
    
    return model

def visualize_attention_pattern():
    """可视化注意力模式"""
    print(f"\n🎨 注意力模式可视化:")
    print("-" * 30)
    
    # 创建简单的序列
    seq = ["I", "love", "machine", "learning", "very", "much"]
    print(f"示例序列: {' '.join(seq)}")
    
    # 模拟注意力权重矩阵
    seq_len = len(seq)
    attention_matrix = torch.rand(seq_len, seq_len)
    attention_matrix = F.softmax(attention_matrix, dim=-1)
    
    print(f"\n注意力权重矩阵:")
    print("    " + "".join(f"{word:>8}" for word in seq))
    for i, word in enumerate(seq):
        weights_str = "".join(f"{attention_matrix[i][j]:>8.3f}" for j in range(seq_len))
        print(f"{word:>4}: {weights_str}")

if __name__ == "__main__":
    # 运行演示
    model = demo_transformer()
    
    # 可视化注意力模式
    visualize_attention_pattern()
    
    print(f"\n✅ Transformer模型演示完成!")
    print(f"\n🔧 主要组件说明:")
    print(f"  1. PositionalEncoding: 位置编码，让模型理解序列顺序")
    print(f"  2. MultiHeadAttention: 多头自注意力机制，捕捉长距离依赖")
    print(f"  3. FeedForward: 前馈网络，增加模型非线性表达能力")
    print(f"  4. TransformerBlock: 完整的Transformer块")
    print(f"  5. SimpleTransformerLM: 完整的生成式语言模型")
