import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding

class _ATTR_NETWORK(nn.Module):
    def __init__(self, vocab_obj, args, device):
        super(_ATTR_NETWORK, self).__init__()

        self.m_device = device
        
        self.m_vocab_size = vocab_obj.vocab_size
        self.m_user_num = vocab_obj.user_num
        self.m_item_num = vocab_obj.item_num

        self.m_attr_embed_size = args.attr_emb_size
        self.m_user_embed_size = args.user_emb_size
        self.m_item_embed_size = args.item_emb_size

        self.m_attn_head_num = args.attn_head_num
        self.m_attn_layer_num = args.attn_layer_num

        self.m_attn_linear_size = args.attn_linear_size

        self.m_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        self.m_conv_1 = nn.Conv2d(1, self.m_attr_embed_size, kernel_size=(1, 3))
        self.m_relu_conv_1 = nn.ReLU()

        self.m_conv_2 = nn.Conv2d(self.m_attr_embed_size, self.m_attr_embed_size, kernel_size=(self.m_attr_embed_size, 1))
        self.m_relu_conv_2 = nn.ReLU()

        self.m_linear_1 = nn.Linear(self.m_attr_embed_size, self.m_attr_embed_size)
        self.m_relu_linear_1 = nn.ReLU()

        self.m_linear_2 = nn.Linear(self.m_attr_embed_size, 1)
        # self.m_relu_linear_2 = nn.ReLU()
        self.m_ac_2 = nn.Sigmoid()

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1
    
        torch.nn.init.uniform_(self.m_attr_embedding.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_user_embedding.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_item_embedding.weight, -initrange, initrange)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def forward(self, user_ids, item_ids, attr_ids):
    
        # """ user """

        user_embed = self.m_user_embedding(user_ids)

        item_embed = self.m_item_embedding(item_ids)

        attr_embed = self.m_attr_embedding(attr_ids)
        
        x_embed = torch.cat([user_embed, item_embed, attr_embed], dim=-1)

        batch_size = x_embed.size(0)

        input_embed = x_embed.reshape(batch_size, 1, self.m_attr_embed_size, 3)

        x_conv_1 = self.m_relu_conv_1(self.m_conv_1(input_embed))
        x_conv_2 = self.m_relu_conv_2(self.m_conv_2(x_conv_1))

        # print("x_conv_2", x_conv_2.size())

        x_conv = x_conv_2.squeeze(-1).squeeze(-1)

        # print("x_conv", x_conv.size())

        x_linear_1 = self.m_relu_linear_1(self.m_linear_1(x_conv))
        # print("x_linear_1", x_linear_1.size())

        x_linear_2 = self.m_linear_2(x_linear_1)
        # print("x_linear_2", x_linear_2.size())
        
        logits = self.m_ac_2(x_linear_2)

        # print("logits", logits.size())

        return logits

   