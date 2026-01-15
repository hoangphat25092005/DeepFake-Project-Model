import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.optim as optim
from models import get_model


class Trainer():
    def __init__(self, opt):
        self.opt = opt
        self.model = get_model(opt)
        
        if opt.resume_path:
            print(f"Loading checkpoint from: {opt.resume_path}")
            state_dict = torch.load(opt.resume_path, map_location='cpu')
            
            # Check what keys are in the checkpoint
            has_model_prefix = any(k.startswith('model.') for k in state_dict.keys())
            has_attention_prefix = any(k.startswith('attention_head.') for k in state_dict.keys())
            
            if has_model_prefix and has_attention_prefix:
                # Full model checkpoint with both CLIP and attention head
                self.model.load_state_dict(state_dict, strict=False)
                print("Loaded full model checkpoint (CLIP + attention head)")
            elif has_attention_prefix:
                # Checkpoint with attention_head prefix only
                self.model.load_state_dict(state_dict, strict=False)
                print("Loaded checkpoint with attention_head prefix")
            else:
                # Only attention head weights without prefix - add prefix
                attention_state_dict = {f'attention_head.{k}': v for k, v in state_dict.items()}
                missing_keys, unexpected_keys = self.model.load_state_dict(attention_state_dict, strict=False)
                print("Loaded attention head weights only (added prefix)")
                if missing_keys:
                    print(f"   Missing keys (will use pretrained CLIP): {len(missing_keys)} keys")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
        
        # Freeze CLIP backbone, only train attention head
        for name, params in self.model.attention_head.named_parameters():
            params.requires_grad = True
        
        if opt.fix_backbone:
            for name, params in self.model.model.named_parameters():
                params.requires_grad = False
            print("CLIP backbone frozen (only training attention head)")
        else:
            for name, params in self.model.model.named_parameters():
                params.requires_grad = True
            print("CLIP backbone unfrozen (training full model)")
        
        self.model = self.model.cuda()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=opt.lr,
            betas=(opt.beta1, 0.999),
            weight_decay=opt.weight_decay
        )
        
        self.total_steps = 0
        self.loss = 0
        
    def set_input(self, input):
        self.input = input[0].cuda()
        self.label = input[1].cuda().float()
    
    def forward(self):
        self.output = self.model(self.input)
    
    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)
    
    def optimize_parameters(self):
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def save_networks(self, name):
        save_filename = name
        save_path = f'{self.opt.checkpoints_dir}/{self.opt.name}/{save_filename}'
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")
    
    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            print(f"Learning rate adjusted to: {param_group['lr']}")
            if param_group['lr'] < min_lr:
                return False
        return True
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()