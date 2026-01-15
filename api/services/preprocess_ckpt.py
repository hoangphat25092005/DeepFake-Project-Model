#This is use for preprocess checkpoint before loading into model (.pth file for example)
import io
import torch


class PreprocessCheckpoint:
    def __init__(self, state_dict):
        self.state_dict = state_dict
    
    def remove_module_prefix(self):
        new_state_dict = {}
        for k, v in self.state_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]  # remove 'module.' prefix
            else:
                new_key = k
            new_state_dict[new_key] = v
        self.state_dict = new_state_dict
        return self.state_dict
    
    def get_state_dict(self):
        return self.state_dict
    
    @staticmethod
    def from_file(file_path):
        state_dict = torch.load(file_path, map_location='cpu')
        return PreprocessCheckpoint(state_dict)
    
    @staticmethod
    def from_bytes(byte_data):
        byte_stream = io.BytesIO(byte_data)
        state_dict = torch.load(byte_stream, map_location='cpu')
        return PreprocessCheckpoint(state_dict)
    
    