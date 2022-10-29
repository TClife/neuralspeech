import torch 
from torch import nn
from math import sqrt 
from hparams import hparams as hps 
from torch.autograd import Variable
from torch.nn import functional as F
from model.layers import ConvNorm, LinearNorm 
from utils.util import *
from utils.audio import * 
import soundfile as sf 
import numpy as np 

class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()
        self.embedding = nn.Embedding(
            hps.n_symbols, hps.symbols_embedding_dim)
        std = sqrt(2.0/(hps.n_symbols+hps.symbols_embedding_dim))
        val = sqrt(3.0)*std
        self.embedding.weight.data.uniform_(-val, val) #weight initialization uniform with low high -val / val best rule of thumb is [-y, y] with y = 1 / sqrt(n) where n is number of samples
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()
    
    
    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data     
        
        embedded_inputs = self.embedding(text_inputs).transpose(1,2)
        
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths)
        
        mel_outputs_postnet = self.postnet(mel_outputs) 
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet 
        
        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments], output_lengths)
     

