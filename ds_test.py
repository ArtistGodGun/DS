import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import glob, os
import torchaudio
import time
import librosa

class DSDataset(Dataset):
    def __init__(self, path, source):
        super().__init__()
        self.path = path

        self.input_data_list = glob.glob(f'{path}/*/wav/*/original.wav')
        self.target_data_list = glob.glob(f'{path}/*/wav/*/{source}.wav')

    def __len__(self):
        return len(self.input_data_list)
    def __getitem__(self, index):
        song_name = self.input_data_list[index].split('/')[4]
        
        audio, sr = torchaudio.load(self.input_data_list[index])
        ride, sr = torchaudio.load(self.target_data_list[index])

        input_data = torch.Tensor([])
        target_data = torch.Tensor([])
        bins = sr * 2
        split = audio.size()[1] // bins + 1
        duration = min(audio.size()[1], ride.size()[1])
        audio, ride = audio[:, :duration], ride[:, :duration]
        zero = torch.zeros(2, split*bins - duration)

        audio, ride = torch.cat((audio, zero), dim = 1),torch.cat((ride, zero), dim = 1)
                
        for i in range(audio.size()[1]//bins):
            stfta = torch.stft(audio[:,i*bins:(i+1)*bins],n_fft=1024,hop_length=128, return_complex=False)
            stftr = torch.stft(ride[:,i*bins:(i+1)*bins], n_fft=1024, hop_length=128, return_complex=False)
            stfta, stftr = stfta.permute(1,2,0,3), stftr.permute(1,2,0,3)
            frq, t, ch, com = stfta.size()
            stfta, stftr = stfta.reshape(frq,t,ch*com), stftr.reshape(frq, t,ch*com)
            input_data = torch.cat((input_data, stfta.unsqueeze(0)))
            target_data = torch.cat((target_data, stftr.unsqueeze(0)))
        return input_data, target_data, song_name


class TransformerModel(nn.Module):
    def __init__(self, nhead, num_layers,input_size,output_size, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_size, nhead, dim_feedforward, dropout),
            num_layers
        )
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        x = x.permute(1,2,0)
        src_mask = self.generate_squaresubsequence_mask(x.size(0)).to(x.device)
        x = self.transformer_encoder(x, src_mask)
        x = x.permute(2,0,1)
        return x
    def generate_squaresubsequence_mask(self, sz):
        mask = (torch.triu(torch.ones(sz,sz))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1,float(0.0))
        return mask

def test(model, path, source_name):
    audio, sr = torchaudio.load(path)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        sr = 16000
    # audio = librosa.effects.percussive(audio.cpu().numpy())
    # audio = torch.Tensor(audio)
    # audio, sr = torchaudio.load('train/1/original.wav')
    # audio = audio[:,:sr*30]
    bins = sr * 2
    original_duration = audio.size(1)
    
    if audio.size(1) < bins:
        audio = torch.cat((audio, torch.zeros(2, bins - audio.size(1))),dim=1)
    spl = audio.size(1) // bins + 1
    pad = spl*bins - audio.size(1)
    if pad > 0:
        audio = torch.cat((audio, torch.zeros(2, pad)),dim=1)
    input_data = torch.Tensor([])
    for i in range(spl):
        stfta = torch.stft(audio[:,i*bins:(i+1)*bins],n_fft=1024,hop_length=128, return_complex=False)
        stfta = stfta.permute(1,2,0,3)
        frq, t, ch, com = stfta.size()
        
        stfta = stfta.reshape(frq, t, ch*com)
        input_data = torch.cat((input_data, stfta.unsqueeze(0)))
    result_audio = torch.Tensor([])
    with torch.no_grad():
        for i in input_data:
            output = model(i.to(device))
            frq, t, c = output.size()
            output = output.reshape(frq, t, 2, 2)
            output = output.permute(2,0,1,3)
            stft = torch.istft(output.cpu(), n_fft=1024,hop_length=128)
            result_audio = torch.cat((result_audio, stft), dim = 1)
        if pad > 0:
            result_audio = result_audio[:, :original_duration]
        torchaudio.save(f'split_{source_name}.wav', result_audio, sr)
  




if __name__ == '__main__':
    # example usage
    num_layers = 6
    num_heads = 3
    dropout = 0.5
    hidden_dim = 1024
    device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu'
    model = TransformerModel(nhead=num_heads, input_size=513, output_size=513, dim_feedforward=hidden_dim, num_layers=num_layers).to(device)
    
    test_path = 'LS-CD1 Demo Track - Drum Loop 12.wav'
    audio, sr = torchaudio.load(test_path)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        sr = 16000
        torchaudio.save(test_path, audio, sr, 16)
        
    sources = ['kick','snare','hihat','ride','tom','crash']

    for source in sources:
        print(f'{source} Start!!!')
        model.load_state_dict(torch.load(f'ds_{source}.pth'))
        test(model, test_path, source)