import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import glob, os
import torchaudio
import time
from tqdm import tqdm


class DSDataset(Dataset):
    def __init__(self, path, source):
        super().__init__()
        self.path = path

        # self.input_data_list = glob.glob(f'{path}/*/wav/*/original.wav')
        # self.target_data_list = glob.glob(f'{path}/*/wav/*/{source}.wav')
        # self.input_data_list = glob.glob(f'{path}/60/0/original.wav')
        # self.target_data_list = glob.glob(f'{path}/60/0/{source}.wav')
        self.input_data_list = glob.glob(f'{path}/*/*/original.wav')
        self.target_data_list = glob.glob(f'{path}/*/*/{source}.wav')

    def __len__(self):
        return len(self.input_data_list)
    def __getitem__(self, index):
        # song_name = self.input_data_list[index].split('/')[4]
        
        audio, sr = torchaudio.load(self.input_data_list[index])
        ride, sr = torchaudio.load(self.target_data_list[index])

        # input_data = torch.Tensor([])
        # target_data = torch.Tensor([])

        duration = min(audio.size()[1], ride.size()[1])
        # bins = sr * 2
        # split = duration // bins + 1
        audio, ride = audio[:, :duration], ride[:, :duration]
        
        stfta = torch.stft(audio, n_fft=1023, hop_length=512, return_complex=False)
        stftr = torch.stft(ride, n_fft=1023, hop_length=512, return_complex=False)
        stfta, stftr = stfta.permute(1,2,0,3), stftr.permute(1,2,0,3)
        frq,t,ch,com = stfta.size()
        stfta, stftr = stfta.reshape(frq,t,ch*com), stftr.reshape(frq, t, ch*com)
        
        
        # zero = torch.zeros(2, split*bins - duration)
        # audio, ride = torch.cat((audio, zero), dim = 1),torch.cat((ride, zero), dim = 1)
        # for i in range(audio.size()[1]//bins):
        #     stfta = torch.stft(audio[:,i*bins:(i+1)*bins],n_fft=1024,hop_length=128, return_complex=False)
        #     stftr = torch.stft(ride[:,i*bins:(i+1)*bins], n_fft=1024, hop_length=128, return_complex=False)
        #     stfta, stftr = stfta.permute(1,2,0,3), stftr.permute(1,2,0,3)
        #     frq, t, ch, com = stfta.size()
        #     stfta, stftr = stfta.reshape(frq,t,ch*com), stftr.reshape(frq, t,ch*com)
        #     input_data = torch.cat((input_data, stfta.unsqueeze(0)))
        #     target_data = torch.cat((target_data, stftr.unsqueeze(0)))
        
        return stfta, stftr

class TransformerModel(nn.Module):
    def __init__(self, nhead, num_layers,input_size,output_size, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_size, nhead, dim_feedforward, dropout),
            num_layers
        )
        self.linear = nn.Linear(input_size, output_size)
        # self.transformer_decoder = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(input_size, nhead, dim_feedforward, dropout),
        #     num_layers
        # )
    def forward(self, x):
        # hihat에 어울리는 모델? #
        x = x.permute(1,2,0)
        src_mask = self.generate_squaresubsequence_mask(x.size(0)).to(x.device)
        x = self.transformer_encoder(x, src_mask)
        # x = self.linear(x)
        x = x.permute(2,0,1)
        # hihat에 어울리는 모델? #
        
        # # ride에 어울리는 모델? #
        # print('1', x.size())
        # x = x.permute(1,2,0)
        # src_mask = self.generate_squaresubsequence_mask(x.size(0)).to(x.device)
        # x = self.transformer_encoder(x, src_mask)
        # print('2',x.size())
        # x = x.permute(2,0,1)
        # print('3',x.size())
        # # ride에 어울리는 모델? #
        
        
        # x = x.permute(2,0,1,3)
        # time, batch, freq, value = x.size()
        # x = x.view(time*batch, freq, value)
        # x = x.permute(0,2,1)
        # src_mask = self.generate_squaresubsequence_mask(x.size(0)).to(x.device)
        # x = self.transformer_encoder(x, src_mask)
        # x = x.permute(0, 1, 2)
        # x = self.linear(x)
        # x = x.permute(0, 2, 1)
        # x = x.view(time, batch, freq, value)
        # x = x.permute(1, 2, 0, 3)
        return x
    def generate_squaresubsequence_mask(self, sz):
        mask = (torch.triu(torch.ones(sz,sz))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1,float(0.0))
        return mask

def train(model,data, epochs, sources, scheduler):
    estart = time.time()
    total_loss = 0
    num = 0
    for ep in range(epochs):
        current_time = time.time()
        for idx, (i,t) in enumerate(data):
            i, t = i.to(device), t.to(device)
            i, t = i.squeeze(0), t.squeeze(0)
            optimizer.zero_grad()
            output = model(i)
            loss = criterion(output, t)
            # print(f'loss : {loss.item()}, time : {time.time() - current_time}')
            num+=1
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            # for inputs, targets in zip(i, t):
            #     inputs,targets = inputs.to(device), targets.to(device)
            #     optimizer.zero_grad()
            #     output = model(inputs)
            #     loss = criterion(output, targets)
            #     num+=1
            #     total_loss += loss.item()
            #     loss.backward()
            #     optimizer.step()
            if (idx+1) % 100 == 0:
                print(f"{sources}_EPOCH : {ep}_{idx+1}, LOSS : {total_loss/num}, TIME : {time.time() - current_time}")
                torch.save(model.state_dict(), f'newds_{sources}.pth')
                current_time = time.time()
        torch.save(model.state_dict(), f'newds_{sources}.pth')
        scheduler.step()
        
    torch.save(model.state_dict(), f'newds_{sources}.pth')
    print(f'{sources} TOTAL TIME : {time.time() - estart}, TOTAL LOSS : {total_loss / num}')

def test(model, path, source_name):
    audio, sr = torchaudio.load(path)
    # audio, sr = torchaudio.load('train/1/original.wav')
    print('length : ', audio.size())
    bins = sr * 2
    input_data = torch.Tensor([])
    for i in range(audio.size()[1]//bins):
        stfta = torch.stft(audio[:,i*bins:(i+1)*bins],n_fft=1023,hop_length=512, return_complex=False)
        stfta = stfta.permute(1,2,0,3)
        frq, t, ch, com = stfta.size()
        
        stfta = stfta.reshape(frq, t, ch*com)
        input_data = torch.cat((input_data, stfta.unsqueeze(0)))
    result_audio = torch.Tensor([])
    print('hello' , input_data.size())
    with torch.no_grad():
        for i in input_data:
            output = model(i.to(device))
            frq, t, c = output.size()
            output = output.reshape(frq, t, 2, 2)
            output = output.permute(2,0,1,3)
            stft = torch.istft(output.cpu(), n_fft=1023,hop_length=512)
            result_audio = torch.cat((result_audio, stft), dim = 1)
        result_audio = result_audio
        print(result_audio.size())
        torchaudio.save(f'{source_name}.wav', result_audio, sr)
  

# example usage
num_layers = 6
num_heads = 8
dropout = 0.2 # 0.1 = 라이드, 0.5 = 심벌
hidden_dim = 1024
device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu'
# drum_path = '/Volumes/VIDEO/DrumSeparateSource/'
drum_path = 'temp'

for source in ['ride','kick','snare','hihat','tom','crash','clap']:
    print(source, ' Start')
    train_data = DSDataset(drum_path, source)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    model = TransformerModel(nhead=num_heads, input_size=512, output_size=512, dim_feedforward=hidden_dim, num_layers=num_layers).to(device)
    # model.load_state_dict(torch.load(f'ds_{source}.pth'))
    # model.load_state_dict(torch.load(f'newds_{source}.pth'))
    
    optimizer = optim.Adam(params=model.parameters(), lr = 0.001) # 0.001 = 라이드가 좋음 0.0001 = 심벌이 좋음
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda= lambda epoch: 0.99 ** epoch)
    train(model, train_loader, 20, source, scheduler)
