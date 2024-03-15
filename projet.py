import os
import torch
import torch.nn.functional as F
from torch import optim,nn
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torch.nn.utils.rnn import pad_sequence
import spacy
import pandas as pd
from torchvision import transforms
from utils.size_calculation import min_size_photo,max_seq_length
from torch.autograd import Variable
import random
from utils import split_dataset 

spacy_tokenizer = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }
        self.stoi = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        self.freq_threshold = freq_threshold

    def tokenize(self, text):
        return [token.text.lower() for token in spacy_tokenizer.tokenizer(text)]

    def make_vocabulary(self, sequences):
        current_idx = 4
        frequencies = {}

        for sequence in sequences:
            for word in self.tokenize(sequence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = current_idx
                    self.itos[current_idx] = word
                    current_idx += 1

    def encode(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

    def decode(self, sequence):
        return [self.itos[token] if token in self.itos else "<UNK>" for token in sequence]

    def __len__(self):
        return len(self.itos)


class CustomPKMNDataset(Dataset):
    def __init__(self, img_dir, data_file,transform):
        self.img_dir=img_dir
        self.data_file=data_file
        self.data=pd.read_csv(self.data_file)
        self.vocab = Vocabulary()
        self.transform=transform
        types = self.data['type'].unique()
        self.type_dict = {type: i for i, type in enumerate(types)}   
        self.vocab.make_vocabulary(self.data['caption'])

    def __len__(self):
        return len(self.data)
    
    def get_vocab_size(self):
        return len(self.vocab)

    def __getitem__(self, idx):
        data_idx=self.data.iloc[idx]
        img=data_idx['image']
        type=data_idx['type']
        legende=data_idx['caption'].strip()
        type_encod=torch.tensor([self.type_dict[type]],dtype=torch.long)
        legende_encod=self.vocab.encode(legende)
        legende_encod.insert(0,1)
        legende_encod.insert(len(legende_encod),2)
        legende_encod=torch.tensor(legende_encod)
        img_encod=self.transform(torch.div(read_image(f'{self.img_dir}/{img}'),255))
        return img_encod,legende_encod,type_encod

class PaddingCollate:
    def __init__(self, pad_idx,max_seq_length):
        self.pad_idx = pad_idx
        self.max_seq_length=max_seq_length

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True,
                               padding_value=self.pad_idx)
        targets = F.pad(targets, (0, self.max_seq_length - targets.size(1)), value=self.pad_idx)
        types = [item[2] for item in batch]
        types = torch.cat(types, dim=0)
        return imgs, targets, types


batch_size=32
max_seq_length=max_seq_length+2

def make_loader(img_dir, data_file, transform, batch_size=batch_size, max_seq_length=max_seq_length,num_workers=0, shuffle=True, pin_memory=True):
    dataset = CustomPKMNDataset(
        img_dir, data_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
        pin_memory=pin_memory, collate_fn=PaddingCollate(pad_idx,max_seq_length),drop_last=True)
    return dataloader, dataset

#initialement je mettais mes photos en 272x272 car plus petite taille du data set (min_size_photo)
#mais pour gagner un peu en temps de calcul et car le resnet resize automatiquement en 256x256 je suis passé en 256x256
transform=transforms.Resize((150,150),antialias=True)

train_dataloader,train_dataset=make_loader('data/images','data/data_train.csv',transform=transform)
test_dataloader,test_dataset=make_loader('data/images','data/data_test.csv',transform=transform)


#CNN pré entrainement
output = 18
resnet = resnet50()
resnet.fc = nn.Linear(resnet.fc.in_features, output)

epochs = 50
lr = 3e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(),lr=lr)

for epoch in range(epochs):
    resnet.train()
    avg_train_loss = 0
    for batch_idx, (imgs, _, types) in enumerate(train_dataloader):
        print(f'epoch: {epoch}, batch: {batch_idx}')
        resnet.zero_grad()
        score = resnet(imgs)
        loss = criterion(score,types)
        loss.backward()
        optimizer.step()
        avg_train_loss += loss.item()
    avg_train_loss /= batch_idx+1

    resnet.eval()
    with torch.no_grad():
        avg_eval_loss = 0
        for batch_idx, (imgs, _, types) in enumerate(test_dataloader):
            score = resnet(imgs)
            loss = criterion(score,types)
            avg_eval_loss += loss.item()
        avg_eval_loss /= batch_idx+1
        
    print(f"Epoch : {epoch}, train loss : {avg_train_loss:.3f}, test loss : {avg_eval_loss:.3f}")

torch.save(resnet.state_dict(), 'projet/parameters.pth')



#RNN  
class RNN(nn.Module):
    def __init__(self, vocab_size,emb_size,max_seq_length, hidden_size):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.emb_size=emb_size
        self.hidden_size=hidden_size
        #TODO ajouter drop out sur LSTM
        self.LSTM=nn.LSTM(emb_size,hidden_size,batch_first=True)
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax=nn.Softmax(-1)

    def forward(self,x):
        x=x.unsqueeze(1)
        h_prev = Variable(torch.zeros(1, batch_size, self.hidden_size))
        c_prev = Variable(torch.zeros(1, batch_size, self.hidden_size))
        hiddens = []
        for _ in range(self.max_seq_length):
            x,(h_prev,c_prev) = self.LSTM(x,(h_prev,c_prev))
            x=self.linear(x)
            probs=self.softmax(x)
            probs=probs[:,0,:]
            next_word_indices = torch.multinomial(probs, 1)
            x=next_word_indices
            x=self.embedding(x)
            hiddens.append(probs)
        logits = torch.stack(hiddens, 1)  
        return logits


class CRNN(nn.Module):
    def __init__(self,vocab_size,emb_size,max_seq_length, hidden_size):
        super().__init__()
        self.CNN=resnet
        self.CNN.load_state_dict(torch.load('projet/parameters.pth'))
        output = emb_size
        self.CNN.fc = nn.Linear(self.CNN.fc.in_features, output)
        for name, param in self.CNN.named_parameters():
            if param.requires_grad:
                if name!="fc.weight" and name!="fc.bias":
                    param.requires_grad=False
        self.RNN=RNN(vocab_size,emb_size,max_seq_length, hidden_size)
        
    def forward(self,x):
        x=self.CNN(x)
        x=self.RNN(x)
        return x

def print_samples(crnn):
    imgs_train, _, _ = next(iter(train_dataloader))
    imgs_test, _, _ = next(iter(test_dataloader))
    logits_train = crnn(imgs_train)
    logits_test = crnn(imgs_test)
    mots_encod_train = []
    mots_encod_test = []
    index=random.randint(0,31)
    for i in range(max_seq_length):
        couche_train = logits_train[index, i, :]
        mot_encod_train = torch.multinomial(couche_train, 1)
        mots_encod_train.append(mot_encod_train)
        couche_test = logits_test[index, i, :]
        mot_encod_test = torch.multinomial(couche_test, 1)
        mots_encod_test.append(mot_encod_test)
    vocab = train_dataset.vocab
    phrase_train = vocab.decode([mot.item() for mot in mots_encod_train])
    phrase_test = vocab.decode([mot.item() for mot in mots_encod_test])
    print(phrase_train)
    print(phrase_test)



emb_size=10
hidden_size=10
crnn=CRNN(train_dataset.get_vocab_size(),emb_size,max_seq_length,hidden_size)
criterion_bis = nn.CrossEntropyLoss()
optimizer_bis = optim.Adam(crnn.parameters(),lr=lr)
step=1

while True:
    crnn.train()
    avg_train_loss = 0
    for batch_idx, (imgs, legende, _) in enumerate(train_dataloader):
        print(f'step: {step}, batch: {batch_idx}')
        crnn.zero_grad()
        logits = crnn(imgs)
        loss = criterion_bis(logits.view(-1, logits.size(-1)),legende.view(-1))
        loss.backward()
        optimizer_bis.step()
        avg_train_loss += loss.item()
    avg_train_loss /= batch_idx+1
    resnet.eval()
    with torch.no_grad():
        avg_eval_loss = 0
        for batch_idx, (imgs, legende, _) in enumerate(test_dataloader):
            score = crnn(imgs)
            loss = criterion_bis(logits.view(-1, logits.size(-1)),legende.view(-1))
            avg_eval_loss += loss.item()
        avg_eval_loss /= batch_idx+1
        print_samples(crnn)
    
    print(f"Step : {step}, train loss : {avg_train_loss:.3f}, test loss : {avg_eval_loss:.3f}")
    step+=1
    

  




