import pandas as pd
import re

def classify(workspace, datapath):
      dataVersion = "v4"
      trainVersion = "1"
      dataVersionPath = dataVersion+"_"+trainVersion
      
      # Field 클래스 정의
      from torchtext.legacy.data import Field
      def tokenizer(text):
            text = re.sub('[\[\]\']', '', str(text))
            text = text.split(', ')
            return text
      TEXT = Field(tokenize=tokenizer)
      LABEL = Field(sequential = False)

      # 데이터 불러오기
      from torchtext.legacy.data import TabularDataset
      tra, validat = TabularDataset.splits(
            path = workspace + '/hashtag_classifier/data/',
            train = 'train_'+dataVersionPath+'.csv',
            validation = 'validation_'+dataVersionPath+'.csv',
            format = 'csv',
            fields = [('text', TEXT), ('label', LABEL)],
            skip_header = True
      )

      test = TabularDataset(
            path = datapath,
            format = 'csv',
            fields = [('text', TEXT), ('label', LABEL)],
            skip_header = True
      )

      # 단어장 및 DataLoader 정의
      import torch
      from torchtext.vocab import Vectors
      from torchtext.legacy.data import BucketIterator

      vectors = Vectors(name = workspace + '/hashtag_classifier/data/tokens_'+dataVersionPath)
      
      TEXT.build_vocab(tra, vectors = vectors, min_freq = 1, max_size = None)
      vocab = TEXT.vocab
      
      LABEL.build_vocab(tra)
      embedded = {}
      for key, value in LABEL.vocab.stoi.items():
            if value != 0:
                  embedded[value-1] = key

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      test_iter = BucketIterator(
            test,
            batch_size = 3,
            device = device,
            shuffle=False
      )

      # TextCNN 모델링
      import torch.nn as nn
      import torch.optim as optim
      import torch.nn.functional as F

      class TextCNN(nn.Module):
            def __init__(self, vocab_built, emb_dim, dim_channel, kernel_wins, num_class):
                  super(TextCNN, self).__init__()
                  self.embed = nn.Embedding(len(vocab_built), emb_dim)
                  self.embed.weight.data.copy_(vocab_built.vectors)
                  self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim))
                                          for w in kernel_wins])
                  self.relu = nn.ReLU()
                  self.dropout = nn.Dropout(0.4)
                  self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)
                  
            def forward(self, x):
                  emb_x = self.embed(x)
                  emb_x = emb_x.unsqueeze(1)
                  con_x = [self.relu(conv(emb_x)) for conv in self.convs]
                  pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2])
                        for x in con_x]
                  fc_x = torch.cat(pool_x, dim=1)
                  fc_x = fc_x.squeeze(-1)
                  fc_x = self.dropout(fc_x)
                  logit = self.fc(fc_x)
                  return logit

      def predict(model, device, itr):
            model.eval()
            results = []
            batchs = 0
            for batch in itr:
                  text = batch.text
                  target = batch.label
                  text = torch.transpose(text, 0, 1)
                  target.data.sub_(1)
                  text, target = text.to(device), target.to(device)
                  logit = model(text)
                  result = torch.max(logit, 1)[1]
                  results.append(result)
                  batchs += 1
            return results, batchs

      def tensortoword(results, batchs):  
            hashtag = []    
            for bat in range(0, batchs):
                  embelist = results[bat].tolist()
                  for index in range(0, 3):
                        print("predict 결과:", embelist[index], embedded[embelist[index]], " ")
                  hashtag.append(embedded[embelist[index]])
                  print()
            return hashtag

      model = TextCNN(vocab, 100, 10, [3, 4, 5], 12).to(device)
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model.load_state_dict(torch.load(workspace + '/hashtag_classifier/data/Best_Validation_'+dataVersionPath, map_location=device))
      model.to(device)
      results, batchs = predict(model, device, test_iter)
      
      hashtag = tensortoword(results, batchs)
      
      return hashtag