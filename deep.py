import os
workspace = os.getcwd()
os.chdir(workspace + '/content/KorBertSum/src')

import sys
sys.path.append(workspace + '/content')
sys.path.append(workspace + '/content/KorBertSum/src')
sys.path.append(workspace + '/hashtag_calssifier')
sys.path.append(workspace + '/hashtag_calssifier/data')

from pyrouge import Rouge155
import torch
import numpy as np
from models import data_loader, model_builder
from models.model_builder import Summarizer
from others.logging import logger, init_logger
from models.data_loader import load_dataset
from transformers import BertConfig, BertTokenizer
from models.stats import Statistics
import easydict
from numpy.lib.function_base import append

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

def build_trainer(args, device_id, model, optim):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    grad_accum_count = args.accum_count
    n_gpu = args.world_size
    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0
    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank)
    if (model):
        n_params = _tally_parameters(model)
    return trainer

class Trainer(object):

    def __init__(self, args, model, optim, grad_accum_count=1, n_gpu=1, gpu_rank=1):
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.loss = torch.nn.BCELoss(reduction='none')
        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def summ(self, test_iter, step, cal_lead=False, cal_oracle=False):
      # Set model in validating mode.
      def _get_ngrams(n, text):
          ngram_set = set()
          text_length = len(text)
          max_index_ngram_start = text_length - n
          for i in range(max_index_ngram_start + 1):
              ngram_set.add(tuple(text[i:i + n]))
          return ngram_set

      def _block_tri(c, p):
          tri_c = _get_ngrams(3, c.split())
          for s in p:
              tri_s = _get_ngrams(3, s.split())
              if len(tri_c.intersection(tri_s))>0:
                  return True
          return False

      if (not cal_lead and not cal_oracle):
          self.model.eval()
      stats = Statistics()

      with torch.no_grad():
          for batch in test_iter:
              src = batch.src
              labels = batch.labels
              segs = batch.segs
              clss = batch.clss
              mask = batch.mask
              mask_cls = batch.mask_cls

              if (cal_lead):
                  selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
              elif (cal_oracle):
                  selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                  range(batch.batch_size)]
              else:
                  sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                  sent_scores = sent_scores + mask.float()
                  sent_scores = sent_scores.cpu().data.numpy()
                  selected_ids = np.argsort(-sent_scores, 1)
      
      index = 0
      i = 0
      for id in selected_ids:
        for number in id:
          if number == 0:
            firstIndex = i
          i = i+1
        id = np.delete(id,firstIndex)
        id =  np.insert(id,0,0)
        if index == 0:
          selected_ids = np.delete(selected_ids,0,0)
          selected_ids = np.insert(selected_ids,0,id,0)
        index = index +1
      return selected_ids

    def _gradient_accumulation(self, true_batchs, normalization, total_stats, report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask
            mask_cls = batch.mask_cls

            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

            loss = self.loss(sent_scores, labels.float())
            loss = (loss*mask.float()).sum()
            (loss/loss.numel()).backward()

            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)


            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model

        model_state_dict = real_model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _maybe_gather_stats(self, stat):
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_save(self, step):
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

class BertData():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def preprocess(self, src):
        if (len(src) == 0):
            return None
        original_src_txt = [' '.join(s) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s) > 1)]
        src = [src[i][:2000] for i in idxs]
        src = src[:1000]
        if (len(src) < 3):
            return None
        src_txt = [' '.join(sent) for sent in src]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = None
        src_txt = [original_src_txt[i] for i in idxs]
        tgt_txt = None
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt

def _lazy_dataset_loader(pt_file):
    yield  pt_file

args = easydict.EasyDict({
    "encoder":'classifier',
    "mode":'test',
    "model_path": workspace,
    "result_path": workspace + '/content/results',
    "temp_dir": workspace + '/content/temp',
    "batch_size":1000,
    "use_interval":True,
    "hidden_size":128,
    "ff_size":512,
    "heads":4,
    "inter_layers":2,
    "rnn_size":512,
    "param_init":0,
    "param_init_glorot":True,
    "dropout":0.1,
    "optim":'adam',
    "lr":2e-3,
    "report_every":1,
    "save_checkpoint_steps":5,
    "block_trigram":True,
    "recall_eval":False,
    
    "accum_count":1,
    "world_size":1,
    "visible_gpus":'-1',
    "gpu_ranks":'0',
    "log_file": workspace + '/content/logs/log.log',
    "test_from": workspace + '/content/model_step_40500.pt'
})
model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers','encoder','ff_actv', 'use_interval','rnn_size']

def test(args, input_list, device_id, pt, step):
    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    cp = args.test_from
    try:
        step = int(cp.split('.')[-2].split('_')[-1])
    except:
        step = 0

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])

    config = BertConfig.from_pretrained('bert-base-multilingual-cased')
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, _lazy_dataset_loader(input_list),
                                       args.batch_size, device, shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    result = trainer.summ(test_iter,step)
    return result, input_list

args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

def txt2input(text):
    data = list(filter(None, text.split('\n')))
    bertdata = BertData()
    txt_data = bertdata.preprocess(data)
    data_dict = {"src":txt_data[0],
                "labels":[0,1,2],
                "segs":txt_data[2],
                "clss":txt_data[3],
                "src_txt":txt_data[4],
                "tgt_txt":None}
    input_data = []
    input_data.append(data_dict)
    return input_data

import urllib.request 
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

import pandas as pd
import classifier_TextCNN
from classifier_TextCNN import classify

import makebow
from makebow import build_bag_of_words


# 중앙일보 크롤링
joongang_article = []
joongang_fin = []
category = ['politics', 'money', 'society']
cate = ['politic', 'economy', 'society']

total = pd.DataFrame()

# 홈페이지 메인 접근
for n in range(3):
    target_url = "https://www.joongang.co.kr/" + category[n]
    req = Request(target_url, headers={'User-Agent': 'Mozilla/5.0'})
    url = urlopen(req).read()
    soup = BeautifulSoup(url, "html.parser")

    # 가장 많이 본 기사 접근
    link = soup.find(class_="aside").find(class_="chain_wrap").find_all(class_="card")[0].find(class_="card_image").find('a').attrs['href']
    article_req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
    article_url = urlopen(article_req).read()
    article = BeautifulSoup(article_url, "html.parser")
    
    # 기사 제목, 저자, 작성일시, 사진
    try:
        title = article.find(class_="article_header").find(class_="headline").get_text().replace('\n', '')
    except:
        title = '제목 없음'
    try:
        author = article.find(class_="byline").find('a').get_text().strip('\n').replace('\n', ' ').replace('\xa0', '')
    except:
        author = '작성자 없음'
    date = article.find(class_="article_header").find_all(class_="date")[0].get_text().replace('입력 ', '')
    try:
        image = article.find(id="article_body").find(class_="photo_center").find('img').attrs['src'].replace('/_ir50_/', '')
    except:
        image = '이미지 없음'
  
    # 기사 내용
    try:
        content_arr = []
        sentence_arr = []
        content = article.select("article.article > div.article_body > p")
        for paragraph in content:
            if paragraph.string == None:
                continue
            else:
                content_arr.append(paragraph.get_text().lstrip().replace('\xa0', '').replace('. ', '.^'))
        content = '^'.join(content_arr).split('^')
        count = 0
        temp = []
        for sentence in content:
            if sentence.count('“') + sentence.count('”') != 0:
                count = count + sentence.count('“') + sentence.count('”')
                if count % 2 == 1:
                    temp.append(sentence)
                else:
                    if temp == []:
                        sentence_arr.append(sentence)
                    else:
                        temp.append(sentence)
                        sentence_arr.append(' '.join(temp))
                        temp = []
                    count = 0
            else:
                sentence_arr.append(sentence)
            content = '\n\n'.join(sentence_arr)
    except:
        content = '원문 내용 없음'

    article_arr = []
    article_arr.append(title.replace("&apos", "'"))
    article_arr.append(author)
    article_arr.append(date)
    article_arr.append(link)
    article_arr.append(image)
    article_arr.append(cate[n])
    article_arr.append(content)
    joongang_article.append(article_arr)
    total = build_bag_of_words(total, content)

joonangtitle = workspace + "/hashtag_classifier/data/craw_j.csv"
total.to_csv(joonangtitle, index=False, encoding="utf-8-sig")
hashtag = classify(workspace, joonangtitle)

indexForClassifier = 0
for article in joongang_article:
    try:
        text = article[6]
        article_original = article[6].replace('\n\n', ' ')
        del article[6]
        article.append(article_original)

        input_data = txt2input(text)
        sum_list = test(args, input_data, -1, '', None)
        article_extractive = ' '.join([list(filter(None, text.split('\n')))[i] for i in sum_list[0][0][:3]])
        article.append(article_extractive)
        
        article.append(hashtag[indexForClassifier])
        indexForClassifier += 1
        
        joongang_fin.append(article)
    except:
        joongang_fin.append([])
        indexForClassifier += 1


# 경향신문 크롤링
khan_article = []
khan_fin = []
category = ['politics', 'economy', 'national']
cate = ['politic', 'economy', 'society']

total = pd.DataFrame()

# 홈페이지 메인 접근
for n in range(3):
    target_url = "https://www.khan.co.kr/" + category[n]
    req = Request(target_url, headers={'User-Agent': 'Mozilla/5.0'})
    url = urlopen(req).read()
    soup = BeautifulSoup(url, "html.parser")
    
    # 가장 많이 본 기사 접근
    link = soup.find(class_="cont-aside").find_all("li")[0].find("a").attrs['href']
    article_req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
    article_url = urlopen(article_req).read()
    article = BeautifulSoup(article_url, "html.parser")
    
    # 기사 제목, 저자, 작성일시, 사진
    try:
        title = article.find(id="article_title").get_text()
    except:
        title = '제목 없음'
    try:
        author = article.find(class_="author").get_text()
    except:
        author = '작성자 없음'
    try:
        date = article.find(class_="byline").find_all('em')[0].get_text().replace('입력 : ', '')
    except:
        date = '작성일 없음'
    try:
        image = 'https:' + article.find(class_="art_photo_wrap").find('img').attrs['src']
    except:
        image = '이미지 없음'
    
    # 기사 내용
    try:
        content_arr = []
        sentence_arr = []
        content = article.select("div.art_body > p.content_text")
        for paragraph in content:
            content_arr.append(paragraph.get_text().strip('\n').replace('. ', '.^'))
        content = '^'.join(content_arr).split('^')
        count = 0
        temp = []
        for sentence in content:
            if sentence.count('“') + sentence.count('”') != 0:
                count = count + sentence.count('“') + sentence.count('”')
                if count % 2 == 1:
                    temp.append(sentence)
                else:
                    if temp == []:
                        sentence_arr.append(sentence)
                    else:
                        temp.append(sentence)
                        sentence_arr.append(' '.join(temp))
                        temp = []
                    count = 0
            else:
                sentence_arr.append(sentence)
        content = '\n\n'.join(sentence_arr)
    except:
        content = '원문 내용 없음'
    
    article_arr = []
    article_arr.append(title)
    article_arr.append(author)
    article_arr.append(date)
    article_arr.append(link)
    article_arr.append(image)
    article_arr.append(cate[n])
    article_arr.append(content)
    khan_article.append(article_arr)
    total = build_bag_of_words(total, content)

khantitle = workspace + "/hashtag_classifier/data/craw_k.csv"
total.to_csv(khantitle, index=False, encoding="utf-8-sig")
hashtag = classify(workspace, khantitle)

indexForClassifier = 0
for article in khan_article:
    try:
        text = article[6]
        article_original = article[6].replace('\n\n', ' ')
        del article[6]
        article.append(article_original)
        
        input_data = txt2input(text)
        sum_list = test(args, input_data, -1, '', None)
        article_extractive = ' '.join([list(filter(None, text.split('\n')))[i] for i in sum_list[0][0][:3]])
        article.append(article_extractive)
        
        article.append(hashtag[indexForClassifier])
        indexForClassifier += 1
        
        khan_fin.append(article)
    except:
        khan_fin.append([])
        indexForClassifier += 1


# 한겨레 크롤링
hani_article = []
hani_fin = []
category = ['politics', 'economy', 'society']
cate = ['politic', 'economy', 'society']

total = pd.DataFrame()

# 홈페이지 메인 접근
for n in range(3):
    target_url = "https://www.hani.co.kr/arti/" + category[n]+"/home01.html"
    req = Request(target_url, headers={'User-Agent': 'Mozilla/5.0'})
    url = urlopen(req).read()
    soup = BeautifulSoup(url, "html.parser")

    # 가장 많이 본 기사 접근
    link = soup.find(class_="article-popularity").find(class_="article-right first").find(class_="photo").find('a').attrs['href']
    article_req = Request("https:"+link, headers={'User-Agent': 'Mozilla/5.0'})
    article_url = urlopen(article_req).read()
    article = BeautifulSoup(article_url, "html.parser")

    # 기사 제목, 저자, 작성일시, 사진
    title = article.find(class_="article-head").find(class_="title").get_text()
    try:
        author = article.find(class_="kizapage-box").find(class_="name").find('strong').get_text().replace('\n', ' ').replace('\xa0', '').replace('☞뉴스레터 공짜 구독하기', '')
    except:
        author = '작성자 없음'
    date = article.find(class_="date-time").find_all('span')[0].get_text().replace('등록 :', '')
    image = "https:"+article.find(class_="a-left").find(class_="image").find('img').attrs['src']

    # 기사 내용
    content_arr = []
    sentence_arr = []
    content = article.select("div.article-text > div.article-text-font-size > div.text")
    content[0].find('div', class_='image-area').decompose()
    while content[0].find('a') != None:
        content[0].find('a').decompose()
    while content[0].find(class_='desc') != None:
        content[0].find(class_='desc').decompose()
    while content[0].find('strong') != None:
        content[0].find('strong').decompose()
    while content[0].find(class_="middleTitle2") != None:
        content[0].find(class_="middleTitle2").decompose()
    while content[0].find('b') != None:
        content[0].find('b').decompose()
    for paragraph in content:
        content_arr.append(paragraph.get_text().replace('\n', ' ').replace('. ', '.^').replace('\r', '').lstrip())
    content = '^'.join(content_arr).split('^')
    
    count = 0
    temp = []
    for sentence in content:
        if sentence.find("☞")!=-1:
            continue
        if sentence.count('“') + sentence.count('”') != 0:
            count = count + sentence.count('“') + sentence.count('”')
            if count % 2 == 1:
                temp.append(sentence)
            else:
                if temp == []:
                    sentence_arr.append(sentence)
                else:
                    temp.append(sentence)
                    sentence_arr.append(' '.join(temp))
                    temp = []
                count = 0
        else:
            sentence_arr.append(sentence)
    content = '\n\n'.join(sentence_arr)

    article_arr = []
    article_arr.append(title)
    article_arr.append(author)
    article_arr.append(date)
    article_arr.append('https:' + link)
    article_arr.append(image)
    article_arr.append(cate[n])
    article_arr.append(content)
    hani_article.append(article_arr)
    total = build_bag_of_words(total, content)

hanititle = workspace + "/hashtag_classifier/data/craw_h.csv"
total.to_csv(hanititle, index=False, encoding="utf-8-sig")
hashtag = classify(workspace, hanititle)

indexForClassifier = 0
for article in hani_article:
    text = article[6] 
    article_original = article[6].replace('\n\n', ' ')
    del article[6]
    article.append(article_original)

    input_data = txt2input(text)
    sum_list = test(args, input_data, -1, '', None)
    article_extractive = ' '.join([list(filter(None, text.split('\n')))[i] for i in sum_list[0][0][:3]])
    article.append(article_extractive)

    article.append(hashtag[indexForClassifier])
    indexForClassifier += 1

    hani_fin.append(article)


# DB 저장 파트
import dbdbdeep
from dbdbdeep import joonangToDB
from dbdbdeep import khanToDB
from dbdbdeep import haniToDB
joonangToDB(joongang_fin)
khanToDB(khan_fin)
haniToDB(hani_fin)
