import json
import torch
import numpy as np

class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self, school,path):
        self.batch_size = 32
        self.ptr = 0
        self.data = []
        self.school = school
        data_file = path+str(school)+'.json'
        config_file = 'config.txt'        

        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
   
   
        config = np.loadtxt(config_file,delimiter = ' ',dtype=int)
        index = np.where(config.T[0] == int(school))[0][0]
        self.student_n,self.exer_n,self.knowledge_n,self.know = config[index][1], config[index][2], config[index][3], config[index][3]

        self.knowledge_dim = int(self.knowledge_n)

      
    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
       
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code] = 1.0
            y = float(log['score'])
            input_stu_ids.append(log['user_id'])
            input_exer_ids.append(log['exer_id'])
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.FloatTensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size> len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, school,path):
        self.ptr = 0
        self.count = 0
        self.data = []
        self.school = school
        data_file = path+str(school)+'.json'
        config_file = 'config.txt'
    
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)

        config = np.loadtxt(config_file,delimiter = ' ',dtype=int)
        index = np.where(config.T[0] == int(school))[0][0]
        self.student_n,self.exer_n,self.knowledge_n = config[index][1], config[index][2], config[index][3]     
        self.knowledge_dim = int(self.knowledge_n)
    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        try: 
            teacher = self.data[self.ptr]['teacher']
        except:
            teacher = 0
        know = [] 
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for log in logs:
            knowitem = []
            input_stu_ids.append(user_id)
            input_exer_ids.append(int(log['exer_id']))
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
              
                knowledge_emb[knowledge_code] = 1.0
                knowitem.append(knowledge_code)
            know.append(knowitem)   
            input_knowledge_embs.append(knowledge_emb)
            y = float(log['score'])
            ys.append(y)
            self.count += 1
            
        self.ptr += 1
    
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.FloatTensor(ys), know, teacher

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
