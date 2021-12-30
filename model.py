import torch
import torch.nn as nn
import torch.nn.functional as F
    
  
class Net(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n):
        self.knowledge_dim = int(knowledge_n)
        self.exer_n = int(exer_n)
        self.emb_num = int(student_n)
        self.stu_dim = int(self.knowledge_dim)

        self.h_dim = 5
        super(Net, self).__init__()

        self.skill_W = nn.Parameter(torch.Tensor(self.knowledge_dim,self.h_dim))
        self.know_W = nn.Parameter(torch.Tensor(self.knowledge_dim,self.h_dim))        
        self.skill_M = nn.Parameter(torch.Tensor(self.h_dim,1))
        nn.init.xavier_uniform_(self.skill_W,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.know_W,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.skill_M,
                                gain=nn.init.calculate_gain('relu'))
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.student_emb.weight.data.copy_(torch.zeros(self.emb_num, self.stu_dim))
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.k_difficulty.weight.data.copy_(torch.zeros(self.exer_n, self.knowledge_dim))
        self.relative_M = nn.Parameter(torch.eye(self.knowledge_dim))       
        self.s = nn.Embedding(self.exer_n, 1)
        self.s.weight.data.copy_(torch.zeros(self.exer_n, 1))
        self.g = nn.Embedding(self.exer_n, 1)
        self.g.weight.data.copy_(torch.zeros(self.exer_n, 1))
    def forward(self, stu_id, exer_id, kn_emb):
        relative_emb = (kn_emb @ (self.relative_M) )
         
        stu_emb = ((self.student_emb(stu_id))*relative_emb).unsqueeze(1)
        skill =   torch.sigmoid(stu_emb @ (self.skill_W) ).squeeze(1)

        k_diffculty = ((self.k_difficulty(exer_id))*relative_emb).unsqueeze(1)
        diffculty = torch.sigmoid( k_diffculty @ (self.know_W) ).squeeze(1)      
        input_x = (skill  - diffculty)@ (self.skill_M)

        slip = torch.sigmoid(self.s(exer_id)-1)
        guess = torch.sigmoid(self.g(exer_id)-1)
        input_x = (1- slip)*input_x + guess * (1-input_x)
        output = torch.sigmoid(input_x)# [0,1]
        #output = 5*torch.sigmoid(input_x)#[0,5]
        
        
        return output.squeeze(-1)




