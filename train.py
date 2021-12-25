import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score, r2_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
import copy
import random
from sklearn.cluster import KMeans
import torch.nn.functional as F
from scipy.spatial.distance import cdist

device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
epoch_n = 5
Epoch = 5

Name = ['skill_W', 'know_W','relative_M']
Matrix = ['student_emb', 'k_difficulty', 's', 'g'] 



def Fedknow(w,weights,know,indice,method):
 
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        if k in Name:
            w_avg[k] = w_avg[k].fill_(0)
    
    student_emb = []
    question_emb = []

    if 'fed' in method:
      for k in w_avg.keys():
          if k in Name:
              if k == 'skill_M':
                  allw = 0
                  for i in range(0, len(w)):
                      r = random.random()
                      if np.sqrt(r) > indice[i][0]:
                          allw += 0.001
                          alpha = 0.001
                      else:
                          allw += weights[i]
                          alpha = weights[i]
                      w_avg[k] += w[i][k] * alpha
                  w_avg[k] /= allw
              else:
                   
                  for i in range(0, len(w)):   
                      r = random.random()            
                      for j in range(0,len(w_avg[k])):
                          item_indice = indice[i][1].detach().cpu().numpy()
                          item_indice = (item_indice)/ max(item_indice)
                          if np.sqrt(r)  > item_indice[j]:
                              know[i][j] = 0.001
                  
                  for i in range(0, len(w)): 
                      alpha = know[i] /np.sum(know)
                      w_avg[k] += w[i][k] * alpha
    

        
    center = []
    for i in range(0, len(w)):
       student_emb = w[i]['student_emb.weight'].detach().cpu()
       clusterMs = KMeans(n_clusters=int(len(student_emb)/5)+1, random_state=0)
       clusterMs.fit_predict(student_emb)
       center.append(clusterMs.cluster_centers_)  
    center = np.vstack(center)  
    clusterMs = KMeans(n_clusters=int(len(center)/10), random_state=0)
    clusterMs.fit_predict(center)    
    
    center = []
    for i in range(0, len(w)):
       question_emb = torch.cat((w[i]['k_difficulty.weight'], w[i]['s.weight'], w[i]['g.weight']),1).detach().cpu() 
       clusterMq = KMeans(n_clusters=int(len(question_emb)/10)+1, random_state=0)
       clusterMq.fit_predict(question_emb)
       center.append(clusterMq.cluster_centers_)
    center = np.vstack(center)

    clusterMq = KMeans(n_clusters=int(len(center)/10), random_state=0)
    clusterMq.fit_predict(center) 

    return w_avg,clusterMs.cluster_centers_,clusterMq.cluster_centers_,[]


def Apply(g_model,local,auc,student_group,question_group,method):
    w = g_model
    l_w = copy.deepcopy(local.state_dict())
    
    if 'fed' in method:
        metricstr = 'euclidean'
        student_emb = l_w['student_emb.weight']
        distance = cdist(student_emb.detach().cpu(),student_group,metric=metricstr)
        q = torch.FloatTensor(auc[1]).to(device).view(1,-1)
        distance[np.isnan(distance)] = 1
        distance = distance/(np.sum(distance,1)[:,None])
        distance = torch.FloatTensor(distance)
        centers = torch.FloatTensor(distance).to(device) @  torch.FloatTensor(student_group).to(device)
        l_w['student_emb.weight'] = l_w['student_emb.weight']*q + centers * (1-q)

        question_emb = torch.cat((l_w['k_difficulty.weight'], l_w['s.weight'], l_w['g.weight']),1)
        distance = cdist(question_emb.detach().cpu(),question_group,metric=metricstr)
        q = torch.FloatTensor(auc[1]).to(device).view(1,-1)
        distance[np.isnan(distance)] = 1
        distance = distance/(np.sum(distance,1)[:,None])
        distance = torch.FloatTensor(distance)
        centers = torch.FloatTensor(distance).to(device) @  torch.FloatTensor(question_group).to(device)
        l_w['k_difficulty.weight'] = l_w['k_difficulty.weight'] * q + centers[:,:-2] * (1-q)
        q = auc[0]
        l_w['s.weight'] = l_w['s.weight'] * q + (centers[:,-2] * (1-q)).view(-1,1)
        l_w['g.weight'] = l_w['g.weight'] * q + (centers[:,-2] * (1-q)).view(-1,1)
        
        
      for k in w.keys():
         
         if k in Name:
             if k == 'skill_M':
                 q = auc[0]
                 l_w[k] = l_w[k]*q + w[k]*(1-q)
             else:          
                 for i in range(0,len(l_w[k])):
                     q = auc[1][i]
                     l_w[k][i] = l_w[k][i]*q + w[k][i]*(1-q) 
                     
    local.load_state_dict(l_w) 

def total(result):
    metric = []
    pred_all = []
    label_all = []
    for i in range(len(result)):
        metric.append(result[i][0])
        pred_all += result[i][1]
        label_all += result[i][2]
    metric = np.array(metric).T
    counts = metric[3]
    acc = np.sum(metric[1] * counts / np.sum(counts))
    r2 = np.sum(metric[4] * counts / np.sum(counts))
    mAP = np.sum(metric[1]) / len(metric[1])
    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    mae = np.mean(np.abs(label_all - pred_all))
    print('federated acc= %f, rmse= %f, mae= %f, map= %f, r2= %f' % (acc, rmse, mae, mAP, r2))
    return acc

def train(method):
    seed = 0
    school_list = [0,1,2,3,4,5,6,7,8,9]

    Nets = []
    random.seed(seed)
    path = 'data/datatest'
    for school in school_list:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
     
        data_loader = TrainDataLoader(school, path)
        val_loader = ValTestDataLoader(school, path)
        net = Net(data_loader.student_n, data_loader.exer_n, data_loader.knowledge_n)
        net = net.to(device)
        Nets.append([school,data_loader,net,copy.deepcopy(net.state_dict()),val_loader]) 
             

    global_model1 = Nets[0][3]
    loss_function = nn.MSELoss(reduction='mean')
    
    Gauc = 0
    for i in range(Epoch):
        AUC = []
        ACC = [] 
        for index in range(len(Nets)):
            school = Nets[index][0]
            net = Nets[index][2]
            data_loader =Nets[index][1]
            val_loader = Nets[index][4]
            optimizer = optim.Adam(net.parameters(), lr=0.001)  
            print('training model...'+str(school))
            best = 0
            best_epoch = 0
            best_knowauc = None
            best_indice = 0
            for epoch in range(epoch_n):
                metric,_,_,know_auc,know_acc = validate(net, epoch, school, path, val_loader)
                auc = metric[1]
                rmse = metric[0]
                indice = metric[1]
                if auc > best:     
                    best = auc
                    best_knowauc = know_auc
                    best_indice = indice
                    best_epoch = epoch
                    best_knowacc = know_acc
                    Nets[index][3]  = copy.deepcopy(net.state_dict())
                if epoch - best_epoch >= 5:
                    break
          
                data_loader.reset()
                running_loss = 0.0
                batch_count = 0
                know_distribution = torch.zeros((data_loader.knowledge_n))
                while not data_loader.is_end():
                   
                    batch_count += 1
                    input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
                    
                    know_distribution += torch.sum(input_knowledge_embs,0)
                    input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
                    loss = loss_function(output, labels)                                                    
                    loss.backward()
                    optimizer.step()
                     

            
            net.load_state_dict(Nets[index][3])
            Nets[index][2] = net
            distribution = know_distribution * best_knowacc
            distribution[distribution == 0] = 0.001 
            Nets[index].append( distribution.unsqueeze(1).to(device) )
            print('Best AUC:',best)    
            AUC.append([best_indice,best_knowacc]) 
            ACC.append(best_indice) 
     

        l_school = [item[0] for item in Nets] 
        l_weights = [len(item[1].data) for item in Nets]
        l_know = [item[5] for item in Nets]
        l_net = [item[3] for item in Nets]
        metric0 = []
        metric1 = []
        metric2 = []
        global_model2, student_group, question_group, _ = Fedknow(l_net,l_weights,l_know,AUC,method)
        print('global test ===========')        
        for k in range(len(Nets)):
            metric2.append(validate(Nets[k][2], i, l_school[k], path, Nets[k][4]))
        globalauc = total(metric2)

        for k in range(len(Nets)):
            Apply(copy.deepcopy(global_model2),Nets[k][2],AUC[k],student_group,question_group,method)
             
def validate(model, epoch, school, path, val_loader):
    school = int(school)  
    data_loader = val_loader
    net = Net(data_loader.student_n, data_loader.exer_n, data_loader.knowledge_n)
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()
    ALLK= []
    KNOW = []
    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    mpred_all, mlabel_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, know, teacher = data_loader.next_batch()            
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1) 

        # compute accuracy
        
        for i in range(len(labels)):
            for i_know in know[i]:
                if i_know not in ALLK:
                    ALLK.append(i_know) 
                KNOW.append(i_know)
                mpred_all += [output[i].to(torch.device('cpu'))]
                mlabel_all += [labels[i].to(torch.device('cpu'))]
            if labels[i] >= output[i] and output[i] > labels[i]-1:
                correct_count += 1
                                   
                
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()
       

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    R2 = r2_score(label_all, pred_all)
    mae = np.mean(np.abs(label_all - pred_all)) 
    print('school= %d, r2= %f, rmse= %f, mae= %f' % (school, R2, rmse, mae))

    know_distribution = torch.ones((data_loader.knowledge_n))
    know_acc = torch.zeros((data_loader.knowledge_n))
    for know in ALLK:
        K_pred = []
        K_label = []
        K_pred2 = []
        for i in range(len(KNOW)):
            if KNOW[i] == know:
                K_pred2.append(1 if (mlabel_all[i] >= mpred_all[i] and mpred_all[i]>mlabel_all[i] -1) else 0)
        know_acc[know] = np.sum(np.array(K_pred2))/len(K_pred2)
    return [mae,accuracy,rmse,exer_count,R2],list(pred_all),list(label_all),know_distribution,know_acc


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    if (len(sys.argv) != 4) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
        print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])
        method = sys.argv[3]
    train(method)
