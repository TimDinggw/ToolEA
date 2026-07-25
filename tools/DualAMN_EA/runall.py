from __future__ import absolute_import

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf
import numpy as np

import sys
import io
# 设置标准输出的编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class NR_GraphAttention(Layer):

    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 depth = 1,
                 use_w = False,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias
        self.use_w = use_w
        self.depth = depth

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        self.biases = []        
        self.attn_kernels = []  
        self.gat_kernels = []
        self.gate_kernels = []

        super(NR_GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        node_F = input_shape[0][-1]
        rel_F = input_shape[1][-1]
        self.ent_F = node_F
        ent_F = self.ent_F
            
        self.gate_kernel = self.add_weight(shape=(ent_F*(self.depth+1),ent_F*(self.depth+1)),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 name='gate_kernel')

        self.proxy = self.add_weight(shape=(64,node_F*(self.depth+1)),
                                   initializer=self.attn_kernel_initializer,
                                   regularizer=self.attn_kernel_regularizer,
                                   constraint=self.attn_kernel_constraint,
                                   name='proxy')

        if self.use_bias:
            self.bias = self.add_weight(shape=(1,ent_F*(self.depth+1)),
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint,
                                   name='bias')
            
        for l in range(self.depth):

            self.attn_kernels.append([])
            for head in range(self.attn_heads):                
                attn_kernel = self.add_weight(shape=(1*node_F ,1),
                                       initializer=self.attn_kernel_initializer,
                                       regularizer=self.attn_kernel_regularizer,
                                       constraint=self.attn_kernel_constraint,
                                       name='attn_kernel_self_{}'.format(head))
                    
                self.attn_kernels[l].append(attn_kernel)
                
        self.built = True
        
    
    def call(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]     
        adj = tf.SparseTensor(K.cast(K.squeeze(inputs[2],axis = 0),dtype = "int64"),
                         K.ones_like(inputs[2][0,:,0]),(self.node_size,self.node_size))
        sparse_indices = tf.squeeze(inputs[3],axis = 0)  
        sparse_val = tf.squeeze(inputs[4],axis = 0)
        
        if self.use_w:
            features = features*self.gcn_kernel
        features = self.activation(features)
        outputs.append(features)
                        
        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]  
                rels_sum = tf.SparseTensor(indices=sparse_indices,values=sparse_val,dense_shape=(self.triple_size,self.rel_size))
                
                rels_sum = tf.sparse_tensor_dense_matmul(rels_sum,rel_emb)
                neighs = K.gather(features,adj.indices[:,1])
                selfs = K.gather(features,adj.indices[:,0])
                
                rels_sum = tf.nn.l2_normalize(rels_sum, 1)
                neighs = neighs - 2 * tf.reduce_sum(neighs * rels_sum, 1, keepdims=True) * rels_sum
                
                att = K.squeeze(K.dot(rels_sum,attention_kernel),axis = -1) #K.concatenate([selfs,neighs,rels_sum])
                att = tf.SparseTensor(indices=adj.indices, values=att, dense_shape=adj.dense_shape)
                att = tf.sparse_softmax(att)
            
                if self.use_w:
                    gat_kernel = self.gat_kernels[head]
                    sums = K.dot(sums,gat_kernel)
                    
                new_features = tf.segment_sum (neighs*K.expand_dims(att.values,axis = -1),adj.indices[:,0])
                features_list.append(new_features)

            if self.attn_heads_reduction == 'concat':
                features = K.concatenate(features_list)  # (N x KF')
            else:
                features = K.mean(K.stack(features_list), axis=0)

            features = self.activation(features)
            outputs.append(features)
        
        outputs = K.concatenate(outputs)
        proxy_att = K.dot(tf.nn.l2_normalize(outputs,axis=-1),K.transpose(tf.nn.l2_normalize(self.proxy,axis=-1)))
        proxy_att = K.softmax(proxy_att,axis = -1)
        proxy_feature = outputs - K.dot(proxy_att,self.proxy)

        if self.use_bias:
            gate_rate = K.sigmoid(K.dot(proxy_feature,self.gate_kernel) + self.bias)
        else:
            gate_rate = K.sigmoid(K.dot(proxy_feature,self.gate_kernel))
        outputs = (gate_rate) * outputs + (1-gate_rate) * proxy_feature
                
        if self.use_w:
            return [outputs] + [self.gcn_kernel]
        else:
            return outputs

    def compute_output_shape(self, input_shape):    
        node_shape = self.node_size, (input_shape[0][-1]) * (self.depth+1)
        if not self.use_w:        
            return node_shape
        else:
            return [node_shape]+[self.gcn_kernel.shape]


import numpy as np
import scipy.sparse as sp
import scipy
import tensorflow as tf
import os
import multiprocessing

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    for line in open(file_name,'r'):
        head,r,tail = [int(item) for item in line.split()]
        entity.add(head); entity.add(tail); rel.add(r+1)
        triples.append((head,r+1,tail))
    return entity,rel,triples

def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1),int(e2)))
    return alignment_pair

def get_matrix(triples,entity,rel):
        ent_size = max(entity)+1
        rel_size = (max(rel) + 1)
        print(ent_size,rel_size)
        adj_matrix = sp.lil_matrix((ent_size,ent_size))
        adj_features = sp.lil_matrix((ent_size,ent_size))
        radj = []
        rel_in = np.zeros((ent_size,rel_size))
        rel_out = np.zeros((ent_size,rel_size))
        
        for i in range(max(entity)+1):
            adj_features[i,i] = 1

        for h,r,t in triples:        
            adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
            adj_features[h,t] = 1; adj_features[t,h] = 1;
            radj.append([h,t,r]); radj.append([t,h,r+rel_size]); 
            rel_out[h][r] += 1; rel_in[t][r] += 1
            
        count = -1
        s = set()
        d = {}
        r_index,r_val = [],[]
        for h,t,r in sorted(radj,key=lambda x: x[0]*10e10+x[1]*10e5):
            if ' '.join([str(h),str(t)]) in s:
                r_index.append([count,r])
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h),str(t)]))
                r_index.append([count,r])
                r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]
        
        rel_features = np.concatenate([rel_in,rel_out],axis=1)
        adj_features = normalize_adj(adj_features)
        rel_features = normalize_adj(sp.lil_matrix(rel_features))    
        return adj_matrix,r_index,r_val,adj_features,rel_features      
    
def load_data(lang,train_ratio = 0.3):             
    entity1,rel1,triples1 = load_triples(lang + 'triples_1')
    entity2,rel2,triples2 = load_triples(lang + 'triples_2')
    if "_en" in lang:
        alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
        np.random.shuffle(alignment_pair)
        train_pair,dev_pair = alignment_pair[0:int(len(alignment_pair)*train_ratio)],alignment_pair[int(len(alignment_pair)*train_ratio):]
    else:
        train_pair = load_alignment_pair(lang + 'sup_ent_ids')
        dev_pair = load_alignment_pair(lang + 'ref_ent_ids')
        ae_features = None
    
    adj_matrix,r_index,r_val,adj_features,rel_features = get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2))

    return np.array(train_pair),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),adj_features,rel_features


import keras
import numpy as np
import numba as nb
from utils import *
from tqdm import *
import tensorflow as tf
import keras.backend as K
from keras.layers import *

class evaluate:
    def __init__(self,dev_pair):
        self.dev_pair = dev_pair
        
        Matrix_A = Input(shape=(None,None))
        Matrix_B = Input(shape=(None,None))

        def dot(tensor):
            k = 10
            A,B = [K.squeeze(matrix,axis=0) for matrix in tensor]
            A_sim = K.dot(A,K.transpose(B))
            return K.expand_dims(A_sim,axis=0)

        results = Lambda(dot)([Matrix_A,Matrix_B])
        self.sim_model = keras.Model(inputs = [Matrix_A,Matrix_B],outputs = results)

        k = 10
        Matrix_A = Input(shape=(None,None))
        results = Lambda(lambda x: K.expand_dims(K.sum(tf.nn.top_k(K.squeeze(x,axis=0),k=k)[0],axis=-1) / k,axis=0))(Matrix_A)
        self.avg_model = keras.Model(inputs = [Matrix_A],outputs = results)

        Matrix_A = Input(shape=(None,None))
        LR_input = Input(shape=(None,))
        RL_input = Input(shape=( None,))
        Ans_input = Input(shape=(None,))
        def CSLS(tensor):
            sim,LR,RL,_ = [K.squeeze(m,axis=0) for m in tensor]
            LR,RL = [K.expand_dims(m,axis=1) for m in [LR,RL]]
            sim = 2*sim - K.transpose(LR)
            sim = sim - RL
            rank = tf.argsort(-sim,axis=-1)
            return K.expand_dims(rank[:,0],axis=0)

        rank = Lambda(CSLS)([Matrix_A,LR_input,RL_input,Ans_input])
        self.rank_model = keras.Model(inputs = [Matrix_A,LR_input,RL_input,Ans_input],outputs = rank)

        Matrix_A = Input(shape=(None,None))
        LR_input = Input(shape=(None,))
        RL_input = Input(shape=(None,))
        Ans_input = Input(shape=(None,))

        def CSLS(tensor):
            sim,LR,RL,ans_rank = [K.squeeze(m,axis=0) for m in tensor]
            LR,RL,ans_rank = [K.expand_dims(m,axis=1) for m in [LR,RL,ans_rank]]
            sim = 2*sim - K.transpose(LR)
            sim = sim - RL
            rank = tf.argsort(-sim,axis=-1)
            results = tf.where(tf.equal(rank,K.cast(K.tile(ans_rank,[1,len(self.dev_pair)]),dtype="int32")))
            return K.expand_dims(results,axis=0)

        results = Lambda(CSLS)([Matrix_A,LR_input,RL_input,Ans_input])
        self.CSLS_model = keras.Model(inputs = [Matrix_A,LR_input,RL_input,Ans_input],outputs = results)

    def CSLS_cal(self, Lvec,Rvec,evaluate = True,batch_size = 1024):
        L_sim,R_sim = [],[]
        for epoch in range(len(Lvec) // batch_size + 1):
            L_sim.append(self.sim_model.predict([np.expand_dims(Lvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Rvec,axis=0)]))
            R_sim.append(self.sim_model.predict([np.expand_dims(Rvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Lvec,axis=0)]))

        LR,RL = [],[]
        for epoch in range(len(Lvec) // batch_size + 1):
            LR.append(self.avg_model.predict([L_sim[epoch]]))
            RL.append(self.avg_model.predict([R_sim[epoch]]))

        if evaluate:
            results = []
            for epoch in range(len(Lvec) // batch_size + 1):
                ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(Lvec)))])
                result = self.CSLS_model.predict([R_sim[epoch],np.concatenate(LR,axis=1),RL[epoch],np.expand_dims(ans_rank,axis=0)])
                results.append(result)
            return np.concatenate(results,axis=1)[0]
        else:
            l_rank,r_rank = [],[]
            for epoch in range(len(Lvec) // batch_size + 1):
                ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(Lvec)))])
                r_rank.append(self.rank_model.predict([R_sim[epoch],np.concatenate(LR,axis=1),RL[epoch],np.expand_dims(ans_rank,axis=0)]))
                l_rank.append(self.rank_model.predict([L_sim[epoch],np.concatenate(RL,axis=1),LR[epoch],np.expand_dims(ans_rank,axis=0)]))

            return np.concatenate(r_rank,axis=1)[0],np.concatenate(l_rank,axis=1)[0] 

    def test(self, Lvec,Rvec):
        results  = self.CSLS_cal(Lvec,Rvec)
        def cal(results):
            hits1,hits5,hits10,mrr = 0,0,0,0
            for x in results[:,1]:
                if x < 1:
                    hits1 += 1
                if x < 5:
                    hits5 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1/(x + 1)
            return hits1,hits5,hits10,mrr
        hits1,hits5,hits10,mrr = cal(results)
        print("Hits@1: ",hits1/len(Lvec)," ","Hits@5: ",hits5/len(Lvec)," ","Hits@10: ",hits10/len(Lvec)," ","MRR: ",mrr/len(Lvec))
        return results

import warnings
warnings.filterwarnings('ignore')

import os
import keras
import numpy as np
import numba as nb
from tqdm import *
import tensorflow as tf
import keras.backend as K
from keras.layers import *
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True  
sess = tf.Session(config=config)  


# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True  
# sess = tf.compat.v1.Session(config=config)



seed = 12306
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)


train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features = load_data("data/ja_en/",train_ratio=0.30)
adj_matrix = np.stack(adj_matrix.nonzero(),axis = 1)
rel_matrix,rel_val = np.stack(rel_features.nonzero(),axis = 1),rel_features.data
ent_matrix,ent_val = np.stack(adj_features.nonzero(),axis = 1),adj_features.data


node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix)
node_hidden = 128
rel_hidden = 128
batch_size = 1024
dropout_rate = 0.3
lr = 0.005
gamma = 1
depth = 2
save_path = "embs.pkl"

def get_embedding(index_a,index_b,vec = None):
    if vec is None:
        inputs = [adj_matrix,r_index,r_val,rel_matrix,ent_matrix]
        inputs = [np.expand_dims(item,axis=0) for item in inputs]
        vec = get_emb.predict_on_batch(inputs)
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec,axis=-1,keepdims=True)+1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec,axis=-1,keepdims=True)+1e-5)
    return Lvec,Rvec

class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings
    
def get_trgat(node_hidden,rel_hidden,triple_size=triple_size,node_size=node_size,rel_size=rel_size,dropout_rate = 0,gamma = 3,lr = 0.005,depth = 2):
    adj_input = Input(shape=(None,2))
    index_input = Input(shape=(None,2),dtype='int64')
    val_input = Input(shape = (None,))
    rel_adj = Input(shape=(None,2))
    ent_adj = Input(shape=(None,2))
    
    ent_emb = TokenEmbedding(node_size,node_hidden,trainable = True)(val_input) 
    rel_emb = TokenEmbedding(rel_size,node_hidden,trainable = True)(val_input)
    
    def avg(tensor,size):
        adj = K.cast(K.squeeze(tensor[0],axis = 0),dtype = "int64")   
        adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:,0],dtype = 'float32'), dense_shape=(node_size,size)) 
        adj = tf.sparse_softmax(adj) 
        return tf.sparse_tensor_dense_matmul(adj,tensor[1])
    
    opt = [rel_emb,adj_input,index_input,val_input]
    ent_feature = Lambda(avg,arguments={'size':node_size})([ent_adj,ent_emb])
    rel_feature = Lambda(avg,arguments={'size':rel_size})([rel_adj,rel_emb])
    
    e_encoder = NR_GraphAttention(node_size,activation="tanh",
                                       rel_size = rel_size,
                                       use_bias = True,
                                       depth = depth,
                                       triple_size = triple_size)
    
    r_encoder = NR_GraphAttention(node_size,activation="tanh",
                                       rel_size = rel_size,
                                       use_bias = True,
                                       depth = depth,
                                       triple_size = triple_size)
    
    out_feature = Concatenate(-1)([e_encoder([ent_feature]+opt),r_encoder([rel_feature]+opt)])
    out_feature = Dropout(dropout_rate)(out_feature)
    
    alignment_input = Input(shape=(None,2))
    
    def align_loss(tensor): 
        
        def squared_dist(x):
            A,B = x
            row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
            row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
            row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
            row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
            return row_norms_A + row_norms_B - 2 * tf.matmul(A, B,transpose_b=True) 
        
        emb = tensor[1]
        l,r = K.cast(tensor[0][0,:,0],'int32'),K.cast(tensor[0][0,:,1],'int32')
        l_emb,r_emb = K.gather(reference=emb,indices=l),K.gather(reference=emb,indices=r)
        
        pos_dis = K.sum(K.square(l_emb-r_emb),axis=-1,keepdims=True)
        r_neg_dis = squared_dist([r_emb,emb])
        l_neg_dis = squared_dist([l_emb,emb])
        
        l_loss = pos_dis - l_neg_dis + gamma
        l_loss = l_loss *(1 - K.one_hot(indices=l,num_classes=node_size) - K.one_hot(indices=r,num_classes=node_size))
        
        r_loss = pos_dis - r_neg_dis + gamma
        r_loss = r_loss *(1 - K.one_hot(indices=l,num_classes=node_size) - K.one_hot(indices=r,num_classes=node_size))
        
        r_loss = (r_loss - K.stop_gradient(K.mean(r_loss,axis=-1,keepdims=True))) / K.stop_gradient(K.std(r_loss,axis=-1,keepdims=True))
        l_loss = (l_loss - K.stop_gradient(K.mean(l_loss,axis=-1,keepdims=True))) / K.stop_gradient(K.std(l_loss,axis=-1,keepdims=True))
        
        lamb,tau = 30, 10
        l_loss = K.logsumexp(lamb*l_loss+tau,axis=-1)
        r_loss = K.logsumexp(lamb*r_loss+tau,axis=-1)
        return K.mean(l_loss + r_loss)

    loss = Lambda(align_loss)([alignment_input,out_feature])

    inputs = [adj_input,index_input,val_input,rel_adj,ent_adj]
    train_model = keras.Model(inputs = inputs + [alignment_input],outputs = loss)
    train_model.compile(loss=lambda y_true,y_pred: y_pred,optimizer=keras.optimizers.rmsprop(lr))
    
    feature_model = keras.Model(inputs = inputs,outputs = out_feature)
    
    return train_model,feature_model


model,get_emb = get_trgat(dropout_rate=dropout_rate,
                          node_size=node_size,
                          rel_size=rel_size,
                          depth=depth,
                          gamma =gamma,
                          node_hidden=node_hidden,
                          rel_hidden=rel_hidden,
                          lr=lr)

evaluater = evaluate(dev_pair)

model.summary()




rest_set_1 = [e1 for e1, e2 in dev_pair]
rest_set_2 = [e2 for e1, e2 in dev_pair]
np.random.shuffle(rest_set_1)
np.random.shuffle(rest_set_2)

epoch = 20
for turn in range(5):
    for i in trange(epoch):
        np.random.shuffle(train_pair)
        for pairs in [train_pair[i*batch_size:(i+1)*batch_size] for i in range(len(train_pair)//batch_size + 1)]:
            if len(pairs) == 0:
                continue
            inputs = [adj_matrix,r_index,r_val,rel_matrix,ent_matrix,pairs]
            inputs = [np.expand_dims(item,axis=0) for item in inputs]
            model.train_on_batch(inputs,np.zeros((1,1)))
        if i==epoch-1:
            Lvec,Rvec = get_embedding(dev_pair[:,0],dev_pair[:,1])
            evaluater.test(Lvec,Rvec)
        new_pair = []   
    Lvec,Rvec = get_embedding(rest_set_1,rest_set_2)
    A,B = evaluater.CSLS_cal(Lvec,Rvec,False)
    for i,j in enumerate(A):
        if  B[j] == i:
            new_pair.append([rest_set_1[j],rest_set_2[i]])
    
    train_pair = np.concatenate([train_pair,np.array(new_pair)],axis = 0)
    for e1,e2 in new_pair:
        if e1 in rest_set_1:
            rest_set_1.remove(e1) 
        
    for e1,e2 in new_pair:
        if e2 in rest_set_2:
            rest_set_2.remove(e2)
    epoch = 5
