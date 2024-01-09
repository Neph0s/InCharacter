from .BaseDB import BaseDB
import random
import string
import os

from math import sqrt

class NaiveDB(BaseDB):
    def __init__(self):
        self.verbose = False
        self.init_db()

    def init_db(self):
        if(self.verbose):
            print("call init_db")
        self.vectors = []
        self.documents = []
        self.norms = []

    def save(self, file_path):
        print( "warning! directly save folder from dbtype NaiveDB has not been implemented yet, try use role_from_hf to load role instead" )

    def load(self, file_path):
        print( "warning! directly load folder from dbtype NaiveDB has not been implemented yet, try use role_from_hf to load role instead" )

    def recompute_norm( self ):
        # 补全这部分代码，self.norms 分别存储每个vector的l2 norm
        # 计算每个向量的L2范数
        self.norms = [sqrt(sum([x**2 for x in vec])) for vec in self.vectors]


    def search(self, query_vector , n_results):

        if(self.verbose):
            print("call search")

        if len(self.norms) != len(self.vectors):
            self.recompute_norm()

        # self.vectors 是list of list of float
        # self.norms 存储了每个vector的l2 norm
        # query_vector是lisft of float
        # 依次计算query_vector和vectors中每个vector的cosine similarity（注意vector的norm已经在self.norm中计算)
        # 并且给出最相近的至多n_results个结果
        # 把对应序号的documents 用list of string的形式return
        # 补全这部分代码
            
        # 计算查询向量的范数
        query_norm = sqrt(sum([x**2 for x in query_vector]))

        # 计算余弦相似度
        similarities = []
        for vec, norm in zip(self.vectors, self.norms):
            dot_product = sum(q * v for q, v in zip(query_vector, vec))
            if query_norm < 1e-20:
                continue
            cosine_similarity = dot_product / (query_norm * norm)
            similarities.append(cosine_similarity)

        # 获取最相似的n_results个结果
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:n_results]
        top_documents = [self.documents[i] for i in top_indices]
        return top_documents

    def init_from_docs(self, vectors, documents):
        if(self.verbose):
            print("call init_from_docs")
        self.vectors = vectors
        self.documents = documents
        self.norms = []