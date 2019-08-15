#!/usr/bin/env python

import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax


'''def softmax_jisuan(u,v):
    return np.exp(u.T*v)

def softmax(outsideVeci,centerVec,U):#U一共为10个词，outsideVecter为2个？还是5个？
    result=0.0
    fenmu=0.0
    for i in range(len(U)):
        fenmu+=softmax_jisuan(U[i],centerVec)
    return float(softmax_jisuan(outsideVeci,centerVec)/fenmu)'''

def sigmoid(x):#对于矩阵而言
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.标量或者矩阵
    s=float(1/(1+exp(-x)))对于标量而言

    Return:
    s -- sigmoid(x)
    """
    s=1/(1+np.exp(-x))
    return s

    ### YOUR CODE HERE
    for i in range(len(x)):
        s[i]=1/(1+np.exp(-x[i]))

    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(#针对单个的loss计算即J(o,V_c,U)
    centerWordVec,#Vc
    outsideWordIdx,#Uo的o
    outsideVectors,#Uos 所有外部向量
    dataset#U
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word对应的外部向量的索引号
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab对应的所有的U
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.#无用

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)#                                                                                          剩一个梯度1
    """

    ### YOUR CODE HERE
    '''gradOutsideVecs=np.empty([outsideVectors.shape[0],outsideVectors.shape[1]])
    loss=-np.log(softmax(outsideVectors[outsideWordIdx],centerWordVec,outsideVectors))#对应的概率计算函数
    gradCenterVec=outsideVectors[outsideWordIdx]
    for i in range(len(outsideVectors)):
        gradCenterVec-=softmax(outsideVectors[i],centerWordVec,outsideVectors)*outsideVectors[i]
    gradCenterVec=-gradCenterVec
    for i in range(len(outsideVectors)):
        if(i!=outsideWordIdx):
            gradOutsideVecs[i]=softmax(outsideVectors[i],centerWordVec,outsideVectors)*centerWordVec
        else:
            gradOutsideVecs[i]=softmax(outsideVectors[i],centerWordVec,outsideVectors)*centerWordVec-centerWordVec

    '''
    y_hat=softmax(np.dot(centerWordVec,outsideVectors.T))#V_c*U.T
    delta=y_hat.copy()#y_hat[i]应该是对应的第i个为U_o的softmax值，因为后面算梯度的时候也要用到其他的softax值，因此此处将所有的softmax值都计算一遍
    delta[outsideWordIdx]-=1#？

    loss=-np.log(y_hat)[outsideWordIdx]#对应的U_o的loss值
    gradCenterVec=np.dot(delta,outsideVectors)#NB! 由于delta对应的是每一个U_w最为U_o时的softmax的值，因此将每一个softmax的值乘以U_w本身就构成了对所有P(O=w|C)*U_w的求和，然后在outsidewordIdx的部分delta减掉了1，刚好对应着-U_o
    gradOutsideVecs=np.dot(delta[:,np.newaxis], centerWordVec[np.newaxis, :])#np.newaxis是指增加一维 delta原来应该是【softmax【i】】进行了newaxis之后应该是【【softmax【i】】】，对centerWordVecc进行横向加维变成【【V_c[i]】】，列向量乘以一个行向量得到一个矩阵



    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 


    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """####################################################？ 2

    negSampleWordIndices = [None] * K
    for k in range(K):#K为10
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(#################################################################################？ 3
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec model

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices#indices for what?#第一个是对应的U_o

    ### YOUR CODE HERE
    gradOutsideVecs=np.zeros(outsideVectors.shape)
    gradCenterVec=np.zeros(centerWordVec.shape)
    loss=0.0#初始化
    z=sigmoid(np.dot(outsideVectors[outsideWordIdx],centerWordVec))#1/(1+e^-(U_o*V_c))
    loss-=np.log(z)
    gradOutsideVecs[outsideWordIdx]+=centerWordVec*(z-1.0)#求导之后是这样的
    gradCenterVec+=outsideVectors[outsideWordIdx]*(z-1.0)

    for k in range(K):
        samp=indices[k+1]
        z=sigmoid(np.dot(-outsideVectors[samp],centerWordVec))#这里前面有一个-号,不要忘记
        loss-=np.log(z)
        gradOutsideVecs[samp]-=centerWordVec*(z-1.0)#这里符号相反要注意
        gradCenterVec-=outsideVectors[samp]*(z-1.0)
    ### Please use your implementation of sigmoid in here.


    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,#计算loss
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words#U_o里的上下文单词
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outsideVectors -- outside word vectors (as rows) for all words in vocab#外部的所有单词
                    (U in pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    gradOutsideVectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE
    currentCenterWordIdx=word2Ind[currentCenterWord]
    centerWordVec=centerWordVectors[currentCenterWordIdx]

    for outsideWord in outsideWords:
        outsideWordIdx=word2Ind[outsideWord]
        (l,gradCenter,gradOutside)=word2vecLossAndGradient(
            centerWordVec,outsideWordIdx,outsideVectors,dataset)
        loss+=l
        gradCenterVecs[currentCenterWordIdx]+=gradCenter
        gradOutsideVectors+=gradOutside

    ### END YOUR CODE

    return loss, gradCenterVecs, gradOutsideVectors

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

'''word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
        negSamplingLossAndGradient)'''#C是滑动窗口的大小


def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, 
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]#中间的单词前半部分
    outsideVectors = wordVectors[int(N/2):,:]#后半部分
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)#？

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient#？
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    print("\n=== Results ===")
    print ("Skip-Gram with naiveSoftmaxLossAndGradient")

    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset) 
        )
    )

    print ("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print ("Skip-Gram with negSamplingLossAndGradient")   
    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:],
            dummy_vectors[5:,:], dataset, negSamplingLossAndGradient)
        )
    )
    print ("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)

if __name__ == "__main__":
    test_word2vec()
