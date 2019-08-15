#!/usr/bin/env python

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 5000

import pickle
import glob
import random
import numpy as np
import os.path as op

def load_saved_params():
    """
    A helper function that loads previously saved parameters and resets
    iteration start.#加载之前存在的参数以及迭代的重新开始
    """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):#判断不是第一次程序迭代
            st = iter

    if st > 0:#确认不是第一次迭代
        params_file = "saved_params_%d.npy" % st
        state_file = "saved_state_%d.pickle" % st
        params = np.load(params_file)
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        return st, params, state#st=之前的迭代次数,params=对应之前的训练参数,state=对应的之前的训练阶段
    else:
        return st, None, None


def save_params(iter, params):#保存参数函数
    params_file = "saved_params_%d.npy" % iter
    np.save(params_file, params)
    with open("saved_state_%d.pickle" % iter, "wb") as f:
        pickle.dump(random.getstate(), f)


'''wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
        negSamplingLossAndGradient),
    wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10)'''#sgd调用实例 word2vec_sgd_wrapper(skipgram,tokens,vec,dataset,C,negSamplingLossAndGradient),返回的loss和grad
#lambda是匿名函数 例: g = lambda x:x+1 g(1)->2

def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,#lambda vec 对应的是f，wordVectors对应的是x0，step是0.3，iterations是40000
        PRINT_EVERY=10):
    """ Stochastic Gradient Descent随机梯度下降

    Implement the stochastic gradient descent method in this function.

    Arguments:
    f -- the function to optimize优化, it should take a single 一个参数 两个输出
         argument and yield two outputs, a loss and the gradient损失函数以及梯度
         with respect to the arguments
    x0 -- the initial point to start SGD from初始化的词向量
    step -- the step size for SGD
    iterations -- total iterations to run SGD for
    postprocessing后处理 -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit单独 length.泛化
    PRINT_EVERY -- specifies how many iterations to output loss输出loss的代数

    Return:
    x -- the parameter value after SGD finishes优化后的x？
    """

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0#x初始化为x0

    if not postprocessing:#对应后处理函数
        postprocessing = lambda x: x

    exploss = None

    for iter in range(start_iter + 1, iterations + 1):#在加载的开始迭代数到最终迭代数之间迭代
        # You might want to print the progress every few iterations.

        loss = None
        ### YOUR CODE HERE 一个计算loss的函数
        loss,gradients=f(x)
        '''if(iter%4000==0):
            print("loss:,",loss)'''
        x-=step*gradients
        ### END YOUR CODE

        x = postprocessing(x)
        if iter % PRINT_EVERY == 0:
            if not exploss:
                exploss = loss
            else:
                exploss = .95 * exploss + .05 * loss
            print("iter %d: %f" % (iter, exploss))

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x


def sanity_check():
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 1 result:", t1)
    assert abs(t1) <= 1e-6

    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print("test 2 result:", t2)
    assert abs(t2) <= 1e-6

    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 3 result:", t3)
    assert abs(t3) <= 1e-6

    print("-" * 40)
    print("ALL TESTS PASSED")
    print("-" * 40)


if __name__ == "__main__":
    sanity_check()
