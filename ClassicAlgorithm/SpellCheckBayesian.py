# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: SpellCheckBayesian.py 
@time: 2017-12-24 15:50

使用贝叶斯方法实现拼写检查
"""
import re
import collections


# 求解：argmaxc P(c|w) -> argmaxc P(w|c)P(c)/P(w)
# P(c)，文章中出现一个正确拼写词c的概率，也就是说，c出现的概率有多大
# P(w|c)，在用户想键入c的情况下敲成w的概率，这个代表用户会以多大的概率把c敲错成w
# argmaxc，用来枚举所有可能的c并且选择概率最大的

# 将语料库中的单词全部抽取出来，转换成小写，并且去除单词中间的特殊符号
def words(text):
    return re.findall("[a-z]+", text.lower())


# 统计语料库中每个单词出现的频数
def train(features):
    # 使用lambda: 1是为了当出现一个语料库中没有的单词时，model这个字典会输出1，而不会输出0
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


NWORDS = train(words(open("words.txt").read()))

# 编辑距离:定义为使用了几次插入(在词中插入一个单字母), 删除(删除一个单字母), 交换(交换相邻两个字母),
# 替换(把一个字母换成另一个)的操作从一个词变到另一个词.

alphabet = 'abcdefghijklmnopqrstuvwxyz'


# 对于一个长度为 n 的单词, 可能有n种删除, n-1中对换, 26n 种
# (译注: 实际上是 25n 种)替换 和 26(n+1) 种插入 (译注: 实际上比这个小, 因为在一个字母前后再插入这个字母构成的词是等价的).
# 这样的话, 一共就是 54n + 25 中情况 (当中还有一点重复).
# 比如说, 和 something 这个单词的编辑距离为1 的词按照这个算来是 511 个, 而实际上是 494 个.
# 返回所有与单词 w 编辑距离为 1 的集合
def edits1(word):
    n = len(word)
    return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion
               [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition
               [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration
               [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion


# 返回所有与单词 w 编辑距离为 2 的集合
# 与 something 编辑距离为2的单词居然达到了 114,324 个
# 优化：在这些编辑距离小于2的词中间, 只把那些正确的词作为候选词, known_edits2('something') 只能返回 3 个单词: 'smoothing', 'something' 和 'soothing'
def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)


# known()函数接受一个单词的集合，将这个单词集合中错误的单词去掉
def known(words):
    return set(w for w in words if w in NWORDS)


# 如果known(set)非空，candidates就会选取这个集合，而不会计算后面的
# 这里选择：编辑距离为0的正确单词比编辑距离为1的优先级高，而编辑距离为1的正确单词比编辑距离为2的优先级高
def correct(word):
    # 如果or关键字中前面有非空值，就不会再计算后面的
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    # NWORDS[w]越大，说明该单词w出现的频数越多，则P(w)越大
    # 对于传进来的单词word，如果该单词本身就是正确的，则candidates=[word]，这时直接输出这个单词即可
    # 如果传进来的单词是经过一次编辑距离得到的错误单词，如tha，则经过known(edits1(word))处理后，
    # 就会出现the、than等正确的单词，并且由于这些正确的单词都是tha经过一次编辑距离得到的，
    # 即这些单词的P(w|c)都是相同的，则只要选出这些单词中NWORDS[w]最大的单词，就是我们所需要的P(w|c)P(c)最大的单词了
    return max(candidates, key=lambda w: NWORDS[w])


while True:
    print("请输入单词：")
    word = input()
    if word == "q":
        break
    else:
        print("可能的正确单词是：", correct(word))
