#!/bin/env python
#coding=utf-8

################################################################
# File: Recommendations.py
# Author: wuhonghuan
# Name: Neal Gavin
# Mail: wuhonghuan@baidu.com
# Created Time: 2015/12/08 09:16:52
# Saying: Everything goes well.
################################################################
import math
critics={'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
     'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 
      'The Night Listener': 3.0},
      'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 
           'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, 
            'You, Me and Dupree': 3.5}, 
      'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
           'Superman Returns': 3.5, 'The Night Listener': 4.0},
      'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
           'The Night Listener': 4.5, 'Superman Returns': 4.0, 
            'You, Me and Dupree': 2.5},
      'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 
           'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
            'You, Me and Dupree': 2.0}, 
      'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
           'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
      'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}

def sim_distance(prefs, persion1, persion2):
    """返回persion1和persion2的欧几里德相似度"""
    si = {}
    for item in prefs[persion1]:
        if item in prefs[persion2]:
            si[item] = 1
    
    #没有共同返回0
    if len(si) == 0:
        return 0
    #计算所有差值的平方和
    sum_of_squares = sum([pow(prefs[persion1][item] - prefs[persion2][item], 2) for item in
        prefs[persion1] if item in prefs[persion2]])
    return 1/(1 + math.sqrt(sum_of_squares))    

def sim_pearson(prefs, persion1, persion2):
    """返回persion1和persion2的皮尔逊相关系数"""
    si = {}
    #找到两个人都评价过的电影
    for item in prefs[persion1]:
        if item in prefs[persion2]:
            si[item] = 1
    n = len(si)
    if n == 0:
        return 1

    #对所有的偏好求和
    sum1 = sum([prefs[persion1][it] for it in si])
    sum2 = sum([prefs[persion2][it] for it in si])

    #求平方和
    sum1Sqre = sum([pow(prefs[persion1][it], 2) for it in si])
    sum2Sqre = sum([pow(prefs[persion2][it], 2) for it in si])

    #求乘积之和
    persionSum = sum([prefs[persion1][it] * prefs[persion2][it] for it in si])

    #计算皮尔逊相关系数
    num = persionSum - (sum1 * sum2 / n)
    den = math.sqrt((sum1Sqre - pow(sum1, 2) / n) * (sum2Sqre - pow(sum2, 2) / n))
    if den == 0:
        return 0
    r = num / den
    return r

def sim_tanimoto(prefs, persion1, persion2):
    """Tanimoto相似度评价"""
    #求persion1与persion2喜欢的交集
    si = {}
    for item in prefs[persion1]:
        if item in prefs[persion2]:
            si[item] = 1
    len_intersect = len(si)
    len_p1 = len(prefs[persion1])
    len_p2 = len(prefs[persion2])
    similar_rate = float(len_intersect) / float(len_p1 + len_p2 - len_intersect)        
    return similar_rate

def topMatches(prefs, persion, n = 5, similarity = sim_pearson):
    """得到Top n 推荐相似度评分"""
    scores = [(similarity(prefs, persion, other), other) for other in prefs if other != persion ]
    scores.sort()
    scores.reverse()
    realn = min(n, len(scores))
#   if realn != n:
#       log.LOG_INFO("推荐个数据不足" + str(n))
    return scores[0:realn]

def getRecommendationsRank(prefs, persion, n = 1000000000, similarity = sim_pearson):
    """相似度加权打分预测persion对目标结果的评分,按评分高低推荐n个"""
    totals = {}
    simSums = {}
    for other in prefs:
        if other == persion:
            continue
        sim = similarity(prefs, persion, other)

        #忽略评价<= 0
        if sim <= 0:
            continue
        #只对自己未看过的电影进行评价预测
        for item in prefs[other]:
            if not item in prefs[persion] or prefs[persion][item] == 0:
                #相似度*评价值 的和
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                simSums.setdefault(item, 0)
                #相似度和
                simSums[item] += sim
    #建立一个归一化列表
    rankings = [ (total / simSums[item], item) for item, total in totals.items()]

    #返回排序的结果
    rankings.sort()
    rankings.reverse()
    realn = min(n, len(rankings))
    return rankings[0:realn]

def transformPrefs(prefs):
    """改变方式"""
    result = {}
    for persion in prefs:
        for item in prefs[persion]:
            result.setdefault(item, {})
            ##人与物品对换
            result[item][persion] = prefs[persion][item]
    return result        


if __name__ == '__main__':
    print sim_distance(critics, 'Toby', 'Jack Matthews')
    print sim_pearson(critics, 'Lisa Rose', 'Gene Seymour')
    print topMatches(critics, 'Toby', n = 3)
    print getRecommendationsRank(critics, 'Toby', n = 3)
    print getRecommendationsRank(critics, 'Toby', n = 3, similarity = sim_distance)
    print getRecommendationsRank(critics, 'Toby', n = 3, similarity = sim_tanimoto), 'tanimoto'
    movies = transformPrefs(critics)
    print topMatches(movies, 'Superman Returns', n = 3)
    print getRecommendationsRank(critics, "Jack Matthews", n = 2)
    print 'test for recommend'
    
