#!/usr/bin/env python
# coding=utf-8
#################################################################
# File: TitanicPredict.py
# Author: Neal Gavin
# Email: nealgavin@126.com
# Created Time: 2016/03/07 16:57:48
# Saying: Fight for freedom ^-^ !
#################################################################
import pandas as pd
import numpy as np
import logging as log
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time
import csv
import re
log.basicConfig(level=log.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
class TitanicPredict(object):
    """TitanicPredict"""
    def readTrain(self):
        """"""
        train = pd.read_csv('./train.csv', header = 0) #chunksize = 100)
        self.train = train
        return train

    def readTest(self):
        """test"""
        test = pd.read_csv('./test.csv', header = 0)
        self.test = test
        return test

    def dataAnalyse(self, train, test):
        """数据分析"""
        df = pd.concat([train, test])
        df.info()
        print df.describe()
        #猜测女性生存率高
        #print df[(df.Sex == 'male')]['Sex']
        people = [df[(df.Sex == 'male')]['Sex'].size, df[(df.Sex == 'female')]['Sex'].size]
        peopleSurvive = [df[(df.Sex == 'male') & (df.Survived == 1)]['Sex'].size, df[(df.Sex == 'female') & (df.Survived == 1)]['Sex'].size]
        print 'male:', people[0], 'female', people[1], 'survive_male', peopleSurvive[0], 'survive_female', peopleSurvive[1]
        #年龄大中小分布#
        ages_bin = pd.cut(df['Age'], [0, 18, 40, 100])
        #根据样本分位数比较均匀划分
        #ages_bin = pd.qcut(df['Age'], 3)
        ages_dummy = pd.get_dummies(ages_bin)
        ages_dummy.columns.values[0] = 'child'
        ages_dummy.columns.values[1] = 'man'
        ages_dummy.columns.values[2] = 'old'
        survived_ages = pd.concat([ages_dummy, df['Survived']], axis = 1)
        print survived_ages[(survived_ages.child == 1)]['child'].size, survived_ages[(survived_ages.man == 1)]['man'].size, survived_ages[(survived_ages.old == 1)]['old'].size
        print survived_ages[(survived_ages.child == 1) & (survived_ages['Survived'] == 1)]['child'].size, survived_ages[(survived_ages.man == 1) & (survived_ages['Survived'] == 1)]['man'].size, survived_ages[(survived_ages.old == 1) & (survived_ages['Survived'] == 1)]['old'].size
        print 'suvived_child' 
        print survived_ages.columns.values
        print ages_dummy.columns.values

    
    def processCabin(self, df, keep_binary = True, keep_scaled = True):
        """船仓定性转化"""
        df['Cabin'][df.Cabin.isnull()] = 'U0'
        df['CabinLetter'] = df['Cabin'].map( lambda x: self.getLetter(x) )
        #factorize:以原始字符为索引映射成数值数据，[0]为数值，[1]为原始索引
        df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0] + 1
        if keep_binary:
            #分拆不同的属性，变成01值的属性，如ABC，分拆后，A(1,0,0) B(0,1,0) C(0,0,1)
            bletters = pd.get_dummies(df['CabinLetter']).rename(columns=lambda x: 'CabinLetter_' + str(x))
            #把分拆的属性特征合并到原始特征数据集中
            df = pd.concat([df, bletters], axis=1)
            #print bletters
        df['CabinNumber'] = df['Cabin'].map( lambda x: self.getNumber(x) )
        if keep_scaled:
            scaler = preprocessing.StandardScaler()
            df['CabinNumber_scaled'] = scaler.fit_transform(df['CabinNumber'])
        return df

    def getLetter(self, cabin):
        """letter"""
        match = re.compile("([a-zA-Z]+)").search(cabin)
        if match:
            return match.group()
        else:
            return 'U'
    
    def getNumber(self, cabin):
        """number"""
        match = re.compile("([0-9]+)").search(cabin)
        if match:
            return match.group()
        else:
            return '0'
    
    def processName(self, df, keep_binary = True, keep_scaled = True, keep_bins = True):
        """处理人名，提取出名望,身份"""
        #名字的长短可能和身份地位相关,增加特征
        names = df['Name'].map(lambda x:len(re.split(' ', x)))
        #每个人的头衔
        #df['Title']
        df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
        #把低频的头衔合并
        df['Title'][df.Title == 'Jonkheer'] = 'Master'
        df['Title'][df.Title.isin(['Ms', 'Mlle'])] = 'Miss'
        df['Title'][df.Title == 'Mme'] = 'Mrs'
        df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col'])] = 'Sir'
        df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
        #...
        #分拆属性
        if keep_binary:
            df = pd.concat([df, pd.get_dummies(df['Title']).rename(columns = lambda x: 'Title_' + str(x))], axis = 1)
        #名字长短数值归一
        if keep_scaled:
            scaler = preprocessing.StandardScaler()
            df['Names_scaled'] = scaler.fit_transform(names)
            #print df['Names_scaled']
        #人名映射成数值
        if keep_bins:
            df['Title_id'] = pd.factorize(df['Title'])[0] + 1
        return df   

    def reduceAndCluster(self, train, test, clusters = 3):
        """数据降维"""
        df = pd.concat([train, test])
        df.reset_index(inplace = True)
        df.drop('index', axis = 1, inplace = True)
        df = df.reindex_axis(train.columns, axis = 1)
        passenger_id =  pd.Series(df['PassengerId'], name = 'PassengerId')
        survived_series = pd.Series(df['Survived'], name = 'Survived')
        df.drop('PassengerId', axis = 1, inplace = True)
        df.drop('Survived', axis = 1, inplace = True)
        print df.columns.values, '所有列'
        print df.columns.size
        X = df.values[:, :]

        #PCA降维
        pca = PCA(n_components = .99)
        #默认在原始数据上处理，也就是执行完后的X的值会改变
        X_transform = pca.fit_transform(X)
#       X_transform = pca.inverse_transform(X_transform)
        pca_dataframe = pd.DataFrame(X_transform)
        print '降维后的维数', pca_dataframe.shape[1]
        #聚类
        kmeans = KMeans(n_clusters = 3, random_state = np.random.RandomState(4), init='random')
        train_cluster_ids = kmeans.fit_predict(X_transform[: train.shape[0]])
        test_cluster_ids = kmeans.predict(X_transform[train.shape[0]:])
        
        cluster_ids = np.concatenate([train_cluster_ids, test_cluster_ids])
        cluster_id_series = pd.Series(cluster_ids, name = 'cluster_ids')
        #处理完后的特征数据
        df = pd.concat([survived_series, cluster_id_series, pca_dataframe, passenger_id], axis = 1)
        train = df[: train.shape[0]]
        test = df[train.shape[0]: ]
        test.reset_index(inplace = True)
        test.drop('index', axis = 1, inplace = True)
        test.drop('Survived', axis = 1, inplace = True)
        
        return train, test

    def processDrop(self, df):
        """drop"""
        raw_drop_list = ['Name', 'Title', 'Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked', 'Cabin', 'CabinLetter', 'CabinNumber', 'Age', 'Fare', 'Ticket']
        string_drop = ['Name', 'Title', 'Cabin', 'Ticket', 'Sex', 'Ticket']
        print 'processDrop:', df.columns.values
        df.drop(raw_drop_list, axis = 1, inplace = True)
        #self.df.drop(string_drop, axis = 1, inplace = True)
        return df

    def processPclass(self, df):
        """pclass 船舱等级"""
        df.Pclass[df.Pclass.isnull()] = df.Pclass.dropna().mode().values
        pclass = pd.get_dummies(df['Pclass']).rename(columns = lambda x: 'Pclass_' + str(x))
        df = pd.concat([df, pclass], axis = 1)
        return df

    def crossFeature(self, df):
        """杂交生成更多特征"""
        columns_list = list(df.columns.values)
        columns_list.remove('Survived')
        new_col_list = list(['Survived'])
        new_col_list.extend(columns_list)
        df = df.reindex(columns = new_col_list)
        print 'start with', df.columns.size, 'feature', df.columns.values
#       numerics = df.loc[:,['Names_scaled', 'CabinNumber_scaled', 'Age_scaled']]
        numerics = df.loc[:,['Names_scaled', 'CabinNumber_scaled']]
        print 'col_num_bef', df.columns.size
        #增加特征
        new_fields_count = 0
        for i in xrange(0, numerics.columns.size):
            for j in xrange(0, numerics.columns.size):
                if i <= j:
                    name = str(numerics.columns.values[i]) + '*' +  str(numerics.columns.values[j])
                    df = pd.concat([df, pd.Series(numerics.iloc[:, i] * numerics.iloc[:, j], name = name)], axis = 1)
                    new_fields_count += 1
                if i < j:
                    name = str(numerics.columns.values[i]) + '+' +  str(numerics.columns.values[j])
                    df = pd.concat([df, pd.Series(numerics.iloc[:, i] + numerics.iloc[:, j], name = name)], axis = 1)
                    new_fields_count += 1
                if not i == j:
#                   name = str(numerics.columns.values[i]) + '/' +  str(numerics.columns.values[j])
#                   df = pd.concat([df, pd.Series(numerics.iloc[:, i] / numerics.iloc[:, j], name = name)], axis = 1)
                    name = str(numerics.columns.values[i]) + '-' +  str(numerics.columns.values[j])
                    df = pd.concat([df, pd.Series(numerics.iloc[:, i] - numerics.iloc[:, j], name = name)], axis = 1)
                    new_fields_count += 2
        return df

    def increaseIndependence(self, df):
        """增加各属性特征间的独立性，去重过于重复的属性"""
        #计算除了Survived,PassengerId外的其它列，两两之间的相关性，用斯皮尔曼相关系数
        df_corr = df.drop(['Survived', 'PassengerId'], axis = 1).corr(method = 'spearman')
        mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
        df_corr = mask * df_corr
        
        #去除和其它属性相关度过大的属性，保证属性的独立性比较强，有区分度
        drops = []
        for col in df_corr.columns.values:
            #第一个在不在第二个里
            if np.in1d([col], drops):
                continue
            #pandas where用法，找出逆相似度大于0.98的
            corr = df_corr[abs(df_corr[col]) > 0.98].index
            #第二个合到第一个里
            drops = np.union1d(drops, corr)
        #按列的drop
        df.drop(drops, axis = 1, inplace = True)
        return df
    
    def processSex(self, df):
        """处理性别"""
        df['Sex_bin'] = np.where(df['Sex'] == 'male', 1, 0)
        return df

    def processAge(self, df, keep_bins = False, keep_scaled = False):
        """age"""
        #空值数据补全,随机森林预测缺失属性值
        #loc 取区域交集
        #age_df = df[['Age', 'Embarked', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Title_id']]
        age_df = df[['Age', 'Sex_bin', 'Title_id']]
        age_null = age_df.loc[df['Age'].isnull()]
        age_no_null = age_df.loc[df['Age'].notnull()]
        print age_null.shape, 'age_null'
        X = age_no_null.values[:, 1:]
        Y = age_no_null.values[:, 0]
        model = RandomForestRegressor(n_estimators = 200, n_jobs = -1)
        model.fit(X, Y)
        predict_ages = model.predict(age_null.values[:, 1:])
        df.loc[(df.Age.isnull()), 'Age'] = predict_ages
        df['Age_num'] = df['Age']
        if keep_bins:
            age_bin = pd.qcut(df['Age'], 7)
            df['Age_bin'] = pd.factorize(age_bin)[0] + 1
        if keep_scaled:
            scaler = preprocessing.StandardScaler()
            df['Age_scaled'] = scaler.fit_transform(df['Age'])
        return df 

    def processSibSp(self, df):
        """在船上有兄弟姐妹，可能会一起逃生"""
        df['SibSp_num'] = df['SibSp']
        return df

    def processParch(self, df):
        """在船上有父母的可能需要按年龄分析一下，不过也该会一起逃生"""
        df['Parch_num'] = df['Parch']
        return df

    def processFare(self, df):
        """价格处理"""
        #print df['Fare']
        df.Fare[df.Fare.isnull()] = df.Fare.dropna().mode().values
        df['Fare_num'] = df['Fare']
        #区间化
        fare_bin = pd.qcut(df['Fare'], 9)
        fare_bin = pd.factorize(fare_bin)[0]
        df['Fare_bin'] = fare_bin
        return df

    def processEmbarked(self, df):
        """上船地点"""
        #用众数实全na值
        df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
        #数值化
        Embarked_num = pd.factorize(df.Embarked)[0]
        df['Embarked_num'] = Embarked_num
        print Embarked_num
        return df

    def dataPreProcess(self, train, test,  pca = False):
        """数据预处理"""
        #合并训练数据和测试数据
        df = pd.concat([train, test])
        df.reset_index(inplace = True)
        df.drop('index', axis=1, inplace=True)
        df.reindex_axis(self.train.columns, axis = 1)
        #特征处理
        df = self.processCabin(df)
        df = self.processName(df)
        df = self.processSex(df)
#       df = self.processAge(df)
        df = self.processPclass(df)
        df = self.processParch(df)
        df = self.processSibSp(df)
        df = self.processFare(df)
        df = self.processEmbarked(df)
        df = self.processDrop(df)
        df = self.crossFeature(df)

        #获取处理完的训练数据和测试数据
        train = df[:train.shape[0]]
        test = df[train.shape[0]:]
        #pca数据降维
        if pca:
            print '降维pca'
            train, test = self.reduceAndCluster(train, test)
        else:
            test.reset_index(inplace = True)
            test.drop('index', axis = 1, inplace = True)
            test.drop('Survived', axis = 1, inplace = True)
            print test.columns.values, 'nopca'

        return (train, test)

    def model(self, train, test):
        """模型训练"""
        print train.columns.values
        print test.columns.values
        test_ids = test['PassengerId']
        train.drop('PassengerId', axis = 1, inplace = True)
        test.drop('PassengerId', axis = 1, inplace = True)

        X = train.values[:, 1:]
        Y = train.values[:, 0]
#        X, X_valid, Y, Y_valid = train_test_split(X, Y, test_size = 50, random_state = 42)
#       test_ids = [x for x in xrange(892, 1310)]
        survived_weight = 0.75
        Y_weight = np.array([survived_weight if s == 0 else 1 for s in Y])
        feature_list = train.columns.values[1:]

        #随机森林
        forest = RandomForestClassifier(oob_score = True, n_estimators = 1000)
#       forest.fit(X, Y, sample_weight = Y_weight)
        forest.fit(X, Y, sample_weight = Y_weight)
#        print "准确率", forest.score(X_valid, Y_valid)

        feature_importance = forest.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        #特征重要度指标
        fi_threshold = 18
        print feature_importance, fi_threshold, '重要度'
        important_idx = np.where(feature_importance > fi_threshold)[0]
        important_features = feature_list[important_idx]
        print 'important_features_num', important_features.shape
        print important_features
        #默认从小到大排序，所以[::-1]倒一下
        sort_idx = np.argsort(feature_importance[important_idx])[::-1]
        self.draw(important_features, feature_importance, important_idx, sort_idx)
        
        #获取排完序的阈值大于18的重要的特征
        X = X[:, important_idx][:, sort_idx]
        #截取测试数据
        test = test.iloc[:, important_idx].iloc[:, sort_idx]

        sqrtfeat = int(np.sqrt(X.shape[1]))
        minsamsplit = int(X.shape[0] * 0.015)
        params_score = {
            "n_estimators":1000,
            "max_features":sqrtfeat,
            "min_samples_split":minsamsplit
        }
        params = params_score
        forest = RandomForestClassifier(n_jobs = -1, oob_score = True, **params)

        test_scores = []
        for i in range(5):
            forest.fit(X, Y, sample_weight = Y_weight)
            #print "准确率", forest.score(X_valid, Y_valid)
            print "OOB:", forest.oob_score_
            test_scores.append(forest.oob_score_)
        test_ans = np.asarray(zip(test_ids, forest.predict(test))).astype(int)
        output = test_ans[test_ans[:, 0].argsort()]
        return output

    def writeCSV(self, output):
        """写入数据"""
        name = 'rfc' + '.csv'
        predict_file = open("./" + name, 'wb')
        open_file_obj = csv.writer(predict_file)
        open_file_obj.writerow(["PassengerId", "Survived"])
        open_file_obj.writerows(output)

    def draw(self, important_features, feature_importance, important_idx, sort_idx):
        """画图"""
        pos = np.arange(sort_idx.shape[0]) + 0.5
        plt.subplot(1, 2, 2)
        plt.title('Feature Importance')
        plt.barh(pos, feature_importance[important_idx][sort_idx[::-1]], color = 'r', align = 'center')
        plt.yticks(pos, important_features[sort_idx[::-1]])
        plt.xlabel('Relative Importance')
        plt.draw()
        plt.show()


    def process(self):
        """process"""
        #特征工程
        train = self.readTrain()
        test = self.readTest()
#       self.dataAnalyse(train, test)
        train, test = self.dataPreProcess(train, test)
#       ##模型训练
        output = self.model(train, test)
        self.writeCSV(output)

if __name__ == '__main__':
    tt = TitanicPredict()
    tt.process()
