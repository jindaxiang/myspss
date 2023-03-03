import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from stargazer.stargazer import Stargazer


class MSStats:
    def __int__(self):
        self.testdata1 = pd.read_excel("data.xls")  # data for testing
        self.testdata2 = pd.read_csv("smarket.csv")  # data for testing

    @classmethod
    def linear_regression(cls, y: list, x: pd.DataFrame, hasconstant="1"):
        if hasconstant:
            x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        stargazer = Stargazer([model])
        output = stargazer.render_html()
        return output

    @classmethod
    def logistic_regression(cls, y: list, x: pd.DataFrame, hasconstant="1"):
        if hasconstant == "true":
            x = sm.add_constant(x)
        model = sm.Logit(y, x).fit()
        stargazer = Stargazer([model])
        output = stargazer.render_html()
        return output

    def frequency(self, df: pd.DataFrame):
        table1 = {}
        table2 = {}
        for i in df:
            if i in table1:
                table1[i] += 1
            else:
                table1[i] = 1
        for j in table1:
            table2[j] = table1[j] / sum(table1.values())
        return table1, table2

    def ttest(self, sample1, sample2, sample_type):
        if sample_type == "singlesample":
            value = stats.ttest_1samp(sample1, sample2, axis=0)
        if sample_type == "independentsample":
            if stats.levene(sample1, sample2).pvalue > 0.05:
                judgement = True
            else:
                judgement = False
            value = stats.ttest_ind(sample1, sample2, equal_var=judgement)
        if sample_type == "pairedsample":
            value = stats.ttest_rel(sample1, sample2)
        return value

    def chi2test(self, df):
        value = chi2_contingency(df)
        print("卡方值=%.4f, p值=%.4f, 自由度=%i expected_frep=%s" % value)
        return value

    def norm_test(self, df, sample_size="small"):
        if sample_size == "small":
            s, p = scipy.stats.shapiro(df)
        else:
            s, p = scipy.stats.kstest(
                df, cdf="norm", args=(), N=20, alternative="two-sided", mode="approx"
            )
        return s, p

    def corr(self, df, plot="false"):
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
        plt.rcParams["axes.unicode_minus"] = False
        sns.set(font="SimHei")
        corr = df.corr()  # 计算各变量的相关性系数
        if plot == "true":
            xticks = list(corr.index)  # x轴标签
            yticks = list(corr.index)  # y轴标签
            fig = plt.figure(figsize=(15, 10))
            ax1 = fig.add_subplot(1, 1, 1)
            sns.heatmap(
                corr,
                annot=True,
                cmap="rainbow",
                ax=ax1,
                linewidths=0.5,
                annot_kws={"size": 9, "weight": "bold", "color": "blue"},
            )
            ax1.set_xticklabels(xticks, rotation=35, fontsize=15)
            ax1.set_yticklabels(yticks, rotation=0, fontsize=15)
            plt.show()
        return corr

    def kmeans(self, df, n_clusters="none"):
        SSE = []
        if n_clusters == "none":
            for k in range(2, 10):
                estimator = KMeans(n_clusters=k)
                estimator.fit(df)
                SSE.append(estimator.inertia_)
                xx = range(2, 10)
            plt.xlabel("k")
            plt.ylabel("SSE")
            plt.plot(xx, SSE, "o-")
            plt.show()

        else:
            estimator = KMeans(n_clusters)
            estimator.fit(df)
            x_label = estimator.labels_
            x_matrix = []
            x_frequency = {}
            for i in range(n_clusters):
                x_new = df[x_label == i]
                x_matrix.append(x_new)
                clusters_type = "cluster_" + str(i + 1)
                x_frequency[clusters_type] = len(x_new)
            return x_matrix, x_frequency

    def ridge(self, y, x, alpha=0):
        if alpha == 0:
            model = RidgeCV(alphas=10 ** np.linspace(-2, 1, 1000))
            model.fit(x, y)
            print("交叉验证最佳alpha值", model.alpha_)
        else:
            model = Ridge(alpha=alpha)
            model.fit(x, y)
        ridge_coef = model.coef_
        ridge_model = model
        print("系数矩阵:\n", model.coef_)
        return ridge_coef, ridge_model

    def lasso(self, y, x, alpha=0):
        if alpha == 0:
            model = LassoCV(alphas=10 ** np.linspace(-2, 1, 1000))
            model.fit(x, y)
            print("交叉验证最佳alpha值", model.alpha_)
        else:
            model = Lasso(alpha=alpha)
            model.fit(x, y)
        lasso_coef = model.coef_
        lasso_model = model
        print("系数矩阵:\n", model.coef_)
        return lasso_coef, lasso_model


if __name__ == "__main__":
    testdata1 = pd.read_excel("data.xls")  # data for testing
    testdata2 = pd.read_csv("smarket.csv")

    ms_stats = MSStats()
    MSStats.linear_regression(
        y=testdata2["Volume"].to_list(),
        x=testdata2[["Lag1", "Lag2", "Lag3"]],
        hasconstant="true",
    )
    MSStats.logistic_regression(
        y=testdata2["Direction"].to_list(),
        x=testdata2[["Lag1", "Lag2", "Lag3"]],
        hasconstant="true",
    )

    temp_json = testdata2[["Lag1", "Lag2", "Lag3"]].to_json(orient="split")

    pd.DataFrame.from_records(
        json.loads(temp_json)["data"],
        columns=json.loads(temp_json)["columns"],
        index=json.loads(temp_json)["index"],
    )
