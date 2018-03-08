import pandas as pd
import numpy
import scipy.stats
import math
import matplotlib.pyplot
import pylab

excel_file = pd.ExcelFile('university_data.xlsx')
df = excel_file.parse('university_data')

cs_score = df['CS Score (USNews)'][0:49]
mu1 = numpy.round(numpy.mean(cs_score), 3)
var1 = numpy.round(numpy.var(cs_score), 3)
sigma1 = numpy.round(numpy.std(cs_score), 3)

research = df['Research Overhead %'][0:49]
mu2 = numpy.round(numpy.mean(research), 3)
var2 = numpy.round(numpy.var(research), 3)
sigma2 = numpy.round(numpy.std(research), 3)

admin_pay = df['Admin Base Pay$'][0:49]
mu3 = numpy.round(numpy.mean(admin_pay), 3)
var3 = numpy.round(numpy.var(admin_pay), 3)
sigma3 = numpy.round(numpy.std(admin_pay), 3)

tuition = df['Tuition(out-state)$'][0:49]
mu4 = numpy.round(numpy.mean(tuition), 3)
var4 = numpy.round(numpy.var(tuition), 3)
sigma4 = numpy.round(numpy.std(tuition), 3)

print("mu1 =", mu1)
print("mu2 =", mu2)
print("mu3 =", mu3)
print("mu4 =", mu4)
print("var1 =", var1)
print("var2 =", var2)
print("var3 =", var3)
print("var4 =", var4)
print("sigma1 =", sigma1)
print("sigma2 =", sigma2)
print("sigma3 =", sigma3)
print("sigma4 =", sigma4)
print()
covarianceMat = numpy.round(numpy.cov([cs_score,research, admin_pay, tuition]), 3)
correlationMat = numpy.round(numpy.corrcoef([cs_score,research, admin_pay, tuition]), 3)

print("covarianceMat =", covarianceMat)
print()
print("correlationMat =", correlationMat)
print()

prob_dist1 = []
for i in range(0, 49):
    a = 1 / (((2*math.pi) ** 0.5) * sigma1)
    b = -0.5 * (((cs_score[i] - mu1)/sigma1) ** 2)
    c = a * (math.e ** b)
    prob_dist1.append(c)

log_like1 = []
for i in range(0, 49):
    a = math.log(prob_dist1[i], math.e)
    log_like1.append(a)

log1 = 0
for i in range(0, 49):
    log1 = log1 + log_like1[i]

prob_dist2 = []
for i in range(0, 49):
    a = 1 /(((2 * math.pi) ** 0.5)*sigma2)
    b = -0.5 * (((research[i] - mu2) / sigma2) ** 2)
    c = a * (math.e ** b)
    prob_dist2.append(c)

log_like2 = []
for i in range(0, 49):
    a = math.log(prob_dist2[i],math.e)
    log_like2.append(a)

log2 = 0
for i in range(0, 49):
    log2 = log2 + log_like2[i]

prob_dist3 = []
for i in range(0, 49):
    a = 1 / (((2 * math.pi) ** 0.5)* sigma3)
    b = -0.5 * (((admin_pay[i] - mu3) / sigma3) ** 2)

    c = a * (math.e ** b)
    prob_dist3.append(c)

log_like3 = []
for i in range(0, 49):
    a = math.log(prob_dist3[i], math.e)
    log_like3.append(a)

log3 = 0
for i in range(0, 49):
    log3 = log3 + log_like3[i]

prob_dist4 = []
for i in range(0, 49):
    a = 1 / (((2 * math.pi) ** 0.5)* sigma4)
    b = -0.5 * (((tuition[i] - mu4) / sigma4) ** 2)
    c = a * (math.e ** b)
    prob_dist4.append(c)

log_like4 = []
for i in range(0, 49):
    a = math.log(prob_dist4[i], math.e)
    log_like4.append(a)

log4 = 0
for i in range(0, 49):
    log4 = log4 + log_like4[i]

univariate_log_likelihood = numpy.round(log1 + log2 + log3 + log4, 3)
print("univariate_log_likelihood =", univariate_log_likelihood)
print()


multi = 1
mean = [mu1, mu2, mu3, mu4]
multiarr = []
for i in range(0, 49):
    row = [cs_score[i], research[i], admin_pay[i], tuition[i]]
    multi =(scipy.stats.multivariate_normal.pdf(row, mean, covarianceMat, True))
    multiarr.append(multi)

multi_log = 0
for item in multiarr:
    multi_log = numpy.round(multi_log + math.log(item,math.e), 3)

print("multi_log =", multi_log)
