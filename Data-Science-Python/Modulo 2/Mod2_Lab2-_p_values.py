"""
Imagine que homens e mulheres tiveram um nível de interesse médio de 5, com um desvio padrão de 3.
"""
# set seed to make random number generation reproducible
import numpy as np
import numpy.random as nr
nr.seed(51120122)

#collect a sample of 100 males
males = nr.normal(5, 3, 100)

#collect a sample of 100 females
females = nr.normal(5, 3, 100)

#print(np.mean(males))
#print(np.mean(females))

"""
Vemos aqui que nossos dois grupos têm resultados de amostra diferentes. Vamos ver o quão grande é a diferença:
"""

#print(np.mean(males)-np.mean(females))
nr.seed(4455)
attitude = nr.normal(2.4, 2.0, 100)
#Qual é a média e o desvio padrão em nossa amostra?
print(np.mean(attitude))
print(np.std(attitude))
from scipy import stats


def t_one_sample(samp, mu=0.0, alpha=0.05):
    '''Function for two-sided one-sample t test'''
    t_stat = stats.ttest_1samp(samp, mu)
    scale = np.std(samp)
    loc = np.mean(samp)
    ci = stats.t.cdf(alpha / 2, len(samp), loc=mu, scale=scale)
    print('Results of one-sample two-sided t test')
    print('Mean         = %4.3f' % loc)
    print('t-Statistic  = %4.3f' % t_stat[0])
    print('p-value      < %4.3e' % t_stat[1])
    print('On degrees of freedom = %4d' % (len(samp) - 1))
    print('Confidence Intervals for alpha =' + str(alpha))
    print('Lower =  %4.3f Upper = %4.3f' % (loc - ci, loc + ci))


t_one_sample(attitude)

"""
O teste avalia quanto os dados discordam dos nulos (ou seja, o efeito; parte superior da fração) em comparação com o 
que você normalmente esperaria por acaso (parte inferior da fração). Assim, podemos literalmente ler o resultado dizendo 
"nosso efeito foi 10,8 vezes maior do que você normalmente esperaria por acaso". Isso parece muito bom para o nosso efeito 
e muito ruim para a hipótese nula.


A informação principal desta função é: t estatística = 10.7, df = 99, valor-p <2.9e-18. Observe que o valor p é exibido 
em notação científica. 2.9e-18 é uma notação científica: 2.9 x 10-18 e significa o mesmo que 0.0000000000000000029. 
Isso é claramente menor que 0,05, para que possamos rejeitar a hipótese nula e concluir que a atitude positiva observada 
entre nossos participantes não foi um acaso estatístico, mas provavelmente uma tendência real na população.
"""


from scipy.stats import t
pvalor = 1 - t.cdf(10.8, df = 99, loc=0, scale=1)
print(pvalor)

pvalorbicaudal =  2.0 * (1 - t.cdf(10.8, df = 99, loc=0, scale=1))