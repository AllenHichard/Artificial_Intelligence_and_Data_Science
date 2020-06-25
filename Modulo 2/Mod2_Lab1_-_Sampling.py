import numpy.random as nr
"""
Neste laboratório, veremos como amostras aleatórias (e as análises de dados que derivam delas) estimam as populações de 
onde elas vêm.
Isso vale repetir: quando você está trabalhando com uma amostra de dados, você está usando isso como uma estimativa da 
população que os gerou.
Então, quão boas são suas estimativas? Ao trabalhar com profissionais e estudantes, eu costumo descobrir que nossas 
intuições humanas geralmente estão erradas. No entanto, podemos brincar de amostragem e ver os resultados.
Primeiro, devemos definir a semente. Uma semente é definida usando a função semente do pacote Numpy.random Python. 
Essa função inicializa a geração de números aleatórios no seu computador como a minha, para que possamos obter os 
mesmos resultados.
"""
nr.seed(12345)
"""
Para gerar uma amostra aleatória normalmente distribuída, usamos o normal (média, padrão, n). Por exemplo, 50 respostas 
de uma população com média de 10 e desvio padrão de 2 são:
"""
nr.normal(10, 2, 50)
"""
Uma matriz Numpy com os valores normalmente distribuídos é retornada.

Também podemos fazer algo semelhante com uma distribuição binomial (os dados podem ter dois resultados, como "curtir" e 
"não curtir" um produto). Aqui está o código que usa: binomial (n, prob, tamanho = 1). O argumento prob representa a 
probabilidade de obter um 1 em vez de um 0. O argumento size altera a natureza da distribuição de uma maneira que não 
discutirei aqui. Se queremos simular 50 respostas de uma população na qual 30% das pessoas gostam do seu produto (1) e 
70% não (0), usamos:
"""
nr.seed(3344)
nr.binomial(1, 0.3, 50)
"""
Nesse caso, cada 1 representa alguém que gosta do seu produto e cada 0 representa alguém que não gosta.

Existem muitas distribuições que podemos usar com várias formas, incluindo distribuições que têm inclinação, 
distribuições que podem se parecer com contagens de coisas (por exemplo, apenas números discretos, a maioria das 
pontuações zero). Vamos ficar com esses dois para este laboratório.

"Gosto" vs "Não gosto"
Vamos tentar o exemplo acima, no qual cada 1 representa alguém que gosta do seu produto e cada 0 representa alguém que não gosta.

Desta vez, executarei a amostra e salvarei o resultado.
"""
nr.seed(3344)
sample1 = nr.binomial(1, 0.3, 50)
"""
Agora podemos examinar o quão bem nossa amostra foi. Nesse caso, sabemos que o valor da população era de 30%, 
porque especificamos esse parâmetro quando executamos o código. Quão perto chegou do nosso verdadeiro valor de 30%? 
Para responder a essa pergunta, execute a função itemfreq no módulo scipy.stats.
python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
"""
from scipy import stats
print(stats.itemfreq(sample1))
print(13.0/(13.0 + 37.0))
"""
Existem 13 curtidas e 37 curtidas. Podemos converter em porcentagens mergulhando pela soma de gostos e desgostos:
Nossa amostra subestimou o número de pessoas que gostam do produto, retornando "26%" em vez de 30%.

Como os dados são codificados em 0 e 1, também podemos fazer com que a matemática retorne uma proporção usando mean () de Numpy:
"""
import numpy as np
np.mean(sample1)

"""
Vamos tentar isso várias vezes. Toda vez que executo o código, uma amostra aleatória será coletada, a proporção de 
pessoas que gostam do produto calculada e relatada a você.
"""

print(np.mean(nr.binomial(1, 0.3, 50)))
print(np.mean(nr.binomial(1, 0.3, 50)))
print(np.mean(nr.binomial(1, 0.3, 50)))
print(np.mean(nr.binomial(1, 0.3, 50)))
print(np.mean(nr.binomial(1, 0.3, 50)))
print(np.mean(nr.binomial(1, 0.3, 50)))

"""
Vemos aqui que nossas amostras estão variando bastante. Podemos executar muitos deles usando uma compreensão de lista. 
Vamos tentar isso 100 vezes. Suponho que você esteja familiarizado com a compreensão de listas em Python.
"""
nr.seed(9977)
results = [np.mean(nr.binomial(1, 0.3, 50)) for _ in range(100)]
print(results)

"""
Vemos uma variação considerável nesses resultados. Podemos histograma-los para ver melhor:
"""
# So the plot appears in line in the noteboook%matplotlib inline

sample_mean = np.mean(results)
import matplotlib.pyplot as plt
plt.hist(results)
plt.vlines(0.3, 0.0, 28.0, color = 'red')
plt.vlines(sample_mean, 0.0, 28.0, color = 'black')
plt.xlabel('Results')
plt.ylabel('Frequency')
plt.title('Histogram of results')
#plt.show()

"""
Vemos aqui que, em média, amostras aleatórias são confiáveis - afinal, elas tendem a 30%. No entanto, amostras 
individuais são menos confiáveis. Alguns resultados são quase tão grandes quanto 135% ou tão baixos quanto 50%. Caramba!

Também podemos subtrair 0,30 de cada pontuação para classificá-las novamente como o grau de erro em cada amostra.
"""
results_error = [round(x - 0.3, 2) for x in results]
print(results_error)

"""
Vemos aqui que a maioria das pontuações da amostra está dentro de cerca de 5% do valor real da população. Ainda assim, 
dependendo do que queremos fazer com os dados, isso pode ser inaceitavelmente grande. A propriedade das amostras para 
"estimar incorretamente" a população é denominada erro de amostragem e é claramente um grande problema, levando a muitas 
decisões erradas. O grau em que suas amostras individuais tendem a "estimar incorretamente" a população (mostrado acima:
 results_error) é algo que queremos estimar. Normalmente, quantificamos isso tomando o desvio padrão desses erros. 
 Isso é chamado de "erro padrão" e é um número único, a que distância as amostras estão, em média:
"""

print(np.std(results_error))
"""
Ah, então vemos que a amostra média está "desligada" do valor da população em 6%. Alguns estão "desligados" por mais; 
alguns estão "desativados" por menos, mas a amostra média está desativada em 6%. Em outras palavras, nosso erro padrão é de 6%.
Curiosidade: você também pode estimar o erro padrão com uma equação simples. Para dados binomiais (pontuações 0 e 1), a equação é:
"""
"""
$$se = \sqrt{\frac{p\left ( 1-p \right )}{n-1}}$$
Aqui, p é a porcentagem na população. Então, inserindo nossos valores:
"""
import math
print(math.sqrt((.30*(1-.3))/(50-1)))

"""
Isso é conveniente, porque nos diz que realmente não precisamos executar simulações como as acima para saber quão 
confiáveis são nossas amostras. De fato, adivinhando um palpite razoável para o valor da população e o tamanho da amostra, 
podemos saber antes de executar um estudo a confiabilidade de uma amostra típica.

Claramente, um grande erro padrão é uma coisa ruim. Podemos reduzir esse problema contando com uma amostra maior. 
Por exemplo, tente usar uma amostra de 700 na equação para o erro padrão mostrado anteriormente:
"""

print(math.sqrt((.30*(1-.3))/(700-1)))

"""
Vemos agora que a amostra típica terá apenas 1,7% do valor da população. Podemos executar um loop semelhante ao feito 
anteriormente e ver isso em ação:
"""

nr.seed(4466)
results = [np.mean(nr.binomial(1, 0.3, 700)) for _ in range(100)]

print(np.std(results))

sample_mean = np.mean(results)
import matplotlib.pyplot as plt
plt.hist(results)
plt.vlines(0.3, 0.0, 28.0, color = 'red')
plt.vlines(sample_mean, 0.0, 28.0, color = 'black')
plt.xlabel('Results')
plt.ylabel('Frequency')
plt.title('Histogram of results')
plt.show()

"""
Vemos aqui, agora que a maioria resulta entre 28,3% e 30,7%, com o resultado típico sendo "desligado" em apenas 1,7% ... 
exatamente como nossa equação de erro padrão previu.

Toda situação de dados tem um erro padrão. O objetivo não é aprender um grande número de equações, mas enfatizar o 
seguinte ponto: as amostras (e as estatísticas que elas produzem) são estimativas defeituosas da população. 
No entanto, eles se tornam cada vez mais precisos à medida que aumentam os tamanhos das amostras.

Descobriremos, em breve, que isso nos dará o conceito de poder estatístico. Amostras grandes produzirão resultados 
fortes o suficiente para que possamos fazer declarações significativas sobre a população 
(em tais situações, temos "bom poder"), onde amostras pequenas contêm tanto erro que não podemos dizer muito significativo 
sobre a população ("poder fraco" )
"""
