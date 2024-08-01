Procurei no Kaggle um banco de dados com jogos e pontos dos dois times numa partida e achei esse, em CSV e SQL. 

Esse [dataset](https://www.kaggle.com/datasets/wyattowalsh/basketball) gigante tem jogos da primeira temporada de 1946 até 2023.

Para o tratamento dos dados, realizei uma limpeza removendo colunas e linhas desnecessárias para o estudo, alterando certos times que mudaram de nome/cidade, entre outros procedimentos. 
Isso resultou no arquivo abaixo.

[Arquivo com os jogos](https://github.com/mths-andrade/pyth_exp/blob/d1565d450591c1158384a8d9a429c76758d7f589/games.csv)

Inicialmente importei as bibliotecas Pandas para procedimentos usuais, Seaborn para gráficos, StatsModels para estatística e Math para matemática. 
Depois, li o arquivo e os armazenei em variáveis como de costume. Criei uma função para a expectativa pitagórica, recebendo os pontos convertidos e sofridos:

```python
def pyth_exp(made,allow):
  return made**2/(made**2+allow**2)
```

Para a porcentagem de vitórias, fiz o comando abaixo. Usei o Washington Wizards, último time na ordem alfabética, como exemplo.

```python
home_w=dados.query("home=='WAS' and wl_home==1")
away_w=dados.query("away=='WAS' and wl_away==1")
home=dados.query("home=='WAS'")
away=dados.query("away=='WAS'")
games=home.shape[0]+away.shape[0]
games_w=home_w.shape[0]+away_w.shape[0]
win_perc=games_w/games
win_perc = round(win_perc, 4)
print("Ganhou %.0f jogos de %.0f, portanto tem um aproveitamento de %.4f."%(games_w,games,win_perc))

Ganhou 2052 jogos de 4516, portanto tem um aproveitamento de 0.4544.
```

Para achar os pontos feitos e sofridos foi mais simples, só o query de método.

```python
pth_m=dados.query("home=='WAS'")["pts_home"].sum()
pth_a=dados.query("home=='WAS'")["pts_away"].sum()
pta_m=dados.query("away=='WAS'")["pts_away"].sum()
pta_a=dados.query("away=='WAS'")["pts_home"].sum()
pt_made=pth_m+pta_m
pt_allow = pth_a + pta_a
print("Os pontos marcados e permitidos são, respectivamente,%.0f e %.0f."%(pt_made,pt_allow))
print("A expectativa pitagórica é de %.4f."%(pyth_exp(pt_made,pt_allow)))

Os pontos marcados e permitidos são, respectivamente,471509 e 477840.
A expectativa pitagórica é de 0.4933.
```

Reuni esses resultados no arquivo abaixo, que tem as vitórias e jogos totais de cada time nas temporadas regulares, além da porcentagem de vitória histórica. 
Coloquei também os dois tipos de pontos e a expectativa pitagórica.

[Expectativa pitagórica](https://github.com/mths-andrade/pyth_exp/blob/d1565d450591c1158384a8d9a429c76758d7f589/pythagorean.csv)

Depois, plotei o gráfico de dispersão entre a expectativa e a porcentagem. 
As linhas tracejadas são as médias de cada eixo. Como a porcentagem real cresce junto com a pitagórica, estima-se que a relação entre as duas variáveis seja quase perfeitamente linear positiva.

```python
x = pyth.pyth_exp
y = pyth.win_perc

ax = sns.scatterplot(data=pyth, x="pyth_exp", y="win_perc")
ax.figure.set_size_inches(10, 6)
ax.set_title("Expectativa pitagórica X porcentagem real de vitórias de cada time")
ax.hlines(y = y.mean(), xmin = x.min(), xmax = x.max(), colors='black', linestyles='dashed')
ax.vlines(x = x.mean(), ymin = y.min(), ymax = y.max(), colors='gray', linestyles='dashed')
ax.text(0.487, 0.5, 'Média da porcentagem de vitórias', fontsize=10, color = 'black', va = "bottom")
ax.text(0.4997, 0.58, 'Média da expectativa pitagórica', fontsize=10, color = 'gray', ha = "left")
plt.show()
```
![Dispersão](https://github.com/user-attachments/assets/2d33e7e4-b315-42c4-a4e5-8ec1fcaeb076)

Em seguida, plotei o gráfico da regressão entre as variáveis. O comportamento linear se mostra ainda mais forte com a reta de regressão.

```python
ax=sns.lmplot(x='pyth_exp',y='win_perc',data=pyth)
ax.fig.suptitle("Expectativa pitagórica X porcentagem real de vitórias",y=1.05)
```
![Regressão](https://github.com/user-attachments/assets/6234659b-e6fd-4f0e-83ef-543b1a98c7e8)

Ajustei o modelo pelo método dos mínimos quadrados ordinários com as funções ols e fit.

```python
pyth_lm=smf.ols(formula='win_perc ~ pyth_exp', data=pyth).fit()
```

O principal resultado que queremos é o coeficiente de correlação $R²$. Vamos chegar a ele de três modos diferentes.

- O primeiro é exibir os dados estatísticos da regressão com o `summary`.

```python
print(pyth_lm.summary())
```

![Modelo](https://github.com/user-attachments/assets/1d36626a-b5e5-41d3-b70f-b348537afd51)

Como pode ser visto acima, temos 30 observações (os 30 times) e 28 graus de liberdade, isto é, 30 observações menos as 2 variáveis. Temos também os coeficientes da nossa reta de regressão, que pode ser escrita assim, arredondando para uma casa decimal:

$$
\boxed{y=-2.9+6.8x}
$$

$x$ é a porcentagem de vitórias e $y$, a expectativa pitagórica. Podemos calcular tais coeficientes manualmente:

```python
x=pyth.pyth_exp
y=pyth.win_perc
SOMA_X = pyth.pyth_exp.sum()
SOMA_Y = pyth.win_perc.sum()
SOMA_X2 = pyth.pyth_exp.apply(lambda x: x**2).sum()
SOMA_Y2 = pyth.win_perc.apply(lambda y: y**2).sum()
SOMA_XY = pyth.apply(lambda data: data.pyth_exp * data.win_perc, axis = 1).sum()
```

$$
\boxed{\beta_2=\frac{n\sum X_iY_i-\sum X_i \sum Y_i}{n\sum X_i^2-(\sum X_i)²}}
$$

```python
n=30 # Número de observações
numerador = n * SOMA_XY - SOMA_X * SOMA_Y
denominador = n * SOMA_X2 - (SOMA_X)**2
beta_2 = numerador / denominador
print("O coeficiente linear da reta de regressão é",beta_2.round(2))

O coeficiente linear da reta de regressão é 6.8
```

$$
\boxed{\beta_1=\frac{\sum X_i²\sum Y_i-\sum X_i\sum X_iY_i}{n\sum X_i²-(\sum X_i)²}\\
=\overline{Y}-\beta_2\overline{X}}
$$

```python
beta_1 = pyth.win_perc.mean() - beta_2 * pyth.pyth_exp.mean()
print("O coeficiente angular da reta de regressão é",beta_1.round(2))

O coeficiente angular da reta de regressão é -2.9
```

$$
\boxed{y=\beta_1+\beta_2x}
$$

```python
print("A reta de regressão é y=%.2f+%.2fx"%(beta_1,beta_2))

A reta de regressão é y=-2.90+6.80x
```

Podemos até criar uma função que retorna a porcentagem de vitórias esperada dada uma expectativa pitagórica:

```python
def previsão(x):
  print("Dada a expectativa pitagórica %.3f, a previsão da porcentagem real de vitórias é %.3f."%(x,beta_1 + beta_2 * x))
  
# Exemplo
previsão(0.515)
Dada a expectativa pitagórica 0.515, a previsão da porcentagem real de vitórias é 0.602.
```

Temos os desvios padrões de cada variável, as estatísticas de teste $t$ de cada uma e o p-valor igual a zero, o que mostra que o modelo tem grande significância estatística. Além disso, também temos os intervalos de confiança ao nível de significância padrão de 5%. Também podemos ver o coeficiente de regressão **$R²=0.983$**, um valor quase igual a 1, comprovando o caráter linear positivo da regressão.

- O segundo é a partir da soma dos quadrados dos erros do modelo, $SQE=\sum{(Y_i - \hat{Y}_i)^2}$, a soma dos quadrados dos pontos da regressão, $SQR=\sum{(\hat{Y}_i - \bar{Y})^2}$, e da soma total desses dois, $SQT=\sum{(Y_i - \bar{Y})^2}$, onde $Y_i$ são as porcentagens reais de vitória e $\hat{Y_i}$ e $\overline{Y}$ respectivamente as expectativas e a média delas.
    
```python
SQE=pyth_lm.ssr
SQR=pyth_lm.ess
SQT=SQE+SQR
print(SQE.round(3)) 0.001
print(SQR.round(3)) 0.073
print(SQT.round(3)) 0.074
```
    
Para achar o coeficiente de correlação, basta dividir SQR por SQT.
    
```python
R2=SQR/SQT
print("O coeficiente de regressão do modelo é",R2.round(3))
    
O coeficiente de regressão do modelo é 0.983
```
    
Achamos $R²=0.983$, exatamente o mesmo valor obtido anteriormente.
    
A título de curiosidade, vamos calcular a média do SQE, que consiste na divisão desse valor pelo número de graus de liberdade, nesse caso 28.
    
```python
# Erro quadrático médio
EQM=pyth_lm.mse_resid
print("O erro quadrático médio é",EQM)

O erro quadrático médio é 4.3977042679913256e-05 # Aproximadamente 0.00004
```
    
Arredondando mais uma vez para 3 casas decimais, temos que ele é zero, aumentando ainda mais a confiabilidade do modelo.
    
- O terceiro modo é simplesmente printar o valor com 3 casas decimais.

```python
pyth_lm.rsquared.round(3)
0.983
```

Novamente temos o mesmo valor dos 2 métodos.

$R²=0.983$, bem próximo de 1, então *a expectativa pitagórica explica muito bem a porcentagem de vitórias convencional*. 
**A relação linear entre elas é quase perfeita**.

Quando vi a fórmula pitagórica pela primeira vez, achei bem aleatória, tirada da cartola, porém, depois desse estudo, entendi a sagacidade de quem a propôs pela primeira vez. 
Imagine conseguir estimar o número de vitórias de um time na temporada só com os pontos que ele fez e sofreu.

Na situação que descobri a fórmula, estava vendo números do Minnesota Timberwolves. 
No caso, vi um número de 57 vitórias pitagóricas de um máximo de 82, ou seja, a porcentagem de vitórias estimada é 0.695.

![exp](https://github.com/user-attachments/assets/5380dde1-6d50-4a2b-951a-dd355eaf324e)

Supondo que esse número obedeça à fórmula da expectativa, está muito longe do nosso modelo histórico (0.403 e 0.488 respectivamente porcentagem real e expectativa).

![excel](https://github.com/user-attachments/assets/f2773200-3bc1-4508-afce-d578093f807f)

Entretanto, a taxa real de vitórias foi de 0.683 (56 vitórias em 82 jogos), apenas uma vitória a menos do que a estimada.

![timb](https://github.com/user-attachments/assets/b6512877-947c-4819-a21c-ccf5a3f59288)

Temos mais uma prova da confiabilidade da expectativa pitagórica, dessa vez, com dados da temporada atual.

---

Todo meu processo está no [notebook](https://github.com/mths-andrade/pyth_exp/blob/c36f98e6b41ff54cbdd532151c941554420e362c/pythagorean.ipynb), usei o Google Colab. O time que está no código é o Washington Wizards, último na ordem alfabética. Para checar os dados de um time em específico, é só substituir pela respectiva sigla, BOS para o Celtics, GSW para o Warriors etc.

Minha inspiração para esse estudo saiu desse [artigo](https://medium.com/@kaantopcu/pythagorean-expectation-in-sports-analytics-in-nba-60061a842d03) do Medium (em inglês).

Como o autor trabalhou só com o período de 2004 a 2021, o modelo ficou um pouquinho menos ajustado que o meu, como consequência, o R² diminuiu.

Obrigado pela atenção! 🏀

