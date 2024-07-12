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
ax.set_title("Expectativa pitagórica X porcentagem real de vitórias")
ax.hlines(y = y.mean(), xmin = x.min(), xmax = x.max(), colors='gray', linestyles='dashed')
ax.vlines(x = x.mean(), ymin = y.min(), ymax = y.max(), colors='gray', linestyles='dashed')
```

![Dispersão](https://github.com/user-attachments/assets/8f95c32c-ed51-40a8-ac21-91aaac435566)

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

![Modelo](https://github.com/user-attachments/assets/a1960e28-b721-45db-b723-a1a75c19bd75)

Como pode ser visto acima, temos 30 observações (os 30 times) e 28 graus de liberdade, isto é, 30 observações menos as 2 variáveis. 
Temos também os coeficientes da nossa reta de regressão, que pode ser escrita assim, arredondando para uma casa decimal:

$$\boxed{y=-2.9+6.8x}$$

$x$ é a porcentagem de vitórias e $y$, a expectativa pitagórica. Podemos calcular tais coeficientes manualmente:










