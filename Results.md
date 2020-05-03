# Porównanie LR, KNN, SVR, CART i MLP
Na podstawie eksperymentu_1 wygenerowałam porównanie regresorów KNN, SVR I drzew CART (na wykresie DTR).
Wyszły porównywalnie dobre co PCALR.

```python
#Metody z użytymi parametrami
methods = {"KNN": KNeighborsRegressor(weights='distance'), "SVR": SVR(), "DTR": DecisionTreeRegressor()}
```

(im więcej wykresów tym gorszy opis osi X)

![figures/experiment_1_1.png](figures/experiment_1_1.png)

Z siecią neuronową nie było już tak prosto, dla porównania zestwiono MLP Z PCALR.
```python
ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
```
Biorąc pod uwagę powyższy błąd zwiększyłam max_iter. Proponowany solver "lbfgs"
```python
methods = {"PCALR": met.PCALR(weights='distance'), "MLP": MLPRegressor(solver="lbfgs", hidden_layer_sizes=(8,), max_iter=5000)}
```

![figures/experiment_1_2.png](figures/experiment_1_2.png)

Zmniejszyłam parameter _hidden_layer_sizes_ do 3, ale nadal wyniki były słabe

```python
methods = {"PCALR": met.PCALR(weights='distance'), "MLP": MLPRegressor(solver="lbfgs", hidden_layer_sizes=(3,), max_iter=5000)}
```

![figures/experiment_1_3.png](figures/experiment_1_3.png)


Ostatecznie zrezygowałam z parametru max_iter, zostawiłam go domyślnie
```python
methods = {"PCALR": met.PCALR(weights='distance'), "MLP": MLPRegressor(solver="lbfgs", hidden_layer_sizes=(3,))}
```

![figures/experiment_1_4.png](figures/experiment_1_4.png)

Ostatecznie zrezygowałam z parametru max_iter, zostawiłam go domyślnie
```python
methods = {"PCALR": met.PCALR(weights='distance'), "MLP": MLPRegressor(solver="lbfgs", hidden_layer_sizes=(5,))}
```

![figures/experiment_1_5.png](figures/experiment_1_5.png)

Wyszło na to, że mniej neuronów = lepiej. 

Znalazłam w dokumentacji, że dla używanego solvera ma jeszcze znaczenie parametr _max_fun_ oznaczający maksymalną liczbę wywołań funkcji
Przy poniższych ustawieniach wynik jak dotąd najlepszy pogorszył się. W niektórym przypadku nawet gorzej niż regresja liniowa.
```python
methods = {"PCALR": met.PCALR(weights='distance'), "LR": LinearRegression(), "MLP": MLPRegressor(solver="lbfgs", hidden_layer_sizes=(3,), max_fun=1000)}
```

![figures/experiment_1_6.png](figures/experiment_1_6.png)

Najlepsze wynik na który wpadłam był dla ustawień:
```python
methods = {"PCALR": met.PCALR(weights='distance'), "LR": LinearRegression(), "MLP": MLPRegressor(solver="lbfgs", hidden_layer_sizes=(1,))}
```

![figures/experiment_1_8.png](figures/experiment_1_8.png)

#PCA dla każdego z regresorów
## n_components: 2
(ustawienia tak jak powyżej, dla MLP ostatniego (chyba najlepszego) uzyskanego wyniku)

![figures/experiment_2.png](figures/experiment_2.png)
![figures/experiment_2_1.png](figures/experiment_2_1.png)

(Wyniki dla MLP wypadały różnie, np.:)
![figures/experiment_2_1.png](figures/experiment_2_4.png)

## n_components: 10
![figures/experiment_2.png](figures/experiment_2_10.png)
![figures/experiment_2_1.png](figures/experiment_2_11.png)