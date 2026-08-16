
Predikcija cijene polovnih automobila – Regresija

Cilj projekta je razviti model mašinskog učenja koji predviđa vrijednost kolone 'priceUSD'. Radi se o regresionom problemu. Model predviđa cijenu, numeričku vrijednost, a ne klasu ili kategoriju.

Dataset

Na raspolaganju je skup podataka:
cars.csv

Dataset sadrži podatke o polovnim automobilima i ima sljedeće kolone:

•	'make' - data marka automobila;
•	'model' - model automobila;
•	'priceUSD' - cijena automobila u dolarima;
•	'year' - godina proizvodnje;
•	'condition' - stanje automobila;
•	'mileage(kilometers)' - pređena kilometraža;
•	'fuel_type' - vrsta goriva;
•	'volume(cm3) - zapremina motora;
•	'color' - boja automobila;
•	'transmission' - tip mjenjača;
•	'drive_unit' - tip pogona;
•	'segment' - klasa automobila.

Ciljna promjenljiva je 'priceUSD'.

Kako se pokreće kod

Instalacija biblioteka iz requirements.txt:
pip install -r requirements.txt
Pokretanje fajla za čišćenje:
python -m src.data_cleaning
Pokretanje fajla za inženjering karakteristika:
python -m src.feature_engineering
Pokretanje fajla za pretprocesiranje:
python src/data_preprocessing.py
Pokretanje fajla za treniranje modela:
python src/model_training.py
Pokretanje fajla za evaluaciju modela:
Python src/model_evaluation.py
Pokretanje fajla za poređenje više modela:
python src/model_comparision.py

Modeli korišteni za testiranje

•	Linear Regression
•	Decision Tree Regressor
•	Random Forest Regressor
•	Gradient Boosting Regressor

Rezultati evaluacije

Model	MAE	RMSE	R2
Random Forest 	1081.82	2864.97	0.88
Decision Tree	1367.24	3465.29	0.83
Gradient Boosting	1533.94	2887.62	0.88
Linear Regression	2042.69	3982.58	0.77

Izbor modela
Najbolje rezultate pokazao je model Random Forest Regressor. 
Izbor je izvršen na osnovu evaluacije metrika koje su prikazane u prethodnoj tabeli.
Vrijednost MAE je 1081.82 USD, što znači da model griješi za 1082 USD po automobilu što predstavlja dobar rezultat kada su automobili u pitanju.
Vrijednost RMSE od 2865 USD predstavlja prosječnu veličinu greške izabranog modela. Znači, da u prosjeku promaši stvarnu cijenu automobila za 2865 USD.
Vrijednost R2 = 0.88 predstavlja dobru vrijednost.











