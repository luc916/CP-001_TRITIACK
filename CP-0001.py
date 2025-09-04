import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Base única (imutável)
df_base = pd.read_csv(
    'household_power_consumption.txt',
    sep=';',
    na_values=['?'],
    low_memory=False
)

cols_to_convert = [
    'Global_active_power', 'Global_reactive_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
    'Sub_metering_3'
]
for col in cols_to_convert:
    df_base[col] = pd.to_numeric(df_base[col])

# Exercicio 1
print("\n--- Exercício 1 ---")
df = df_base.copy()
print(df.head(10))
print("\n" + "="*50 + "\n")

# Exercicio 2
print("\n--- Exercício 2 ---")
df = df_base.copy()
print("Global active power é o consumo total da casa, Global reactive power é o consumo necessário para manter a operação de certos dispositivos, mas que não faz trabalho útil.")
print("\n" + "="*50 + "\n")

# Exercicio 3
print("\n--- Exercício 3 ---")
df = df_base.copy()
nulos = df.isnull().sum()
print(nulos)
print("\n" + "="*50 + "\n")

# Exercicio 4
print("\n--- Exercício 4 ---")
df = df_base.copy()
df["Date"] = pd.to_datetime(df["Date"], format='%d/%m/%Y')
df["Dia_da_Semana"] = df["Date"].dt.day_name()
print("Coluna 'Dia_da_Semana' criada:")
print(df[['Dia_da_Semana']].head())
print("\n" + "="*50 + "\n")

# Exercicio 5
print("\n--- Exercício 5 ---")
df = df_base.copy()
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
ano = 2007
df_ano = df[df["Date"].dt.year == ano]
global_active_power_2007 = df_ano["Global_active_power"]
media = global_active_power_2007.mean()
print(f"A média de consumo diário da coluna Global Active Power no ano de 2007 foi de: {media:.2f} kW")
print("\n" + "="*50 + "\n")

# Exercicio 6
print("\n--- Exercício 6 ---")
df = df_base.copy()
print("Gráfico")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
dia = pd.to_datetime("2007-12-29")
df_dia = df[df["Date"].dt.date == dia.date()]
plt.figure(figsize=(10, 6))  
plt.plot(df_dia['Time'], df_dia['Global_active_power'], marker='o', linestyle='-', color='b')
plt.title(f'Variação de Global Active Power em {dia}')
plt.xlabel('Hora')
plt.ylabel('Global Active Power (kilowatts)')
plt.xticks(rotation=45)
plt.tight_layout() 
plt.show()
print("\n" + "="*50 + "\n")

# Exercicio 7
print("\n--- Exercício 7 ---")
df = df_base.copy()
print("Gráfico")
plt.figure(figsize=(10, 6))
plt.hist(df['Voltage'], bins=50, alpha=0.7, color='blue')
plt.title('Distribuição da Tensão (Voltage)')
plt.xlabel('Voltage (V)')
plt.ylabel('Frequência')
plt.xticks(rotation=45) 
plt.tight_layout() 
plt.show()
print("\n" + "="*50 + "\n")

# Exercicio 8
print("\n--- Exercício 8 ---")
df = df_base.copy()
media_mes = df["Global_active_power"].mean()
print(f'O consumo médio de energia durante o período foi de {media_mes:.2f} kW')
print("\n" + "="*50 + "\n")

# Exercicio 9
print("\n--- Exercício 9 ---")
df = df_base.copy()
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
consumo = df.groupby("Date")["Global_active_power"].sum()
dia_max = consumo.idxmax()
consumo_max = consumo.max()
print(f'O dia com mais consumo de energia foi {dia_max.date()} com {consumo_max:.2f} kW')
print("\n" + "="*50 + "\n")

# Exercicio 10
print("\n--- Exercício 10 ---")
df = df_base.copy()
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df["Dias_da_Semana"] = df["Date"].dt.dayofweek
df["Tipo_de_Dia"] = df["Dias_da_Semana"].apply(lambda x: "Final_de_semana" if x >= 5 else "Semana")
consumo_medio_dia_da_semana = df.groupby("Tipo_de_Dia")["Global_active_power"].mean()
print(consumo_medio_dia_da_semana)
print("\n" + "="*50 + "\n")

# Exercicio 11
print("\n--- Exercício 11 ---")
df = df_base.copy()
colunas = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
correlacao = df[colunas].corr()
print("Matriz de correlação:")
print(correlacao)
print("\n" + "="*50 + "\n")

# Exercicio 12
print("\n--- Exercício 12 ---")
df = df_base.copy()
df["Total_Sub_Metering"] = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
print("Coluna 'Total_Sub_Metering' criada:")
print(df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Total_Sub_Metering']].head())
print("\n" + "="*50 + "\n")

# Exercicio 13
print("\n--- Exercício 13 ---")
df = df_base.copy()
df["Total_Sub_Metering"] = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month
media_geral_gap = df['Global_active_power'].mean()
consumo_mensal = df.groupby(['Year', 'Month'])['Total_Sub_Metering'].mean()
meses_alto_consumo = consumo_mensal[consumo_mensal > media_geral_gap]
print(f"Referência (Média global de Global_active_power): {media_geral_gap:.2f} kW")
print("\nMeses onde a média de 'Total_Sub_Metering' foi MAIOR que a referência:")
for (year, month), valor in meses_alto_consumo.items():
    print(f"{year}-{month:02d}: {valor:.2f} kW")
df.drop(columns=['Year', 'Month'], inplace=True)
print("\n" + "="*50 + "\n")

# Exercicio 14
print("\n--- Exercício 14 ---")
df = df_base.copy()
print("Gráfico")
df['DateTime'] = pd.to_datetime(df['Date'] + " " + df['Time'], format="%d/%m/%Y %H:%M:%S")
df_2008 = df[df['DateTime'].dt.year == 2008]
df_2008_temp = df_2008.copy()
df_2008_temp.set_index('DateTime', inplace=True)
df_2008_daily = df_2008_temp['Voltage'].resample('D').mean()
plt.figure(figsize=(15, 6))
plt.plot(df_2008_daily.index, df_2008_daily.values, linewidth=1)
plt.title('Série Temporal - Voltage Médio Diário (2008)')
plt.xlabel('Data')
plt.ylabel('Voltage (V)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("\n" + "="*50 + "\n")

# Exercicio 15
print("\n--- Exercício 15 ---")
df = df_base.copy()
df['DateTime'] = pd.to_datetime(df['Date'] + " " + df['Time'], format="%d/%m/%Y %H:%M:%S")
df['Month'] = df['DateTime'].dt.month
consumo_verao = df[df['Month'].isin([6, 7, 8])]['Global_active_power'].mean()
consumo_inverno = df[df['Month'].isin([12, 1, 2])]['Global_active_power'].mean()
print(f"Verão (Jun-Ago): {consumo_verao:.2f} kW")
print(f"Inverno (Dez-Fev): {consumo_inverno:.2f} kW")
diferenca = ((consumo_inverno - consumo_verao) / consumo_verao) * 100
print(f"Inverno é {diferenca:+.1f}% comparado ao verão")
df.drop(columns=['Month'], inplace=True)
print("\n" + "="*50 + "\n")

# Exercicio 16
print("\n--- Exercício 16 ---")
df = df_base.copy()
amostra = df.sample(frac=0.01, random_state=0)
base_media = df['Global_active_power'].mean()
base_std = df['Global_active_power'].std()
amostra_media = amostra['Global_active_power'].mean()
amostra_std = amostra['Global_active_power'].std()
dif_media = abs((amostra_media - base_media) / base_media) * 100
dif_std = (amostra_std - base_std / base_std) * 100
print(f"\nComparação da coluna '{'Global_active_power'}':")
print(f"Base completa - Média: {base_media:.3f} | Desvio: {base_std:.3f}")
print(f"Amostra 1%    - Média: {amostra_media:.3f} | Desvio: {amostra_std:.3f}")
print(f"\nDiferença:")
print(f"Média: {dif_media:.2f}%")
print(f"Desvio: {dif_std:.2f}%")
print("\n" + "="*50 + "\n")

# Exercicio 17
print("\n--- Exercício 17 ---")
df = df_base.copy()
colunas = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
for col in colunas:
    df[f"{col}_scaled"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
print("\nComparação das médias antes e depois da normalização:")    
for col in colunas:
    original = df[col].mean()   
    scaled = df[f"{col}_scaled"].mean()
    print(f"Média original de {col}: {original:.3f} | Média escalada: {scaled:.3f}")
print("\n" + "="*50 + "\n")

# Exercicio 18
print("\n--- Exercício 18 ---")
df = df_base.copy()
df['DateTime'] = pd.to_datetime(df['Date'] + " " + df['Time'], format="%d/%m/%Y %H:%M:%S")
df_diario = df.groupby(df['DateTime'].dt.date)['Global_active_power'].mean().reset_index()
df_diario.columns = ['Date', 'Consumo_Medio']
df_diario = df_diario.dropna(subset=['Consumo_Medio'])
kmeans = KMeans(n_clusters=3, random_state=0)
df_diario['Cluster'] = kmeans.fit_predict(df_diario[['Consumo_Medio']])
print(f"Total de dias válidos: {len(df_diario)}")
for i in range(3):
    dados = df_diario[df_diario['Cluster'] == i]
    consumo_medio = dados['Consumo_Medio'].mean()
    print(f"Cluster {i}: {len(dados)} dias (consumo médio: {consumo_medio:.2f} kW)")
centroides = kmeans.cluster_centers_.flatten()
ordem = sorted(range(3), key=lambda i: centroides[i])
nomes = ['Baixo consumo', 'Consumo médio', 'Alto consumo']
for i in range(3):
    cluster_id = ordem[i]
    valor = centroides[cluster_id]
    print(f"Cluster {cluster_id}: {nomes[i]} ({valor:.2f} kW)")
print("\n" + "="*50 + "\n")

# Exercicio 19
print("\n--- Exercício 19 ---")
df = df_base.copy()
df['DateTime'] = pd.to_datetime(df['Date'] + " " + df['Time'], format="%d/%m/%Y %H:%M:%S")
serie = df.set_index('DateTime')['2010-06-30':'2010-12-31']['Global_active_power'].resample('D').mean()
tendencia = serie.rolling(7, center=True).mean()
sazonal = (serie - tendencia).groupby(serie.index.dayofweek).transform('mean')
residuo = serie - tendencia - sazonal
plt.figure(figsize=(12, 8))
plt.subplot(4,1,1); plt.plot(serie); plt.title('Original')
plt.subplot(4,1,2); plt.plot(tendencia, 'r'); plt.title('Tendência')
plt.subplot(4,1,3); plt.plot(sazonal, 'g'); plt.title('Sazonalidade')
plt.subplot(4,1,4); plt.plot(residuo, 'orange'); plt.title('Resíduo')
plt.tight_layout()
plt.show()
print("Gráfico")
print(f"Tendência: {tendencia.dropna().iloc[-1] - tendencia.dropna().iloc[0]:+.3f} kW")
print(f"Amplitude sazonal: {sazonal.max() - sazonal.min():.3f} kW")
print("\n" + "="*50 + "\n")

# Exercicio 20
print("\n--- Exercício 20 ---")
df = df_base.copy()
x = df["Global_intensity"]
y = df["Global_active_power"]
x_mean, y_mean = x.mean(), y.mean()
b1 = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()    
b0 = y_mean - b1 * x_mean
pred = b0 + b1 * x
rmse = math.sqrt(((pred - y) ** 2).mean())
r2 = 1 - ((pred - y) ** 2).sum() / ((y - y_mean) ** 2).sum()    
print(f"Modelo: y = {b0:.4f} + {b1:.4f} * x")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print("\n" + "="*50 + "\n")

# Exercicio 21
print("\n--- Exercício 21 ---")
df = df_base.copy()
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.set_index("Datetime")
media_hora = df['Global_active_power'].resample('h').mean()
media_ordenada = media_hora.sort_values(ascending=False)
print("Top 5 horas de maior consumo: ")
print(media_ordenada.head(10))

# Exercicio 22
print("\n--- Exercício 22 ---")
df = df_base.copy()
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.set_index("Datetime")
serie = df['Global_active_power'].resample('h').mean()
lag_1h = serie.autocorr(lag=1)  
lag_24h = serie.autocorr(lag=24)  
lag_48h = serie.autocorr(lag=48)  
print(f"Autocorrelação (1h):  {lag_1h:.3f}")
print(f"Autocorrelação (24h): {lag_24h:.3f}")
print(f"Autocorrelação (48h): {lag_48h:.3f}")

# Exercicio 23
print("\n--- Exercício 23 ---")
df = df_base.copy()
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.set_index("Datetime")
new_df = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
new_df = new_df.dropna()
scaler = StandardScaler()
df_scaled = scaler.fit_transform(new_df)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
print("Variância por cada componente:", pca.explained_variance_ratio_)
print("Variância total:", pca.explained_variance_ratio_.sum())

# Exercicio 24
print("\n--- Exercício 24 ---")
df = df_base.copy()
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.set_index("Datetime")
new_df = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
new_df = new_df.dropna()
scaler = StandardScaler()
df_scaled = scaler.fit_transform(new_df)
pca = PCA(n_components=2)
new_df = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(new_df, columns=["PC1", "PC2"])
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(new_df)
df_pca["Cluster"] = clusters 
plt.figure(figsize=(8,6))
for cluster_id in df_pca["Cluster"].unique():
    plt.scatter(
        df_pca[df_pca["Cluster"] == cluster_id]["PC1"],
        df_pca[df_pca["Cluster"] == cluster_id]["PC2"],
        label=f"Cluster {cluster_id}",
        alpha=0.5
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clusters KMeans depois do PCA")
plt.legend()
plt.show()

# Exercicio 25
print("\n--- Exercício 25 ---")
df = df_base.copy()
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.set_index("Datetime")
df = df[['Global_active_power', 'Voltage']].dropna()
X = df[['Voltage']].values
y = df['Global_active_power'].values
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)
rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)
rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
print(f"RMSE Linear: {rmse_linear:.4f}")
print(f"RMSE Polinomial (grau 2): {rmse_poly:.4f}")

# Exercicio 26
print("\n--- Exercício 26 ---")

# Base única (imutável)
df_base2 = pd.read_csv(
    'energydata_complete.csv',
    sep=',',
    na_values=['?'],
    low_memory=False
)
df = df_base2.copy()
print(df.info())
print(df.describe())

# Exercicio 27
print("\n--- Exercício 27 ---")
df = df_base2.copy()
plt.figure(figsize=(8, 5))
plt.hist(df["Appliances"], bins=30, color="skyblue", edgecolor="black")
plt.title("Histograma de Appliances")
plt.xlabel("Consumo de Energia (Wh)")
plt.ylabel("Frequência")
plt.grid(alpha=0.3)
plt.show()
df["date"] = pd.to_datetime(df["date"])
plt.figure(figsize=(20, 8))
plt.plot(df['date'], df["Appliances"], color="orange", linewidth=1)
plt.title("Série temporal - Appliances")
plt.xlabel("Data")
plt.ylabel("Consumo de Energia (Wh)")
plt.grid(alpha=0.3)
plt.show()

# Exercicio 28
print("\n--- Exercício 28 ---")
df = df_base2.copy()
cols = ["Appliances", 'T1', 'T2', 'T3', 'T_out', 'RH_1', 'RH_2', 'RH_3']
df_corr = df_base2[cols].corr()
print(df_corr)

# Exercicio 29
print("\n--- Exercício 29 ---")
df = df_base2.copy()
df_numerico = df.select_dtypes(include=["number"])
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_numerico)
print(df_scaled)

df_scaled = pd.DataFrame(df_scaled, columns=df_numerico.columns, index=df_numerico.index)

# Exercicio 30
print("\n--- Exercício 30 ---")
new_df = df_scaled.dropna()
scaler = StandardScaler()
df_scaled = scaler.fit_transform(new_df)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
plt.figure(figsize=(8, 5))
plt.scatter(df_pca[:, 0], df_pca[:, 1], alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - 2 Componentes Principais")
plt.show()

# Exercicio 31
print("\n--- Exercício 31 ---")
df = df_base2.copy()
X = df.drop(columns=["Appliances", "date"]).select_dtypes(include="number")
y = df["Appliances"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse_lr = math.sqrt(mse)
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE (Linear): {rmse_lr:.4f}")

# Exercicio 32
print("\n--- Exercício 32 ---")
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred_rf)
rmse_rf = math.sqrt(mse)
print(f"RMSE (Random Forest): {rmse_rf:.4f}")
print("Comparação: Linear vs Random Forest")
print(f"RMSE Linear: {rmse_lr:.4f} | RMSE RF: {rmse_rf:.4f}")

# Exercicio 33
print("\n--- Exercício 33 ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
for k in [3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)
    df[f"cluster_{k}"] = clusters
    print(f"\n--- K={k} ---")
    print(df.groupby(f"cluster_{k}")["Appliances"].describe())

# Exercicio 34
print("\n--- Exercício 34 ---")
median_appliances = y_train.median()
y_train_bin = (y_train > median_appliances).astype(int)
y_test_bin = (y_test > median_appliances).astype(int)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
log_reg = LogisticRegression(max_iter=1000, random_state=0)
log_reg.fit(X_train_scaled, y_train_bin)
y_pred_log = log_reg.predict(X_test_scaled)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)
rf_clf.fit(X_train, y_train_bin)
y_pred_rf_clf = rf_clf.predict(X_test)
print("Classificador treinado com sucesso")

# Exercicio 35
print("\n--- Exercício 35 ---")
def avaliar_modelo(nome, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n{nome}")
    print("Matriz de confusão:")
    print(cm)
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Sensibilidade: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    tn, fp, fn, tp = cm.ravel()
    recall_baixo = tn / (tn + fp) if (tn+fp) else 0
    recall_alto = tp / (tp + fn) if (tp+fn) else 0
    if recall_alto < recall_baixo:
        print("→ O modelo erra mais para ALTO consumo")
    elif recall_baixo < recall_alto:
        print("→ O modelo erra mais para BAIXO consumo")
    else:
        print("→ O modelo erra igualmente para as duas classes")
avaliar_modelo("Regressão Logística", y_test_bin, y_pred_log)
avaliar_modelo("Random Forest Classifier", y_test_bin, y_pred_rf_clf)
