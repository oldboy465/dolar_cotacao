#%%

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.formula.api import ols
#%%
# Função para coletar os dados do Yahoo Finance
def coletar_dados(tickers, start_date, end_date):
    dados = {}
    for ticker in tickers:
        dados[ticker] = yf.download(ticker, start=start_date, end=end_date)['Close']
    return pd.DataFrame(dados)
#%%
# Configuração dos tickers e período
tickers = {
    "petro": "PBR",
    "volks": "VWAGY",
    "dolar": "BRL=X",
    "euro": "EURUSD=X",
    "apple": "AAPL",
    "amazon": "AMZN",
    "microsoft": "MSFT",
    "coca_cola": "KO",
     }
#%%
start_date = "2024-01-01"
end_date = "2024-11-21"

# Coleta e tratamento dos dados
dados_brutos = coletar_dados(tickers.values(), start_date, end_date)
dados_brutos.columns = tickers.keys()
dados_brutos.dropna(inplace=True)

# Escalonamento dos dados
scaler = MinMaxScaler()
dados_escalados = pd.DataFrame(scaler.fit_transform(dados_brutos), columns=dados_brutos.columns, index=dados_brutos.index)
#%%
# Gráfico das variáveis escaladas
plt.figure(figsize=(16, 9))
for coluna in dados_escalados.columns:
    plt.plot(dados_escalados.index, dados_escalados[coluna], label=coluna)
plt.title("Variáveis Escaladas (0 a 1)")
plt.xlabel("Data")
plt.ylabel("Valor Escalado")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()
#%%
# Fórmula da regressão (dólar como variável dependente)
formula = "dolar ~ " + " + ".join([coluna for coluna in dados_escalados.columns if coluna != "dolar"])

# Criação do modelo de regressão generalizado
modelo = ols(formula=formula, data=dados_escalados).fit()

# Visualização das fórmulas e resultados do modelo
print("Resumo do Modelo de Regressão:")
print(modelo.summary())

# %%
