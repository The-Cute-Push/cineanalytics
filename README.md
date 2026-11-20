# CineAnalytics

Aplicativo acadêmico em Flask para análise de preferências e recomendação de filmes. Compara Árvore de Decisão, Random Forest e SVM por usuário e usa K-means para selecionar candidatos do catálogo MovieLens.

## Dados

Baixe o [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) e extraia `movies.csv` e `ratings.csv` para a mesma pasta de `app.py`. A aplicação carrega a base completa em memória ao iniciar.

## Executar no Windows

Use Python 3.12. No PowerShell, na pasta do repositório, execute

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Abra [http://127.0.0.1:3000](http://127.0.0.1:3000). O painel abre com o usuário #28.

## Capas dos filmes

O painel funciona com cartões de iniciais por padrão. Para habilitar as capas, configure `TMDB_API_KEY` no mesmo terminal antes de iniciar o aplicativo.

```powershell
$env:TMDB_API_KEY = 'sua-chave-do-tmdb'
.\.venv\Scripts\python.exe app.py
```
