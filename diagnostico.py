
import pandas as pd

# Carregamento dos dados
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# 1. VALORES AUSENTES
print("\n=== VALORES AUSENTES ===")

for nome, df in [("MOVIES", movies), ("RATINGS", ratings)]:
    print(f"\n{nome}")
    ausentes = df.isnull().sum()

    for coluna in df.columns:
        qtd = int(ausentes[coluna])
        percentual = qtd / len(df) * 100
        print(f"{coluna}: {qtd} ({percentual:.4f}%)")

# 2. OUTLIERS PELO MÉTODO IQR
print("\n=== OUTLIERS ===")

stats = ratings.groupby("movieId")["rating"].agg(
    nota_media="mean",
    total_votos="count"
).reset_index()

def analisar_outliers(df, coluna):
    q1 = df[coluna].quantile(0.25)
    q3 = df[coluna].quantile(0.75)
    iqr = q3 - q1

    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr

    outliers = df[
        (df[coluna] < limite_inf) |
        (df[coluna] > limite_sup)
    ]

    print(f"\n{coluna}")
    print(f"Limite inferior: {limite_inf:.4f}")
    print(f"Limite superior: {limite_sup:.4f}")
    print(f"Outliers: {len(outliers)}")
    print(f"Percentual: {len(outliers)/len(df)*100:.4f}%")

for coluna in ["rating", "nota_media", "total_votos"]:
    analisar_outliers(
        ratings if coluna == "rating" else stats,
        coluna
    )

# 3. DESBALANCEAMENTO POR USUÁRIO
print("\n=== DISTRIBUIÇÃO DAS CLASSES ===")

for user_id in [28, 175325, 17035, 55653,
                123465, 10202, 49305]:

    notas = ratings.loc[
        ratings["userId"] == user_id, "rating"
    ]

    if len(notas) < 10:
        print(f"\nUsuário {user_id}: avaliações insuficientes")
        continue

    media = notas.mean()
    classes = (notas > media).astype(int)
    distribuicao = classes.value_counts().sort_index()

    print(f"\nUsuário {user_id}")
    print(f"Classe 0: {distribuicao.get(0, 0)} "
          f"({(classes == 0).mean()*100:.2f}%)")
    print(f"Classe 1: {distribuicao.get(1, 0)} "
          f"({(classes == 1).mean()*100:.2f}%)")
