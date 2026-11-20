import os
import re
import json
import math
import urllib.request
import urllib.parse
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, silhouette_score

app = Flask(__name__)

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
poster_cache = {}

def obter_poster_tmdb(titulo_completo):
    if not TMDB_API_KEY:
        return None

    if titulo_completo in poster_cache:
        return poster_cache[titulo_completo]

    match = re.search(r'\s*\((\d{4})\)', titulo_completo)
    ano = match.group(1) if match else None
    nome = re.sub(r'\s*\(\d{4}\)', '', titulo_completo).strip()

    if ', The' in nome:
        nome = 'The ' + nome.replace(', The', '')
    elif ', A' in nome:
        nome = 'A ' + nome.replace(', A', '')
    elif ', An' in nome:
        nome = 'An ' + nome.replace(', An', '')
    nome = nome.strip()

    query = urllib.parse.quote(nome)
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
    if ano:
        url += f"&year={ano}"

    poster_url = None
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
            if data.get('results') and data['results'][0].get('poster_path'):
                poster_url = f"https://image.tmdb.org/t/p/w500{data['results'][0]['poster_path']}"
    except Exception:
        poster_url = None

    poster_cache[titulo_completo] = poster_url
    return poster_url

print("Carregando base de filmes e estatísticas...")
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'])

stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
stats.columns = ['movieId', 'nota_media', 'total_votos']

df_catalogo = pd.merge(movies, stats, on='movieId', how='inner')
generos_catalogo = df_catalogo['genres'].str.get_dummies(sep='|')
X_catalogo = pd.concat([df_catalogo[['nota_media', 'total_votos']], generos_catalogo], axis=1)

print("Executando K-Means no catálogo...")
scaler_kmeans = StandardScaler()
X_catalogo_scaled = scaler_kmeans.fit_transform(X_catalogo)

kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
df_catalogo['cluster'] = kmeans.fit_predict(X_catalogo_scaled)

silhueta = silhouette_score(X_catalogo_scaled, df_catalogo['cluster'], sample_size=5000, random_state=42)

clusters_info = []
contagem = df_catalogo['cluster'].value_counts().sort_index()
for c_id, qtd in contagem.items():
    grupo = df_catalogo[df_catalogo['cluster'] == c_id]
    genero_principal = generos_catalogo.loc[grupo.index].mean().idxmax()
    votos_med = grupo['total_votos'].mean()
    
    if votos_med > 10000:
        rotulo = f"Super Clássicos / Blockbusters ({genero_principal})"
    elif c_id == 7:
        rotulo = "Animation (Curta-Metragem)"
    elif c_id == 17:
        rotulo = "Animation (Longa-Metragem)"
    else:
        rotulo = str(genero_principal)

    clusters_info.append({
        'id': int(c_id),
        'qtd': int(qtd),
        'genero_dominante': rotulo
    })

print(f"Dados prontos! Silhouette Score K-Means: {silhueta:.3f}")

top_users_sugestoes = [28, 65, 354]

def processar_usuario(user_id):
    notas_user = ratings[ratings['userId'] == user_id][['movieId', 'rating']]
    total_avaliacoes = len(notas_user)
    
    if total_avaliacoes < 10:
        return None, f"O Usuário #{user_id} possui poucas avaliações ({total_avaliacoes}). São necessárias pelo menos 10 para treinar os modelos."

    df_u = pd.merge(notas_user, movies, on='movieId')
    df_u = pd.merge(df_u, stats, on='movieId')
    
    if len(df_u) < 10:
        return None, f"Filmes insuficientes encontrados no catálogo para o Usuário #{user_id}."

    generos_u = df_u['genres'].str.get_dummies(sep='|')
    for col in generos_catalogo.columns:
        if col not in generos_u.columns:
            generos_u[col] = 0
    generos_u = generos_u[generos_catalogo.columns]

    X = pd.concat([df_u[['nota_media', 'total_votos']], generos_u], axis=1)
    
    media_pessoal = float(df_u['rating'].mean())
    y = (df_u['rating'] > media_pessoal).astype(int)
    
    if y.nunique() < 2:
        return None, f"Todas as avaliações do Usuário #{user_id} caíram na mesma classe em relação à sua média."

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    arvore = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
    arvore.fit(X_train, y_train)
    prev_dt = arvore.predict(X_test)
    dt_acc = float(accuracy_score(y_test, prev_dt) * 100)
    dt_prec = float(precision_score(y_test, prev_dt, zero_division=0) * 100)
    dt_rec = float(recall_score(y_test, prev_dt, zero_division=0) * 100)
    dt_rules = export_text(arvore, feature_names=list(X.columns), max_depth=4)

    floresta = RandomForestClassifier(n_estimators=100, random_state=42)
    floresta.fit(X_train, y_train)
    prev_rf = floresta.predict(X_test)
    rf_acc = float(accuracy_score(y_test, prev_rf) * 100)
    rf_prec = float(precision_score(y_test, prev_rf, zero_division=0) * 100)
    rf_rec = float(recall_score(y_test, prev_rf, zero_division=0) * 100)

    importancias = pd.Series(floresta.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_fatores = []
    for nome, peso in importancias.head(5).items():
        top_fatores.append({
            'nome': nome,
            'peso': float(peso * 100),
            'peso_formatado': f"{peso * 100:.2f}%"
        })

    scaler_svm = StandardScaler()
    X_train_scaled = scaler_svm.fit_transform(X_train)
    X_test_scaled = scaler_svm.transform(X_test)

    modelo_svm = SVC(kernel='rbf', random_state=42)
    modelo_svm.fit(X_train_scaled, y_train)
    prev_svm = modelo_svm.predict(X_test_scaled)
    svm_acc = float(accuracy_score(y_test, prev_svm) * 100)
    svm_prec = float(precision_score(y_test, prev_svm, zero_division=0) * 100)
    svm_rec = float(recall_score(y_test, prev_svm, zero_division=0) * 100)

    df_user_cat = pd.merge(notas_user, df_catalogo, on='movieId')
    
    grp_clusters = df_user_cat.groupby('cluster').agg(
        qtd=('rating', 'count'),
        media_cluster=('rating', 'mean'),
        favs=('rating', lambda s: (s > media_pessoal).sum())
    )
    
    grp_validos = grp_clusters[grp_clusters['qtd'] >= 2]
    
    nichos_tematicos = grp_validos[grp_validos.index != 14].copy()
    
    if len(nichos_tematicos) > 0:
        nichos_tematicos['score'] = (nichos_tematicos['media_cluster'] - media_pessoal) * nichos_tematicos['favs'].apply(lambda x: math.log1p(max(0, x)))
        cluster_top = int(nichos_tematicos.sort_values(by=['score', 'media_cluster'], ascending=[False, False]).index[0])
        qtd_favoritos_cluster = int(nichos_tematicos.loc[cluster_top, 'favs'])
    elif len(grp_validos) > 0:
        cluster_top = int(grp_validos.sort_values(by='media_cluster', ascending=False).index[0])
        qtd_favoritos_cluster = int(grp_validos.loc[cluster_top, 'favs'])
    else:
        cluster_top = 0
        qtd_favoritos_cluster = 0

    nao_vistos = df_catalogo[
        (~df_catalogo['movieId'].isin(notas_user['movieId'])) &
        (df_catalogo['cluster'] == cluster_top) &
        (df_catalogo['total_votos'] >= 200)
    ].sort_values(by='nota_media', ascending=False)

    recomendacoes = []
    for _, row in nao_vistos.head(5).iterrows():
        recomendacoes.append({
            'title': row['title'],
            'genres': row['genres'],
            'nota_media': f"{row['nota_media']:.2f}"
        })

    modelos_dict = {
        'Árvore de Decisão': {'acc': dt_acc, 'prec': dt_prec, 'rec': dt_rec},
        'Random Forest': {'acc': rf_acc, 'prec': rf_prec, 'rec': rf_rec},
        'SVM (RBF)': {'acc': svm_acc, 'prec': svm_prec, 'rec': svm_rec}
    }

    melhor_acc_nome, melhor_acc_val = max(modelos_dict.items(), key=lambda item: item[1]['acc'])
    melhor_prec_nome, melhor_prec_val = max(modelos_dict.items(), key=lambda item: item[1]['prec'])
    melhor_rec_nome, melhor_rec_val = max(modelos_dict.items(), key=lambda item: item[1]['rec'])

    destaques = {
        'acc': {'valor': f"{melhor_acc_val['acc']:.1f}%", 'modelo': melhor_acc_nome},
        'prec': {'valor': f"{melhor_prec_val['prec']:.1f}%", 'modelo': melhor_prec_nome},
        'rec': {'valor': f"{melhor_rec_val['rec']:.1f}%", 'modelo': melhor_rec_nome}
    }

    top10_df = df_u.sort_values(by=['rating', 'nota_media', 'total_votos'], ascending=[False, False, False]).head(10)
    filmes_mais_assistidos = []
    
    cores_capa = [
        "from-indigo-600 to-blue-800",
        "from-rose-600 to-red-800",
        "from-emerald-600 to-teal-800",
        "from-amber-600 to-orange-800",
        "from-purple-600 to-indigo-800",
        "from-cyan-600 to-blue-700",
        "from-fuchsia-600 to-pink-800",
        "from-violet-600 to-purple-800",
        "from-teal-600 to-emerald-800",
        "from-blue-600 to-cyan-800"
    ]
    
    for idx, (_, row) in enumerate(top10_df.iterrows()):
        title_full = row['title']
        match = re.search(r'\s*\((\d{4})\)', title_full)
        ano = match.group(1) if match else 'N/A'
        nome_limpo = re.sub(r'\s*\(\d{4}\)', '', title_full).strip()
        
        palavras = [p for p in re.split(r'[^A-Za-z0-9]', nome_limpo) if p]
        iniciais = ''.join([p[0].upper() for p in palavras[:2]]) if palavras else "MV"
        
        poster_url = obter_poster_tmdb(title_full)

        filmes_mais_assistidos.append({
            'rank': idx + 1,
            'title': nome_limpo,
            'title_full': title_full,
            'ano': ano,
            'genres': row['genres'],
            'nota_usuario': f"{float(row['rating']):.1f}",
            'nota_media_geral': f"{float(row['nota_media']):.2f}",
            'total_votos': f"{int(row['total_votos']):,}".replace(',', '.'),
            'iniciais': iniciais,
            'poster_url': poster_url,
            'cor': cores_capa[idx % len(cores_capa)]
        })

    dados = {
        'user_id': user_id,
        'total_avaliacoes': total_avaliacoes,
        'media_pessoal': f"{media_pessoal:.2f}",
        'silhouette': f"{silhueta:.3f}",
        'cluster_top': cluster_top,
        'qtd_favoritos_cluster': qtd_favoritos_cluster,
        'destaques': destaques,
        'filmes_mais_assistidos': filmes_mais_assistidos,
        'metricas': {
            'dt': {'acc': round(dt_acc, 1), 'prec': round(dt_prec, 1), 'rec': round(dt_rec, 1)},
            'rf': {'acc': round(rf_acc, 1), 'prec': round(rf_prec, 1), 'rec': round(rf_rec, 1)},
            'svm': {'acc': round(svm_acc, 1), 'prec': round(svm_prec, 1), 'rec': round(svm_rec, 1)}
        },
        'top_fatores': top_fatores,
        'dt_rules': dt_rules,
        'recomendacoes': recomendacoes,
        'clusters_info': clusters_info
    }
    return dados, None


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CineMatch</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
    code, pre { font-family: 'JetBrains Mono', monospace; }
    .custom-scrollbar::-webkit-scrollbar { width: 6px; height: 6px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: #1e293b; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">

  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-500/30">
          <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
          </svg>
        </div>
        <div>
          <h1 class="font-bold text-lg text-white">CineMatch</h1>
          <p class="text-xs text-slate-400">4 Modelos de Machine Learning Integrados</p>
        </div>
      </div>
      
      <form method="GET" action="/" class="flex items-center space-x-2">
        <label for="userIdInput" class="text-xs text-slate-300 font-medium hidden sm:inline">Usuário ID:</label>
        <input 
          type="number" 
          id="userIdInput" 
          name="user_id" 
          value="{{ data.user_id }}" 
          class="w-24 sm:w-32 bg-slate-800 border border-slate-700 text-white text-xs rounded-lg px-2.5 py-1.5 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
          placeholder="Ex: 28" 
          required
        />
        <button type="submit" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-3 py-1.5 rounded-lg transition shadow-md shadow-indigo-600/30">
          Carregar
        </button>
      </form>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
    
    {% if error %}
    <div class="bg-rose-500/10 border border-rose-500/30 text-rose-300 p-4 rounded-xl text-sm flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <svg class="w-5 h-5 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
        </svg>
        <span>{{ error }}</span>
      </div>
      <a href="/?user_id=28" class="text-xs underline hover:text-white">Voltar ao Usuário #28</a>
    </div>
    {% endif %}

    <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center space-x-2">
        <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Atalhos de Usuários Ativos:</span>
        <div class="flex flex-wrap gap-1.5">
          {% for u in suggestions %}
          <a href="/?user_id={{ u }}" class="text-xs px-2.5 py-1 rounded-md border transition {% if u == data.user_id %}bg-indigo-600 text-white border-indigo-500 font-bold{% else %}bg-slate-800/80 hover:bg-slate-700 text-slate-300 border-slate-700{% endif %}">
            #{{ u }}
          </a>
          {% endfor %}
        </div>
      </div>
      <div class="text-xs text-slate-400">
        Avaliações do Usuário Atual: <strong class="text-white">{{ data.total_avaliacoes }} filmes</strong> | Média Pessoal: <strong class="text-amber-400">{{ data.media_pessoal }} ★</strong>
      </div>
    </div>

    <section class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6 pb-4 border-b border-slate-800">
        <div>
          <span class="text-xs font-semibold uppercase tracking-wider text-indigo-400">Histórico de Visualizações</span>
          <h3 class="text-lg sm:text-xl font-bold text-white flex items-center gap-2 mt-0.5">
            <svg class="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            Top 10 Melhores Filmes Avaliados pelo Usuário #{{ data.user_id }}
          </h3>
        </div>
        <span class="text-xs text-slate-400 font-medium">Ordenados pela maior nota pessoal atribuída</span>
      </div>

      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {% for filme in data.filmes_mais_assistidos %}
        <div class="bg-slate-950/80 rounded-xl border border-slate-800 overflow-hidden flex flex-col justify-between hover:border-indigo-500/50 hover:shadow-lg hover:shadow-indigo-500/10 transition group">
          <div>
            <div class="relative h-56 bg-slate-900 overflow-hidden flex flex-col justify-between">
              {% if filme.poster_url %}
              <img 
                src="{{ filme.poster_url }}" 
                alt="{{ filme.title }}" 
                class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                loading="lazy"
              />
              {% else %}
              <div class="w-full h-full bg-gradient-to-br {{ filme.cor }} flex flex-col justify-between p-3.5 shadow-inner">
                <div class="text-center my-auto z-10">
                  <span class="text-3xl font-black text-white/30 tracking-widest font-mono select-none">
                    {{ filme.iniciais }}
                  </span>
                </div>
                <div class="flex items-center justify-between text-[11px] text-white/90 z-10">
                  <span class="font-medium bg-black/40 px-2 py-0.5 rounded">{{ filme.ano }}</span>
                  <span class="text-[10px] text-white/70">Média: {{ filme.nota_media_geral }}</span>
                </div>
              </div>
              {% endif %}

              <div class="absolute top-2 left-2 right-2 flex items-center justify-between z-20 pointer-events-none">
                <span class="text-[11px] font-mono font-bold bg-slate-950/85 text-slate-200 px-2 py-0.5 rounded backdrop-blur border border-white/10 shadow">
                  #{{ filme.rank }}
                </span>
                <span class="text-[11px] font-bold bg-amber-400 text-slate-950 px-2 py-0.5 rounded-full shadow">
                  ★ {{ filme.nota_usuario }}
                </span>
              </div>
            </div>

            <div class="p-3.5">
              <h4 class="font-bold text-xs sm:text-sm text-white line-clamp-2 group-hover:text-indigo-300 transition" title="{{ filme.title_full }}">
                {{ filme.title }}
              </h4>
              <p class="text-[11px] text-slate-400 mt-1 line-clamp-1" title="{{ filme.genres }}">
                {{ filme.genres }}
              </p>
            </div>
          </div>

          <div class="px-3.5 pb-3 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-500">
            <span>Público global:</span>
            <span class="font-mono text-slate-300 font-semibold">{{ filme.total_votos }} votos</span>
          </div>
        </div>
        {% endfor %}
      </div>
    </section>

    <section class="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xl">
      <div class="max-w-3xl space-y-3">
        <span class="text-xs font-semibold uppercase tracking-wider text-indigo-400">Painel Unificado de Machine Learning</span>
        <h2 class="text-2xl sm:text-3xl font-extrabold text-white">Análise Preditiva para o Usuário #{{ data.user_id }}</h2>
        <p class="text-slate-300 text-sm sm:text-base leading-relaxed">
          Dados processados dinamicamente através dos 4 algoritmos: <strong>Árvore de Decisão</strong>, <strong>Random Forest</strong>, <strong>SVM (RBF)</strong> e <strong>K-Means</strong>.
        </p>
      </div>

      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-8 pt-6 border-t border-slate-800/80">
        <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-800">
          <p class="text-xs text-slate-400 font-medium">Melhor Acurácia</p>
          <p class="text-2xl font-bold text-emerald-400 mt-1">{{ data.destaques.acc.valor }}</p>
          <p class="text-xs text-slate-400 mt-0.5 font-medium">{{ data.destaques.acc.modelo }}</p>
        </div>
        <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-800">
          <p class="text-xs text-slate-400 font-medium">Melhor Precisão</p>
          <p class="text-2xl font-bold text-indigo-400 mt-1">{{ data.destaques.prec.valor }}</p>
          <p class="text-xs text-slate-400 mt-0.5 font-medium">{{ data.destaques.prec.modelo }}</p>
        </div>
        <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-800">
          <p class="text-xs text-slate-400 font-medium">Melhor Recall</p>
          <p class="text-2xl font-bold text-purple-400 mt-1">{{ data.destaques.rec.valor }}</p>
          <p class="text-xs text-slate-400 mt-0.5 font-medium">{{ data.destaques.rec.modelo }}</p>
        </div>
        <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-800">
          <p class="text-xs text-slate-400 font-medium">Silhouette Score (K-Means)</p>
          <p class="text-2xl font-bold text-amber-400 mt-1">{{ data.silhouette }}</p>
          <p class="text-xs text-slate-500 mt-0.5">20 Nichos Identificados</p>
        </div>
      </div>
    </section>

    <section class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      <div class="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between mb-4">
            <div>
              <h3 class="text-lg font-bold text-white flex items-center gap-2">
                <svg class="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Comparativo de Modelos Supervisionados
              </h3>
              <p class="text-xs text-slate-400">Classificação binária (Gosta / Não Gosta com base na média do Usuário #{{ data.user_id }})</p>
            </div>
          </div>
          <div class="relative h-64 sm:h-72 w-full">
            <canvas id="modelsChart"></canvas>
          </div>
        </div>
        <div class="grid grid-cols-3 gap-2 mt-4 pt-4 border-t border-slate-800 text-center text-xs">
          <div>
            <span class="text-slate-400 block">Árvore de Decisão</span>
            <span class="font-semibold text-slate-200">Acurácia: {{ data.metricas.dt.acc }}%</span>
          </div>
          <div>
            <span class="text-slate-400 block">Random Forest</span>
            <span class="font-semibold text-emerald-400">Acurácia: {{ data.metricas.rf.acc }}%</span>
          </div>
          <div>
            <span class="text-slate-400 block">SVM (RBF)</span>
            <span class="font-semibold text-purple-400">Acurácia: {{ data.metricas.svm.acc }}%</span>
          </div>
        </div>
      </div>

      <div class="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
        <div>
          <h3 class="text-lg font-bold text-white mb-2">Tabela de Métricas Detalhadas</h3>
          <p class="text-xs text-slate-400 mb-4">Divisão de dados 80% treino / 20% teste para Usuário #{{ data.user_id }}.</p>
          
          <div class="overflow-x-auto">
            <table class="w-full text-left text-sm">
              <thead class="text-xs text-slate-400 uppercase bg-slate-800/60 border-b border-slate-800">
                <tr>
                  <th class="py-3 px-3">Modelo</th>
                  <th class="py-3 px-3 text-center">Acurácia</th>
                  <th class="py-3 px-3 text-center">Precisão</th>
                  <th class="py-3 px-3 text-center">Recall</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-slate-800 text-slate-300">
                <tr class="hover:bg-slate-800/40 transition">
                  <td class="py-3.5 px-3 font-medium text-white flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                    Árvore de Decisão
                  </td>
                  <td class="py-3.5 px-3 text-center font-mono {% if data.metricas.dt.acc >= data.metricas.rf.acc and data.metricas.dt.acc >= data.metricas.svm.acc %}font-bold text-amber-500{% endif %}">
                    {{ data.metricas.dt.acc }}%
                  </td>
                  <td class="py-3.5 px-3 text-center font-mono {% if data.metricas.dt.prec >= data.metricas.rf.prec and data.metricas.dt.prec >= data.metricas.svm.prec %}font-bold text-amber-500{% endif %}">
                    {{ data.metricas.dt.prec }}%
                  </td>
                  <td class="py-3.5 px-3 text-center font-mono {% if data.metricas.dt.rec >= data.metricas.rf.rec and data.metricas.dt.rec >= data.metricas.svm.rec %}font-bold text-amber-500{% endif %}">
                    {{ data.metricas.dt.rec }}%
                  </td>
                </tr>
                <tr class="hover:bg-slate-800/40 transition">
                  <td class="py-3.5 px-3 font-medium text-white flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                    Random Forest
                  </td>
                  <td class="py-3.5 px-3 text-center font-mono {% if data.metricas.rf.acc >= data.metricas.dt.acc and data.metricas.rf.acc >= data.metricas.svm.acc %}font-bold text-emerald-400{% endif %}">
                    {{ data.metricas.rf.acc }}%
                  </td>
                  <td class="py-3.5 px-3 text-center font-mono {% if data.metricas.rf.prec >= data.metricas.dt.prec and data.metricas.rf.prec >= data.metricas.svm.prec %}font-bold text-emerald-400{% endif %}">
                    {{ data.metricas.rf.prec }}%
                  </td>
                  <td class="py-3.5 px-3 text-center font-mono {% if data.metricas.rf.rec >= data.metricas.dt.rec and data.metricas.rf.rec >= data.metricas.svm.rec %}font-bold text-emerald-400{% endif %}">
                    {{ data.metricas.rf.rec }}%
                  </td>
                </tr>
                <tr class="hover:bg-slate-800/40 transition">
                  <td class="py-3.5 px-3 font-medium text-white flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-purple-500"></span>
                    SVM (RBF)
                  </td>
                  <td class="py-3.5 px-3 text-center font-mono {% if data.metricas.svm.acc >= data.metricas.dt.acc and data.metricas.svm.acc >= data.metricas.rf.acc %}font-bold text-purple-400{% endif %}">
                    {{ data.metricas.svm.acc }}%
                  </td>
                  <td class="py-3.5 px-3 text-center font-mono {% if data.metricas.svm.prec >= data.metricas.dt.prec and data.metricas.svm.prec >= data.metricas.rf.prec %}font-bold text-purple-400{% endif %}">
                    {{ data.metricas.svm.prec }}%
                  </td>
                  <td class="py-3.5 px-3 text-center font-mono {% if data.metricas.svm.rec >= data.metricas.dt.rec and data.metricas.svm.rec >= data.metricas.rf.rec %}font-bold text-purple-400{% endif %}">
                    {{ data.metricas.svm.rec }}%
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="mt-6 bg-slate-950/60 rounded-xl p-4 border border-slate-800/80 text-xs text-slate-400 space-y-2">
          <p><strong class="text-slate-200">Acurácia:</strong> Percentual de acertos gerais em prever se o usuário aprova ou não o filme.</p>
          <p><strong class="text-slate-200">Precisão vs Recall:</strong> Equilíbrio entre não recomendar filmes indesejados e não deixar filmes bons de fora.</p>
        </div>
      </div>

    </section>

    <section class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <div class="max-w-3xl">
        <h3 class="text-lg font-bold text-white flex items-center gap-2 mb-1">
          <svg class="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Top Fatores Decisivos (Random Forest)
        </h3>
        <p class="text-xs text-slate-400 mb-6">Importância calculada das features no gosto do Usuário #{{ data.user_id }}</p>
      </div>
      
      <div class="space-y-4">
        {% for fator in data.top_fatores %}
        <div>
          <div class="flex justify-between text-xs mb-1">
            <span class="font-medium text-slate-200">{{ loop.index }}. {{ fator.nome }}</span>
            <span class="font-mono text-emerald-400 font-semibold">{{ fator.peso_formatado }}</span>
          </div>
          <div class="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
            <div class="bg-gradient-to-r from-emerald-500 to-teal-400 h-2.5 rounded-full" style="width: {{ fator.peso }}%"></div>
          </div>
        </div>
        {% endfor %}
      </div>

      <div class="mt-6 p-4 rounded-xl bg-slate-950/70 border border-slate-800 text-xs text-slate-400">
        <p><span class="text-emerald-400 font-bold">Resumo:</span> Os dois primeiros fatores agregam o maior peso estatístico para orientar as recomendações deste perfil.</p>
      </div>
    </section>

    <section class="bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xl space-y-6">
      
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <span class="text-xs font-semibold uppercase tracking-wider text-amber-400">Agrupamento Não Supervisionado</span>
          <h3 class="text-xl sm:text-2xl font-bold text-white">K-Means: Cluster Favorito & Recomendações</h3>
          <p class="text-xs sm:text-sm text-slate-400 mt-1">
            Segmentação de filmes similares por nota, volume e gênero. Silhouette Score: <span class="text-amber-400 font-mono font-bold">{{ data.silhouette }}</span>.
          </p>
        </div>

        <div class="bg-amber-500/10 border border-amber-500/30 rounded-xl px-4 py-3 flex items-center gap-3">
          <div class="w-9 h-9 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold">
            #{{ data.cluster_top }}
          </div>
          <div>
            <p class="text-xs text-amber-200/80 font-medium">Cluster Favorito do Usuário #{{ data.user_id }}</p>
            <p class="text-sm font-bold text-amber-400">{{ data.qtd_favoritos_cluster }} Filmes Favoritos Neste Nicho</p>
          </div>
        </div>
      </div>

      <div>
        <h4 class="text-base font-bold text-white flex items-center gap-2 mb-4">
          <svg class="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
          </svg>
          Top 5 Filmes Recomendados (Ainda Não Vistos pelo Usuário #{{ data.user_id }})
        </h4>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          {% for rec in data.recomendacoes %}
          <div class="bg-slate-950/80 border border-slate-800 rounded-xl p-4 flex flex-col justify-between hover:border-amber-500/40 transition">
            <div>
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 font-mono">Rank #{{ loop.index }}</span>
                <span class="text-xs font-mono font-bold text-emerald-400">★ {{ rec.nota_media }}</span>
              </div>
              <h5 class="font-bold text-sm text-white line-clamp-2">{{ rec.title }}</h5>
            </div>
            <div class="mt-4 pt-3 border-t border-slate-800/80">
              <span class="inline-block text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">{{ rec.genres }}</span>
            </div>
          </div>
          {% endfor %}
        </div>
      </div>

      <div class="pt-6 border-t border-slate-800">
        <div class="flex items-center justify-between mb-4">
          <h4 class="text-base font-bold text-white">Distribuição dos 20 Clusters no Catálogo Completo</h4>
          <span class="text-xs text-slate-400">Total: 84.433 filmes indexados</span>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 max-h-96 overflow-y-auto pr-2 custom-scrollbar">
          {% for c in data.clusters_info %}
          <div class="p-3 rounded-lg border transition {% if c.id == data.cluster_top %}bg-amber-500/10 border-2 border-amber-500/60 shadow-lg shadow-amber-500/10{% else %}bg-slate-950/60 border-slate-800{% endif %} flex items-center justify-between">
            <div>
              <div class="flex items-center gap-1.5">
                <p class="text-xs font-mono font-bold {% if c.id == data.cluster_top %}text-amber-300{% else %}text-slate-300{% endif %}">Cluster #{{ '%02d' % c.id }}</p>
                {% if c.id == data.cluster_top %}
                <span class="text-[10px] bg-amber-500 text-slate-950 font-bold px-1.5 rounded-full">FAVORITO</span>
                {% endif %}
              </div>
              <p class="text-xs {% if c.id == data.cluster_top %}text-amber-200{% else %}text-slate-400{% endif %}">{{ c.genero_dominante }}</p>
            </div>
            <span class="text-xs font-mono px-2 py-1 rounded {% if c.id == data.cluster_top %}bg-amber-500/20 text-amber-300 font-bold{% else %}bg-slate-800 text-slate-300{% endif %}">
              {{ c.qtd }} filmes
            </span>
          </div>
          {% endfor %}
        </div>
      </div>

    </section>

  </main>

  <footer class="border-t border-slate-800/80 mt-12 py-6 text-center text-xs text-slate-500">
    <p>CineAnalytics ML • Aplicação Flask Completa com Árvore de Decisão, Random Forest, SVM e K-Means.</p>
  </footer>

  <script>
    const ctx = document.getElementById('modelsChart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Árvore de Decisão', 'Random Forest', 'SVM (RBF)'],
        datasets: [
          {
            label: 'Acurácia (%)',
            data: [{{ data.metricas.dt.acc }}, {{ data.metricas.rf.acc }}, {{ data.metricas.svm.acc }}],
            backgroundColor: 'rgba(59, 130, 246, 0.8)',
            borderColor: 'rgb(59, 130, 246)',
            borderWidth: 1,
            borderRadius: 6
          },
          {
            label: 'Precisão (%)',
            data: [{{ data.metricas.dt.prec }}, {{ data.metricas.rf.prec }}, {{ data.metricas.svm.prec }}],
            backgroundColor: 'rgba(16, 185, 129, 0.8)',
            borderColor: 'rgb(16, 185, 129)',
            borderWidth: 1,
            borderRadius: 6
          },
          {
            label: 'Recall (%)',
            data: [{{ data.metricas.dt.rec }}, {{ data.metricas.rf.rec }}, {{ data.metricas.svm.rec }}],
            backgroundColor: 'rgba(168, 85, 247, 0.8)',
            borderColor: 'rgb(168, 85, 247)',
            borderWidth: 1,
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 }, usePointStyle: true, boxWidth: 8 }
          },
          tooltip: {
            backgroundColor: '#0f172a',
            borderColor: '#334155',
            borderWidth: 1,
            titleFont: { family: 'Inter', weight: 'bold' },
            bodyFont: { family: 'JetBrains Mono' },
            padding: 10
          }
        },
        scales: {
          y: {
            min: 50,
            max: 100,
            ticks: {
              color: '#64748b',
              callback: function(value) { return value + '%'; }
            },
            grid: { color: 'rgba(51, 65, 85, 0.4)' }
          },
          x: {
            ticks: { color: '#cbd5e1', font: { family: 'Inter', weight: 500 } },
            grid: { display: false }
          }
        }
      }
    });
  </script>
</body>
</html>
"""

cache_usuarios = {}

@app.route('/')
def index():
    user_id_param = request.args.get('user_id', 28)
    try:
        user_id = int(user_id_param)
    except ValueError:
        user_id = 28

    if user_id in cache_usuarios:
        dados = cache_usuarios[user_id]
        error = None
    else:
        dados, error = processar_usuario(user_id)
        if error:
            dados, _ = processar_usuario(28)
        else:
            cache_usuarios[user_id] = dados

    return render_template_string(
        HTML_TEMPLATE,
        data=dados,
        error=error,
        suggestions=top_users_sugestoes
    )

if __name__ == '__main__':
    print("Iniciando servidor CineAnalytics na porta 3000...")
    print("Acesse no navegador: http://127.0.0.1:3000")
    app.run(host='0.0.0.0', port=3000, debug=False)