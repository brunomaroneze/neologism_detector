# train_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, csr_matrix # Importar csr_matrix para as features explícitas
import joblib # Para salvar/carregar modelos
import os
import re
import unicodedata # Para normalizar palavras, se o CSV tiver acentos e as regras forem sem acento
import numpy as np # Para criar arrays

# --- Configurações e Caminhos ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

CLASSIFIED_NEOLOGISMS_CSV = os.path.join(DATA_DIR, 'neologismos_classificados.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'neologism_classifier_model.pkl')
VECTORIZER_PATH = os.path.join(DATA_DIR, 'char_vectorizer.pkl')
EXPLICIT_FEATURE_NAMES_PATH = os.path.join(DATA_DIR, 'explicit_feature_names.pkl')
COMMON_PREFIXES_PATH = os.path.join(DATA_DIR, 'common_prefixes.pkl')
COMMON_SUFFIXES_PATH = os.path.join(DATA_DIR, 'common_suffixes.pkl')
FOREIGN_PATTERNS_PATH = os.path.join(DATA_DIR, 'foreign_patterns.pkl')


# --- Definição das Heurísticas (Listas de Prefixes, Sufixos, Padrões Estrangeiros) ---
# Essas listas serão salvas e carregadas no services.py
COMMON_PREFIXES = [
    'pré-', 'pós-', 'super-', 'mega-', 'macro-', 'micro-', 'anti-', 'auto-', 'co-', 'des-', 'in-', 're-', 'hiper-', 'inter-'
]

COMMON_SUFFIXES = [
    '-dade', '-dades', '-ção', '-ções', '-agem', '-agens', '-mento', '-mentos', '-ico', '-ica', '-icos', '-icas', '-izar', '-ismo', '-ismos', '-eiro', '-eira', '-eiras', '-eiras', '-vel', '-veis', '-ificar'
]

FOREIGN_PATTERNS = {
    'letters': ['k', 'w', 'y'],
    'start_patterns': ['st', 'sp', 'sl', 'sm', 'sf', 'sc', 'sk', 'sh', 'th', 'ph', 'ch'], # st- para startswith
    'end_patterns': ['q', 't', 'p', 'j', 'h', 'g', 'f', 'd', 'c', 'v', 'b', 'er', 'or', 'ar', 'ing', 'est'], # Pode precisar de ajuste fino
    'internal_patterns': ['zz', 'th', 'ph', 'mm', 'gg', 'tt', 'sh', 'zh', 'ck']
}
# As terminações para estrangeirismo são mais complexas, "er", "or", "ar" são comuns em português.
# O ML aprenderá o contexto, mas se for um problema, ajuste essas listas.

PORTUGUESE_ENDINGS = ['ão', 'ões'] # Forte indicativo de não ser estrangeirismo


# --- Função de Engenharia de Features ---
def create_features(word):
    """
    Cria um dicionário de features booleanas e numéricas para uma dada palavra.
    """
    features = {}
    
    # Normaliza a palavra para minúsculas e sem acentos para algumas regras
    # (importante que isso seja consistente com como as regras foram definidas)
    word_normalized = unicodedata.normalize('NFKD', word.lower()).encode('ascii', 'ignore').decode('utf-8')
    word_lower = word.lower() # Mantém a versão original em minúsculas para outras regras

    # Feature 1: Presença de hífen
    features['has_hyphen'] = 1 if '-' in word_lower else 0

    # Feature 2: Comprimento da palavra
    features['word_length'] = len(word_lower)

    # Feature 3: Proporção de vogais (pode ser útil para estrangeirismos)
    num_vowels = sum(1 for char in word_lower if char in 'aeiouáéíóúãõ')
    features['vowel_ratio'] = num_vowels / len(word_lower) if len(word_lower) > 0 else 0

    # Feature 4: Heurísticas de Derivação Prefixal
    for prefix in COMMON_PREFIXES:
        # Verifica se a palavra começa com o prefixo
        # Ex: 'pós-operação' -> has_prefix_pos- = 1
        features[f'has_prefix_{prefix.replace("-", "")}'] = 1 if word_lower.startswith(prefix) else 0

    # Feature 5: Heurísticas de Derivação Sufixal
    for suffix in COMMON_SUFFIXES:
        # Verifica se a palavra termina com o sufixo
        # Ex: 'bolsonarização' -> has_suffix_acao = 1 (se '-ção' for 'acao' normalizado)
        # É melhor comparar com a versão 'word_lower' e o sufixo exato.
        features[f'has_suffix_{suffix.replace("-", "")}'] = 1 if word_lower.endswith(suffix) else 0

    # Feature 6: Heurísticas de Estrangeirismo
    for letter in FOREIGN_PATTERNS['letters']:
        features[f'has_foreign_letter_{letter}'] = 1 if letter in word_lower else 0
    
    for pattern in FOREIGN_PATTERNS['start_patterns']:
        features[f'starts_foreign_{pattern}'] = 1 if word_lower.startswith(pattern) else 0

    for pattern in FOREIGN_PATTERNS['end_patterns']:
        features[f'ends_foreign_{pattern}'] = 1 if word_lower.endswith(pattern) else 0
        
    for pattern in FOREIGN_PATTERNS['internal_patterns']:
        features[f'has_internal_foreign_{pattern}'] = 1 if pattern in word_lower else 0

    # Feature 7: Heurística de NÃO ser Estrangeirismo
    for ending in PORTUGUESE_ENDINGS:
        features[f'ends_portuguese_{ending}'] = 1 if word_lower.endswith(ending) else 0

    # Para composição, podemos usar a presença de hífen (já coberta) ou tentar dividir em palavras conhecidas,
    # mas isso é mais complexo e talvez seja melhor para o ML inferir dos n-grams/outras features.
    
    return features


def train_classifier():
    print("Carregando dados de treinamento...")
    try:
        # Carregar o CSV sem cabeçalho e pegar todas as colunas como 'col0', 'col1', etc.
        df_raw = pd.read_csv(CLASSIFIED_NEOLOGISMS_CSV)

        # === NOVO: Selecionar apenas as colunas de interesse ===
        # Se a coluna 'neologismo' é a segunda (índice 1) e 'classificação' é a terceira (índice 2)
        # Ajuste os índices (0-based) conforme a posição REAL das colunas no seu CSV.
        # Por exemplo, se a sua lista é:
        # "data_hora", "neologismo", "classificação", "nome_classificador"
        # Então, o neologismo estaria no índice 1 e a classificação no índice 2.
        
        # Exemplo: df_raw[1] para neologismo, df_raw[2] para classificação
        df = df_raw[['neologismo', 'classificacao']].copy()
        df.columns = ['palavra', 'classe'] # Renomeia as colunas para facilitar o uso
    except FileNotFoundError:
        print(f"Erro: Arquivo {CLASSIFIED_NEOLOGISMS_CSV} não encontrado.")
        print("Certifique-se de que o CSV com neologismos classificados está na pasta 'data/'.")
        return
    except KeyError as e:
        print(f"Erro: As colunas especificadas não foram encontradas no CSV. Verifique os índices. Erro: {e}")
        print("Certifique-se de que o CSV não tem cabeçalho ou ajuste 'header=None' e os índices.")
        print(f"Colunas disponíveis no CSV: {list(df_raw.columns)}")
        return
    except Exception as e:
        print(f"Erro ao carregar ou processar o CSV: {e}")
        return

    # Limpeza e normalização básica
    df['palavra_limpa'] = df['palavra'].apply(lambda x: unicodedata.normalize('NFKD', x.lower()).encode('ascii', 'ignore').decode('utf-8'))
    df['classe'] = df['classe'].str.lower() # Garante que as classes sejam minúsculas

    print(f"Total de amostras carregadas: {len(df)}")
    print("Distribuição das classes:")
    print(df['classe'].value_counts())

    # --- 1. Engenharia de Features Explícitas (Heurísticas) ---
    print("\nCriando features explícitas (heurísticas)...")
    explicit_features_list = [create_features(word) for word in df['palavra']]
    explicit_feature_df = pd.DataFrame(explicit_features_list)
    
    # Preenche qualquer NaN que possa surgir se uma feature não foi gerada para uma palavra
    explicit_feature_df = explicit_feature_df.fillna(0) 

    # IMPORTANTE: Salvar os nomes das colunas de features explícitas para garantir a ordem na predição
    explicit_feature_names = list(explicit_feature_df.columns)
    joblib.dump(explicit_feature_names, EXPLICIT_FEATURE_NAMES_PATH)
    print(f"Nomes das features explícitas salvas em: {EXPLICIT_FEATURE_NAMES_PATH}")

    # Converte o DataFrame de features explícitas para uma matriz esparsa
    # (para poder combinar com TF-IDF que também é esparso)
    # Certifique-se que o DataFrame está com tipos numéricos, o .sparse.to_coo() precisa de um SparseDtype
    # A forma mais simples é converter para numpy array e depois para csr_matrix
    explicit_features_sparse = csr_matrix(explicit_feature_df.values)
    print(f"Dimensões das features explícitas: {explicit_features_sparse.shape}")

    # --- 2. Engenharia de Features de Caracteres (N-grams) ---
    print("Criando features de caracteres (n-grams)...")
    char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), max_features=5000) # Limitar max_features
    char_features = char_vectorizer.fit_transform(df['palavra_limpa']) # Use palavra_limpa para n-grams
    print(f"Dimensões das features de n-grams: {char_features.shape}")

    # --- 3. Combinar Todas as Features ---
    X = hstack([explicit_features_sparse, char_features])
    y = df['classe']
    print(f"Dimensões totais das features (X): {X.shape}")

    # --- 4. Treinamento do Modelo ---
    print("\nDividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Treinando modelo RandomForestClassifier (isso pode levar um tempo para 15k amostras)...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced') # Aumentar n_estimators, usar class_weight
    model.fit(X_train, y_train)

    # --- 5. Avaliação do Modelo ---
    print("\nAvaliando o modelo...")
    y_pred = model.predict(X_test)
    print("Acurácia geral:", accuracy_score(y_test, y_pred))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # --- 6. Salvar Modelo e Ferramentas ---
    print("\nSalvando modelo e vetorizador...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(char_vectorizer, VECTORIZER_PATH)
    joblib.dump(COMMON_PREFIXES, COMMON_PREFIXES_PATH)
    joblib.dump(COMMON_SUFFIXES, COMMON_SUFFIXES_PATH)
    joblib.dump(FOREIGN_PATTERNS, FOREIGN_PATTERNS_PATH) # Salvar o dicionário de padrões estrangeiros
    joblib.dump(PORTUGUESE_ENDINGS, os.path.join(DATA_DIR, 'portuguese_endings.pkl'))


    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Vetorizador salvo em: {VECTORIZER_PATH}")
    print("\nTreinamento concluído com sucesso!")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    train_classifier()