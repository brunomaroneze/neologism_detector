# neologism_app/services.py

import os
import re
import requests
from bs4 import BeautifulSoup
import spacy
import json
from collections import deque # Para um cache LRU simples
import html
import unicodedata
import joblib # Para carregar modelos de ML
import numpy as np # Para manipulação de arrays
import pandas as pd # Para criar DataFrames de features (para consistência)
from scipy.sparse import hstack, csr_matrix # Para combinar features esparsas

from django.db.models import Exists, OuterRef 
from neologism_app.models import LexiconWord, CustomAddition, NeologismValidated 

# --- Configurações e Caminhos ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DICIO_CACHE_PATH = os.path.join(DATA_DIR, 'dicio_cache.json')

# --- Mapeamento de Classes Gramaticais (PARA EXIBIÇÃO E FILTRAGEM) ---
CANDIDATE_POS_TAGS = {"NOUN", "ADJ", "VERB"}

POS_MAPPING = {
    "NOUN": "Substantivo",
    "PROPN": "Substantivo Próprio", 
    "ADJ": "Adjetivo",
    "VERB": "Verbo",
    "ADV": "Advérbio",
    "PRON": "Pronome",
    "DET": "Determinante",
    "ADP": "Preposição",
    "AUX": "Verbo Auxiliar",
    "CCONJ": "Conjunção Coordenativa",
    "SCONJ": "Conjunção Subordinativa",
    "NUM": "Numeral",
    "INTJ": "Interjeição",
    "PART": "Partícula",
    "SYM": "Símbolo",
    "X": "Outros", 
    "SPACE": "Espaço",
    "PUNCT": "Pontuação",
}

# --- Cache para Dicio.com.br (código existente) ---
DICIO_CACHE = {}
MAX_CACHE_SIZE = 1000
DICIO_CACHE_KEYS = deque() 

def load_dicio_cache():
    global DICIO_CACHE, DICIO_CACHE_KEYS
    try:
        with open(DICIO_CACHE_PATH, 'r', encoding='utf-8') as f:
            DICIO_CACHE = json.load(f)
            DICIO_CACHE_KEYS = deque(DICIO_CACHE.keys())
    except (FileNotFoundError, json.JSONDecodeError):
        DICIO_CACHE = {}
        DICIO_CACHE_KEYS = deque()

def save_dicio_cache():
    with open(DICIO_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(DICIO_CACHE, f, ensure_ascii=False, indent=4)

load_dicio_cache()

# --- Dicio.com.br Scraper (código existente) ---
def is_word_in_dicio(word):
    word_lower = word.lower()
    dicio_query_word = unicodedata.normalize('NFKD', word_lower).encode('ascii', 'ignore').decode('utf-8')

    if dicio_query_word in DICIO_CACHE:
        if dicio_query_word in DICIO_CACHE_KEYS:
            DICIO_CACHE_KEYS.remove(dicio_query_word)
        DICIO_CACHE_KEYS.append(dicio_query_word)
        return DICIO_CACHE[dicio_query_word]

    print(f"Buscando '{word_lower}' (query: '{dicio_query_word}') no Dicio.com.br...")
    url = f"https://www.dicio.com.br/{dicio_query_word}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')

        word_nf_element = soup.find('p', class_='significado word-nf')
        
        if word_nf_element:
            is_present = False
        else:
            definition_indicators = [
                soup.find('p', class_='significado'),
                soup.find('div', class_='tit-sub'),
                soup.find('div', class_='conjugacao'),
                soup.find('h2', class_='tit-section'),
            ]
            is_present = any(indicator is not None for indicator in definition_indicators)

        if len(DICIO_CACHE_KEYS) >= MAX_CACHE_SIZE:
            oldest_key = DICIO_CACHE_KEYS.popleft()
            DICIO_CACHE.pop(oldest_key, None)

        DICIO_CACHE[dicio_query_word] = is_present 
        DICIO_CACHE_KEYS.append(dicio_query_word)
        save_dicio_cache()

        return is_present
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error {e.response.status_code} ao consultar Dicio para '{word_lower}' (query: '{dicio_query_word}'): {e}. Assumindo não encontrada.")
        DICIO_CACHE[dicio_query_word] = False 
        DICIO_CACHE_KEYS.append(dicio_query_word)
        save_dicio_cache()
        return False
    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão/timeout ao consultar Dicio para '{word_lower}' (query: '{dicio_query_word}'): {e}. Assumindo não encontrada.")
        DICIO_CACHE[dicio_query_word] = False 
        DICIO_CACHE_KEYS.append(dicio_query_word)
        save_dicio_cache()
        return False
    except Exception as e:
        print(f"Erro inesperado ao parsear Dicio para '{word_lower}' (query: '{dicio_query_word}'): {e}. Assumindo não encontrada.")
        DICIO_CACHE[dicio_query_word] = False 
        DICIO_CACHE_KEYS.append(dicio_query_word)
        save_dicio_cache()
        return False

# --- NOVAS CATEGORIAS DE NEOLOGISMOS (ATUALIZADO) ---
FORMATION_PROCESS_OPTIONS = [ # Lista de opções para o dropdown no modal
    "composto neoclássico",
    "derivado prefixal",
    "estrangeirismo",
    "derivado sufixal",
    "splinter",
    "composto",
    "sigla",
    "outros" # Ajuste para 'outros' em vez de 'outros casos' para consistência
]

# --- Carregar Modelos de ML e Ferramentas (NOVO BLOCO DE CÓDIGO) ---
CLASSIFIER_MODEL = None
CHAR_VECTORIZER = None
EXPLICIT_FEATURE_NAMES = []
COMMON_PREFIXES_ML = []
COMMON_SUFFIXES_ML = []
FOREIGN_PATTERNS_ML = {}
PORTUGUESE_ENDINGS_ML = []

try:
    CLASSIFIER_MODEL = joblib.load(os.path.join(DATA_DIR, 'neologism_classifier_model.pkl'))
    CHAR_VECTORIZER = joblib.load(os.path.join(DATA_DIR, 'char_vectorizer.pkl'))
    EXPLICIT_FEATURE_NAMES = joblib.load(os.path.join(DATA_DIR, 'explicit_feature_names.pkl'))
    COMMON_PREFIXES_ML = joblib.load(os.path.join(DATA_DIR, 'common_prefixes.pkl'))
    COMMON_SUFFIXES_ML = joblib.load(os.path.join(DATA_DIR, 'common_suffixes.pkl'))
    FOREIGN_PATTERNS_ML = joblib.load(os.path.join(DATA_DIR, 'foreign_patterns.pkl'))
    PORTUGUESE_ENDINGS_ML = joblib.load(os.path.join(DATA_DIR, 'portuguese_endings.pkl'))

    print("Modelos de classificação de neologismos carregados com sucesso.")
except FileNotFoundError as e:
    print(f"Aviso: Arquivo de modelo de classificação de neologismos não encontrado: {e}. A classificação automática não estará disponível.")
except Exception as e:
    print(f"Erro ao carregar modelos de classificação de neologismos: {e}. A classificação automática não estará disponível.")


# --- Função de Engenharia de Features para Predição (NOVO) ---
# Precisa ser IDÊNTICA à usada no treinamento em train_classifier.py
def create_prediction_features(word):
    features = {}
    word_normalized = unicodedata.normalize('NFKD', word.lower()).encode('ascii', 'ignore').decode('utf-8')
    word_lower = word.lower()

    features['has_hyphen'] = 1 if '-' in word_lower else 0
    features['word_length'] = len(word_lower)
    num_vowels = sum(1 for char in word_lower if char in 'aeiouáéíóúãõ')
    features['vowel_ratio'] = num_vowels / len(word_lower) if len(word_lower) > 0 else 0

    for prefix in COMMON_PREFIXES_ML: # Usar as listas carregadas
        features[f'has_prefix_{prefix.replace("-", "")}'] = 1 if word_lower.startswith(prefix) else 0

    for suffix in COMMON_SUFFIXES_ML:
        features[f'has_suffix_{suffix.replace("-", "")}'] = 1 if word_lower.endswith(suffix) else 0

    if FOREIGN_PATTERNS_ML: # Verificar se o dicionário foi carregado
        for letter in FOREIGN_PATTERNS_ML.get('letters', []):
            features[f'has_foreign_letter_{letter}'] = 1 if letter in word_lower else 0
        for pattern in FOREIGN_PATTERNS_ML.get('start_patterns', []):
            features[f'starts_foreign_{pattern}'] = 1 if word_lower.startswith(pattern) else 0
        for pattern in FOREIGN_PATTERNS_ML.get('end_patterns', []):
            features[f'ends_foreign_{pattern}'] = 1 if word_lower.endswith(pattern) else 0
        for pattern in FOREIGN_PATTERNS_ML.get('internal_patterns', []):
            features[f'has_internal_foreign_{pattern}'] = 1 if pattern in word_lower else 0

    for ending in PORTUGUESE_ENDINGS_ML:
        features[f'ends_portuguese_{ending}'] = 1 if word_lower.endswith(ending) else 0
    
    return features

# --- Detector de Neologismos Principal ---
class NeologismDetector:
    def __init__(self):
        self.nlp = self._load_spacy_model()

    def _load_spacy_model(self):
        """Carrega o modelo spaCy para português."""
        try:
            return spacy.load("pt_core_news_sm")
        except OSError:
            print("Modelo spaCy 'pt_core_news_sm' não encontrado. Baixando...")
            spacy.cli.download("pt_core_news_sm")
            return spacy.load("pt_core_news_sm")
    
    def process_text(self, text):
        doc = self.nlp(text)
        processed_html_parts = []
        neologism_candidates = []
        total_words = 0
        num_neologisms = 0
        
        seen_neologism_candidates = set()
        
        sentences = [sent.text for sent in doc.sents]

        for token in doc:
            # DEBUG: Imprime o token original e suas propriedades iniciais
            # print(f"\n--- Processando Token: '{token.text}' (Index: {token.idx}) ---")
            # print(f"  POS: {token.pos_}, Lemma: {token.lemma_}, Ent_Type: {token.ent_type_}")

            if token.is_space:
                # print(f"  Ignorando: Espaço.")
                processed_html_parts.append(token.text)
                continue
            if token.is_punct or token.like_num:
                # print(f"  Ignorando: Pontuação ou Número.")
                processed_html_parts.append(token.text + token.whitespace_)
                continue

            word_lower = token.text.lower()
            original_word = token.text
            
            clean_word_lower = re.sub(r'^\W+|\W+$', '', word_lower)
            clean_original_word = re.sub(r'^\W+|\W+$', '', original_word)

            if not clean_word_lower:
                # print(f"  Ignorando: Vazio após limpeza ('{original_word}').")
                processed_html_parts.append(original_word + token.whitespace_)
                continue

            total_words += 1
            is_neologism_candidate = False

            if token.pos_ == "PROPN" or token.ent_type_ in ["PERSON", "LOC", "ORG", "MISC"]:
                # print(f"  Ignorando: Nome Próprio ou Entidade Nomeada ({token.pos_}/{token.ent_type_}).")
                processed_html_parts.append(original_word + token.whitespace_)
                continue

            if token.pos_ not in CANDIDATE_POS_TAGS:
                # print(f"  Ignorando: Classe Gramatical ('{token.pos_}') não é candidata a neologismo.")
                processed_html_parts.append(original_word + token.whitespace_)
                continue
         
            # 4. Verificar no LÉXICO DO BANCO DE DADOS
            found_by_word_form_in_lexicon = LexiconWord.objects.filter(word=clean_word_lower).exists()
            found_by_word_form_in_custom = CustomAddition.objects.filter(word=clean_word_lower).exists()
            
            found_by_lemma_in_lexicon = False
            found_by_lemma_in_custom = False
            lemma_to_check = token.lemma_.lower()

            if ' ' in lemma_to_check:
                main_lemma_part = lemma_to_check.split(' ')[0]
                found_by_lemma_in_lexicon = LexiconWord.objects.filter(word=main_lemma_part).exists()
                found_by_lemma_in_custom = CustomAddition.objects.filter(word=main_lemma_part).exists()
            else:
                found_by_lemma_in_lexicon = LexiconWord.objects.filter(word=lemma_to_check).exists()
                found_by_lemma_in_custom = CustomAddition.objects.filter(word=lemma_to_check).exists()

            is_word_in_db_lexicon = found_by_word_form_in_lexicon or \
                                   found_by_word_form_in_custom or \
                                   found_by_lemma_in_lexicon or \
                                   found_by_lemma_in_custom

            # print(f"  Verificando léxico (limpo: '{clean_word_lower}', lema: '{token.lemma_}')")
            # print(f"    Encontrado por forma limpa no léxico: {found_by_word_form_in_lexicon}")
            # print(f"    Encontrado por forma limpa nas custom: {found_by_word_form_in_custom}")
            # print(f"    Encontrado por lema no léxico: {found_by_lemma_in_lexicon}")
            # print(f"    Encontrado por lema nas custom: {found_by_lemma_in_custom}")
            # print(f"    Total no DB Léxico: {is_word_in_db_lexicon}")

            
            if not is_word_in_db_lexicon: # Se a palavra (ou seu lema) NÃO está no léxico do DB
                # 5. Enriquecer o filtro com Dicio.com.br
                is_in_dicio = is_word_in_dicio(clean_word_lower)
                # print(f"  Verificando Dicio.com.br para '{clean_word_lower}': {is_in_dicio}")

                if not is_in_dicio:
                    is_neologism_candidate = True
                    # print(f"  Marcado como NEOLOGISMO CANDIDATO.")
                else:
                    self.add_to_custom_additions(clean_word_lower)
                    # print(f"  NÃO é neologismo: Encontrado no Dicio. Adicionado a CustomAdditions.")
            else:
                # print(f"  NÃO é neologismo: Encontrado no Léxico DB.")
                pass # Não faça nada aqui, a palavra não é candidata

            if is_neologism_candidate:
                num_neologisms += 1
                
                # NOVO: Classificação automática via ML
                predicted_formation = "Não classificado (ML indisponível)"
                if CLASSIFIER_MODEL and CHAR_VECTORIZER and EXPLICIT_FEATURE_NAMES:
                    try:
                        word_explicit_features_dict = create_prediction_features(clean_original_word)
                        explicit_features_array = np.array([[word_explicit_features_dict.get(name, 0) for name in EXPLICIT_FEATURE_NAMES]])
                        explicit_features_sparse = csr_matrix(explicit_features_array)
                        
                        char_features_single = CHAR_VECTORIZER.transform([clean_original_word])
                        
                        X_single_word = hstack([explicit_features_sparse, char_features_single])
                        
                        predicted_formation = CLASSIFIER_MODEL.predict(X_single_word)[0]
                        # print(f"  Classificação ML para '{clean_original_word}': {predicted_formation}")
                    except Exception as e:
                        # print(f"  Erro na classificação ML para '{clean_original_word}': {e}")
                        predicted_formation = "Erro na classificação ML"

                processed_html_parts.append(
                    f'<span class="neologism" data-word="{html.escape(clean_original_word)}" '
                    f'data-original-pos="{token.pos_}" data-pos="{POS_MAPPING.get(token.pos_, token.pos_)}" data-lemma="{html.escape(token.lemma_)}" '
                    f'data-sent-idx="{self._get_sentence_index(token, doc)}" '
                    f'data-sentence-text="{html.escape(sentences[self._get_sentence_index(token, doc)])}" '
                    f'data-predicted-formation="{html.escape(predicted_formation)}">' # NOVO DATA-ATTRIBUTE
                    f'{html.escape(original_word)}</span>{token.whitespace_}'
                )
                if clean_word_lower not in seen_neologism_candidates:
                    neologism_candidates.append({
                        'word': clean_original_word,
                        'word_lower': clean_word_lower,
                        'original_pos': token.pos_,
                        'pos': POS_MAPPING.get(token.pos_, token.pos_),
                        'lemma': token.lemma_,
                        'sentence_idx': self._get_sentence_index(token, doc),
                        'sentence_text': sentences[self._get_sentence_index(token, doc)],
                        'predicted_formation': predicted_formation # NOVO CAMPO NO neologism_candidates
                    })
                    seen_neologism_candidates.add(clean_word_lower)
            else:
                processed_html_parts.append(original_word + token.whitespace_)

        # print("\n--- Fim do Processamento do Texto ---")
        return {
            'processed_text_html': "".join(processed_html_parts),
            'neologism_candidates': neologism_candidates,
            'total_words': total_words,
            'num_neologisms': num_neologisms,
            'sentences': sentences
        }

    def _get_sentence_index(self, token, doc):
        """Retorna o índice da sentença à qual o token pertence."""
        for i, sent in enumerate(doc.sents):
            if token.idx >= sent.start_char and token.idx < sent.end_char:
                return i
        return -1 

    # Funções de adição/validação (AGORA INTERAGEM COM OS MODELOS DB)
    def add_to_custom_additions(self, word):
        """Adiciona uma palavra à tabela CustomAddition (não neologismos)."""
        word_lower = word.lower()
        try:
            _, created = CustomAddition.objects.get_or_create(word=word_lower)
            NeologismValidated.objects.filter(word=word_lower).delete()
            return created
        except Exception as e:
            print(f"Erro ao adicionar '{word_lower}' a CustomAddition: {e}")
            return False

    # Ajustar add_to_neologisms_validated para receber a predição como default
    def add_to_neologisms_validated(self, word, original_pos_tag=None, corrected_pos_tag=None, lemma=None, formation_process=None, predicted_formation=None):
        word_lower = word.lower()
        try:
            neologism, created = NeologismValidated.objects.get_or_create(
                word=word_lower,
                defaults={
                    'pos_tag': corrected_pos_tag or POS_MAPPING.get(original_pos_tag, original_pos_tag),
                    'lemma': lemma,
                    'formation_process': formation_process or predicted_formation # Usa a do usuário, senão a predita
                }
            )
            if not created:
                if corrected_pos_tag: neologism.pos_tag = corrected_pos_tag
                elif original_pos_tag: neologism.pos_tag = POS_MAPPING.get(original_pos_tag, original_pos_tag)
                if lemma: neologism.lemma = lemma
                if formation_process: neologism.formation_process = formation_process
                elif predicted_formation and not neologism.formation_process: # Atualiza com predição se não houver já uma definição
                    neologism.formation_process = predicted_formation
                neologism.save()
            
            CustomAddition.objects.filter(word=word_lower).delete()

            return True
        except Exception as e:
            print(f"Erro ao adicionar/atualizar '{word_lower}' em NeologismValidated: {e}")
            return False

    # Ajustar export_results_to_csv para usar a predição ML
    def export_results_to_csv(self, results, filename="neologisms.csv"):
        import csv
        filepath = os.path.join(DATA_DIR, filename)
        
        pos_mapping = POS_MAPPING 

        def get_formation_process_for_csv(word_lower, original_pos_tag, predicted_formation_from_detection=None):
            validated_neo = NeologismValidated.objects.filter(word=word_lower).first()
            if validated_neo and validated_neo.formation_process:
                return validated_neo.formation_process # Prioriza a do usuário no DB
            
            if predicted_formation_from_detection: # Usa a predição da detecção
                return predicted_formation_from_detection
            
            # Heurística simples como fallback (se ML indisponível ou falhou)
            if original_pos_tag:
                 if word_lower.endswith("mente") and original_pos_tag == "ADJ": return "Derivação sufixal (Heurística)"
                 if word_lower.startswith("des") and original_pos_tag == "VERB": return "Derivação prefixal (Heurística)"
                 if re.match(r'^[a-zA-Z]+[_-][a-zA-Z]+$', word_lower): return "Composição (Heurística)"
                 if any(char in 'kqwy' for char in word_lower) and len(word_lower) > 3: return "Estrangeirismo (Heurística)?"
            return "Outros (Heurística)" # Ajuste para "Outros"

        fieldnames = ['neologismo', 'lema', 'classe_gramatical', 'processo_formacao', 'sentenca']
        
        unique_candidates = {}
        for candidate in results['neologism_candidates']:
            word_lower = candidate['word_lower']
            if word_lower not in unique_candidates:
                unique_candidates[word_lower] = {
                    'word': candidate['word'],
                    'original_pos': candidate['original_pos'],
                    'lemma': candidate['lemma'],
                    'sentences_idx': [candidate['sentence_idx']],
                    'predicted_formation': candidate.get('predicted_formation') # Recupera a predição
                }
            else:
                if candidate['sentence_idx'] not in unique_candidates[word_lower]['sentences_idx']:
                    unique_candidates[word_lower]['sentences_idx'].append(candidate['sentence_idx'])

        data_to_export = []
        for word_lower, details in unique_candidates.items():
            original_word = details['word']
            original_pos_tag_spacy = details['original_pos']
            original_lemma_spacy = details['lemma']
            predicted_formation_from_detection = details.get('predicted_formation') # Passa a predição


            validated_neo = NeologismValidated.objects.filter(word=word_lower).first()
            
            csv_pos_tag = validated_neo.pos_tag if validated_neo and validated_neo.pos_tag else pos_mapping.get(original_pos_tag_spacy, 'Não identificado')
            csv_lemma = validated_neo.lemma if validated_neo and validated_neo.lemma else original_lemma_spacy

            csv_formation_process = get_formation_process_for_csv(word_lower, original_pos_tag_spacy, predicted_formation_from_detection)

            for s_idx in details['sentences_idx']:
                sentence_text = results['sentences'][s_idx] if s_idx != -1 else "Sentença não encontrada"
                pattern = r'\b' + re.escape(original_word) + r'\b'
                highlighted_sentence = re.sub(pattern, f'<{original_word}>', sentence_text, flags=re.IGNORECASE)

                data_to_export.append({
                    'neologismo': original_word,
                    'lema': csv_lemma,
                    'classe_gramatical': csv_pos_tag,
                    'processo_formacao': csv_formation_process,
                    'sentenca': highlighted_sentence
                })
        
        data_to_export.sort(key=lambda x: x['neologismo'].lower())

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                writer.writerows(data_to_export)
            return filepath
        except Exception as e:
            print(f"Erro ao exportar CSV: {e}")
            return None


detector = NeologismDetector()