# neologism_app/services.py

import os
import re
import requests
from bs4 import BeautifulSoup
import spacy
import json
from collections import deque 
import html
import unicodedata
import joblib 
import numpy as np 
import pandas as pd 
from scipy.sparse import hstack, csr_matrix 

from django.db.models import Exists, OuterRef 
from neologism_app.models import LexiconWord, CustomAddition, NeologismValidated 

# --- Configurações e Caminhos ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DICIO_CACHE_PATH = os.path.join(DATA_DIR, 'dicio_cache.json')

# --- Configurações do spaCy para textos longos ---
SPACY_CHUNK_SIZE = 50_000 # <-- Mantido em 50k, ajuste se ainda tiver MemoryError
SPACY_MAX_LENGTH_NORMAL = 1_000_000 
SPACY_MAX_LENGTH_LARGE_TEXT_PROCESSING = 2_000_000 

TEXT_SIZE_THRESHOLD_FOR_LIGHT_SPACY = 1_000_000 

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
FORMATION_PROCESS_OPTIONS = [ 
    "composto neoclássico",
    "derivado prefixal",
    "estrangeirismo",
    "derivado sufixal",
    "splinter",
    "composto",
    "sigla",
    "outros" 
]

# --- Carregar Modelos de ML e Ferramentas (EXISTENTE) ---
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

# --- Função Auxiliar para Lidar com Pronomes Enclíticos (EXISTENTE) ---
def normalize_enclitic_pronoun_word(word_form, lemma_from_spacy):
    """
    Normaliza a forma da palavra e/ou o lema do spaCy para lidar com pronomes enclíticos.
    Retorna uma lista de palavras a serem verificadas no léxico.
    """
    word_form_lower = word_form.lower()
    lemma_lower = lemma_from_spacy.lower() if lemma_from_spacy else ''
    
    candidates_for_lexicon_check = []

    candidates_for_lexicon_check.append(word_form_lower)
    candidates_for_lexicon_check.append(lemma_lower)
    
    if ' ' in lemma_lower:
        candidates_for_lexicon_check.append(lemma_lower.split(' ')[0])

    pattern1_endings_with_hyphen = ['-o', '-a', '-os', '-as', '-no', '-na', '-nos', '-nas', '-se', '-lhe', '-lhes']
    
    for ending in pattern1_endings_with_hyphen:
        if word_form_lower.endswith(ending):
            part_before_pronoun = word_form_lower.rsplit(ending, 1)[0]
            if part_before_pronoun:
                candidates_for_lexicon_check.append(part_before_pronoun)
                
                if part_before_pronoun.endswith('r'):
                    candidates_for_lexicon_check.append(part_before_pronoun) 
                    candidates_for_lexicon_check.append(part_before_pronoun[:-1]) 
                    
    match_ar = re.match(r'(.+)á-(?:lo|la|los|las)$', word_form_lower)
    if match_ar:
        candidates_for_lexicon_check.append(match_ar.group(1) + 'ar') 

    match_er = re.match(r'(.+)ê-(?:lo|la|los|las)$', word_form_lower)
    if match_er:
        candidates_for_lexicon_check.append(match_er.group(1) + 'er') 

    match_ir = re.match(r'(.+)i-(?:lo|la|los|las)$', word_form_lower)
    if match_ir:
        candidates_for_lexicon_check.append(match_ir.group(1) + 'ir') 
    
    match_or = re.match(r'(.+)ô-(?:lo|la|los|las)$', word_form_lower)
    if match_or:
        candidates_for_lexicon_check.append(match_or.group(1) + 'or') 

    return list(set(candidates_for_lexicon_check))

# --- Função de Engenharia de Features para Predição (EXISTENTE) ---
def create_prediction_features(word):
    features = {}
    word_normalized = unicodedata.normalize('NFKD', word.lower()).encode('ascii', 'ignore').decode('utf-8')
    word_lower = word.lower()

    features['has_hyphen'] = 1 if '-' in word_lower else 0
    features['word_length'] = len(word_lower)
    num_vowels = sum(1 for char in word_lower if char in 'aeiouáéíóúãõ')
    features['vowel_ratio'] = num_vowels / len(word_lower) if len(word_lower) > 0 else 0

    for prefix in COMMON_PREFIXES_ML: 
        features[f'has_prefix_{prefix.replace("-", "")}'] = 1 if word_lower.startswith(prefix) else 0

    for suffix in COMMON_SUFFIXES_ML:
        features[f'has_suffix_{suffix.replace("-", "")}'] = 1 if word_lower.endswith(suffix) else 0

    if FOREIGN_PATTERNS_ML: 
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
        self.nlp = self._load_spacy_model(use_light_config=False) 
        try:
            self.nlp.max_length = SPACY_MAX_LENGTH_LARGE_TEXT_PROCESSING
        except ValueError as e:
            print(f"Aviso: Não foi possível definir nlp.max_length para {SPACY_MAX_LENGTH_LARGE_TEXT_PROCESSING}: {e}. Usando valor padrão ou menor.")

        self.light_nlp = None 

    def _load_spacy_model(self, use_light_config=False):
        """Carrega o modelo spaCy para português, opcionalmente com uma configuração leve."""
        if use_light_config:
            print("Carregando modelo spaCy com configuração leve (desabilitando parser e NER) para economia de memória.")
            try:
                return spacy.load("pt_core_news_sm", disable=["parser", "ner"])
            except OSError:
                print("Modelo spaCy 'pt_core_news_sm' não encontrado. Baixando...")
                spacy.cli.download("pt_core_news_sm") 
                return spacy.load("pt_core_news_sm", disable=["parser", "ner"])
        else:
            print("Carregando modelo spaCy completo (pt_core_news_sm).")
            try:
                return spacy.load("pt_core_news_sm")
            except OSError:
                print("Modelo spaCy 'pt_core_news_sm' não encontrado. Baixando...")
                spacy.cli.download("pt_core_news_sm")
                return spacy.load("pt_core_news_sm")
    
    def process_text(self, text):
        IS_LARGE_TEXT_FOR_DISPLAY = len(text) > 50000 

        nlp_instance_to_use = self.nlp 

        if len(text) > TEXT_SIZE_THRESHOLD_FOR_LIGHT_SPACY:
            if self.light_nlp is None:
                self.light_nlp = self._load_spacy_model(use_light_config=True)
            nlp_instance_to_use = self.light_nlp
        
        use_full_ner_filter = (nlp_instance_to_use == self.nlp) 

        temp_doc_for_sents = None
        sentences = [] 

        try:
            temp_doc_for_sents = nlp_instance_to_use(text) 
            sentences = [sent.text for sent in temp_doc_for_sents.sents]
        except Exception as e:
            print(f"ERRO: Falha ao segmentar sentenças do texto completo com spaCy ({e}). Não será possível obter contexto de sentenças exato para neologismos. Prosseguindo sem contexto de sentenças.")
            sentences = [text] 

        all_neologism_candidates = []
        seen_neologism_candidates_global = set()
        total_words = 0
        num_neologisms = 0
        
        processed_html_parts = []

        text_chunks = [text[i:i + SPACY_CHUNK_SIZE] for i in range(0, len(text), SPACY_CHUNK_SIZE)]
        print(f"Processando {len(text)} caracteres em {len(text_chunks)} chunks de até {SPACY_CHUNK_SIZE} chars.")


        for chunk_idx, chunk_text in enumerate(text_chunks):
            print(f"  Processando Chunk {chunk_idx + 1}/{len(text_chunks)}")
            try:
                chunk_doc = nlp_instance_to_use(chunk_text) 
            except Exception as e:
                print(f"ERRO: Falha ao processar chunk {chunk_idx + 1} com spaCy: {e}. Pulando este chunk.")
                continue 

            for token in chunk_doc:
                # CORREÇÃO AQUI: Definir original_word e word_lower (a partir do token.text)
                # para que eles existam no escopo do loop.
                original_word = token.text # <-- CORREÇÃO
                word_lower = token.text.lower() # <-- CORREÇÃO
                
                if token.is_space:
                    if not IS_LARGE_TEXT_FOR_DISPLAY:
                        processed_html_parts.append(original_word + token.whitespace_) # Usa original_word
                    continue
                if token.is_punct or token.like_num:
                    if not IS_LARGE_TEXT_FOR_DISPLAY:
                        processed_html_parts.append(original_word + token.whitespace_) # Usa original_word
                    continue

                # === LIMPEZA MAIS RIGOROSA DE CARACTERES NÃO-ALFABÉTICOS E HÍFENS ===
                # Estas são as variáveis que passarão por limpeza profunda.
                # Não sobrescreva original_word e word_lower que vêm do token.text
                temp_clean_word_lower = re.sub(r'[^a-z0-9áàâãéêíóôõúüç\-]', '', word_lower)
                temp_clean_original_word = re.sub(r'[^a-zA-Z0-9áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ\-]', '', original_word) # Adicionado caracteres maiúsculos

                temp_clean_word_lower = re.sub(r'-{2,}', '-', temp_clean_word_lower).strip('-')
                temp_clean_original_word = re.sub(r'-{2,}', '-', temp_clean_original_word).strip('-')

                clean_word_lower = temp_clean_word_lower # A versão limpa e minúscula para consultas
                clean_original_word = temp_clean_original_word # A versão limpa, mantendo o case original, para dados e display

                if not clean_word_lower: # Se a palavra ficou vazia após a limpeza rigorosa
                    if not IS_LARGE_TEXT_FOR_DISPLAY:
                        processed_html_parts.append(original_word + token.whitespace_) # Usa original_word
                    continue

                # NOVO FILTRO: Ignorar tokens com caracteres especiais internos que não são hífens
                # E que a limpeza os removeu. Ex: "abaixo).Até" -> "abaixate" (removido ')', '.')
                # Essa heurística tenta pegar IDs de arquivos, URLs parciais, etc.
                # Se `original_word` continha algo que não `clean_original_word` tem (que não seja hífen/letra/número)
                if len(original_word) > len(clean_original_word) and \
                   re.search(r'[^\w\s\-]', original_word): # Procura por caracteres não alfanuméricos, não espaço, não hífen
                    # print(f"  Ignorando: Token original muito diferente da versão limpa ('{original_word}'). Provável string malformada/ID.")
                    if not IS_LARGE_TEXT_FOR_DISPLAY:
                        processed_html_parts.append(original_word + token.whitespace_) # Usa original_word
                    continue

                # === NOVO FILTRO: IGNORAR PADRÕES DE HORA (ex: 9h, 22h30) ===
                # Padrão: Começa com 1 ou 2 dígitos, seguido por 'h', opcionalmente seguido por 2 dígitos.
                # Regex: ^\d{1,2}h(\d{2})?$
                # ^\d{1,2}: Início da string, 1 ou 2 dígitos (ex: 9, 22)
                # h: o caractere 'h' literal
                # (\d{2})?: Opcionalmente, 2 dígitos (ex: 30 em 22h30)
                
                # Para ser mais abrangente, considerar variações como "9h30min" ou "14:00h" se necessário,
                # mas o pedido inicial é focado em "Xh" e "XhXX".
                
                # Vamos testar a `clean_word_lower` para este padrão.
                if re.match(r'^\d{1,2}h(\d{2})?$', clean_word_lower):
                    print(f"  Ignorando: Padrão de hora detectado ('{original_word}').")
                    if not IS_LARGE_TEXT_FOR_DISPLAY:
                        processed_html_parts.append(original_word + token.whitespace_)
                    continue

                total_words += 1
                is_neologism_candidate = False

                # === LÓGICA DE FILTRAGEM DE NOMES PRÓPRIOS (AJUSTADA) ===
                if (token.pos_ == "PROPN") or \
                   (use_full_ner_filter and token.ent_type_ in ["PERSON", "LOC", "ORG", "MISC"]):
                    if not IS_LARGE_TEXT_FOR_DISPLAY:
                        processed_html_parts.append(original_word + token.whitespace_) # Usa original_word
                    continue

                if token.pos_ not in CANDIDATE_POS_TAGS:
                    if not IS_LARGE_TEXT_FOR_DISPLAY:
                        processed_html_parts.append(original_word + token.whitespace_) # Usa original_word
                    continue
            
                # 4. VERIFICAR NO LÉXICO DO BANCO DE DADOS (USANDO normalize_enclitic_pronoun_word)
                words_to_check_in_lexicon = normalize_enclitic_pronoun_word(original_word, token.lemma_) # Passar original_word para normalizer
                
                is_word_in_db_lexicon = False
                for check_word in words_to_check_in_lexicon:
                    if LexiconWord.objects.filter(word=check_word).exists():
                        is_word_in_db_lexicon = True
                        break
                    if CustomAddition.objects.filter(word=check_word).exists():
                        is_word_in_db_lexicon = True
                        break
                
                if not is_word_in_db_lexicon:
                    is_in_dicio = is_word_in_dicio(clean_word_lower)

                    if not is_in_dicio:
                        is_neologism_candidate = True
                    else:
                        self.add_to_custom_additions(clean_word_lower)
                else:
                    pass

                # Obter índice da sentença e texto para o token atual
                original_token_start_char = token.idx + (chunk_idx * SPACY_CHUNK_SIZE)
                
                sentence_idx_for_token = self._get_sentence_index_from_full_text(original_token_start_char, temp_doc_for_sents)
                
                sentence_text_for_token = sentences[sentence_idx_for_token] if (sentence_idx_for_token != -1 and sentences and sentence_idx_for_token < len(sentences)) else "Sentença não encontrada"

                if is_neologism_candidate:
                    num_neologisms += 1
                    
                    predicted_formation = "Não classificado (ML indisponível)"
                    if CLASSIFIER_MODEL and CHAR_VECTORIZER and EXPLICIT_FEATURE_NAMES:
                        try:
                            word_explicit_features_dict = create_prediction_features(clean_original_word) # Usar clean_original_word
                            explicit_features_array = np.array([[word_explicit_features_dict.get(name, 0) for name in EXPLICIT_FEATURE_NAMES]])
                            explicit_features_sparse = csr_matrix(explicit_features_array)
                            
                            char_features_single = CHAR_VECTORIZER.transform([clean_original_word]) # Usar clean_original_word
                            
                            X_single_word = hstack([explicit_features_sparse, char_features_single])
                            
                            predicted_formation = CLASSIFIER_MODEL.predict(X_single_word)[0]
                        except Exception as e:
                            print(f"  Erro na classificação ML para '{clean_original_word}': {e}")
                            predicted_formation = "Erro na classificação ML"

                    if not IS_LARGE_TEXT_FOR_DISPLAY:
                        processed_html_parts.append(
                            f'<span class="neologism" data-word="{html.escape(clean_original_word)}" ' # data-word é limpo
                            f'data-original-pos="{token.pos_}" data-pos="{POS_MAPPING.get(token.pos_, token.pos_)}" data-lemma="{html.escape(token.lemma_)}" '
                            f'data-sent-idx="{sentence_idx_for_token}" '
                            f'data-sentence-text="{html.escape(sentence_text_for_token)}" '
                            f'data-predicted-formation="{html.escape(predicted_formation)}">'
                            f'{html.escape(original_word)}</span>{token.whitespace_}' # Aqui original_word (sem limpeza)
                        )
                    else:
                        processed_html_parts.append(original_word + token.whitespace_) # Aqui original_word (sem limpeza)

                    if clean_word_lower not in seen_neologism_candidates_global: 
                        all_neologism_candidates.append({ 
                            'word': clean_original_word, # Usar clean_original_word aqui
                            'word_lower': clean_word_lower, # Usar clean_word_lower aqui
                            'original_pos': token.pos_,
                            'pos': POS_MAPPING.get(token.pos_, token.pos_),
                            'lemma': token.lemma_,
                            'sentence_idx': sentence_idx_for_token, 
                            'sentence_text': sentence_text_for_token, 
                            'predicted_formation': predicted_formation
                        })
                        seen_neologism_candidates_global.add(clean_word_lower) 
                else:
                    if not IS_LARGE_TEXT_FOR_DISPLAY:
                        processed_html_parts.append(original_word + token.whitespace_) # Usa original_word
                    else:
                        processed_html_parts.append(original_word + token.whitespace_) # Usa original_word

        return {
            'processed_text_html': "".join(processed_html_parts) if not IS_LARGE_TEXT_FOR_DISPLAY else "", 
            'neologism_candidates': all_neologism_candidates, 
            'total_words': total_words,
            'num_neologisms': num_neologisms,
            'sentences': sentences, 
            'is_large_text_for_display': IS_LARGE_TEXT_FOR_DISPLAY 
        }

    def _get_sentence_index_from_full_text(self, char_offset_in_full_text, full_doc_for_sents):
        """
        Retorna o índice da sentença à qual o offset de caractere pertence no documento completo.
        Retorna -1 se não encontrar ou se full_doc_for_sents não puder ser processado.
        """
        if not full_doc_for_sents:
            return -1
        for i, sent in enumerate(full_doc_for_sents.sents):
            if hasattr(sent, 'start_char') and hasattr(sent, 'end_char') and \
               sent.start_char is not None and sent.end_char is not None:
                if char_offset_in_full_text >= sent.start_char and char_offset_in_full_text < sent.end_char:
                    return i
        return -1 

    # Funções de adição/validação (código existente)
    def add_to_custom_additions(self, word):
        word_lower = word.lower()
        try:
            _, created = CustomAddition.objects.get_or_create(word=word_lower)
            NeologismValidated.objects.filter(word=word_lower).delete()
            return created
        except Exception as e:
            print(f"Erro ao adicionar '{word_lower}' a CustomAddition: {e}")
            return False

    def add_to_neologisms_validated(self, word, original_pos_tag=None, corrected_pos_tag=None, lemma=None, formation_process=None, predicted_formation=None):
        word_lower = word.lower()
        try:
            neologism, created = NeologismValidated.objects.get_or_create(
                word=word_lower,
                defaults={
                    'pos_tag': corrected_pos_tag or POS_MAPPING.get(original_pos_tag, original_pos_tag),
                    'lemma': lemma,
                    'formation_process': formation_process or predicted_formation 
                }
            )
            if not created:
                if corrected_pos_tag: neologism.pos_tag = corrected_pos_tag
                elif original_pos_tag: neologism.pos_tag = POS_MAPPING.get(original_pos_tag, original_pos_tag)
                if lemma: neologism.lemma = lemma
                if formation_process: neologism.formation_process = formation_process
                elif predicted_formation and not neologism.formation_process: 
                    neologism.formation_process = predicted_formation
                neologism.save()
            
            CustomAddition.objects.filter(word=word_lower).delete()

            return True
        except Exception as e:
            print(f"Erro ao adicionar/atualizar '{word_lower}' em NeologismValidated: {e}")
            return False

    # Exportar resultados para CSV (código existente)
    def export_results_to_csv(self, results, filename="neologisms.csv"):
        import csv
        filepath = os.path.join(DATA_DIR, filename)
        
        pos_mapping = POS_MAPPING 

        def get_formation_process_for_csv(word_lower, original_pos_tag, predicted_formation_from_detection=None):
            validated_neo = NeologismValidated.objects.filter(word=word_lower).first()
            if validated_neo and validated_neo.formation_process:
                return validated_neo.formation_process 
            
            if predicted_formation_from_detection: 
                return predicted_formation_from_detection
            
            if original_pos_tag:
                 if word_lower.endswith("mente") and original_pos_tag == "ADJ": return "Derivação sufixal (Heurística)"
                 if word_lower.startswith("des") and original_pos_tag == "VERB": return "Derivação prefixal (Heurística)"
                 if re.match(r'^[a-zA-Z]+[_-][a-zA-Z]+$', word_lower): return "Composição (Heurística)"
                 if any(char in 'kqwy' for char in word_lower) and len(word_lower) > 3: return "Estrangeirismo (Heurística)?"
            return "Outros (Heurística)" 

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
                    'predicted_formation': candidate.get('predicted_formation') 
                }
            else:
                if candidate['sentence_idx'] not in unique_candidates[word_lower]['sentences_idx']:
                    unique_candidates[word_lower]['sentences_idx'].append(candidate['sentence_idx'])

        data_to_export = []
        for word_lower, details in unique_candidates.items():
            original_word = details['word']
            original_pos_tag_spacy = details['original_pos']
            original_lemma_spacy = details['lemma']
            predicted_formation_from_detection = details.get('predicted_formation') 


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