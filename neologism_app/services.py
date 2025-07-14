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

from django.db.models import Exists, OuterRef # Mantenha este import se já tiver
from neologism_app.models import LexiconWord, CustomAddition, NeologismValidated # <--- IMPORTANTE: Importar os modelos

# --- Configurações e Caminhos ---
# Obtenha o caminho absoluto do diretório 'data'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DICIO_CACHE_PATH = os.path.join(DATA_DIR, 'dicio_cache.json')

# --- Mapeamento de Classes Gramaticais (PARA EXIBIÇÃO E FILTRAGEM) ---
# Classes gramaticais que são candidatas a neologismos (substantivo, adjetivo, verbo)
CANDIDATE_POS_TAGS = {"NOUN", "ADJ", "VERB"}

# Mapeia POS tags do spaCy para classes gramaticais mais legíveis em português
POS_MAPPING = {
    "NOUN": "Substantivo",
    "PROPN": "Substantivo Próprio", # PROPN (Nomes Próprios) geralmente são eliminados por NER, mas mantido no mapping para completude
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
    "X": "Outros", # Para palavras não classificadas ou estrangeiras
    "SPACE": "Espaço", # spaCy pode ter tokens de espaço
    "PUNCT": "Pontuação", # spaCy pode ter tokens de pontuação
}

# --- Cache para Dicio.com.br ---
DICIO_CACHE = {}
# LRU Cache para Dicio (evitar que o cache cresça indefinidamente)
MAX_CACHE_SIZE = 1000
DICIO_CACHE_KEYS = deque() # Para gerenciar a ordem de uso

def load_dicio_cache():
    global DICIO_CACHE, DICIO_CACHE_KEYS
    try:
        with open(DICIO_CACHE_PATH, 'r', encoding='utf-8') as f:
            DICIO_CACHE = json.load(f)
            # Reconstroi o deque para manter a ordem, se possível, ou apenas limpa
            DICIO_CACHE_KEYS = deque(DICIO_CACHE.keys())
    except (FileNotFoundError, json.JSONDecodeError):
        DICIO_CACHE = {}
        DICIO_CACHE_KEYS = deque()

def save_dicio_cache():
    with open(DICIO_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(DICIO_CACHE, f, ensure_ascii=False, indent=4)

load_dicio_cache()

# --- Dicio.com.br Scraper ---
def is_word_in_dicio(word):
    """
    Verifica se uma palavra existe no Dicio.com.br usando web scraping.
    Usa um cache em memória e em disco para evitar requisições repetidas.
    Lida com palavras acentuadas e detecção robusta de "não encontrada".
    """
    word_lower = word.lower() # Mantém a palavra original para o cache do Python
    
    # NOVO: Remover acentos para a consulta ao Dicio.com.br e para a chave do cache
    # Normaliza para a forma de decomposição de caracteres e então ignora caracteres não-ASCII
    dicio_query_word = unicodedata.normalize('NFKD', word_lower).encode('ascii', 'ignore').decode('utf-8')

    # 1. Verifica no cache em memória usando a forma sem acento para a chave
    if dicio_query_word in DICIO_CACHE:
        # Move a chave para o final do deque (mais recentemente usada)
        if dicio_query_word in DICIO_CACHE_KEYS:
            DICIO_CACHE_KEYS.remove(dicio_query_word)
        DICIO_CACHE_KEYS.append(dicio_query_word)
        return DICIO_CACHE[dicio_query_word]

    # 2. Se não estiver no cache, tenta buscar
    print(f"Buscando '{word_lower}' (query: '{dicio_query_word}') no Dicio.com.br...")
    url = f"https://www.dicio.com.br/{dicio_query_word}/" # <--- USAR A PALAVRA SEM ACENTO AQUI
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status() # Lança um HTTPError para 4xx/5xx erros (e.g., 404 Not Found)
        soup = BeautifulSoup(response.text, 'html.parser')

        # NOVO: DETECÇÃO DE "NÃO ENCONTRADA"
        # 1. Verificar se a classe específica de "não encontrada" está presente.
        word_nf_element = soup.find('p', class_='significado word-nf')
        
        if word_nf_element:
            # Se encontrou 'significado word-nf', a palavra NÃO existe no Dicio.
            is_present = False
        else:
            # Se não há a classe de "não encontrada", procurar por indicadores de definição real.
            # Isso é para o caso de a palavra existir, mas não ter a classe 'significado' ou se tiver outras seções.
            definition_indicators = [
                soup.find('p', class_='significado'), # Procura por p.significado SEM o word-nf (já filtrado acima)
                soup.find('div', class_='tit-sub'),
                soup.find('div', class_='conjugacao'),
                soup.find('h2', class_='tit-section'),
            ]
            # A palavra é considerada presente se pelo menos um dos indicadores for encontrado.
            is_present = any(indicator is not None for indicator in definition_indicators)

        # 3. Adiciona ao cache usando a forma sem acento para a chave
        if len(DICIO_CACHE_KEYS) >= MAX_CACHE_SIZE:
            oldest_key = DICIO_CACHE_KEYS.popleft()
            DICIO_CACHE.pop(oldest_key, None)

        DICIO_CACHE[dicio_query_word] = is_present # <--- Chave do cache é a palavra sem acento
        DICIO_CACHE_KEYS.append(dicio_query_word)
        save_dicio_cache()

        return is_present
    except requests.exceptions.HTTPError as e:
        # Se for um erro HTTP 4xx/5xx (e.g., 404 Not Found), a palavra não existe.
        print(f"HTTP Error {e.response.status_code} ao consultar Dicio para '{word_lower}' (query: '{dicio_query_word}'): {e}. Assumindo não encontrada.")
        DICIO_CACHE[dicio_query_word] = False # <--- Cache com a palavra sem acento
        DICIO_CACHE_KEYS.append(dicio_query_word)
        save_dicio_cache()
        return False
    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão/timeout ao consultar Dicio para '{word_lower}' (query: '{dicio_query_word}'): {e}. Assumindo não encontrada.")
        DICIO_CACHE[dicio_query_word] = False # <--- Cache com a palavra sem acento
        DICIO_CACHE_KEYS.append(dicio_query_word)
        save_dicio_cache()
        return False
    except Exception as e:
        print(f"Erro inesperado ao parsear Dicio para '{word_lower}' (query: '{dicio_query_word}'): {e}. Assumindo não encontrada.")
        DICIO_CACHE[dicio_query_word] = False # <--- Cache com a palavra sem acento
        DICIO_CACHE_KEYS.append(dicio_query_word)
        save_dicio_cache()
        return False

FORMATION_PROCESS_OPTIONS = [ # Lista de opções para o dropdown no modal
    "Derivação prefixal",
    "Derivação sufixal",
    "Composição",
    "Estrangeirismo",
    "Outros casos"
]

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
            print(f"\n--- Processando Token: '{token.text}' (Index: {token.idx}) ---")
            print(f"  POS: {token.pos_}, Lemma: {token.lemma_}, Ent_Type: {token.ent_type_}")

            if token.is_space:
                print(f"  Ignorando: Espaço.")
                processed_html_parts.append(token.text)
                continue
            if token.is_punct or token.like_num:
                print(f"  Ignorando: Pontuação ou Número.")
                processed_html_parts.append(token.text + token.whitespace_)
                continue

            word_lower = token.text.lower()
            original_word = token.text
            
            clean_word_lower = re.sub(r'^\W+|\W+$', '', word_lower)
            clean_original_word = re.sub(r'^\W+|\W+$', '', original_word)

            if not clean_word_lower:
                print(f"  Ignorando: Vazio após limpeza ('{original_word}').")
                processed_html_parts.append(original_word + token.whitespace_)
                continue

            total_words += 1
            is_neologism_candidate = False

            if token.pos_ == "PROPN" or token.ent_type_ in ["PERSON", "LOC", "ORG", "MISC"]:
                print(f"  Ignorando: Nome Próprio ou Entidade Nomeada ({token.pos_}/{token.ent_type_}).")
                processed_html_parts.append(original_word + token.whitespace_)
                continue

            if token.pos_ not in CANDIDATE_POS_TAGS:
                print(f"  Ignorando: Classe Gramatical ('{token.pos_}') não é candidata a neologismo.")
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

            print(f"  Verificando léxico (limpo: '{clean_word_lower}', lema: '{token.lemma_}')")
            print(f"    Encontrado por forma limpa no léxico: {found_by_word_form_in_lexicon}")
            print(f"    Encontrado por forma limpa nas custom: {found_by_word_form_in_custom}")
            print(f"    Encontrado por lema no léxico: {found_by_lemma_in_lexicon}")
            print(f"    Encontrado por lema nas custom: {found_by_lemma_in_custom}")
            print(f"    Total no DB Léxico: {is_word_in_db_lexicon}")

            
            if not is_word_in_db_lexicon: # Se a palavra (ou seu lema) NÃO está no léxico do DB
                # 5. Enriquecer o filtro com Dicio.com.br
                is_in_dicio = is_word_in_dicio(clean_word_lower)
                print(f"  Verificando Dicio.com.br para '{clean_word_lower}': {is_in_dicio}")

                if not is_in_dicio:
                    is_neologism_candidate = True
                    print(f"  Marcado como NEOLOGISMO CANDIDATO.")
                else:
                    self.add_to_custom_additions(clean_word_lower)
                    print(f"  NÃO é neologismo: Encontrado no Dicio. Adicionado a CustomAdditions.")
            else:
                print(f"  NÃO é neologismo: Encontrado no Léxico DB.")

            if is_neologism_candidate:
                num_neologisms += 1
                processed_html_parts.append(
                    f'<span class="neologism" data-word="{html.escape(clean_original_word)}" '
                    f'data-original-pos="{token.pos_}" data-pos="{POS_MAPPING.get(token.pos_, token.pos_)}" data-lemma="{html.escape(token.lemma_)}" ' # Lema original do spaCy, para o usuário ver e corrigir
                    f'data-sent-idx="{self._get_sentence_index(token, doc)}" '
                    f'data-sentence-text="{html.escape(sentences[self._get_sentence_index(token, doc)])}">'
                    f'{html.escape(original_word)}</span>{token.whitespace_}'
                )
                if clean_word_lower not in seen_neologism_candidates:
                    neologism_candidates.append({
                        'word': clean_original_word,
                        'word_lower': clean_word_lower,
                        'original_pos': token.pos_,
                        'pos': POS_MAPPING.get(token.pos_, token.pos_),
                        'lemma': token.lemma_, # Lema original do spaCy
                        'sentence_idx': self._get_sentence_index(token, doc),
                        'sentence_text': sentences[self._get_sentence_index(token, doc)]
                    })
                    seen_neologism_candidates.add(clean_word_lower)
            else:
                processed_html_parts.append(original_word + token.whitespace_)

        print("\n--- Fim do Processamento do Texto ---")
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
        return -1 # Caso não encontre (nunca deve acontecer se o token vier do doc)

    # Funções de adição/validação (AGORA INTERAGEM COM OS MODELOS DB)
    def add_to_custom_additions(self, word):
        """Adiciona uma palavra à tabela CustomAddition (não neologismos)."""
        word_lower = word.lower()
        try:
            _, created = CustomAddition.objects.get_or_create(word=word_lower)
            # Se a palavra é rejeitada, certifique-se de que não está em NeologismValidated
            NeologismValidated.objects.filter(word=word_lower).delete()
            return created
        except Exception as e:
            print(f"Erro ao adicionar '{word_lower}' a CustomAddition: {e}")
            return False

    def add_to_neologisms_validated(self, word, original_pos_tag=None, corrected_pos_tag=None, lemma=None, formation_process=None):
        """
        Adiciona ou atualiza uma palavra na tabela NeologismValidated
        com a classe gramatical corrigida e o processo de formação.
        """
        word_lower = word.lower()
        try:
            neologism, created = NeologismValidated.objects.get_or_create(
                word=word_lower,
                defaults={
                    'pos_tag': corrected_pos_tag or POS_MAPPING.get(original_pos_tag, original_pos_tag), # Salva o POS corrigido (ou o original mapeado)
                    'lemma': lemma,
                    'formation_process': formation_process
                }
            )
            if not created:
                if corrected_pos_tag: neologism.pos_tag = corrected_pos_tag
                elif original_pos_tag: neologism.pos_tag = POS_MAPPING.get(original_pos_tag, original_pos_tag) # Atualiza se já existia mas não tinha o POS
                if lemma: neologism.lemma = lemma
                if formation_process: neologism.formation_process = formation_process
                neologism.save()
            
            # Se a palavra é validada como neologismo, certifique-se de que não está em CustomAddition
            CustomAddition.objects.filter(word=word_lower).delete()

            return True
        except Exception as e:
            print(f"Erro ao adicionar/atualizar '{word_lower}' em NeologismValidated: {e}")
            return False

    # Exportar resultados para CSV (ajustado para buscar do DB para formação se validado)
    def export_results_to_csv(self, results, filename="neologisms.csv"):
        import csv
        filepath = os.path.join(DATA_DIR, filename)
        
        pos_mapping = POS_MAPPING 

        def get_formation_process_for_csv(word_lower, original_pos_tag):
            # Tenta buscar a classificação do DB NeologismValidated
            validated_neo = NeologismValidated.objects.filter(word=word_lower).first()
            if validated_neo and validated_neo.formation_process:
                return validated_neo.formation_process
            
            # Heurística simples se não estiver no DB ou não classificado
            if original_pos_tag:
                 if word_lower.endswith("mente") and original_pos_tag == "ADJ": return "Derivação sufixal (sugerido)"
                 if word_lower.startswith("des") and original_pos_tag == "VERB": return "Derivação prefixal (sugerido)"
                 if re.match(r'^[a-zA-Z]+[_-][a-zA-Z]+$', word_lower): return "Composição (sugerido)"
                 if any(char in 'kqwy' for char in word_lower) and len(word_lower) > 3: return "Estrangeirismo (sugerido)?"
            return "Outros casos / Não classificado"

        fieldnames = ['neologismo', 'lema', 'classe_gramatical', 'processo_formacao', 'sentenca']
        
        unique_candidates = {}
        for candidate in results['neologism_candidates']:
            word_lower = candidate['word_lower']
            if word_lower not in unique_candidates:
                unique_candidates[word_lower] = {
                    'word': candidate['word'],
                    'original_pos': candidate['original_pos'],
                    'lemma': candidate['lemma'],
                    'sentences_idx': [candidate['sentence_idx']]
                }
            else:
                if candidate['sentence_idx'] not in unique_candidates[word_lower]['sentences_idx']:
                    unique_candidates[word_lower]['sentences_idx'].append(candidate['sentence_idx'])

        data_to_export = []
        for word_lower, details in unique_candidates.items():
            original_word = details['word']
            original_pos_tag_spacy = details['original_pos'] # O POS do spaCy
            original_lemma_spacy = details['lemma'] # <--- PEGAR O LEMA ORIGINAL DO SPACY
            
            validated_neo = NeologismValidated.objects.filter(word=word_lower).first()
            
            # Classe gramatical para CSV: usa a corrigida pelo usuário, senão a do spaCy mapeada
            csv_pos_tag = validated_neo.pos_tag if validated_neo and validated_neo.pos_tag else pos_mapping.get(original_pos_tag_spacy, 'Não identificado')

            csv_lemma = validated_neo.lemma if validated_neo and validated_neo.lemma else original_lemma_spacy

            # Processo de formação: usa o corrigido pelo usuário, senão a heurística
            csv_formation_process = get_formation_process_for_csv(word_lower, original_pos_tag_spacy)

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

# Instancie o detector uma vez para reuso (global ou passado por DI)
# Em um ambiente Django, pode ser melhor instanciar no request ou em um AppConfig
# Por simplicidade inicial, faremos global aqui, mas considere injeção de dependência.
detector = NeologismDetector()