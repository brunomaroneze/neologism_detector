# neologism_app/services.py

import os
import re
import requests
from bs4 import BeautifulSoup
import spacy
import json
from collections import deque # Para um cache LRU simples
import html

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
    """
    word_lower = word.lower()

    # 1. Verifica no cache em memória
    if word_lower in DICIO_CACHE:
        # Move a chave para o final do deque (mais recentemente usada)
        if word_lower in DICIO_CACHE_KEYS:
            DICIO_CACHE_KEYS.remove(word_lower)
        DICIO_CACHE_KEYS.append(word_lower)
        return DICIO_CACHE[word_lower]

    # 2. Se não estiver no cache, tenta buscar
    print(f"Buscando '{word_lower}' no Dicio.com.br...")
    url = f"https://www.dicio.com.br/{word_lower}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status() # Lança um HTTPError para 4xx/5xx erros
        soup = BeautifulSoup(response.text, 'html.parser')

        # Dicio.com.br retorna 404 ou uma página de "não encontrado" para palavras inexistentes.
        # Procuramos por elementos que indicam uma definição.
        # Elementos comuns que indicam definição: <p class="significado">, <div class="conjugacao">, etc.
        # Ou a ausência de uma mensagem de "palavra não encontrada".
        # Vamos procurar por uma div com a classe "significado" ou "tit-sub" (para conjugação, etc.)
        definition_found = soup.find('p', class_='significado') or \
                           soup.find('div', class_='tit-sub')

        is_present = definition_found is not None

        # 3. Adiciona ao cache
        if len(DICIO_CACHE_KEYS) >= MAX_CACHE_SIZE:
            oldest_key = DICIO_CACHE_KEYS.popleft() # Remove o item menos usado
            DICIO_CACHE.pop(oldest_key, None)

        DICIO_CACHE[word_lower] = is_present
        DICIO_CACHE_KEYS.append(word_lower)
        save_dicio_cache() # Salva o cache em disco após cada adição

        return is_present
    except requests.exceptions.RequestException as e:
        print(f"Erro ao consultar Dicio para '{word_lower}': {e}")
        # Em caso de erro, assumimos que a palavra não foi encontrada para não travar.
        # Podemos cachear como False ou não cachear, dependendo da estratégia.
        DICIO_CACHE[word_lower] = False # Cacheia como False para evitar novas tentativas com erros
        DICIO_CACHE_KEYS.append(word_lower)
        save_dicio_cache()
        return False
    except Exception as e:
        print(f"Erro inesperado ao parsear Dicio para '{word_lower}': {e}")
        DICIO_CACHE[word_lower] = False
        DICIO_CACHE_KEYS.append(word_lower)
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
            if token.is_space:
                processed_html_parts.append(token.text)
                continue
            if token.is_punct or token.like_num:
                processed_html_parts.append(token.text + token.whitespace_)
                continue

            word_lower = token.text.lower()
            original_word = token.text
            
            clean_word_lower = re.sub(r'^\W+|\W+$', '', word_lower)
            clean_original_word = re.sub(r'^\W+|\W+$', '', original_word)

            if not clean_word_lower:
                processed_html_parts.append(original_word + token.whitespace_)
                continue

            total_words += 1
            is_neologism_candidate = False

            if token.pos_ == "PROPN" or token.ent_type_ in ["PERSON", "LOC", "ORG", "MISC"]:
                processed_html_parts.append(original_word + token.whitespace_)
                continue

            if token.pos_ not in CANDIDATE_POS_TAGS:
                processed_html_parts.append(original_word + token.whitespace_)
                continue

            # ================================================================
            # NOVA LÓGICA CHAVE: Verificação do Léxico no Banco de Dados
            # ================================================================

            # 1. Verificar a forma limpa da palavra (ex: "amá-lo", "oferta-relâmpago")
            found_by_word_form = LexiconWord.objects.filter(word=clean_word_lower).exists() or \
                                 CustomAddition.objects.filter(word=clean_word_lower).exists()

            # 2. Verificar o lema do spaCy
            found_by_lemma = False
            lemma_to_check = token.lemma_.lower() # Converte o lema para minúsculas

            if ' ' in lemma_to_check:
                # Se o lema é composto (ex: "amar ele"), verifica apenas o primeiro componente (o verbo)
                main_lemma_part = lemma_to_check.split(' ')[0]
                if LexiconWord.objects.filter(word=main_lemma_part).exists() or \
                   CustomAddition.objects.filter(word=main_lemma_part).exists():
                    found_by_lemma = True
            else:
                # Se o lema é uma única palavra (ex: "casa", "vender", "elegiar")
                if LexiconWord.objects.filter(word=lemma_to_check).exists() or \
                   CustomAddition.objects.filter(word=lemma_to_check).exists():
                    found_by_lemma = True

            is_word_in_db_lexicon = found_by_word_form or found_by_lemma

            # ================================================================
            # FIM DA NOVA LÓGICA
            # ================================================================

            if not is_word_in_db_lexicon: # Se a palavra (ou seu lema) NÃO está no léxico do DB
                # 5. Enriquecer o filtro com Dicio.com.br
                if not is_word_in_dicio(clean_word_lower): # Ainda consulta Dicio com a forma limpa
                    is_neologism_candidate = True
                else:
                    # Se encontrada no Dicio, adicionar a CustomAddition (via DB)
                    # Adicionamos a forma limpa da palavra que foi encontrada no Dicio.
                    self.add_to_custom_additions(clean_word_lower)

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