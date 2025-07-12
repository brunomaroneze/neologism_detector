# neologism_app/services.py

import os
import re
import requests
from bs4 import BeautifulSoup
import spacy
import json
from collections import deque # Para um cache LRU simples

# --- Configurações e Caminhos ---
# Obtenha o caminho absoluto do diretório 'data'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

LEXICON_PATH = os.path.join(DATA_DIR, 'lexicon.txt')
CUSTOM_ADDITIONS_PATH = os.path.join(DATA_DIR, 'custom_additions.txt')
NEOLOGISMS_VALIDATED_PATH = os.path.join(DATA_DIR, 'neologisms_validated.txt')
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

# --- Funções Auxiliares ---
def load_wordlist(file_path):
    """Carrega palavras de um arquivo de texto para um conjunto."""
    words = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word and not word.startswith('#'): # Ignora linhas vazias ou comentários
                    words.add(word)
    except FileNotFoundError:
        print(f"Aviso: Arquivo '{file_path}' não encontrado. Criando um vazio.")
        open(file_path, 'a', encoding='utf-8').close() # Cria o arquivo se não existir
    return words

def add_word_to_file(word, file_path):
    """Adiciona uma palavra a um arquivo, se ela ainda não existir."""
    word = word.lower()
    existing_words = load_wordlist(file_path)
    if word not in existing_words:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(word + '\n')
        return True
    return False

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


# --- Detector de Neologismos Principal ---
class NeologismDetector:
    def __init__(self):
        self.lexicon = self._load_full_lexicon()
        self.nlp = self._load_spacy_model()

    def _load_full_lexicon(self):
        """Carrega o léxico base e as adições personalizadas do usuário."""
        base_lexicon = load_wordlist(LEXICON_PATH)
        custom_additions = load_wordlist(CUSTOM_ADDITIONS_PATH)
        return base_lexicon.union(custom_additions)

    def _load_spacy_model(self):
        """Carrega o modelo spaCy para português."""
        try:
            return spacy.load("pt_core_news_sm")
        except OSError:
            print("Modelo spaCy 'pt_core_news_sm' não encontrado. Baixando...")
            spacy.cli.download("pt_core_news_sm")
            return spacy.load("pt_core_news_sm")

    def reload_lexicon(self):
        """Recarrega o léxico após uma atualização."""
        self.lexicon = self._load_full_lexicon()

    def process_text(self, text):
        """
        Processa o texto para detectar neologismos.
        Retorna o texto HTML marcado e uma lista de candidatos.
        """
        doc = self.nlp(text)
        processed_html_parts = []
        neologism_candidates = []
        total_words = 0
        num_neologisms = 0

        # Para controlar palavras já processadas (para evitar duplicatas em neologismo_candidates)
        # e palavras já checadas como NEs para não refazer a busca no Dicio
        seen_neologism_candidates = set()
        
        # Sentenças para exportação CSV
        sentences = [sent.text for sent in doc.sents]

        for token in doc:
            # 1. Manter espaços e ignorar pontuação/números se não forem palavras para análise
            if token.is_space: # Adiciona o espaço e continua para o próximo token
                processed_html_parts.append(token.text)
                continue
            if token.is_punct or token.like_num: # Ignora pontuação ou números
                processed_html_parts.append(token.text + token.whitespace_)
                continue

            word_lower = token.text.lower()
            original_word = token.text
            
            # Limpeza adicional para remover pontuação anexada à palavra
            clean_word_lower = re.sub(r'^\W+|\W+$', '', word_lower)
            clean_original_word = re.sub(r'^\W+|\W+$', '', original_word)

            if not clean_word_lower: # Se a palavra ficou vazia após limpeza, pular
                processed_html_parts.append(original_word + token.whitespace_)
                continue

            total_words += 1
            is_neologism_candidate = False

            # 2. Eliminar Nomes Próprios (Feature 1)
            # spaCy marca Nomes Próprios (PROPN) e Entidades Nomeadas (PERSON, LOC, ORG, etc.)
            if token.pos_ == "PROPN" or token.ent_type_ in ["PERSON", "LOC", "ORG", "MISC"]:
                processed_html_parts.append(original_word + token.whitespace_)
                continue

            # 3. FILTRAR POR CLASSE GRAMATICAL (NOVA CONDIÇÃO)
            # Somente processa palavras que podem ser neologismos (substantivo, adjetivo, verbo)
            if token.pos_ not in CANDIDATE_POS_TAGS:
                # print(f"DEBUG: '{original_word}' é '{token.pos_}'. Ignorando por tipo.")
                processed_html_parts.append(original_word + token.whitespace_)
                continue

            # 4. Verificar no Léxico Local
            if clean_word_lower not in self.lexicon:
                # 5. Enriquecer o filtro com Dicio.com.br (Feature 2)
                if not is_word_in_dicio(clean_word_lower):
                    is_neologism_candidate = True
                else:
                    self.add_to_custom_additions(clean_word_lower)
            
            if is_neologism_candidate:
                num_neologisms += 1
                processed_html_parts.append(
                    f'<span class="neologism" data-word="{clean_original_word}" '
                    f'data-pos="{POS_MAPPING.get(token.pos_, token.pos_)}" data-lemma="{token.lemma_}" ' # <--- USAR POS_MAPPING AQUI
                    f'data-sent-idx="{self._get_sentence_index(token, doc)}">'
                    f'{original_word}</span>{token.whitespace_}' # <--- ADICIONAR O WHITESPACE
                )
                if clean_word_lower not in seen_neologism_candidates:
                    # Feature 4: Classe Gramatical
                    neologism_candidates.append({
                        'word': clean_original_word,
                        'word_lower': clean_word_lower,
                        'pos': POS_MAPPING.get(token.pos_, token.pos_), # <--- USAR POS_MAPPING AQUI
                        'lemma': token.lemma_,
                        'sentence_idx': self._get_sentence_index(token, doc)
                    })
                    seen_neologism_candidates.add(clean_word_lower)
            else:
                processed_html_parts.append(original_word + token.whitespace_) # <--- ADICIONAR O WHITESPACE

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

    # Feature 3: Validação pelo Usuário
    def add_to_custom_additions(self, word):
        """Adiciona uma palavra à lista de adições personalizadas (não neologismos)."""
        if add_word_to_file(word, CUSTOM_ADDITIONS_PATH):
            self.reload_lexicon() # Recarrega o léxico para incluir a nova palavra
            return True
        return False

    def add_to_neologisms_validated(self, word):
        """Adiciona uma palavra à lista de neologismos validados (para ML futuro)."""
        return add_word_to_file(word, NEOLOGISMS_VALIDATED_PATH)

    # Feature 6: Exportar para CSV
    def export_results_to_csv(self, results, filename="neologisms.csv"):
        """
        Exporta os resultados da detecção de neologismos para um arquivo CSV.
        `results` deve ser o dicionário retornado por `process_text`.
        """
        import csv

        filepath = os.path.join(DATA_DIR, filename) # Salva no diretório data
        
        # Mapeia POS tags do spaCy para classes gramaticais mais legíveis
        pos_mapping = POS_MAPPING

        # Simulação para 'processo de formação' (Feature 5 - futura)
        # Por enquanto, será "Não classificado" ou "Outros"
        # Em uma implementação futura, isso seria preenchido por um modelo de ML
        def get_formation_process(word, pos):
            # Lógica simples/placeholder, para ser substituída por ML
            if word and pos:
                 # Exemplo: identificar possível sufixação/prefixação muito simples
                 if word.endswith("mente") and pos == "ADV": return "Derivação Sufixal"
                 if word.startswith("des") and pos == "VERB": return "Derivação Prefixal"
                 if re.match(r'^[a-zA-Z]+[_-][a-zA-Z]+$', word): return "Composição" # Ex: "web-site"
                 # Você pode adicionar uma heurística simples para estrangeirismos
                 # Por exemplo, se a palavra for muito curta e não tiver vogais (ou poucas),
                 # ou tiver padrões de letras incomuns em português.
                 # Mas isso é complexo e ML será melhor.
                 if any(char in 'kqwy' for char in word.lower()): return "Estrangeirismo?" # Heurística fraca
            return "Outros casos / Não classificado"


        fieldnames = ['neologismo', 'classe_gramatical', 'processo_formacao', 'sentenca']
        
        # Filtra os candidatos únicos para o CSV (mesmo que o texto original tenha repetições)
        unique_candidates = {}
        for candidate in results['neologism_candidates']:
            word_lower = candidate['word_lower']
            if word_lower not in unique_candidates:
                unique_candidates[word_lower] = {
                    'word': candidate['word'],
                    'pos': candidate['pos'],
                    'lemma': candidate['lemma'],
                    'sentences_idx': [candidate['sentence_idx']] # Armazena todos os índices de sentenças
                }
            else:
                if candidate['sentence_idx'] not in unique_candidates[word_lower]['sentences_idx']:
                    unique_candidates[word_lower]['sentences_idx'].append(candidate['sentence_idx'])

        data_to_export = []
        for word_lower, details in unique_candidates.items():
            original_word = details['word']
            pos_tag = details['pos']
            sentence_indices = details['sentences_idx']

            for s_idx in sentence_indices:
                sentence_text = results['sentences'][s_idx] if s_idx != -1 else "Sentença não encontrada"
                
                # Highlight the neologism in the sentence
                # Using regex for word boundary to avoid matching "casa" in "casar"
                pattern = r'\b' + re.escape(original_word) + r'\b'
                highlighted_sentence = re.sub(pattern, f'<{original_word}>', sentence_text, flags=re.IGNORECASE)


                data_to_export.append({
                    'neologismo': original_word,
                    'classe_gramatical': pos_mapping.get(pos_tag, 'Não identificado'),
                    'processo_formacao': get_formation_process(original_word, pos_tag), # Placeholder
                    'sentenca': highlighted_sentence # Sentença com a palavra marcada
                })
        
        # Ordena para melhor visualização
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