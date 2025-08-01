    document.addEventListener('DOMContentLoaded', function() {
    // Obtenção de elementos DOM
    const rejectButtons = document.querySelectorAll('.validate-btn[data-action="reject_neologism"]');
    const classifyButtons = document.querySelectorAll('.classify-btn');
    const processedTextDisplay = document.getElementById('processed-text-display');
    const neologismCandidateContainer = document.getElementById('neologism-candidate-container');
    const numNeologismsSpan = document.getElementById('num-neologisms');
    const ajaxMessageContainer = document.getElementById('ajax-message-container');

    // Modal elements
    const modal = document.getElementById('neologismModal');
    const closeButtons = modal.querySelectorAll('.close-button');
    const modalWord = document.getElementById('modal-word');
    const modalOriginalPos = document.getElementById('modal-original-pos');
    const modalLemmaInput = document.getElementById('modal-lemma-input');
    const modalCorrectedPos = document.getElementById('modal-corrected-pos');
    const modalFormationProcess = document.getElementById('modal-formation-process');
    const modalSentenceContext = document.getElementById('modal-sentence-context');
    const modalSaveButton = document.getElementById('modal-save-button');
    const modalRejectButton = modal.querySelector('.modal-reject-btn');
    const modalPredictedFormation = document.getElementById('modal-predicted-formation'); // NOVO ELEMENTO

    // Obter CSRF token do meta tag
    const csrftoken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');

    // Função para exibir mensagens AJAX
    function showAjaxMessage(message, type = 'info') {
        ajaxMessageContainer.textContent = message;
        ajaxMessageContainer.classList.remove('hidden', 'status-processing', 'status-complete', 'status-error', 'status-info', 'status-success');
        ajaxMessageContainer.classList.add(`status-${type}`);
        ajaxMessageContainer.classList.remove('hidden'); 
        setTimeout(() => {
            ajaxMessageContainer.classList.add('hidden');
        }, 5000);
    }

    // Função para remover a palavra do DOM (lista e texto marcado)
    function removeWordFromUI(wordLower) {
        // Remover da lista de candidatos
        const candidateItem = neologismCandidateContainer.querySelector(`.candidate-item[data-word-lower="${wordLower}"]`);
        if (candidateItem) {
            candidateItem.remove();
            let currentNumNeologisms = parseInt(numNeologismsSpan.textContent);
            numNeologismsSpan.textContent = Math.max(0, currentNumNeologisms - 1);
        }

        // Remover a marcação do texto processado
        const markedWords = processedTextDisplay.querySelectorAll(`.neologism[data-word="${wordLower}"]`); // Case-insensitive
        markedWords.forEach(el => {
            const textNode = document.createTextNode(el.textContent);
            el.parentNode.replaceChild(textNode, el);
        });

        // Se não houver mais candidatos, exibir mensagem
        if (neologismCandidateContainer && neologismCandidateContainer.children.length === 0) {
            const candidateListSection = neologismCandidateContainer.closest('.candidate-list');
            if (candidateListSection) {
                candidateListSection.innerHTML = '<p class="status-message status-complete">Todos os candidatos foram validados!</p>';
            }
        }
    }

    // Função para lidar com requisições AJAX
    async function sendValidationRequest(word, action, payload = {}) {
        showAjaxMessage('Processando...', 'processing');
        try {
            const formData = new URLSearchParams();
            formData.append('word', word);
            formData.append('action', action);
            for (const key in payload) {
                formData.append(key, payload[key]);
            }

            const response = await fetch('/validate_neologism/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrftoken,
                    'X-Requested-With': 'XMLHttpRequest',
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: formData.toString()
            });

            const data = await response.json();
            showAjaxMessage(data.message, data.status);
            return data;
        } catch (error) {
            console.error('Erro na requisição AJAX:', error);
            showAjaxMessage('Ocorreu um erro ao processar sua solicitação.', 'error');
            return { status: 'error', message: 'Erro de rede ou servidor.' };
        }
    }

    // --- Event Listeners ---

    // 1. Botões "Não é neologismo" na lista de candidatos
    rejectButtons.forEach(button => {
        button.addEventListener('click', async function() {
            const word = this.dataset.word;
            const data = await sendValidationRequest(word, 'reject_neologism');
            if (data.status === 'success' || data.status === 'info') {
                removeWordFromUI(word.toLowerCase());
            }
        });
    });

    // 2. Botões "Classificar" na lista de candidatos
    classifyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const word = this.dataset.word;
            const originalPos = this.dataset.originalPos;
            const pos = this.dataset.pos;
            const lemma = this.dataset.lemma;
            const sentenceText = this.dataset.sentenceText;
            const predictedFormation = this.dataset.predictedFormation; // <--- NOVO: Pegar a predição
            
            openModal(word, originalPos, pos, lemma, sentenceText, predictedFormation);
        });
    });

    // 3. Clique nas palavras marcadas no texto
    processedTextDisplay.addEventListener('click', function(event) {
        const target = event.target;
        if (target.classList.contains('neologism')) {
            const word = target.dataset.word;
            const originalPos = target.dataset.originalPos;
            const pos = target.dataset.pos;
            const lemma = target.dataset.lemma;
            const sentenceIdx = target.dataset.sentIdx;
            const sentenceText = target.dataset.sentenceText || ''; 
            const predictedFormation = target.dataset.predictedFormation; // <--- NOVO: Pegar a predição
            // Precisa do texto da sentença. Poderíamos passar no data-attribute do span.
            // Ou buscar do array 'sentences' guardado na sessão, se for passado ao JS.
            // Por simplicidade, faremos um fetch ou guardaremos um mapa.
            // Por agora, vamos buscar a sentença do neologism_candidates array
            // (assumindo que a sessão de results contém neologism_candidates com sentence_text)
            // Ou podemos fazer um AJAX para buscar a sentença específica por ID se ela não estiver na UI.
            // A melhor forma é já vir no data-attribute ou no objeto passado para o JS.
            // Como já incluímos `sentence_text` no `neologism_candidates` e `data-sentence-text` no botão classify-btn,
            // vamos buscar a sentença correspondente da lista de candidatos na sessão.
            
            // Para simplificar, vou assumir que `data-sentence-text` foi adicionado ao SPAN.
            // Se não estiver, precisará buscar essa informação de algum lugar.
            // ALTERNATIVA: `const sentenceText = target.closest('.processed-text').textContent.split(/[.!?]\s*/)[parseInt(sentenceIdx)] || '';`
            // Essa alternativa é frágil. Idealmente a sentença completa viria via data-attribute ou um objeto JS pré-carregado.
            // Pela sua pergunta original "extraída por meio de algum algoritmo de NLP que faça a segmentação do texto",
            // a sentença já está sendo gerada e armazenada no backend.
            // Se o span não tiver 'data-sentence-text', precisamos fazer um loop nos 'neologism_candidates' (se disponível em JS)
            // ou retornar a sentença completa do backend.
            // No services.py, já adicionei `sentence_text` ao neologism_candidates.
            // Vou adicionar ao data-attribute do span também para facilitar.

            // Por enquanto, se o `neologism` span não tiver `data-sentence-text`, será vazio.
            // const sentenceText = target.dataset.sentenceText || ''; 
            
            openModal(word, originalPos, pos, lemma, sentenceText, predictedFormation);
        }
    });

    // 4. Botão "Salvar Classificação" dentro do modal
    modalSaveButton.addEventListener('click', async function() {
        const word = modalWord.textContent;
        const originalPosTag = modalOriginalPos.dataset.originalPos; // Pega o POS do spaCy
        const correctedPosTag = modalCorrectedPos.value;
        const lemma = modalLemmaInput.value;
        const formationProcess = modalFormationProcess.value;

        // Validar se pelo menos uma opção foi selecionada para classe ou formação (opcional, mas boa prática)
        if (!correctedPosTag && !formationProcess) {
            showAjaxMessage('Selecione ao menos a classe gramatical ou o processo de formação.', 'error');
            return;
        }

        const payload = {
            original_pos_tag: originalPosTag,
            corrected_pos_tag: correctedPosTag,
            lemma: lemma,
            formation_process: formationProcess
        };

        const data = await sendValidationRequest(word, 'save_classification', payload);
        if (data.status === 'success') {
            closeModal();
            // A palavra não é removida da UI, pois foi validada como neologismo real
            // Se desejar, pode-se mudar a cor ou um ícone para indicar que foi classificada.
            // Por enquanto, ela permanece visível mas com a classificação salva.
        }
    });

    // 5. Botão "Não é neologismo" dentro do modal
    modalRejectButton.addEventListener('click', async function() {
        const word = modalWord.textContent;
        const data = await sendValidationRequest(word, 'reject_neologism');
        if (data.status === 'success' || data.status === 'info') {
            removeWordFromUI(word.toLowerCase());
            closeModal();
        }
    });

    // --- Funções do Modal ---
    function openModal(word, originalPos, pos, lemma, sentenceText, predictedFormation) {
        modalWord.textContent = word;
        modalOriginalPos.textContent = pos; // Exibe o POS mapeado (Substantivo, Adjetivo, Verbo)
        modalOriginalPos.dataset.originalPos = originalPos; // Guarda o POS original do spaCy
        modalLemmaInput.value = lemma;
        modalSentenceContext.textContent = sentenceText;

        modalPredictedFormation.textContent = predictedFormation || 'N/A'; // Exibe a predição ML
        // Pre-selecionar o select do processo de formação com a sugestão ML
        modalFormationProcess.value = predictedFormation || ''; // Pré-seleciona, se houver. Senão, fica vazio.

        // Pre-selecionar a classe gramatical sugerida se for uma das opções do select
        const mappedOriginalPos = POS_MAPPING[originalPos] || 'Outros'; // Use o mapa JS para consistency
        modalCorrectedPos.value = mappedOriginalPos;
        
        modal.style.display = 'flex'; // Exibe o modal
    }

    function closeModal() {
        modal.style.display = 'none'; // Esconde o modal
    }

    closeButtons.forEach(btn => btn.addEventListener('click', closeModal));

    // Fechar modal ao clicar fora do conteúdo
    window.addEventListener('click', function(event) {
        if (event.target == modal) {
            closeModal();
        }
    });
    
    // Mapeamento de POS do spaCy para strings amigáveis em JS para uso no modal (se necessário para a lógica JS)
    // Isso é opcional se você já passa a string mapeada via data-attribute
    const POS_MAPPING = {
        "NOUN": "Substantivo",
        "ADJ": "Adjetivo",
        "VERB": "Verbo",
        "PROPN": "Substantivo Próprio",
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
        "Outros": "Outros (não substantivo/adjetivo/verbo)"
    };
});