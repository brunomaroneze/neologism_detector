# neologism_app/views.py

from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
import json
import os

# Importar o detector do services.py
from .services import detector, DICIO_CACHE_PATH, POS_MAPPING, FORMATION_PROCESS_OPTIONS # <--- Importar mapeamentos e opções

def index(request):
    """View para a página inicial com o formulário de entrada de texto."""
    if request.method == 'POST':
        text = request.POST.get('text_input', '').strip()
        if not text:
            messages.error(request, "Por favor, insira um texto para análise.")
            return render(request, 'neologism_app/index.html')

        # Armazena o texto na sessão para reuso se o usuário voltar à página
        request.session['text_to_process'] = text

        # Processa o texto usando o detector
        # A instância 'detector' é global e persistente, evitando recarregar o modelo spaCy
        results = detector.process_text(text)
        
        # Armazena os resultados na sessão para serem acessados pela página de resultados
        request.session['detection_results'] = results

        messages.success(request, "Texto processado com sucesso!")
        return redirect('neologism_app:results')

    return render(request, 'neologism_app/index.html')

def results(request):
    """View para exibir os resultados da detecção."""
    detection_results = request.session.get('detection_results', None)
    text_to_process = request.session.get('text_to_process', '')

    if not detection_results:
        messages.info(request, "Nenhum texto foi processado ainda. Por favor, insira um texto na página inicial.")
        return redirect('neologism_app:index')

    context = {
        'processed_text_html': detection_results.get('processed_text_html', ''),
        'neologism_candidates': detection_results.get('neologism_candidates', []),
        'total_words': detection_results.get('total_words', 0),
        'num_neologisms': detection_results.get('num_neologisms', 0),
        'original_text': text_to_process,
    }
    return render(request, 'neologism_app/results.html', context)

def validate_neologism(request):
    """
    Endpoint para validação de neologismos via AJAX.
    Lida com:
    - 'reject_neologism': usuário diz que não é neologismo.
    - 'save_classification': usuário fornece classificação detalhada.
    """
    if request.method == 'POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        word = request.POST.get('word')
        action = request.POST.get('action') # 'reject_neologism' ou 'save_classification'

        if not word or not action:
            return JsonResponse({'status': 'error', 'message': 'Dados inválidos.'}, status=400)

        if action == 'reject_neologism': # Usuário diz "não é neologismo"
            if detector.add_to_custom_additions(word):
                message = f"'{word}' adicionado ao seu léxico pessoal e removido dos neologismos."
                status = 'success'
            else:
                message = f"'{word}' já está no seu léxico pessoal."
                status = 'info'
            
            # Limpa o cache do Dicio para a palavra rejeitada, se houver
            try:
                with open(DICIO_CACHE_PATH, 'r+', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    if word.lower() in cache_data:
                        del cache_data[word.lower()]
                        f.seek(0)
                        json.dump(cache_data, f, ensure_ascii=False, indent=4)
                        f.truncate()
            except Exception as e:
                print(f"Erro ao atualizar cache do Dicio para '{word}': {e}")

        elif action == 'save_classification': # Usuário classificou o neologismo
            original_pos_tag = request.POST.get('original_pos_tag') # POS do spaCy
            corrected_pos_tag = request.POST.get('corrected_pos_tag') # POS corrigido pelo usuário (em português)
            lemma = request.POST.get('lemma')
            formation_process = request.POST.get('formation_process')

            if detector.add_to_neologisms_validated(word, original_pos_tag, corrected_pos_tag, lemma, formation_process):
                message = f"'{word}' classificado e salvo com sucesso para futuro treinamento de ML!"
                status = 'success'
            else:
                message = f"Erro ao salvar a classificação de '{word}'."
                status = 'error'
            
            # Não remove do cache do Dicio aqui, pois a palavra é um neologismo validado.

        else:
            return JsonResponse({'status': 'error', 'message': 'Ação inválida.'}, status=400)
        
        return JsonResponse({'status': status, 'message': message})
    return JsonResponse({'status': 'error', 'message': 'Método não permitido.'}, status=405)

def export_csv(request):
    """
    Exporta os resultados da última detecção para um arquivo CSV (Feature 6).
    """
    detection_results = request.session.get('detection_results', None)
    
    # Verifica se há neologismos detectados, não apenas se a chave existe
    if not detection_results or not detection_results.get('neologism_candidates'):
        messages.error(request, "Nenhum neologismo detectado para exportar para CSV.")
        return redirect('neologism_app:results')

    csv_filepath = detector.export_results_to_csv(detection_results)

    if csv_filepath and os.path.exists(csv_filepath):
        # A mensagem será exibida apenas se o usuário navegar de volta ou a página for recarregada
        messages.success(request, f"Arquivo CSV '{os.path.basename(csv_filepath)}' gerado com sucesso!")
        
        with open(csv_filepath, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(csv_filepath)}"'
            return response
    else:
        messages.error(request, "Erro ao gerar o arquivo CSV.")
        return redirect('neologism_app:results')