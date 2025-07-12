# neologism_app/views.py

from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
import json
import os

# Importar o detector do services.py
from .services import detector, DICIO_CACHE_PATH # Importa a instância do detector e o caminho do cache

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
    Endpoint para validação de neologismos via AJAX (Feature 3).
    O usuário pode marcar uma palavra como neologismo real ou como "não é neologismo" (erro/palavra comum).
    """
    if request.method == 'POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        word = request.POST.get('word')
        action = request.POST.get('action') # 'accept_neologism' ou 'reject_neologism'

        if not word or not action:
            return JsonResponse({'status': 'error', 'message': 'Dados inválidos.'}, status=400)

        if action == 'reject_neologism': # Usuário diz "não é neologismo", adicionar ao léxico local
            if detector.add_to_custom_additions(word):
                message = f"'{word}' adicionado ao seu léxico pessoal. Não será mais marcado como neologismo."
                status = 'success'
            else:
                message = f"'{word}' já está no seu léxico pessoal."
                status = 'info'
            # Remove a palavra do cache do Dicio se ela foi rejeitada,
            # para forçar uma nova consulta se ela for marcada como neologismo por engano
            # ou para limpar o cache de palavras que o usuário validou como 'comuns'.
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


        elif action == 'accept_neologism': # Usuário diz "é um neologismo", para o treinamento futuro de ML
            if detector.add_to_neologisms_validated(word):
                message = f"'{word}' marcado como neologismo validado. Ótimo para o futuro treinamento de ML!"
                status = 'success'
            else:
                message = f"'{word}' já está na lista de neologismos validados."
                status = 'info'
        else:
            return JsonResponse({'status': 'error', 'message': 'Ação inválida.'}, status=400)
        
        # Após a validação, se quiser reprocessar o texto automaticamente, você pode fazer
        # isso aqui e retornar os novos resultados HTML/candidatos.
        # Por enquanto, apenas retornamos a mensagem de sucesso.
        
        # O ideal seria que a página de resultados fosse atualizada dinamicamente
        # após a validação, removendo o item da lista de candidatos e do texto marcado.
        # Isso pode ser feito com JavaScript no `results.html`.

        return JsonResponse({'status': status, 'message': message})
    return JsonResponse({'status': 'error', 'message': 'Método não permitido.'}, status=405)


def export_csv(request):
    """
    Exporta os resultados da última detecção para um arquivo CSV (Feature 6).
    """
    detection_results = request.session.get('detection_results', None)
    if not detection_results or not detection_results.get('neologism_candidates'):
        messages.error(request, "Nenhum resultado de neologismos para exportar.")
        return redirect('neologism_app:results') # Redireciona para a página de resultados

    csv_filepath = detector.export_results_to_csv(detection_results)

    if csv_filepath and os.path.exists(csv_filepath):
        # Adiciona uma mensagem de sucesso antes de retornar o arquivo
        messages.success(request, f"Arquivo CSV '{os.path.basename(csv_filepath)}' gerado com sucesso!")
        
        # O Django Messages não é exibido automaticamente se o retorno for um HttpResponse de arquivo.
        # Uma estratégia comum é forçar um redirecionamento *após* o download ou lidar com isso via JS.
        # No entanto, o navegador geralmente lida com o download e não exibe a mensagem de redirecionamento.
        # Para ver a mensagem, o usuário precisaria voltar à página de resultados.
        #
        # A melhor abordagem para feedback imediato para download é via JavaScript.
        # Por enquanto, manteremos a mensagem do Django (que é útil se o download for via link e não JS).
        
        with open(csv_filepath, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(csv_filepath)}"'
            return response
    else:
        messages.error(request, "Erro ao gerar o arquivo CSV.")
        return redirect('neologism_app:results')