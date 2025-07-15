# neologism_app/views.py

from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
import json
import os
import io # Para lidar com arquivos em memória

# Importar o detector do services.py
from .services import detector, DICIO_CACHE_PATH, POS_MAPPING, FORMATION_PROCESS_OPTIONS # <--- Importar mapeamentos e opções

# Adicione o max_upload_size para prevenir ataques de upload de arquivos muito grandes
MAX_UPLOAD_SIZE = 5 * 1024 * 1024 # 5 MB

def index(request):
    if request.method == 'POST':
        text = None
        text_input_from_form = request.POST.get('text_input', '').strip()
        uploaded_file = request.FILES.get('file_upload')

        if text_input_from_form:
            text = text_input_from_form
            source_type = "textarea"
        elif uploaded_file:
            if uploaded_file.size > MAX_UPLOAD_SIZE:
                messages.error(request, f"O arquivo '{uploaded_file.name}' é muito grande. O tamanho máximo permitido é {MAX_UPLOAD_SIZE / (1024 * 1024):.1f} MB.")
                return render(request, 'neologism_app/index.html')
            
            if not uploaded_file.name.lower().endswith('.txt'):
                messages.error(request, f"Tipo de arquivo não suportado para '{uploaded_file.name}'. Por favor, envie apenas arquivos TXT.")
                return render(request, 'neologism_app/index.html')

            try:
                file_content_bytes = uploaded_file.read()
                try:
                    text = file_content_bytes.decode('utf-8')
                    source_type = f"arquivo '{uploaded_file.name}' (UTF-8)"
                except UnicodeDecodeError:
                    text = file_content_bytes.decode('latin-1')
                    source_type = f"arquivo '{uploaded_file.name}' (Latin-1)"
                
            except Exception as e:
                messages.error(request, f"Erro ao ler o arquivo '{uploaded_file.name}': {e}")
                return render(request, 'neologism_app/index.html')
        
        if not text:
            messages.error(request, "Por favor, insira um texto ou faça o upload de um arquivo TXT para análise.")
            return render(request, 'neologism_app/index.html')

        # Processa o texto usando o detector
        results = detector.process_text(text)
        
        # NOVO: Lógica para textos grandes
        if results.get('is_large_text_for_display'):
            # Para textos grandes, não salvamos na sessão para display, apenas exportamos o CSV
            # O `export_csv` já pega o `detection_results` da sessão. Então vamos colocá-lo lá temporariamente.
            request.session['detection_results'] = results
            
            # Gerar o nome do arquivo CSV
            csv_filename = f"neologismos_{uploaded_file.name.replace('.txt', '') if uploaded_file else 'analise'}.csv"

            # Chamar a função de exportação do service
            csv_filepath = detector.export_results_to_csv(results, filename=csv_filename) # Passa 'results' diretamente

            if csv_filepath and os.path.exists(csv_filepath):
                messages.success(request, f"O texto é muito longo para exibição. Um arquivo CSV ('{os.path.basename(csv_filepath)}') foi gerado e será baixado automaticamente.")
                with open(csv_filepath, 'rb') as f:
                    response = HttpResponse(f.read(), content_type='text/csv')
                    response['Content-Disposition'] = f'attachment; filename="{os.path.basename(csv_filepath)}"'
                    return response # Retorna o arquivo CSV diretamente para download
            else:
                messages.error(request, "Erro ao gerar o arquivo CSV para texto longo.")
                return redirect('neologism_app:index') # Redireciona para o início com erro

        else:
            # Para textos menores, salva na sessão e redireciona para a página de resultados normal
            request.session['text_to_process'] = text # O texto original, para reuso
            request.session['detection_results'] = results
            messages.success(request, f"Texto processado com sucesso! Fonte: {source_type}")
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
    detection_results = request.session.get('detection_results', None)
    if not detection_results or not detection_results.get('neologism_candidates'):
        messages.error(request, "Nenhum neologismo detectado para exportar.")
        return redirect('neologism_app:results')

    # Para textos menores, o CSV ainda será gerado pelo link na página de resultados.
    # O `detector.export_results_to_csv` já está adaptado para receber `results`.
    csv_filepath = detector.export_results_to_csv(detection_results)

    if csv_filepath and os.path.exists(csv_filepath):
        messages.success(request, f"Arquivo CSV '{os.path.basename(csv_filepath)}' gerado com sucesso!")
        with open(csv_filepath, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(csv_filepath)}"'
            return response
    else:
        messages.error(request, "Erro ao gerar o arquivo CSV.")
        return redirect('neologism_app:results')