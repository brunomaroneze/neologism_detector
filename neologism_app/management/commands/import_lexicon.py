# neologism_app/management/commands/import_lexicon.py

import os
from django.core.management.base import BaseCommand, CommandError
from neologism_app.models import LexiconWord, CustomAddition, NeologismValidated # Importe todos os modelos
from django.db import transaction

class Command(BaseCommand):
    help = 'Imports words from a given text file into the LexiconWord or CustomAddition database table.'

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='The path to the text file containing words, one per line.')
        parser.add_argument('--target', type=str, default='lexicon', 
                            help='Target table: "lexicon" for LexiconWord or "custom" for CustomAddition. Default is lexicon.')
        parser.add_argument('--batch-size', type=int, default=1000, 
                            help='Number of words to insert in a single batch. Default is 1000.')
        parser.add_argument('--delete-existing', action='store_true', 
                            help='Deletes all existing records in the target table before import.')


    def handle(self, *args, **options):
        file_path = options['file_path']
        target_table = options['target'].lower()
        batch_size = options['batch_size']
        delete_existing = options['delete_existing']

        if not os.path.exists(file_path):
            raise CommandError(f'File "{file_path}" does not exist.')

        if target_table not in ['lexicon', 'custom', 'validated']: # Adicionado 'validated' caso queira importar neologismos validados
            raise CommandError('Invalid target. Must be "lexicon", "custom", or "validated".')

        model = None
        if target_table == 'lexicon':
            model = LexiconWord
        elif target_table == 'custom':
            model = CustomAddition
        elif target_table == 'validated':
            model = NeologismValidated # Note: este modelo tem mais campos, a importação aqui seria mais simples (só a palavra)

        if delete_existing:
            self.stdout.write(self.style.WARNING(f'Deleting all existing records from {model.__name__}...'))
            model.objects.all().delete()
            self.stdout.write(self.style.SUCCESS('Existing records deleted.'))

        self.stdout.write(self.style.SUCCESS(f'Starting import from "{file_path}" to "{model.__name__}"...'))
        
        total_imported = 0
        total_skipped = 0
        words_to_create = []

        # Para lidar com codificações
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
        file_processed_successfully = False

        for encoding in encodings_to_try:
            try:
                line_num = 0
                with open(file_path, 'r', encoding=encoding) as f:
                    for line_num, line in enumerate(f, 1):
                        word = line.strip()
                        
                        if not word or word.startswith('#'):
                            continue
                        
                        # Para LexiconWord e CustomAddition, só precisamos da palavra
                        if model in [LexiconWord, CustomAddition]:
                            words_to_create.append(model(word=word.lower()))
                        elif model == NeologismValidated:
                            # Se for NeologismValidated, pode ser que o arquivo tenha mais dados
                            # Por simplicidade, aqui assumimos que também é apenas uma palavra por linha.
                            # Se seu arquivo de neologismos validados tiver mais colunas (ex: CSV),
                            # você precisará de uma lógica de parseamento mais complexa aqui.
                            words_to_create.append(model(word=word.lower()))

                        if len(words_to_create) >= batch_size:
                            try:
                                with transaction.atomic():
                                    created_objs = model.objects.bulk_create(words_to_create, ignore_conflicts=True)
                                    total_imported += len(created_objs)
                                    total_skipped += len(words_to_create) - len(created_objs)
                                    self.stdout.write(self.style.SUCCESS(f'Batch {line_num // batch_size} processed. Imported: {len(created_objs)}, Skipped: {len(words_to_create) - len(created_objs)}.'))
                            except Exception as e:
                                self.stderr.write(self.style.ERROR(f'Error importing batch at line {line_num}: {e}'))
                            words_to_create = []
                
                self.stdout.write(self.style.SUCCESS(f"'{file_path}' processado com sucesso usando codificação '{encoding}'."))
                file_processed_successfully = True
                break
            except UnicodeDecodeError as e:
                self.stdout.write(self.style.WARNING(f"Erro de codificação em '{file_path}' na linha {line_num or 'inicial'} com '{encoding}': {e}. Tentando a próxima codificação..."))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Erro inesperado ao ler '{file_path}' com '{encoding}' na linha {line_num or 'inicial'}: {e}. Quebrando..."))
                file_processed_successfully = False
                break 
        
        if not file_processed_successfully:
            raise CommandError(f"Erro fatal: Não foi possível ler '{file_path}' com nenhuma das codificações tentadas.")

        # Importa o restante
        if words_to_create:
            try:
                with transaction.atomic():
                    created_objs = model.objects.bulk_create(words_to_create, ignore_conflicts=True)
                    total_imported += len(created_objs)
                    total_skipped += len(words_to_create) - len(created_objs)
                    self.stdout.write(self.style.SUCCESS(f'Final batch processed. Imported: {len(created_objs)}, Skipped: {len(words_to_create) - len(created_objs)}.'))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f'Error importing final batch: {e}'))

        self.stdout.write(self.style.SUCCESS('Import complete!'))
        self.stdout.write(self.style.SUCCESS(f'Total imported words: {total_imported}'))
        self.stdout.write(self.style.WARNING(f'Total skipped (already exist): {total_skipped}'))