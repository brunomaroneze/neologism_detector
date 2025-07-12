# neologism_app/models.py

from django.db import models

class LexiconWord(models.Model):
    """
    Representa uma palavra no léxico base (não-neologismo).
    Este modelo armazenará as ~500k palavras consolidadas.
    """
    word = models.CharField(max_length=255, unique=True, db_index=True)

    class Meta:
        verbose_name = "Palavra do Léxico"
        verbose_name_plural = "Palavras do Léxico"

    def __str__(self):
        return self.word

class CustomAddition(models.Model):
    """
    Palavras adicionadas pelo usuário como não-neologismos (erros do algoritmo, etc.).
    Estas são as palavras que antes eram salvas em 'custom_additions.txt'.
    """
    word = models.CharField(max_length=255, unique=True, db_index=True)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Adição Personalizada"
        verbose_name_plural = "Adições Personalizadas"

    def __str__(self):
        return self.word

class NeologismValidated(models.Model):
    """
    Neologismos que foram validados e/ou classificados pelo usuário,
    para uso futuro em treinamento de Machine Learning.
    Estas são as palavras que antes eram salvas em 'neologisms_validated.txt'.
    """
    word = models.CharField(max_length=255, unique=True, db_index=True)
    pos_tag = models.CharField(max_length=50, blank=True, null=True) # Classe gramatical corrigida
    lemma = models.CharField(max_length=255, blank=True, null=True) # Lema
    formation_process = models.CharField(max_length=100, blank=True, null=True) # Processo de formação
    validated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Neologismo Validado"
        verbose_name_plural = "Neologismos Validados"

    def __str__(self):
        return self.word