from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Dictionary(models.Model):
    """Dicionário base de palavras conhecidas"""
    word = models.CharField(max_length=100, unique=True)
    is_common = models.BooleanField(default=True)
    word_class = models.CharField(max_length=50, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Palavra do Dicionário"
        verbose_name_plural = "Palavras do Dicionário"
    
    def __str__(self):
        return self.word

class UserDictionary(models.Model):
    """Dicionário personalizado do usuário"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    word = models.CharField(max_length=100)
    is_neologism = models.BooleanField(default=False)
    added_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['user', 'word']
        verbose_name = "Palavra Personalizada"
        verbose_name_plural = "Palavras Personalizadas"
    
    def __str__(self):
        return f"{self.user.username}: {self.word}"

class TextAnalysis(models.Model):
    """Análise de texto realizada"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    original_text = models.TextField()
    processed_text = models.TextField()
    word_count = models.IntegerField()
    neologism_count = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Análise de Texto"
        verbose_name_plural = "Análises de Texto"
    
    def __str__(self):
        return f"Análise {self.id} - {self.created_at.strftime('%d/%m/%Y')}"

class DetectedNeologism(models.Model):
    """Neologismo detectado em uma análise"""
    WORD_CLASSES = [
        ('substantivo', 'Substantivo'),
        ('adjetivo', 'Adjetivo'),
        ('verbo', 'Verbo'),
        ('advérbio', 'Advérbio'),
        ('outros', 'Outros'),
    ]
    
    FORMATION_TYPES = [
        ('derivação_prefixal', 'Derivação Prefixal'),
        ('derivação_sufixal', 'Derivação Sufixal'),
        ('composição', 'Composição'),
        ('estrangeirismo', 'Estrangeirismo'),
        ('outros', 'Outros'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pendente'),
        ('confirmed', 'Confirmado'),
        ('rejected', 'Rejeitado'),
    ]
    
    analysis = models.ForeignKey(TextAnalysis, on_delete=models.CASCADE)
    word = models.CharField(max_length=100)
    word_class = models.CharField(max_length=20, choices=WORD_CLASSES)
    formation_type = models.CharField(max_length=30, choices=FORMATION_TYPES)
    sentence_context = models.TextField()
    confidence_score = models.FloatField(default=0.0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    validated_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Neologismo Detectado"
        verbose_name_plural = "Neologismos Detectados"
    
    def __str__(self):
        return f"{self.word} ({self.get_status_display()})"