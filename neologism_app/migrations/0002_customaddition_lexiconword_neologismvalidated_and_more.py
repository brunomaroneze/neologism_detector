# Generated by Django 5.2.4 on 2025-07-12 22:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('neologism_app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='CustomAddition',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('word', models.CharField(db_index=True, max_length=255, unique=True)),
                ('added_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'verbose_name': 'Adição Personalizada',
                'verbose_name_plural': 'Adições Personalizadas',
            },
        ),
        migrations.CreateModel(
            name='LexiconWord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('word', models.CharField(db_index=True, max_length=255, unique=True)),
            ],
            options={
                'verbose_name': 'Palavra do Léxico',
                'verbose_name_plural': 'Palavras do Léxico',
            },
        ),
        migrations.CreateModel(
            name='NeologismValidated',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('word', models.CharField(db_index=True, max_length=255, unique=True)),
                ('pos_tag', models.CharField(blank=True, max_length=50, null=True)),
                ('lemma', models.CharField(blank=True, max_length=255, null=True)),
                ('formation_process', models.CharField(blank=True, max_length=100, null=True)),
                ('validated_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'verbose_name': 'Neologismo Validado',
                'verbose_name_plural': 'Neologismos Validados',
            },
        ),
        migrations.RemoveField(
            model_name='detectedneologism',
            name='analysis',
        ),
        migrations.DeleteModel(
            name='Dictionary',
        ),
        migrations.RemoveField(
            model_name='textanalysis',
            name='user',
        ),
        migrations.AlterUniqueTogether(
            name='userdictionary',
            unique_together=None,
        ),
        migrations.RemoveField(
            model_name='userdictionary',
            name='user',
        ),
        migrations.DeleteModel(
            name='DetectedNeologism',
        ),
        migrations.DeleteModel(
            name='TextAnalysis',
        ),
        migrations.DeleteModel(
            name='UserDictionary',
        ),
    ]
