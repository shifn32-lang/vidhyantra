from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0062_storeprofile_age_storeprofile_city_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='sitecustomization',
            name='ai_brand_name',
            field=models.CharField(
                default='Vidhyora', max_length=60,
                help_text=(
                    "The AI assistant's name — shown across the chat UI (header, title, model picker) and used "
                    'in its own replies ("I\'m an AI model built by the ... team"). Changing this takes effect '
                    'immediately, no restart needed.'
                ),
            ),
        ),
    ]
