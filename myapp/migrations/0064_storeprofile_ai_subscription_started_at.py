from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0063_sitecustomization_ai_brand_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='storeprofile',
            name='ai_subscription_started_at',
            field=models.DateTimeField(
                blank=True, null=True,
                help_text='When the current AI subscription period began.',
            ),
        ),
    ]
