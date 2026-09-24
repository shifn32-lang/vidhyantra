from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0061_aiapiaccess_aiapikey'),
    ]

    operations = [
        migrations.AddField(
            model_name='storeprofile',
            name='age',
            field=models.PositiveSmallIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='storeprofile',
            name='state',
            field=models.CharField(blank=True, max_length=100),
        ),
        migrations.AddField(
            model_name='storeprofile',
            name='city',
            field=models.CharField(blank=True, max_length=100),
        ),
    ]
