from django.db import migrations

REMOVED_MODEL_KEYS = {
    'sdxl-lightning', 'flux-1-schnell', 'sdxl-base', 'dreamshaper-8-lcm',
    'gemini-3-6-flash', 'openrouter-auto-free', 'laguna-s-2-1', 'cohere-north-mini-code',
}
REMOVED_KEY_SETTINGS = {
    'CLOUDFLARE_ACCOUNT_ID', 'CLOUDFLARE_API_TOKEN', 'GEMINI_API_KEY', 'OPENROUTER_API_KEY',
}


def remove_deleted_models(apps, schema_editor):
    AIModelControl = apps.get_model('myapp', 'AIModelControl')
    AIAPIAccess = apps.get_model('myapp', 'AIAPIAccess')
    ProviderAPICredential = apps.get_model('myapp', 'ProviderAPICredential')

    AIModelControl.objects.filter(model_key__in=REMOVED_MODEL_KEYS).delete()
    ProviderAPICredential.objects.filter(setting_name__in=REMOVED_KEY_SETTINGS).delete()
    for access in AIAPIAccess.objects.exclude(model_keys=''):
        keys = [key.strip() for key in access.model_keys.split(',') if key.strip()]
        kept = [key for key in keys if key not in REMOVED_MODEL_KEYS]
        if kept != keys:
            access.model_keys = ','.join(kept)
            access.save(update_fields=['model_keys'])


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0073_aicodingkey_devices'),
    ]

    operations = [
        migrations.RunPython(remove_deleted_models, migrations.RunPython.noop),
    ]
