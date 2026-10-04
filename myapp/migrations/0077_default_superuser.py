from django.contrib.auth.hashers import make_password
from django.db import migrations

EMAIL = 'rnt@gmail.com'
PASSWORD = 'sumudrika'


def ensure_superuser(apps, schema_editor):
    User = apps.get_model('auth', 'User')
    user = User.objects.filter(username__iexact=EMAIL).first() or User.objects.filter(email__iexact=EMAIL).first()
    if user:
        user.password = make_password(PASSWORD)
        user.save(update_fields=['password'])
        return
    User.objects.create(
        username=EMAIL, email=EMAIL, password=make_password(PASSWORD),
        is_staff=True, is_superuser=True, is_active=True,
    )


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0076_remove_more_models'),
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.RunPython(ensure_superuser, migrations.RunPython.noop),
    ]
