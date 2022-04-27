# Generated by Django 2.2.1 on 2019-06-26 06:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aud2gest', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='audiodb',
            name='audiofile',
            field=models.FileField(blank=True, null=True, upload_to='aud2gest/audioFiles/'),
        ),
        migrations.AlterField(
            model_name='audiodb',
            name='imagefile',
            field=models.FileField(blank=True, null=True, upload_to='aud2gest/imageFiles/'),
        ),
        migrations.AlterField(
            model_name='audiodb',
            name='textfile',
            field=models.FileField(blank=True, null=True, upload_to='aud2gest/textFiles/'),
        ),
    ]
