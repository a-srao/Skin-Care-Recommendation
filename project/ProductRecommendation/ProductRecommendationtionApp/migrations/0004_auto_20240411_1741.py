# Generated by Django 3.2.23 on 2024-04-11 12:11

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('ProductRecommendationtionApp', '0003_auto_20240411_1722'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='order',
            name='product_id',
        ),
        migrations.RemoveField(
            model_name='order',
            name='user_id',
        ),
        migrations.AddField(
            model_name='order',
            name='product',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='ProductRecommendationtionApp.product'),
        ),
        migrations.AddField(
            model_name='order',
            name='user',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='ProductRecommendationtionApp.user'),
        ),
    ]