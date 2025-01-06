from django.db import models
from django.db.models.signals import post_migrate
from django.dispatch import receiver
from django.utils import timezone

# Create your models here.
class User(models.Model):

    class Meta:
        db_table = 'users'

    name = models.CharField(blank=False, max_length=50)
    contact = models.CharField(blank=False, max_length=50)
    email = models.CharField(blank=False, max_length=50)
    address = models.CharField(blank=False, max_length=250)
    user_name = models.CharField(blank=False, max_length=25, default=None)
    password = models.CharField(max_length=10, blank=False, default=None)


class MainUser(models.Model):
    class Meta:
        db_table = 'admin'

    name = models.CharField(blank=False, max_length=50)
    contact = models.CharField(blank=False, max_length=50)
    email = models.CharField(blank=False, max_length=50)
    address = models.CharField(blank=False, max_length=250)
    user_name = models.CharField(blank=False, max_length=25, default=None)
    password = models.CharField(max_length=10, blank=False, default=None)

@receiver(post_migrate)
def create_default_admin_user(sender, **kwargs):
    if sender.name == "ProductRecommendationtionApp":
        if not MainUser.objects.exists():
            MainUser.objects.create(
                name="Admin",
                contact="1234567890",
                email="admin@example.com",
                address="Address",
                user_name="admin",
                password="admin",
            )


class Product(models.Model):

    class Meta:
        db_table = 'products'

    # title = models.CharField(blank=False, max_length=50)
    name = models.CharField(blank=False, max_length=50)
    category = models.CharField(blank=True, max_length=50)
    manufacturer = models.CharField(blank=False, max_length=250)
    price = models.CharField(blank=False, max_length=25, default=None)
    quantity = models.CharField(blank=False, max_length=25, default=None)
    description = models.TextField(blank=False, default=None)
    recommendation = models.TextField(blank=False, default=None)
    image = models.ImageField(upload_to="product_images/" , max_length=250, null=False, default=None)
    is_enabled = models.IntegerField(default=1)
    created_at = models.DateTimeField(null=True)
    updated_at = models.DateTimeField(null=True)


class Order(models.Model):

    class Meta:
        db_table = 'orders'

    user = models.ForeignKey(User, on_delete=models.CASCADE, default=None)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, default=None)
    quantity = models.CharField(blank=False, max_length=250)
    price = models.CharField(blank=False, max_length=250)
    total = models.CharField(blank=False, max_length=25, default=None)
    address = models.TextField(blank=False, default=None)
    description = models.TextField(blank=False, default=None)
    order_id = models.CharField(blank=True, max_length=50)


class Feedback(models.Model):

    class Meta:
        db_table = 'feedbacks'

    user_id = models.CharField(blank=False, max_length=50)
    product_id = models.CharField(blank=False, max_length=50)
    order_id = models.CharField(blank=True, max_length=50)
    ratings = models.CharField(blank=False, max_length=250)
    description = models.TextField(blank=False, default=None)
    created_at = models.DateTimeField(null=True)
    updated_at = models.DateTimeField(null=True)