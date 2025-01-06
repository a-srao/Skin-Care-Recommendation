
from pyexpat.errors import messages
import random
from django.shortcuts import render, redirect
from ProductRecommendationtionApp.models import *
from django.db.models import Q, Sum
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from ProductRecommendationtionApp import predict
from django.utils import timezone
import shutil
import os
from django.http import JsonResponse
from django.core.serializers.json import DjangoJSONEncoder
import cv2
from django.http import HttpResponse



utc_now = timezone.now()

# Convert UTC time to Indian Standard Time (IST)
ist_now = utc_now.astimezone(timezone.get_fixed_timezone(330))  # UTC+5:30 for Indian Standard Time

# Format the datetime as a string if needed
formatted_time = ist_now.strftime('%Y-%m-%d %H:%M:%S')

# Create your views here.
def index(request):
    products = Product.objects.filter(is_enabled=1).all()
    return render(request,'index.html', {'products' : products})

def user(request):
    return render(request,'user/index.html')

def registration(request):
    return render(request,'user/registration.html')

def saveUser(request):
    if request.method == 'POST':
        farmername = request.POST['uname']
        contactNo = request.POST['contactNo']
        emailId = request.POST['emailId']
        address = request.POST['address']
        username = request.POST['username']
        password = request.POST['password']

        user = User.objects.filter(
            Q(email=emailId) | Q(contact=contactNo) | Q(user_name=username)
        ).first()

        has_error = False
        error = ''

        if user != None and user.user_name == username:
            has_error = True
            error = 'Duplicate user name'

        if user != None and user.email == emailId:
            has_error = True
            error = 'Duplicate email'

        if user != None and user.contact == contactNo:
            has_error = True
            error = 'Duplicate contact number'

        if has_error:
            return render(request, "user/registration.html", {'error': error})

        user = User(name=farmername, contact=contactNo, email=emailId,
                    address=address, user_name=username, password=password)
        user.save()

        return render(request, "user/registration.html", {'success': 'User Added Successfully'})
    else:
        return render(request, 'user/registration.html')

def userlogin(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = User.objects.values_list('password', 'id', 'name').\
            filter(user_name=request.POST['username'])

        user = User.objects.filter(
            user_name=username, password=password).first()

        if user == None:
            return render(request, 'user/index.html', {'error': 'Invalid login credentials'})

        request.session['userid'] = user.id # type: ignore
        request.session['userName'] = user.name

        products = Product.objects.filter(is_enabled=1).all()
        if products != None:
            return render(request, 'user/userHome.html', {'products' : products, 'success' : "Login Success..."})
        else:
            return render(request, 'user/userHome.html', {'error' : "Not found"})
    else:
        return render(request, 'user/index.html')

def homepage(request):
    products = Product.objects.filter(is_enabled = 1).all()
    return render(request, 'user/userHome.html', {'products' : products})

def adminLogin(request):
    return render(request,'admin/index.html')

def adminlogin(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = MainUser.objects.values_list('password', 'id', 'name').\
            filter(user_name=request.POST['username'])

        user = MainUser.objects.filter(
            user_name=username, password=password).first()

        if user == None:
            return render(request, 'admin/index.html', {'error': 'Invalid login credentials'})

        request.session['userid'] = user.id
        request.session['userName'] = user.name
        products = Product.objects.filter(is_enabled = 1).all()
        return render(request, 'admin/adminHome.html', {'success': 'Login success', 'products' : products})

    else:
        return render(request, 'admin/index.html')
    
def addProduct(request):
    if request.method == 'POST':
        name = request.POST['name']
        category = request.POST['category']
        manufacturer = request.POST['manufacturer']
        quantity = request.POST['qty']
        price = request.POST['price']
        description = request.POST['description']
        recommendation = request.POST['recommendation']

        product = Product(name = name, manufacturer = manufacturer, category = category, quantity = quantity, price = price, description = description, recommendation = recommendation, created_at = formatted_time, image = request.FILES['images'])
        product.save()
        products = Product.objects.filter(is_enabled = 1).all()
        return render(request, 'admin/adminHome.html', {'success': 'Product added successfully', 'products' : products})
    
def viewProduct(request):
    products = Product.objects.all()
    return render(request, 'admin/viewProduct.html', {'products' : products})

def homePage(request):
    products = Product.objects.filter(is_enabled = 1).all()
    return render(request,'admin/adminHome.html', {'products' : products})

def changeStatus(request, id, category):
    product = Product.objects.filter(id = id).first()
    if product != None:
        product.is_enabled = category
        product.save()
        products = Product.objects.all()
        return render(request, 'admin/viewProduct.html', {'products' : products, 'success' : "Product updated..."})
    else:
        return render(request, 'admin/viewProduct.html', {'error' : "Something went wrong"})
    
def deleteNow(request, id):
    product = Product.objects.get(id = id)
    product.delete()
    products = Product.objects.all()
    
    return render(request, 'admin/viewProduct.html', {'products' : products, 'success': 'Product deleted...'})

def uploadImage(request):
    return render(request,'user/upload.html')

def home(request):
    if request.method == "POST":
        image = request.FILES['test1']

        shutil.rmtree(os.getcwd() + '\\media\\uploaded_image\\')

        path = default_storage.save(
            os.getcwd() + '\\media\\uploaded_image\\input.png', ContentFile(image.read()))
        

        res = predict.process()
        print(res[0])

        category = 0
        if res[0] == "DRY-NATURAL":
            category = 1
        elif res[0] == "OILY":
            category = 2
        else:
            category = 3


        products = Product.objects.filter(category = category).all()
        
    return render(request, "user/result.html", {'products' : products, "result": res})

def testagain(request):
    return render(request, "user/upload.html")

def orderNow(request, id):
    if id == 0:
        return render(request, "user/index.html", {'error': "Please login before order..."})
    
def getProductInfo(request):
    if request.method == 'GET':
        product_id = request.GET.get('id', None)
        if product_id != None:
            try:
                product = Product.objects.get(id=product_id)
                # Convert food details to a dictionary
                product_details = {
                    'id': product.id, # type: ignore
                    'name': product.name,
                    'manufacturer': product.manufacturer,
                    'category': product.category,
                    'quantity': product.quantity,
                    'price': product.price,
                    'description': product.description,
                    'recommendation': product.recommendation,
                    # 'image': food.image,
                }
                return JsonResponse(product_details, safe=False, encoder=DjangoJSONEncoder)
            except Product.DoesNotExist:
                return JsonResponse({'error': 'Book item not found'}, status=404)
        else:
            return JsonResponse({'error': 'Missing Book ID parameter'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    
def updateProduct(request):
    if request.method == 'POST':
        id = request.POST['id']
        name = request.POST['name']
        category = request.POST['category']
        manufacturer = request.POST['manufacturer']
        quantity = request.POST['qty']
        price = request.POST['price']
        description = request.POST['description']
        recommendation = request.POST['recommendation']

        product = Product.objects.filter(id=id).first()
        product.name = name
        product.category = category
        product.manufacturer = manufacturer
        product.quantity = quantity
        product.price = price
        product.description = description
        product.recommendation = recommendation
        product.image = request.FILES['images']
        product.save()
        
    products = Product.objects.all()
    return render(request, 'admin/viewProduct.html', {'products': products, 'success': 'Product Updated...'})

def orderProduct(request):
    if request.method == "POST":
        import random
        order_id = random.randint(10000, 99999)
        user_id = request.session['userid']
        product_id = request.POST['id']
        quantity = request.POST['quantity']
        price = request.POST['price']
        total = request.POST['total']
        address = request.POST['address']
        description = request.POST['description']

        order = Order(order_id = order_id, user_id = user_id, product_id = product_id, quantity = quantity, price = price, total = total, address = address, description = description)
        order.save()
        products = Product.objects.filter(is_enabled = 1).all()
        return render(request, "user/userHome.html", {'success': 'Product ordered successfully...', 'products' : products})

def history(request):
    user_id = request.session['userid']
    order = Order.objects.filter(user_id = user_id).all()
    return render(request, "user/history.html", {'products' : order})

def orders(request):
    order = Order.objects.all()
    return render(request, "admin/orders.html", {'products' : order})


def capture_image(request):
    # shutil.rmtree(os.getcwd() + '\\media\\uploaded_image\\input.png')
    # Define the path where the image will be saved
    save_path = os.path.join('media', 'uploaded_image', 'input.png')

    # Initialize the OpenCV cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the default camera (typically the webcam)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Save the frame as an image if any key is pressed
        key = cv2.waitKey(1)
        if key != -1:
            cv2.imwrite(save_path, frame)
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

    res = predict.process()
    print("Result Values",res)

    category = 0
    if res == "DRY":
        category = 1
    elif res == "OILY":
        category = 2
    else:
        category = 3

    print(res)

    products = Product.objects.filter(category = category).all()
        
    return render(request, "user/result.html", {'products' : products, "result": res})
