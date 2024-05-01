from django.shortcuts import render
from django.http import HttpResponse
from work import dbconnection
from django.http import HttpResponseRedirect
from django.core.files.storage import FileSystemStorage
from datetime import date
# Create your views here.

def home(request):
    return render(request,'index.html',{})

def login(request):
    if request.method=='POST':
        a=request.POST.get('uid')
        b=request.POST.get('pas')
        sql="select * from user_log where uid='"+a+"' and pas='"+b+"'"
        data=dbconnection.logindata(sql)
        if data:                        
            if data[3]=="admin":
                request.session['x']=a
                return HttpResponseRedirect("http://127.0.0.1:8000/adminhome")
            if data[3]=="stf":
                request.session['s']=a
                return HttpResponseRedirect("http://127.0.0.1:8000/stafhome")            
    return render(request,'login.html',{})

def adhome(request):
    return render(request,'office/index.html',{})

def addstaff(request):
    query="select * from staff_data"
    data=dbconnection.selectalldata(query)
    if request.method=='POST':
        nme=request.POST.get('nme')
        addr=request.POST.get('addr')
        con=request.POST.get('con')        
        qry="INSERT INTO `staff_data`(`nme`, `addr`, `con`) VALUES ('"+nme+"','"+addr+"','"+con+"')"
        dbconnection.insertdata(qry)
        qry1="INSERT INTO `user_log`(`uid`, `pas`, `typ`) VALUES ('"+con+"','stf','stf')"
        dbconnection.insertdata(qry1)
        return HttpResponseRedirect("http://127.0.0.1:8000/addstaff")
    return render(request,'office/addstaff.html',{'st':data})

def mstaff(request):
    query="select * from staff_data"
    data=dbconnection.selectalldata(query)
    tip="select * from tips_data"
    tips=dbconnection.selectalldata(tip)
    return render(request,'office/liststaff.html',{'st':data,'tips':tips})

def report(request,uid):
    query="select * from staff_data where id='"+uid+"'"
    data=dbconnection.selectdata(query)
    q1="select count(*) from analyse where stid='"+str(data[0])+"' and emoval='1'"
    d1=dbconnection.selectdata(q1)
    q2="select count(*) from analyse where stid='"+str(data[0])+"' and emoval='0'"
    d2=dbconnection.selectdata(q2) 
    print(data[0]) 
    if d2[0]>=1: 
        sum=d1[0]+d2[0]
        p=(d1[0]/sum)*100 
        n=(d2[0]/sum)*100
        if n>p:
            tip="select * from tips_data"
            tips=dbconnection.selectalldata(tip)
        else:
            tips="" 
        return render(request,'office/report.html',{'st':data,'d1':d1,'d2':d2,'p':p,'n':n,'t':tips})
    else:
        msg="No Data Found"
        return render(request,'office/report.html',{'st':data,'m':msg}) 

def addtips(request):
    if request.method=='POST':
        tip=request.POST.get('tip')
        qry="INSERT INTO `tips_data`(`tips`, `st`) VALUES ('"+tip+"','1')"
        dbconnection.insertdata(qry)
    return HttpResponseRedirect("http://127.0.0.1:8000/mstaff")

def stafhome(request):
    uid=request.session['s']
    #print(uid)
    #print("hai")
    query="select * from staff_data where con='"+uid+"'"
    data=dbconnection.selectdata(query)
    q1="select count(*) from analyse where stid='"+str(data[0])+"' and emoval='1'"
    d1=dbconnection.selectdata(q1)
    q2="select count(*) from analyse where stid='"+str(data[0])+"' and emoval='0'"
    d2=dbconnection.selectdata(q2) 
    print(data[0]) 
    if d2[0]>=1: 
        sum=d1[0]+d2[0]
        p=(d1[0]/sum)*100 
        n=(d2[0]/sum)*100
        if n>p:
            tip="select * from tips_data"
            tips=dbconnection.selectalldata(tip)
        else:
            tips=""
        return render(request,'staffdata/index.html',{'st':data,'d1':d1,'d2':d2,'p':p,'n':n,'t':tips})
    else:
        msg="No Data Found"
        return render(request,'staffdata/index.html',{'st':data,'m':msg})

def scan(request,uid):    
    q="delete from analyse where stid='"+uid+"'"
    dbconnection.selectdata(q)
    import numpy as np
    import argparse
    import matplotlib.pyplot as plt
    import cv2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import os
    import os
    from werkzeug.utils import secure_filename
       
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    import os
    basepath = os.path.dirname(__file__)
    model.load_weights(os.path.join(basepath, 'static\\model', secure_filename('model.h5')))

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Tensed", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Stressed", 6: "Surprised"}
    i=0
    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier(os.path.join(basepath, 'static\\model', secure_filename('haarcascade_frontalface_default.xml')))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            #print(emotion_dict[maxindex]) 
            val=emotion_dict[maxindex] 
            print("value.....................",val)
            if(val=="Tensed"):
                x1=0
            if(val=="Disgusted"):
                x1=0
            if(val=="Fearful"):
                x1=0
            if(val=="Happy"):
                x1=1
            if(val=="Neutral"):
                x1=1
            if(val=="Stressed"):
                x1=0
            if(val=="Surprised"):
                x1=1           
            today = str(date.today())  
            chk="select * from analyse order by id desc"
            chdata=dbconnection.selectdata(chk)
            if(chdata):
                if(chdata[2]==val):
                    print("no")
                else:
                    qry="INSERT INTO `analyse`(`stid`, `emo`, `emoval`, `dt`) VALUES ('"+str(uid)+"','"+emotion_dict[maxindex]+"','"+str(x1)+"','"+today+"')" 
                    dbconnection.insertdata(qry)  
                    print(emotion_dict[maxindex]) 
                    i=i+1
                    print(i)
                    if(i>10):
                        cap.release()
                        cv2.destroyAllWindows()
                        return HttpResponseRedirect("http://127.0.0.1:8000/mstaff")
            else:
                qry="INSERT INTO `analyse`(`stid`, `emo`, `emoval`, `dt`) VALUES ('"+str(uid)+"','"+emotion_dict[maxindex]+"','"+str(x1)+"','"+today+"')" 
                dbconnection.insertdata(qry)  

            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(750,750),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render(request,'stress.html',{})

def stanalyse(request):
    uid=request.session['s']
    query="select * from staff_data where con='"+uid+"'"
    data=dbconnection.selectdata(query)
    q="delete from analyse where stid='"+uid+"'"
    dbconnection.selectdata(q)
    import numpy as np
    import argparse
    import matplotlib.pyplot as plt
    import cv2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import os
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    import os
    from werkzeug.utils import secure_filename
    from gevent.pywsgi import WSGIServer
    basepath = os.path.dirname(__file__)
    model.load_weights(os.path.join(basepath, 'static\\model', secure_filename('model.h5')))
    
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Tensed", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Stressed", 6: "Surprised"}
    i=0
    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier(os.path.join(basepath, 'static\\model', secure_filename('haarcascade_frontalface_default.xml')))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            #print(emotion_dict[maxindex])q 
            val=emotion_dict[maxindex] 
            if(val=="Tensed"):
                x1=0
            if(val=="Disgusted"):
                x1=0
            if(val=="Fearful"):
                x1=0
            if(val=="Happy"):
                x1=1
            if(val=="Neutral"):
                x1=1
            if(val=="Stressed"):
                x1=0
            if(val=="Surprised"):
                x1=1 
            today = str(date.today())  
            chk="select * from analyse order by id desc"
            chdata=dbconnection.selectdata(chk)
            if(chdata):
                if(chdata[2]==val):
                    print("no")
                else:
                    x=1
                    qry="INSERT INTO `analyse`(`stid`, `emo`, `emoval`, `dt`) VALUES ('"+str(data[0])+"','"+emotion_dict[maxindex]+"','"+str(x1)+"','"+today+"')" 
                    dbconnection.insertdata(qry)  
                    print(emotion_dict[maxindex]) 
                    i=i+1
                    print(i)
                    if(i>10):
                        cap.release()
                        cv2.destroyAllWindows()
                        return HttpResponseRedirect("http://127.0.0.1:8000/staffdata")
            else:
                qry="INSERT INTO `analyse`(`stid`, `emo`, `emoval`, `dt`) VALUES ('"+str(data[0])+"','"+emotion_dict[maxindex]+"','"+str(x1)+"','"+today+"')" 
                dbconnection.insertdata(qry)  

            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(750,750),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render(request,'stanalyse.html',{})

def pic(request):  
    if request.method=='POST':
        nme=request.POST.get('nme')
        up=request.FILES['up']
        fs=FileSystemStorage()
        from datetime import datetime
        cdt = datetime.today().strftime('%Y-%m-%d')
        filename=fs.save("work/static/scanpic/"+up.name,up)     
        qry="INSERT INTO `picdata`(`finfo`, `pic`, `dt`) VALUES ('"+nme+"','"+up.name+"','"+cdt+"')"
        dbconnection.insertdata(qry)  
    qry="select * from picdata order by id desc"
    pdata=dbconnection.selectalldata(qry)
    return render(request,'office/picture.html',{'pic':pdata})

def picanalyse(request):  
    pid=request.GET.get("pid")
    import numpy as np
    import argparse
    import matplotlib.pyplot as plt
    import cv2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from keras.preprocessing import image
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)



    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    import os
    from werkzeug.utils import secure_filename
    from gevent.pywsgi import WSGIServer
    basepath = os.path.dirname(__file__)
    model.load_weights(os.path.join(basepath, 'static\\model', secure_filename('model.h5')))

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    img = image.load_img(os.path.join(basepath, 'static\\scanpic', secure_filename(pid)),target_size = (48,48),color_mode = "grayscale")
    img = np.array(img)
    plt.imshow(img)
    print(img.shape) 

    img = np.expand_dims(img,axis = 0) #makes image shape (1,48,48)
    img = img.reshape(1,48,48,1)
    result = model.predict(img)
    result = list(result[0])
    print(result)

    img_index = result.index(max(result))
    rslt=emotion_dict[img_index]


    return render(request,'office/picanalyse.html',{'pic':pid,'rslt':rslt})

def delstf(request):
    sid=request.GET.get('id')
    qry="delete from staff_data where id='"+sid+"'"
    delt=dbconnection.deletedata(qry)
    return HttpResponseRedirect("http://127.0.0.1:8000/addstaff")

def video(request):  
    if request.method=='POST':
        nme=request.POST.get('nme')
        up=request.FILES['up']
        fs=FileSystemStorage()
        from datetime import datetime
        cdt = datetime.today().strftime('%Y-%m-%d')
        filename=fs.save("work/static/scanvideo/"+up.name,up)     
        qry="INSERT INTO `vdodata`(`finfo`, `pic`, `dt`) VALUES ('"+nme+"','"+up.name+"','"+cdt+"')"
        dbconnection.insertdata(qry)  
    qry="select * from vdodata order by id desc"
    pdata=dbconnection.selectalldata(qry)  
    return render(request,'office/video.html',{'pic':pdata})

def videoanalyse(request):
    pid=request.GET.get("pid")
    import numpy as np
    import argparse
    import matplotlib.pyplot as plt
    import cv2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from keras.preprocessing import image
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)



    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    import os
    from werkzeug.utils import secure_filename
    from gevent.pywsgi import WSGIServer
    basepath = os.path.dirname(__file__)
    model.load_weights(os.path.join(basepath, 'static\\model', secure_filename('model.h5')))

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    cap = cv2.VideoCapture(os.path.join(basepath, 'static\\scanvideo', secure_filename(pid)))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        facecasc = cv2.CascadeClassifier(os.path.join(basepath, 'static\\model', secure_filename('haarcascade_frontalface_default.xml')))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return HttpResponseRedirect("http://127.0.0.1:8000/video")