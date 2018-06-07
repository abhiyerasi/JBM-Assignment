
# coding: utf-8

# In[33]:


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model


# In[37]:


img_width, img_height = 150, 150
model_path = 'model.h5'
model_weights_path = 'modelConv2d.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)


# In[38]:


from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)


test_set = test_datagen.flow_from_directory(
    'E:/JBM/trying/test-data/test_61326',
    target_size=(150, 150),
    batch_size=16,
    class_mode=None,
shuffle=False)


# In[46]:


import os
list_file=os.listdir("E:/JBM/trying/test-data/test_61326/")


# In[48]:


##Predicting using test data
import numpy as np
from keras.preprocessing import image
for i in list_file:
    stri = str(i)
    test_image = image.load_img('E:/JBM/trying/test-data/test_61326/' + stri, target_size = (150,150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis = 0)
    result = model.predict_classes(test_image,verbose=1)
    
    if result==0:
        print('61326_ok_back')
    elif result==1:
        print('61326_ok_front')
    elif result==2:
        print('61326_scratch_mark')
    elif result==3:
        print('61326_slot_damage')
    elif result==4:
        print('61326_thinning')
    elif result==5:
        print('61326_wrinkle')
    else:
        print('none')

