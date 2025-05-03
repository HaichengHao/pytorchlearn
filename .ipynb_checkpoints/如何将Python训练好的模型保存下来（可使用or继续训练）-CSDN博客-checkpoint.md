 

背景描述
----

Python提供了许多机器学习框架，例如Scikit-learn、TensorFlow和PyTorch。这些框架是使用Python编写的，可以方便地训练模型。但是，模型训练是一项昂贵的任务，需要大量的计算资源和时间。一旦模型训练完成，将其保存以便以后使用是非常重要的。

解决办法
----

保存Python训练好的模型有多种方法，下面介绍其中几种。

### 方法一： 使用pickle——通用

pickle是[Python标准库](https://so.csdn.net/so/search?q=Python%E6%A0%87%E5%87%86%E5%BA%93&spm=1001.2101.3001.7020)中的一个模块，它可以将Python对象序列化为二进制格式，以便于存储和传输。可以使用pickle将训练好的模型保存到磁盘上，以备将来使用。

**保存模型**

```py
import pickle

# train the model
model = ...

# save the model
with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)

```

**加载（使用）模型**

```py
import pickle

# load the saved model
with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)

# predict using the loaded model
model.predict(X)
```

**加载模型继续训练**

```py
import pickle

# load the saved model
with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)

# continue training the model
model.fit(X_train, y_train)

# save the updated model
with open('my_updated_model.pkl', 'wb') as f:
    pickle.dump(model, f)

```

### 方法二：使用joblib——大型模型

joblib是一个用于将Python对象序列化为磁盘文件的库，专门用于 **大型数组**。它可以高效地处理大型数据集和模型。对于`大型机器学习模型`，使用joblib可能比pickle更快。

**保存模型**

```py
import joblib

# train the model
model = ...

# save the model
joblib.dump(model, 'my_model.joblib')
```

**加载（使用）模型**

```py
import joblib

# load the saved model
model = joblib.load('my_model.joblib')

# predict using the loaded model
model.predict(X)

```

**继续训练**

```py
import joblib

# load the saved model
model = joblib.load('my_model.joblib')

# continue training the model
model.fit(X_train, y_train)

# save the updated model
joblib.dump(model, 'my_updated_model.joblib')

```

在这个例子中，我们使用`joblib.load()`函数加载之前保存的模型，并在**新数据上继续训练模型**。最后，我们使用joblib.dump()函数将更新后的模型保存到文件中。

### 方法三：使用HDF5——大型模型（保存权重）

HDF5是一种用于存储大型科学数据集的文件格式，常用于**存储深度学习模型的`权重`**。

**保存模型（权重）**

```py
# train the model
model = ...

# save the model weights to HDF5 file
model.save_weights('my_model_weights.h5')

```

**使用模型**

```py
import tensorflow as tf

# define the model architecture
model = ...

# load the saved model weights
model.load_weights('my_model_weights.h5')

# predict using the loaded model weights
model.predict(X)

```

在这个例子中，我们首先定义了模型的架构，然后使用model.load\_weights()函数加载之前保存的模型权重，并在新数据上进行预测。

**使用模型权重继续优化模**

```py
import tensorflow as tf

# define the model architecture
model = ...

# load the saved model weights
model.load_weights('my_model_weights.h5')

# continue training the model
model.fit(X_train, y_train)

# save the updated model weights to HDF5 file
model.save_weights('my_updated_model_weights.h5')

```

> 需要注意的是，在使用保存的模型权重初始化新模型时，**新模型的架构应该与原始模型相同**。如果新模型的架构不同，您需要重新定义模型，并使用保存的权重`初始化`它。

在这个例子中，我们首先定义了模型的架构。

然后使用`model.load_weights()`函数加载之前保存的模型权重，并在新数据上继续优化模型。

最后，我们使用`model.save_weights()`函数将更新后的模型权重保存到HDF5文件中。

### 方法四：使用ONNX——不同平台

ONNX是一种开放式的格式，可以用于表示机器学习模型。使用ONNX，您可以将模型从一个框架转换为另一个框架，或者在**不同平台上**使用模型。

**保存模型**

```py
import onnx

# train the model
model = ...

# convert the model to ONNX format
onnx_model = onnx.convert(model)

# save the model
onnx.save_model(onnx_model, 'my_model.onnx')

```

**加载（使用）模型**

```py
import onnxruntime

# load the saved model
onnx_session = onnxruntime.InferenceSession('my_model.onnx')

# predict using the loaded model
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name
result = onnx_session.run([output_name], {input_name: X})

```

**继续训练**

由于ONNX格式是为**模型转换**而设计的，因此它`不直接支持模型的进一步训练`。

但是，您可以使用ONNX格式将模型从一个框架**转换为另一个框架**，并在新框架中继续训练模型。

例如，您可以使用ONNX格式将PyTorch模型转换为`TensorFlow模型`，并在TensorFlow中继续训练模型。

常见问题解答
------

1.  **我可以使用pickle将任何Python对象保存到磁盘上吗？**  
    是的，pickle可以保存任何Python对象，包括模型、数据集、字典、列表等等。
    
2.  **如何将保存在pickle中的模型加载到另一个Python程序中？**  
    您可以使用pickle.load()函数从pickle文件中加载模型，并将其存储在变量中。然后，您可以在另一个Python程序中使用该变量。
    
3.  **如何在保存模型的同时保存模型的元数据？**  
    您可以将元数据存储在字典中，并将其一起保存到pickle文件中。然后，您可以在加载模型时从pickle文件中读取字典。
    
4.  **我可以在保存模型时使用不同的格式吗？**  
    是的，您可以根据需要使用pickle、joblib、HDF5或ONNX格式保存模型。
    
5.  **如何在加载模型后继续训练模型？**  
    您可以使用模型的.fit()方法在加载模型后继续训练模型。

本文转自 <https://blog.csdn.net/qq_22841387/article/details/130194553>，如有侵权，请联系删除。