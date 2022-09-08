# KerasTransferLearning
Cats and Dogs example of transfer learning

There are two main scripts to run

## eval.py
This script evaluate the accuracy of the **OM** saved checkpoints

MobileNetV2 results using 

10 images:<br/>
Validation accuracy : 0.9559999704360962<br/>
Validation loss : 0.5512717366218567<br/>

100 images:<br/>
Validation accuracy : 0.9729999899864197<br/>
Validation loss : 0.5023120641708374<br/>

All 1000 images:<br/>
Validation accuracy : 0.968999981880188<br/>
Validation loss : 0.49909040331840515<br/>

## train.py
This script trains the model based on standard SGD.

10 images, epochs: 100, lr: 0.0005<br/>
Validation accuracy : 0.9419999718666077<br/>
Validation loss : 0.3839479982852936<br/><br/>

10 images, epochs: 100, lr: 0.001<br/>
Validation accuracy : 0.953000009059906<br/>
Validation loss : 0.3696589171886444<br/><br/>

100 images, epochs: 100, lr: 0.0005<br/>
Validation accuracy : 0.9710000157356262<br/>
Validation loss : 0.3566269278526306<br/><br/>

100 images, epochs: 50, lr: 0.001<br/>
Validation accuracy : 0.9789999723434448<br/>
Validation loss : 0.3392152190208435<br/>

1000 images, epochs: 10, lr: 0.001<br/>
Validation accuracy : 0.9829999804496765<br/>
Validation loss : 0.32968994975090027<br/>


