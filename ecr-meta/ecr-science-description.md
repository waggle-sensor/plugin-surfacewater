# Science

Classifies image that is collected from bottom camera on Sage/Waggle node. The method is based on the ResNet50 and head for image classification.


# AI at Edge

The code runs a ResNet50 based model with a given time interval. In each run, it takes a still image from a given camera (bottom) and outputs if standing water is detected or not (1 or 0). The plugin crops images in a certain area to see part of the image where water filled on the ground when it rains or snow melts, and then resize the image to 224x224 as the model was trained with the size.

# Ontology

The code publishes measurement with topic `env.binary.surfacewater`.

# Inference from Sage codes
To query the output from the plugin, you can do with python library 'sage_data_client':
```
import sage_data_client

# query and load data into pandas data frame
df = sage_data_client.query(
    ## start and end date option 1
    start="yyyy-mm-dd",
    end="yyyy-mm-dd",
    ## start and end date option 2
    start="-1h",

    filter={
        "name": "env.binary.surfacewater",
    }
)

# print results in data frame
print(df)
# print results by its name
print(df.name.value_counts())
# print filter names
print(df.name.unique())
```
For more information, please see [Access and use data documentation](https://docs.waggle-edge.ai/docs/tutorials/accessing-data) and [sage_data_client](https://pypi.org/project/sage-data-client/).
