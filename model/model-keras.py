import json
from tensorflow.keras.models import model_from_json
from tensorflow.keras.initializers import GlorotUniform

# Step 1: Load the model JSON file
with open('model.json', 'r') as json_file:
    model_json = json_file.read()

# Step 2: Parse the JSON file into a Python dictionary
model_config = json.loads(model_json)

# Step 3: Access the model's layers
if 'modelTopology' in model_config and 'model_config' in model_config['modelTopology']:
    layers_config = model_config['modelTopology']['model_config']['config']['layers']
else:
    raise KeyError("Model configuration not found in JSON.")

# Step 4: Adjust the kernel initializer for Conv2D layers
for layer in layers_config:
    if 'kernel_initializer' in layer['config'] and layer['config']['kernel_initializer']['class_name'] == 'GlorotUniform':
        # Replace with a manually deserialized initializer
        layer['config']['kernel_initializer'] = {'class_name': 'GlorotUniform', 'config': {}}

# Step 5: Convert the updated config back to JSON
model_json_updated = json.dumps(model_config)

# Step 6: Load the model using model_from_json
model = model_from_json(model_json_updated, custom_objects={'GlorotUniform': GlorotUniform})

# Step 7: Load the weights
model.load_weights('group1-shard1of1.bin')

# Step 8: Load class names
with open('class_names.txt', 'r') as file:
    class_names = file.read().splitlines()

print("Model loaded successfully!")