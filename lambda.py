# data generation lambda

import json
import boto3
import base64
import urllib

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    bucket = event['s3_bucket']
    key = urllib.parse.unquote_plus(event['s3_key'], encoding='utf-8')
    
    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, '/tmp/image.png')
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

# classification lambda

import json
import base64
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

ENDPOINT = "image-classification-2025-07-20-14-32-47-316"

def lambda_handler(event, context):
    image_bytes = base64.b64decode(event["image_data"])

    predictor = Predictor(
        endpoint_name=ENDPOINT,
        serializer=IdentitySerializer(content_type="image/png"),
        deserializer=JSONDeserializer()
    )

    inference = predictor.predict(image_bytes)

    event["inferences"] = inference

    return {
        "statusCode": 200,
        "body": json.dumps(event["inferences"])
    }


# inference threshhold

import json

THRESHOLD = .93

def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event["inference"]
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(i > THRESHOLD for i in inferences)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
