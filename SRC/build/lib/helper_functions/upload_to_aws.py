import boto3
import os 


BUCKET_NAME = 'cv-bottle-detection-bucket'

client = boto3.client('s3')

print(client.list_buckets())
# def check_path_exists()

# def upload_to_aws(local_file, bucket, s3_file):
    
    