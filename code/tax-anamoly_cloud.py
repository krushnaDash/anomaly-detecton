import time
from datetime import datetime, timezone
from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.anomalydetector.models import *




SUBSCRIPTION_KEY =  "****"
ANOMALY_DETECTOR_ENDPOINT = "https://tax-nomaly.cognitiveservices.azure.com/"

ad_client = AnomalyDetectorClient(ANOMALY_DETECTOR_ENDPOINT, AzureKeyCredential(SUBSCRIPTION_KEY))

time_format = "%Y-%m-%dT%H:%M:%SZ"
blob_url = "https://krushbua.blob.core.windows.net/hackathon-data/sample_data_5_3000.csv"

train_body = ModelInfo(
    data_source=blob_url,
    start_time=datetime.strptime("2021-01-02T00:00:00Z", time_format),
    end_time=datetime.strptime("2021-01-02T05:00:00Z", time_format),
    data_schema="OneTable",
    display_name="sample",
    sliding_window=200,
    align_policy=AlignPolicy(
        align_mode=AlignMode.OUTER,
        fill_n_a_method=FillNAMethod.LINEAR,
        padding_value=0,
    ),
)

batch_inference_body = MultivariateBatchDetectionOptions(
       data_source=blob_url,
       top_contributor_count=10,
       start_time=datetime.strptime("2021-01-02T00:00:00Z", time_format),
       end_time=datetime.strptime("2021-01-02T05:00:00Z", time_format),
   )


print("Training new model...(it may take a few minutes)")
model = ad_client.train_multivariate_model(train_body)
model_id = model.model_id
print("Training model id is {}".format(model_id))

## Wait until the model is ready. It usually takes several minutes
model_status = None
model = None

while model_status != ModelStatus.READY and model_status != ModelStatus.FAILED:
    model = ad_client.get_multivariate_model(model_id)
    print(model)
    model_status = model.model_info.status
    print("Model is {}".format(model_status))
    time.sleep(30)
if model_status == ModelStatus.READY:
    print("Done.\n--------------------")
    # Return the latest model id

# Detect anomaly in the same data source (but a different interval)
result = ad_client.detect_multivariate_batch_anomaly(model_id, batch_inference_body)
result_id = result.result_id

# Get results (may need a few seconds)
anomaly_results = ad_client.get_multivariate_batch_detection_result(result_id)
print("Get detection result...(it may take a few seconds)")


while anomaly_results.summary.status != MultivariateBatchDetectionStatus.READY and anomaly_results.summary.status != MultivariateBatchDetectionStatus.FAILED and anomaly_results.summary.status !=MultivariateBatchDetectionStatus.CREATED:
    anomaly_results = ad_client.get_multivariate_batch_detection_result(result_id)
    print("Detection is {}".format(anomaly_results.summary.status))
    time.sleep(30)
    
   
print("Result ID:\t", anomaly_results.result_id)
print("Result status:\t", anomaly_results.summary.status)
print("Result length:\t", len(anomaly_results.results))

# See detailed inference result
for r in anomaly_results.results:
    print(
        "timestamp: {}, is_anomaly: {:<5}, anomaly score: {:.4f}, severity: {:.4f}, contributor count: {:<4d}".format(
            r.timestamp,
            r.value.is_anomaly,
            r.value.score,
            r.value.severity,
            len(r.value.interpretation) if r.value.is_anomaly else 0,
        )
    )
    if r.value.interpretation:
        for contributor in r.value.interpretation:
            print(
                "\tcontributor variable: {:<10}, contributor score: {:.4f}".format(
                    contributor.variable, contributor.contribution_score
                )
            )