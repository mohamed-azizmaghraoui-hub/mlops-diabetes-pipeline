
from prefect import serve

from pipeline_prefect import all_flow, train_flow


if __name__ == "__main__":

    deployment_all = all_flow.to_deployment(
        name="ml-pipeline-all",
        cron="0 0 * * *"
    )

    deployment_train = train_flow.to_deployment(
        name="ml-pipeline-train",
        cron="0 0 * * *"
    )

    serve(
        deployment_all,
        deployment_train
    )
