import logging
import os
import json
import pyodbc
import requests
import azure.functions as func

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="AffectionHttpTrigger", methods=["POST"])
def affection_http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("HTTP trigger received")

    try:
        data = req.get_json()

        # Inference
        if os.getenv("ENV") == "local":
            label = 1
        else:
            resp = requests.post(
                os.getenv("AZUREML_ENDPOINT_URL"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('AZUREML_KEY')}"
                },
                json=data,
                timeout=10
            )
            resp.raise_for_status()
            result = resp.json()

            if isinstance(result, str):
                result = json.loads(result)

            label = int(result.get("prediction", result.get("label", 0)))

        # DB connection
        conn = pyodbc.connect(
            f"Driver={{ODBC Driver 17 for SQL Server}};"
            f"Server={os.getenv('SQL_SERVER')};"
            f"Database={os.getenv('SQL_DATABASE')};"
            f"Uid={os.getenv('SQL_USERNAME')};"
            f"Pwd={os.getenv('SQL_PASSWORD')};"
        )
        cursor = conn.cursor()

        # Insert data (Id auto increment)
        cursor.execute(
            """
            INSERT INTO TempAffection
            (ACC_X, ACC_Y, ACC_Z, BVP, EDA, TEMP, Label)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            data["ACC_X"],
            data["ACC_Y"],
            data["ACC_Z"],
            data["BVP"],
            data["EDA"],
            data["TEMP"],
            label
        )
        conn.commit()

        # Check buffer count
        cursor.execute("SELECT COUNT(*) FROM TempAffection")
        buffer_count = cursor.fetchone()[0]

        # AUTO trigger when exactly 500
        if buffer_count == 500:
            cursor.execute(
                """
                INSERT INTO Affection
                (ACC_X, ACC_Y, ACC_Z, BVP, EDA, TEMP, LABEL)
                SELECT
                ACC_X, ACC_Y, ACC_Z, BVP, EDA, TEMP, Label
                FROM TempAffection
                """
            )
            cursor.execute("TRUNCATE TABLE TempAffection")
            conn.commit()

            if os.getenv("ENV") != "local":
                requests.post(
                    os.getenv("AZUREML_PIPELINE_TRIGGER_URL"),
                    headers={
                        "Authorization": f"Bearer {os.getenv('AZUREML_PIPELINE_TOKEN')}",
                        "Content-Type": "application/json"
                    },
                    timeout=10
                )

        return func.HttpResponse(
            json.dumps({
                "label": label,
                "buffer_count": buffer_count
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)