from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from analyses import analyze_labels, analyze_signals
from cleanup import clean_signals

app = FastAPI()

@app.post("/upload/{subject_number}/")
async def upload_files(
    subject_number: int,
    file_etiquetas: UploadFile = File(...),
    file_t1: UploadFile = File(...), file_t2: UploadFile = File(...),
    file_b1: UploadFile = File(...), file_b2: UploadFile = File(...)
    ):
    try:
        content_etiquetas = await file_etiquetas.read()
        df_etiquetas = pd.read_excel(content_etiquetas)

        file_names = [
            f"Suj_{subject_number}_T1.xlsx",
            f"Suj_{subject_number}_B1.xlsx",
            f"Suj_{subject_number}_T2.xlsx",
            f"Suj_{subject_number}_B2.xlsx"
        ]

        dfs = []
        for idx, file in enumerate([file_t1, file_b1, file_t2, file_b2]):
            content = await file.read()
            if "T2" in file_names[idx] or "B2" in file_names[idx]:
                df = pd.read_excel(content, header=None)
                df.columns = ["Signal1", "Signal2", "Signal3", "Signal4"]
            else:
                df = pd.read_excel(content)
            dfs.append(df)

        combined_df = pd.concat(dfs, axis=1)
        cleaned_signals = clean_signals(combined_df)

        labels_result = analyze_labels(df_etiquetas)

        signals_result = analyze_signals(cleaned_signals)

        return {
            "labels_analysis": labels_result,
            "signals_analysis": signals_result
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
