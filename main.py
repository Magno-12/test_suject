from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from analyses import analyze_labels, analyze_signals_fixed_v2, post_hoc_tukey_analysis
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
        labels_result = analyze_labels(df_etiquetas)

        analysis_results = {}
        files = [file_t1, file_t2, file_b1, file_b2]
        labels = ["T1", "T2", "B1", "B2"]
        dfs = {}

        for file, label in zip(files, labels):
            content = await file.read()
            df = pd.read_excel(content)
            df.columns = ["Signal1", "Signal2", "Signal3", "Signal4"]
            cleaned_df = clean_signals(df)
            analysis_results[f"Suj_{subject_number}_{label}_analysis"] = analyze_signals_fixed_v2(cleaned_df, label)
            dfs[f"Suj_{subject_number}_{label}.xlsx"] = df

        tukey_result = post_hoc_tukey_analysis(dfs)

        return JSONResponse(content={
            "labels_analysis": labels_result,
            **analysis_results,
            "post_hoc_tukey_analysis": tukey_result
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
