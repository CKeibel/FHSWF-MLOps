import os
import mlflow
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from jinja2 import Template
import datetime

class AdultIncomeDataVisualisation:
    def __init__(self, mlflowRun, dataToVisOrigin, dataToVis):
        self.mlflow_run = mlflowRun
        self.artifact_dir = "visualizations"
        os.makedirs(self.artifact_dir, exist_ok=True)

        # Erstelle einen Bericht
        def compare_datasets(df_orig, df_new):
            report = {}
            orig_columns = set(df_orig.columns)
            new_columns = set(df_new.columns)
            
            report["fehlende_spalten"] = list(orig_columns - new_columns)
            report["neue_spalten"] = list(new_columns - orig_columns)
            
            dtype_changes = {col: (str(df_orig[col].dtype), str(df_new[col].dtype)) 
                            for col in orig_columns.intersection(new_columns) if df_orig[col].dtype != df_new[col].dtype}
            report["geaenderte_datentypen"] = dtype_changes
            
            numeric_cols = df_orig.select_dtypes(include=[np.number]).columns.intersection(
                df_new.select_dtypes(include=[np.number]).columns
            )
            range_changes = {}
            
            for col in numeric_cols:
                orig_min, orig_max = df_orig[col].min(), df_orig[col].max()
                new_min, new_max = df_new[col].min(), df_new[col].max()
                range_changes[col] = {"original": (orig_min, orig_max), "neu": (new_min, new_max)}
            report["numerische_abweichungen"] = range_changes
            
            categorical_cols = df_orig.select_dtypes(include=["object"]).columns.intersection(
                df_new.select_dtypes(include=["object"]).columns
            )
            category_changes = {}
            
            for col in categorical_cols:
                orig_categories = set(df_orig[col].dropna().unique())
                new_categories = set(df_new[col].dropna().unique())
                category_changes[col] = {"original": list(orig_categories), "neu": list(new_categories)}
            report["kategorische_abweichungen"] = category_changes
            
            return report

        report = compare_datasets(dataToVisOrigin, dataToVis)

        # Funktion Verteilungsanalyse der unabhängigen Variablen
        def generate_side_by_side_plots(df_orig, df_new, col, prefix):
            if col == "income":  
                return None  

            fig, axes = plt.subplots(1, 2, figsize=(24, 10)) 
            plt.subplots_adjust(wspace=1)  

            if df_orig[col].dtype == "object":
                sns.countplot(x=col, data=df_orig, hue="income", ax=axes[0])
                sns.countplot(x=col, data=df_new, hue="income", ax=axes[1])
                axes[0].set_xticks(range(len(axes[0].get_xticklabels())))
                axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

                axes[1].set_xticks(range(len(axes[1].get_xticklabels())))
                axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

            else:
                sns.histplot(df_orig, x=col, hue="income", bins=30, kde=False, ax=axes[0])
                sns.histplot(df_new, x=col, hue="income", bins=30, kde=False, ax=axes[1])
            
            axes[0].set_title(f"{col} - Ursprungsdatensatz")
            axes[1].set_title(f"{col} - Neuer Datensatz")
            
            plot_path = f"{prefix}_{col}.png"
            plt.savefig(f"{self.artifact_dir}/{plot_path}")
            plt.close()
            return plot_path

        # Verteilungen der Zielvariablen income 
        def plot_income_distribution(df_orig, df_new):
            fig, axes = plt.subplots(1, 2, figsize=(24, 10))
            plt.subplots_adjust(wspace=1)

            
            sns.countplot(x="income", data=df_orig, ax=axes[0], hue="income", palette="Set1")
            sns.countplot(x="income", data=df_new, ax=axes[1], hue="income", palette="Set1")

            axes[0].set_title("Verteilung der Zielvariable income - Ursprungsdatensatz")
            axes[1].set_title("Verteilung der Zielvariable income - Neuer Datensatz")
            
            plot_path = f"income_comparison.png"
            plt.savefig(f"{self.artifact_dir}/{plot_path}")
            plt.close()
            return plot_path

        income_plot = plot_income_distribution(dataToVisOrigin, dataToVis)
        comparison_plots = [generate_side_by_side_plots(dataToVisOrigin, dataToVis, col, "comparison") 
                            for col in dataToVisOrigin.columns.intersection(dataToVisOrigin.columns) if col != "income"]
        comparison_plots = [p for p in comparison_plots if p]  
        # HTML Bericht
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Datenvergleichsbericht</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2 { color: #333; }
                table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f4f4f4; }
                img { margin: 10px; max-width: 1200px; display: block; }
                .image-container { display: flex; justify-content: center; }
            </style>
        </head>
        <body>
            <h1>Datenvergleichsbericht</h1>
            <h2>Erstellt am {{ date }}</h2>
            
            <h2>Fehlende und neue Spalten</h2>
            <table>
                <tr><th>Fehlende Spalten</th><th>Neue Spalten</th></tr>
                <tr>
                    <td>{{ report.fehlende_spalten }}</td>
                    <td>{{ report.neue_spalten }}</td>
                </tr>
            </table>
            
            <h2>Numerische und Kategorische Abweichungen</h2>
            <h3>Numerische Variablen</h3>
            <table>
                <tr><th>Spalte</th><th>Vorher (Min, Max)</th><th>Jetzt (Min, Max)</th></tr>
                {% for col, changes in report.numerische_abweichungen.items() %}
                <tr>
                    <td>{{ col }}</td><td>{{ changes.original }}</td><td>{{ changes.neu }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h3>Kategorische Variablen</h3>
            <table>
                <tr><th>Spalte</th><th>Vorher Kategorien</th><th>Jetzt Kategorien</th></tr>
                {% for col, changes in report.kategorische_abweichungen.items() %}
                <tr>
                    <td>{{ col }}</td><td>{{ changes.original }}</td><td>{{ changes.neu }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Verteilung der Zielvariablen Income</h2>
            <img src="{{ income_plot }}">
            
            <h2>Verteilungsanalyse der unabhaengigen Variablen</h2>
            {% for plot in comparison_plots %}
                <div class="image-container">
                    <img src="{{ plot }}">
                </div>
            {% endfor %}
        </body>
        </html>
        """

        template = Template(html_template)
        html_content = template.render(date=datetime.datetime.now(), income_plot=income_plot, comparison_plots=comparison_plots, report=report)

        with open(f"{self.artifact_dir}/report.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        print("HTML-Bericht wurde erstellt: report.html")

    def safeToRun(self):
        for file in os.listdir(self.artifact_dir):
            file_path = os.path.join(self.artifact_dir, file)
            mlflow.log_artifact(file_path)

def main():
    load_dotenv(override=True)

    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path(os.getcwd())

    root_dir = script_dir.parent.parent.parent

    dataPath = os.getenv("DATA_PATH")
    mlflowUri = os.getenv("MLFLOW_TRACKING_URI")

    dataPath = Path(dataPath)
    mlflowUri = Path(mlflowUri)

    if not dataPath.is_absolute():
        dataPath = root_dir / dataPath
    
    if not mlflowUri.is_absolute():
        mlflowUri = root_dir / mlflowUri

    print(dataPath)
    print(mlflowUri)

    import sys

    sys.path.append(os.path.abspath(root_dir / Path('backend/models/income')))

    from data_preparation import AdultIncomeDataPreparation

    data = AdultIncomeDataPreparation(dataPath)
    
    mlflow.set_tracking_uri(mlflowUri)

    mlflow.set_experiment("Adult Income")

    with mlflow.start_run() as run:
        vis = AdultIncomeDataVisualisation(run, data.originData, data.originData)
        vis.safeToRun()

if __name__ == "__main__":
    main()
