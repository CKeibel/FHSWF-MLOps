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
from scipy.stats import chi2_contingency, pearsonr

# HTML Bericht
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Automatisierte Untersuchung des neuen Datensatzes</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2 { color: #333; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 40px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f4f4f4; }
        img { margin: 10px; max-width: 1200px; display: block; }
        .image-container { display: flex; justify-content: center; }
    </style>
</head>
<body>
    <h1>Automatisierte Untersuchung des neuen Datensatzes</h1>
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
    
     <h2>Darstellung neuer Variablen</h2>
    {% for plot in new_variable_plots %}
        <div class="image-container">
            <img src="{{ plot }}">
        </div>
    {% endfor %}

    <h2>Darstellung fehlender Variablen</h2>
    {% for plot in deleted_variable_plots %}
        <div class="image-container">
            <img src="{{ plot }}">
        </div>
    {% endfor %}
     <h2>Eigenschaften und Hinweise zum neuen Datensatz</h2>
        <table>
        <tr>
            <th>Kategorie</th>
            <th>Ergebnisse</th>
        </tr>
        {% for row in report_s %}
        <tr>
            <td>{{ row.Kategorie }}</td>
            <td>{{ row.Ergebnisse }}</td>
        </tr>
        {% endfor %}
    </table>
    <h2>Korrelationsmatrix des neuen Datensatzes</h2>
        <img src="{{ correlation_plot }}">

</body>
</html>
"""


class AdultIncomeDataVisualisation:
    def __init__(self, mlflowRun, dataToVisOrigin, dataToVis):
        self.mlflow_run = mlflowRun
        self.artifact_dir = "visualizations"
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.dataToVisOrigin = dataToVisOrigin
        self.dataToVis = dataToVis

    # Erstelle einen Bericht
    def compare_datasets(self, df_orig, df_new):
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

    # Funktion Verteilungsanalyse der unabhängigen Variablen
    def generate_side_by_side_plots(self, df_orig, df_new, col, prefix):
        if col == "income":  
            return None  

        fig, axes = plt.subplots(1, 2, figsize=(16, 6)) 
        plt.subplots_adjust(wspace=0.7)    

        if df_orig[col].dtype == "object":
            sns.countplot(x=col, data=df_orig, hue="income", ax=axes[0])
            sns.countplot(x=col, data=df_new, hue="income", ax=axes[1])
            axes[0].tick_params(axis="x", rotation=45)
            axes[1].tick_params(axis="x", rotation=45)
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
    def plot_income_distribution(self, df_orig, df_new):
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

    def plot_new_variables(self, df_new, new_columns):
        new_plots = []
        for col in new_columns:
            if col == "income":
                continue

            plt.figure(figsize=(12, 6))
            if df_new[col].dtype == "object":
                sns.countplot(x=col, data=df_new, hue="income", order=df_new[col].value_counts().index)
                plt.xticks(rotation=45, ha="right")
            else:
                sns.histplot(df_new, x=col, hue="income", bins=30, kde=True)

            plt.title(f"Neue Variable: {col}")
            plot_path = f"plots/new_variable_{col}.png"
            plt.savefig(plot_path)
            plt.close()
            new_plots.append(plot_path)

        return new_plots

    # Funktion zur visualisierung der gelöschten Variablen - welche Informationen fehlen fürs Modell
    def plot_deleted_variables(self, df_orig, deleted_columns):
        deleted_plots = []
        for col in deleted_columns:
            if col == "income":
                continue

            plt.figure(figsize=(12, 6))
            if df_orig[col].dtype == "object":
                sns.countplot(x=col, data=df_orig, hue="income", order=df_orig[col].value_counts().index)
                plt.xticks(rotation=45, ha="right")
            else:
                sns.histplot(df_orig, x=col, hue="income", bins=30, kde=True)

            plt.title(f"Gelöschte Variable: {col}")
            plot_path = f"plots/deleted_variable_{col}.png"
            plt.savefig(plot_path)
            plt.close()
            deleted_plots.append(plot_path)

        return deleted_plots

    def analyze_dataset(self, target_col="income", numeric_threshold=0.7, categorical_p_value=0.01, imbalance_threshold=80):
    
        # Daten laden
        df = self.dataToVis
        report_s = {}
        
        # Allgemeine Beschreibung des Datensatzes
        num_rows, num_cols = df.shape
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        report_s["Allgemeine Datenbeschreibung"] = (f"Der Datensatz hat {num_rows} Zeilen und {num_cols} Spalten.\n"
                                                f"- {len(numeric_cols)} numerische Spalten\n"
                                                f"- {len(categorical_cols)} kategoriale Spalten\n")
        
        # Duplikate prüfen (kritisch ab 0.5%)
        duplicate_count = df.duplicated().sum()
        duplicate_ratio = duplicate_count / num_rows * 100
        report_s["Duplikate"] = f"Es gibt {duplicate_count} Duplikate ({duplicate_ratio:.2f}%). - Da weniger als 0,5% der Daten Duplikate sind, gehen wir davon aus, dass es wahre Beobachtungen sind und keine Bereinigung notwendig ist." \
            if duplicate_ratio < 0.5 else f"Achtung! Es gibt {duplicate_count} Duplikate ({duplicate_ratio:.2f}%). Eine Bereinigung könnte notwendig sein."
        
        # Korrelationen (Pearson für numerisch, Cramér's V für kategorial, Eta-Squared für gemischt)
        cols = df.columns
        corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

        def cramers_v(contingency_table):
            chi2, _, _, _ = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            phi2 = chi2 / n
            r, k = contingency_table.shape
            return np.sqrt(phi2 / min(r-1, k-1))
        
        def eta_squared(num_col, cat_col):
            groups = [num_col[cat_col == cat].dropna() for cat in np.unique(cat_col)]
            ss_total = np.var(num_col, ddof=1) * (len(num_col) - 1)
            ss_between = sum(len(group) * (np.mean(group) - np.mean(num_col))**2 for group in groups)
            return ss_between / ss_total if ss_total > 0 else 0

        for col1 in cols:
            for col2 in cols:
                if col1 == col2:
                    corr_matrix.loc[col1, col2] = 1.0
                elif df[col1].dtype in ["int64", "float64"] and df[col2].dtype in ["int64", "float64"]:
                    corr_matrix.loc[col1, col2] = pearsonr(df[col1].dropna(), df[col2].dropna())[0]
                elif df[col1].dtype == "object" and df[col2].dtype == "object":
                    contingency = pd.crosstab(df[col1], df[col2])
                    corr_matrix.loc[col1, col2] = cramers_v(contingency)
                else:
                    cat_col, num_col = (df[col1], df[col2]) if df[col1].dtype == "object" else (df[col2], df[col1])
                    corr_matrix.loc[col1, col2] = eta_squared(num_col, cat_col)

        # Top 3 Korrelationen zur Zielvariable
        if target_col in corr_matrix.columns:
            target_correlation = corr_matrix[target_col].drop(index=target_col, errors="ignore").abs().sort_values(ascending=False)
            report_s["Top 3 Variablen mit hoechster Korrelation zur Zielvariable"] = target_correlation.head(3).to_string()
            report_s["Top 3 Variablen mit geringster Korrelation zur Zielvariable"] = target_correlation.tail(3).to_string()
        
        # Starke gegenseitige Korrelationen (> 0.7)
        high_corr_pairs = [(row, col, corr_matrix.loc[row, col]) for row in corr_matrix.index for col in corr_matrix.columns
                        if row != col and abs(corr_matrix.loc[row, col]) > numeric_threshold]
        report_s["Variablen mit hoher gegenseitiger Korrelation (>0,7)"] = "\n".join(
            [f" - {var1} & {var2} (Korrelation: {corr:.2f}) - ggfs. nur eine Variable fuer das Model relevant" for var1, var2, corr in high_corr_pairs])


        # Ungleich verteilte Variablen**
        skewed_vars = []
        for col in df.columns:
            most_common_ratio = df[col].value_counts(normalize=True, dropna=True).max() * 100  # Anteil der häufigsten Kategorie
            if most_common_ratio > imbalance_threshold:
                skewed_vars.append((col, most_common_ratio))

        if skewed_vars:
            report_s["Ungleich verteilte Variablen"] = "Folgende Variablen sind stark unausgewogen (>80% in einer Kategorie):\n"
            for var, ratio in skewed_vars:
                report_s["Ungleich verteilte Variablen"] += f" - {var}: {ratio:.2f}% der Werte entfallen auf eine einzige Kategorie\n. Eine Transformation koennte notwendig sein."
        else:
            report_s["Ungleich verteilte Variablen"] = "Keine extrem ungleich verteilten Variablen gefunden."

        
        # Fehlende Werte (NaN und "?"-Zeichen prüfen)
        missing_values = {}

        for col in df.columns:
            num_missing_nan = df[col].isna().sum()  # Anzahl NaN-Werte
            num_missing_question = (df[col] == "?").sum()  # Anzahl "?"-Werte
            total_missing = num_missing_nan + num_missing_question  # Gesamte fehlende Werte

            if total_missing > 0:
                missing_ratio = total_missing / len(df) * 100
                missing_values[col] = (total_missing, missing_ratio, num_missing_nan, num_missing_question)

        if missing_values:
            report_s["Fehlende Werte"] = "Folgende Spalten enthalten fehlende Werte (NaN oder '?'):\n"
            for var, (total, ratio, nan_count, question_count) in missing_values.items():
                report_s["Fehlende Werte"] += f" - {var}: {total} fehlende Werte ({ratio:.2f}%)\n"
                report_s["Fehlende Werte"] += f"   - NaN-Werte: {nan_count}, '?' Werte: {question_count}\n"
                if ratio < 5:
                    report_s["Fehlende Werte"] += "   - Empfehlung: Modus oder Mittelwert verwenden\n"
                else:
                    report_s["Fehlende Werte"] += "   - Empfehlung: Imputation oder Entfernen pruefen\n"
        else:
            report_s["Fehlende Werte"] = "Keine fehlenden Werte (NaN oder '?') gefunden."


        # Zielvariable überprüfen 
        target_col = "income"  # Zielvariable
        if target_col in df.columns:
            target_distribution = df[target_col].value_counts(normalize=True) * 100
            imbalance_threshold = 70
            majority_class = target_distribution.idxmax()
            majority_ratio = target_distribution.max()

            if majority_ratio > imbalance_threshold:
                report_s["Zielvariable-Verteilung"] = (f"Achtung! Die Zielvariable '{target_col}' ist unausgeglichen.\n"
                                                    f"- Mehrheit der Daten ({majority_ratio:.2f}%) sind in der Klasse '{majority_class}'.\n"
                                                    "- Dies kann das Modell stark beeinflussen. In Betracht ziehen: Sampling-Methoden.")
            else:
                report_s["Zielvariable-Verteilung"] = f"Die Zielvariable '{target_col}' ist relativ ausgewogen verteilt."
        else:
            report_s["Zielvariable-Verteilung"] = "Zielvariable 'income' nicht im Datensatz gefunden."


        return report_s

    #Korrelationsmatrix erstellen und Bild abspeichern

    def compute_correlation_matrix(self, df):
        cols = df.columns
        corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

        def cramers_v(contingency_table):
            chi2, _, _, _ = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            phi2 = chi2 / n
            r, k = contingency_table.shape
            return np.sqrt(phi2 / min(r-1, k-1)) if min(r-1, k-1) > 0 else 0

        def eta_squared(num_col, cat_col):
            groups = [num_col[cat_col == cat].dropna() for cat in np.unique(cat_col)]
            ss_total = np.var(num_col, ddof=1) * (len(num_col) - 1)
            ss_between = sum(len(group) * (np.mean(group) - np.mean(num_col))**2 for group in groups)
            return ss_between / ss_total if ss_total > 0 else 0

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols[i:], i):  # Berechnet nur die obere Hälfte der Matrix
                if col1 == col2:
                    corr_matrix.loc[col1, col2] = 1.0
                elif df[col1].dtype in ["int64", "float64"] and df[col2].dtype in ["int64", "float64"]:
                    corr_matrix.loc[col1, col2] = corr_matrix.loc[col2, col1] = pearsonr(df[col1].dropna(), df[col2].dropna())[0]
                elif df[col1].dtype == "object" and df[col2].dtype == "object":
                    contingency = pd.crosstab(df[col1], df[col2])
                    value = cramers_v(contingency)
                    corr_matrix.loc[col1, col2] = corr_matrix.loc[col2, col1] = value
                else:
                    cat_col, num_col = (df[col1], df[col2]) if df[col1].dtype == "object" else (df[col2], df[col1])
                    value = eta_squared(num_col, cat_col)
                    corr_matrix.loc[col1, col2] = corr_matrix.loc[col2, col1] = value

        return corr_matrix
    
    def safeToRun(self):
        for file in os.listdir(self.artifact_dir):
            file_path = os.path.join(self.artifact_dir, file)
            mlflow.log_artifact(file_path)

    def createPlots(self):
        report = self.compare_datasets(self.dataToVisOrigin, self.dataToVis)
        income_plot = self.plot_income_distribution(self.dataToVisOrigin, self.dataToVis)
        comparison_plots = [self.generate_side_by_side_plots(self.dataToVisOrigin, self.dataToVis, col, "comparison") 
                            for col in self.dataToVisOrigin.columns.intersection(self.dataToVisOrigin.columns) if col != "income"]
        comparison_plots = [p for p in comparison_plots if p] 

        new_variable_plots = self.plot_new_variables(self.dataToVis, report["neue_spalten"])
        deleted_variable_plots = self.plot_deleted_variables(self.dataToVisOrigin, report["fehlende_spalten"])

        correlation_matrix = self.compute_correlation_matrix(self.dataToVis)

        # Funktion ausführen und Ergebnisse anzeigen -  ggfs. Dateipfad anpassen!!
        final_analysis_report = self.analyze_dataset()


        # Bericht als DataFrame formatieren
        report_df = pd.DataFrame(list(final_analysis_report.items()), columns=["Kategorie", "Ergebnisse"])


        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Korrelationsmatrix des neuen Datensatzes")
        correlation_plot_fileName = "correlation_matrix.png"

        plt.savefig(f"{self.artifact_dir}/{correlation_plot_fileName}")
        plt.close()

        template = Template(html_template)
        html_content = template.render(date=datetime.datetime.now(), income_plot=income_plot, comparison_plots=comparison_plots, new_variable_plots=new_variable_plots, deleted_variable_plots=deleted_variable_plots, correlation_plot=correlation_plot_fileName, report_s=report_df.to_dict(orient="records"), report=report)

        with open(f"{self.artifact_dir}/report.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        print("HTML-Bericht wurde erstellt: report.html")

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
        vis.createPlots()
        vis.safeToRun()

if __name__ == "__main__":
    main()
