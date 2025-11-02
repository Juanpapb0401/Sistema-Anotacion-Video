"""
Script para evaluar los modelos entrenados en el conjunto de test
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    f1_score,
    precision_score,
    recall_score
)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Rutas
DATA_PATH = "../data"
MODELS_PATH = "../models"
FIGURES_PATH = "../reports/figures"


def load_test_data():
    """Carga los datos de test"""
    print("📂 Cargando datos de test...")
    
    data_path = Path(DATA_PATH)
    test_df = pd.read_csv(data_path / "test.csv")
    
    # Separar features y labels
    feature_cols = [col for col in test_df.columns if col not in ['label', 'person', 'video_id', 'video_speed']]
    
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    print(f"✅ Test set cargado: {len(X_test):,} muestras, {X_test.shape[1]} features")
    
    return X_test, y_test, test_df


def load_models():
    """Carga todos los modelos entrenados"""
    print("\n📦 Cargando modelos entrenados...")
    
    models_path = Path(MODELS_PATH)
    
    models = {
        'SVM': joblib.load(models_path / "svm_model.pkl"),
        'Random Forest': joblib.load(models_path / "random_forest_model.pkl")
    }
    
    # Cargar XGBoost si existe
    xgb_path = models_path / "xgboost_model.pkl"
    if xgb_path.exists():
        models['XGBoost'] = joblib.load(xgb_path)
    
    label_encoder = joblib.load(models_path / "label_encoder.pkl")
    
    print(f"✅ {len(models)} modelos cargados")
    
    return models, label_encoder


def evaluate_model(model, model_name, X_test, y_test, label_encoder):
    """
    Evalúa un modelo en el conjunto de test
    """
    print(f"\n{'='*80}")
    print(f"📊 Evaluando: {model_name}")
    print('='*80)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas globales
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n🎯 Métricas Generales:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Reporte de clasificación por clase
    class_names = label_encoder.classes_
    report = classification_report(
        y_test, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    print(f"\n📋 Reporte por Clase:")
    print(f"   {'Clase':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"   {'-'*75}")
    
    for class_name in class_names:
        class_report = report[class_name]
        print(f"   {class_name:<25} {class_report['precision']:<12.4f} {class_report['recall']:<12.4f} {class_report['f1-score']:<12.4f} {int(class_report['support']):<10}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    return results, cm


def plot_confusion_matrix(cm, class_names, model_name, save_path):
    """
    Genera y guarda la matriz de confusión
    """
    plt.figure(figsize=(10, 8))
    
    # Calcular porcentajes
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Crear anotaciones que muestren conteo y porcentaje
    annotations = np.array([[f'{count}\n({pct:.1f}%)' 
                            for count, pct in zip(row_counts, row_pcts)]
                           for row_counts, row_pcts in zip(cm, cm_pct)])
    
    sns.heatmap(
        cm, 
        annot=annotations,
        fmt='',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Número de muestras'}
    )
    
    plt.title(f'Matriz de Confusión - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Etiqueta Verdadera', fontsize=12)
    plt.xlabel('Etiqueta Predicha', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
    print(f"   ✅ Matriz de confusión guardada: {filename}")
    plt.close()


def plot_model_comparison(all_results, save_path):
    """
    Genera gráfico comparativo de todos los modelos
    """
    print("\n📊 Generando gráfico comparativo...")
    
    # Preparar datos
    models = [r['model_name'] for r in all_results]
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, metric in enumerate(metrics):
        values = [r[metric] for r in all_results]
        
        bars = axes[idx].bar(models, values, color=colors)
        axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].set_ylim([0, 1.1])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.4f}',
                          ha='center', va='bottom', fontweight='bold')
        
        # Rotar etiquetas si es necesario
        axes[idx].tick_params(axis='x', rotation=15)
    
    plt.suptitle('Comparación de Modelos - Conjunto de Test', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(save_path / "model_comparison.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico comparativo guardado: model_comparison.png")
    plt.close()


def plot_per_class_performance(all_results, class_names, save_path):
    """
    Genera gráfico de desempeño por clase para cada modelo
    """
    print("\n📊 Generando gráfico de desempeño por clase...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['precision', 'recall', 'f1-score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Preparar datos para cada modelo
        x = np.arange(len(class_names))
        width = 0.25
        
        for i, result in enumerate(all_results):
            values = [result['classification_report'][class_name][metric] 
                     for class_name in class_names]
            ax.bar(x + i*width, values, width, label=result['model_name'])
        
        ax.set_xlabel('Clase', fontsize=10)
        ax.set_ylabel(metric.replace('-', ' ').title(), fontsize=10)
        ax.set_title(f'{metric.replace("-", " ").title()} por Clase', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    plt.suptitle('Desempeño por Clase - Todos los Modelos', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(save_path / "per_class_performance.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico por clase guardado: per_class_performance.png")
    plt.close()


def generate_evaluation_report(all_results, best_model_name):
    """
    Genera un reporte completo de evaluación
    """
    print("\n📝 Generando reporte de evaluación...")
    
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("REPORTE DE EVALUACIÓN - CONJUNTO DE TEST")
    report_lines.append("Sistema de Anotación de Video - Entrega 2")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Resumen comparativo
    report_lines.append("📊 RESUMEN COMPARATIVO")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Modelo':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    report_lines.append("-" * 80)
    
    for result in sorted(all_results, key=lambda x: x['f1_score'], reverse=True):
        marker = "🏆" if result['model_name'] == best_model_name else "  "
        report_lines.append(
            f"{marker} {result['model_name']:<18} "
            f"{result['accuracy']:<12.4f} "
            f"{result['precision']:<12.4f} "
            f"{result['recall']:<12.4f} "
            f"{result['f1_score']:<12.4f}"
        )
    
    report_lines.append("")
    
    # Detalles por modelo
    report_lines.append("🔍 DETALLES POR MODELO")
    report_lines.append("-" * 80)
    
    for result in all_results:
        report_lines.append(f"\n{result['model_name']}:")
        report_lines.append(f"  Accuracy:  {result['accuracy']:.4f}")
        report_lines.append(f"  Precision: {result['precision']:.4f}")
        report_lines.append(f"  Recall:    {result['recall']:.4f}")
        report_lines.append(f"  F1-Score:  {result['f1_score']:.4f}")
        
        report_lines.append(f"\n  Desempeño por clase:")
        report_lines.append(f"    {'Clase':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        report_lines.append(f"    {'-'*60}")
        
        for class_name, metrics in result['classification_report'].items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                report_lines.append(
                    f"    {class_name:<25} "
                    f"{metrics['precision']:<12.4f} "
                    f"{metrics['recall']:<12.4f} "
                    f"{metrics['f1-score']:<12.4f}"
                )
    
    # Mejor modelo
    report_lines.append("")
    report_lines.append("🏆 MODELO RECOMENDADO")
    report_lines.append("-" * 80)
    
    best_result = next(r for r in all_results if r['model_name'] == best_model_name)
    report_lines.append(f"Modelo: {best_model_name}")
    report_lines.append(f"F1-Score: {best_result['f1_score']:.4f}")
    report_lines.append(f"Accuracy: {best_result['accuracy']:.4f}")
    report_lines.append("")
    report_lines.append("Este modelo ha demostrado el mejor desempeño en el conjunto de test")
    report_lines.append("y está disponible como 'best_model.pkl'")
    
    # Análisis de errores
    report_lines.append("")
    report_lines.append("⚠️  ANÁLISIS DE ERRORES")
    report_lines.append("-" * 80)
    
    cm = np.array(best_result['confusion_matrix'])
    class_names = [k for k in best_result['classification_report'].keys() 
                   if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Encontrar las confusiones más comunes
    confusions = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                confusions.append((class_names[i], class_names[j], cm[i][j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    if confusions:
        report_lines.append("Confusiones más comunes (verdadero → predicho):")
        for true_class, pred_class, count in confusions[:5]:
            report_lines.append(f"  {true_class} → {pred_class}: {count} muestras")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Guardar reporte
    report_text = "\n".join(report_lines)
    report_file = Path(DATA_PATH) / "evaluation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ Reporte guardado: evaluation_report.txt")
    
    return report_text


def save_evaluation_results(all_results, best_model_name):
    """
    Guarda los resultados de evaluación en JSON
    """
    print("\n💾 Guardando resultados de evaluación...")
    
    # Preparar datos serializables (remover numpy arrays)
    results_serializable = []
    for result in all_results:
        result_copy = result.copy()
        result_copy['confusion_matrix'] = result['confusion_matrix']  # Ya está como lista
        results_serializable.append(result_copy)
    
    evaluation_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_evaluated': len(all_results),
        'best_model': best_model_name,
        'results': results_serializable
    }
    
    output_file = Path(DATA_PATH) / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    print(f"✅ Resultados guardados: evaluation_results.json")


def main():
    """Función principal"""
    print("🚀 Iniciando Evaluación de Modelos\n")
    print("=" * 80)
    
    # 1. Cargar datos de test
    X_test, y_test, test_df = load_test_data()
    
    # 2. Cargar modelos
    models, label_encoder = load_models()
    
    # 3. Evaluar cada modelo
    all_results = []
    
    figures_path = Path(FIGURES_PATH)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    for model_name, model in models.items():
        results, cm = evaluate_model(model, model_name, X_test, y_test, label_encoder)
        all_results.append(results)
        
        # Generar matriz de confusión
        plot_confusion_matrix(cm, label_encoder.classes_, model_name, figures_path)
    
    # 4. Identificar mejor modelo
    best_result = max(all_results, key=lambda x: x['f1_score'])
    best_model_name = best_result['model_name']
    
    print(f"\n🏆 Mejor modelo: {best_model_name} (F1-Score: {best_result['f1_score']:.4f})")
    
    # 5. Generar gráficos comparativos
    plot_model_comparison(all_results, figures_path)
    plot_per_class_performance(all_results, label_encoder.classes_, figures_path)
    
    # 6. Guardar resultados
    save_evaluation_results(all_results, best_model_name)
    
    # 7. Generar reporte
    report = generate_evaluation_report(all_results, best_model_name)
    
    print("\n" + "=" * 80)
    print(report)
    
    print("\n✅ Evaluación completada!")
    print(f"📁 Resultados guardados en: {DATA_PATH}/")
    print("   - evaluation_results.json")
    print("   - evaluation_report.txt")
    print(f"\n📊 Gráficos guardados en: {FIGURES_PATH}/")
    print("   - confusion_matrix_*.png (3 archivos)")
    print("   - model_comparison.png")
    print("   - per_class_performance.png")


if __name__ == "__main__":
    main()
