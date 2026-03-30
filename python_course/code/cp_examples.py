"""
Módulo refactorizado para demostración de Conformal Prediction
Utiliza clases y buenas prácticas de programación
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch

# Sistema de caché para consistencia de índices de visualización
class CacheManager:
    """
    Gestiona el caché de índices de visualización para comparaciones consistentes
    """
    def __init__(self):
        self.visual_indices = []      # Índices de las 3 imágenes mostradas inicialmente
        self.is_cached = False        # Estado del caché
        self.model_trained = False    # Estado del entrenamiento del modelo
        self.cached_model = None      # Modelo cacheado
        
    def cache_visualization_indices(self, indices: List[int]):
        """
        Guarda los índices de la primera visualización
        
        Args:
            indices: Lista de índices de las imágenes mostradas
        """
        self.visual_indices = indices.copy()
        self.is_cached = True
        print(f"🔒 Índices de visualización cacheados: {self.visual_indices}")
    
    def get_visualization_indices(self) -> List[int]:
        """
        Obtiene los índices cacheados o devuelve vacío si no hay caché
        
        Returns:
            Lista de índices cacheados o lista vacía
        """
        return self.visual_indices.copy() if self.is_cached else []
    
    def clear_cache(self):
        """Limpia el caché de índices"""
        self.visual_indices = []
        self.is_cached = False
        print("🗑️ Caché de visualización limpiado")
    
    def set_model_trained(self, trained: bool = True):
        """Marca si el modelo ya fue entrenado"""
        self.model_trained = trained
        if trained:
            print("✅ Modelo marcado como entrenado")
    
    def is_model_trained(self) -> bool:
        """Verifica si el modelo ya fue entrenado"""
        return self.model_trained
    
    def set_cached_model(self, model):
        """Guarda el modelo entrenado"""
        self.cached_model = model
        print("✅ Modelo cacheado")
    
    def get_cached_model(self):
        """Obtiene el modelo cacheado"""
        return self.cached_model

# Instancia global del caché
cache_manager = CacheManager()


# Importaciones del paquete cp_models (asumiendo que están disponibles)
try:
    from cp_models.classification import SplitConformalClassifier
    from cp_models.models.mlp import GenericMLP
    from cp_models.models.utils import get_data
except ImportError as e:
    print(f"Error importando módulos cp_models: {e}")
    # Acá se podría implementar versiones mock para desarrollo
    SplitConformalClassifier = None
    GenericMLP = None
    get_data = None


@dataclass
class PredictionResults:
    """Contenedor para los resultados de la predicción conformal"""
    accuracy: float
    predictions: Any
    prediction_sets: Any
    probabilities: Any
    scores: Any #para el caso de clasificacion seran las "1 - probabilidades"
    multi_class_indices: Dict[int, Any]


class ConformalPredictionDemo:
    """
    Clase principal para demostrar Conformal Prediction
    Encapsula toda la lógica de carga de datos, entrenamiento y visualización
    """
    
    def __init__(self, alpha: float = 0.05, model_type: str = "mlp", epochs: int = 1):
        """
        Inicializa la demostración
        
        Args:
            alpha: Nivel de significancia para conformal prediction
            model_type: Tipo de modelo ('mlp' o 'cnn')
            epochs: Número de épocas de entrenamiento
        """
        self.alpha       = alpha
        self.model_type  = model_type
        self.epochs      = epochs
        self.model = None
        self.create_model()
        self.cp_classifier = SplitConformalClassifier(self.model, alpha=self.alpha)
        self.data        = None
        
    def load_and_process_data(self, source: str = "mnist", size_calib: int = 50, noise_level: float = 0.0) -> bool:
        """
        Carga y procesa los datos para el modelo
        
        Args:
            source: Fuente de datos ('mnist')
            size_calib: Tamaño del conjunto de calibración
            noise_level: Nivel de ruido gaussiano (0.0 = sin ruido, 1.0 = máximo ruido)
            
        Returns:
            bool: True si los datos se cargaron correctamente
        """
        try:
            if get_data is None:
                raise ImportError("get_data no está disponible")
                
            X_train, y_train, X_test, y_test, X_cal, y_cal = get_data(
                source=source, flatten=False, size_calib=size_calib
            )
            
            # Agregar ruido gaussiano si noise_level > 0
            if noise_level > 0.0:
                noise_std = noise_level * 0.5  # MNIST está en [0,1], así que 0.5 es reasonable
                
                # Agregar ruido a todos los conjuntos
                #X_train = X_train + torch.randn_like(X_train) * noise_std
                X_test = X_test + torch.randn_like(X_test) * noise_std
                X_cal  = X_cal + torch.randn_like(X_cal) * noise_std
                
                # Asegurar que los valores permanezcan en [0,1]
                #X_train = torch.clamp(X_train, 0, 1)
                X_test = torch.clamp(X_test, 0, 1)
                X_cal  = torch.clamp(X_cal, 0, 1)
            
            # Para MLP hay que aplanar las imágenes
            if self.model_type == "mlp":
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test  = X_test.reshape(X_test.shape[0], -1)
                X_cal   = X_cal.reshape(X_cal.shape[0], -1)
            
            self.data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'X_cal': X_cal,
                'y_cal': y_cal
            }
            
            print(f"Datos cargados - Train: {X_train.shape}, Test: {X_test.shape}, Cal: {X_cal.shape}")
            if noise_level > 0:
                print(f"🔊 Ruido aplicado con nivel {noise_level:.2f}")
            return True
            
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return False
    
    def create_model(self) -> bool:
        """
        Crea el modelo base según el tipo especificado
        
        Returns:
            bool: True si el modelo se creó correctamente
        """
        try:
            if GenericMLP is None:
                raise ImportError("GenericMLP no está disponible")
                
            if self.model_type == "mlp":
                self.model = GenericMLP(input_dim=784, num_classes=10, epochs=self.epochs)
            elif self.model_type == "cnn":
                # Descomentar cuando CNN esté disponible
                # self.model = GenericCNN(input_channels=1, num_classes=10)
                raise NotImplementedError("CNN no implementado aún")
            else:
                raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
                
            print(f"Modelo {self.model_type} creado exitosamente")
            return True
            
        except Exception as e:
            print(f"Error creando modelo: {e}")
            return False
    
    def train_model(self) -> bool:
        """
        Entrena el modelo base (solo una vez)
        
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        try:
            if self.model is None or self.data is None:
                raise ValueError("Modelo o datos no inicializados")
            
            # Verificar si el modelo ya fue entrenado
            if cache_manager.is_model_trained():
                print("🔄 Modelo ya entrenado - omitiendo entrenamiento")
                self.model = cache_manager.get_cached_model()
                self.cp_classifier = SplitConformalClassifier(self.model, self.alpha)
                return True
            
            print("🚀 Entrenando modelo base...")
            # Entrenar el modelo (sin epochs en fit())
            self.cp_classifier.fit(self.data['X_train'], self.data['y_train'])
            
            # Marcar como entrenado
            cache_manager.set_model_trained(True)
            cache_manager.set_cached_model(self.model)
            print(f"✅ Modelo entrenado")
            return True
            
        except Exception as e:
            print(f"Error entrenando modelo: {e}")
            return False
    
    def calibrate(self) -> bool:
        """
        Calibra el clasificador conformal (se puede ejecutar múltiples veces)
        
        Returns:
            bool: True si la calibración fue exitosa
        """
        try:
            if SplitConformalClassifier is None:
                raise ImportError("SplitConformalClassifier no está disponible")
                
            if self.model is None or self.data is None:
                raise ValueError("Modelo o datos no inicializados")
            
            # Crear nuevo clasificador conformal con el alpha actual
            self.cp_classifier = SplitConformalClassifier(self.model, alpha=self.alpha)
            
            # Entrenar y calibrar
            print(f"🎯 Calibrando clasificador conformal con alpha={self.alpha}")
            self.train_model()
       
            self.cp_classifier.calibrate(self.data['X_cal'], self.data['y_cal'])
            
            print(f"✅ Clasificador conformal calibrado")
            return True
            
        except Exception as e:
            print(f"Error calibrando clasificador conformal: {e}")
            return False

    def evaluate_model(self, use_cache: bool = False) -> Optional[PredictionResults]:
        """
        Evalúa el modelo y genera predicciones conformales
        
        Args:
            use_cache: Si usar índices de referencia cacheados
            
        Returns:
            PredictionResults con todos los resultados
        """
        try:
            if self.model is None or self.data is None or self.cp_classifier is None:
                raise ValueError("Modelo, datos o clasificador conformal no inicializados")
            
            # Predicciones del modelo
            y_pred = self.model.predict(self.data['X_test'])
            accuracy = (y_pred == self.data['y_test']).float().mean().item()
            
            # Predicciones conjuntos y probabilidades
            pred_sets = self.cp_classifier.predict_set(self.data['X_test'])
            probabilidades = self.cp_classifier.predict_proba(self.data['X_test'])
            scores = 1 - probabilidades
            
            # Encontrar casos con múltiples clases (usando caché si corresponde)
            multi_class_indices = self._find_multi_class_predictions(scores, use_cache=use_cache)
            
            results = PredictionResults(
                accuracy           = accuracy,
                predictions        = y_pred,
                prediction_sets    = pred_sets,
                probabilities      = probabilidades,
                scores             = scores,
                multi_class_indices= multi_class_indices
            )
            
            print(f"Accuracy del modelo: {accuracy:.4f}")
            print(f"Casos con múltiples clases: {len(multi_class_indices)}")
            
            return results
            
        except Exception as e:
            print(f"Error evaluando modelo: {e}")
            return None
    
    def _find_multi_class_predictions(self, scores: Any, max_samples: int = 3, use_cache: bool = False) -> Dict[int, Any]:
        """
        Encuentra predicciones con múltiples clases
        
        Args:
            scores: Scores del modelo
            max_samples: Máximo de muestras a retornar
            use_cache: Si usar índices cacheados de visualización
            
        Returns:
            Dict con índices y clases válidas
        """
        threshold = self.cp_classifier.q_hat
        valid_classes = scores <= threshold

        # Si usamos caché y hay índices cacheados, priorizarlos
        if use_cache:
            cached_indices = cache_manager.get_visualization_indices()
            
            if cached_indices:
                multi_class_indices = {}
                
                # Primero buscar en índices cacheados
                for i in cached_indices:
                    if i < len(valid_classes) and valid_classes[i].sum() > 1:
                        multi_class_indices[i] = valid_classes[i]
                        if len(multi_class_indices) >= max_samples:
                            return multi_class_indices
                
                # Si no hay suficientes multi-clase en caché, buscar en resto
                if len(multi_class_indices) < max_samples:
                    for i, pred_set in enumerate(valid_classes):
                        if i not in cached_indices and pred_set.sum() > 1:
                            multi_class_indices[i] = pred_set
                            if len(multi_class_indices) >= max_samples:
                                break
                
                # Completar con casos normales si aún no hay suficientes
                if len(multi_class_indices) < max_samples:
                    # Primero usar índices cacheados que no son multi-clase
                    for i in cached_indices:
                        if i < len(valid_classes) and i not in multi_class_indices:
                            multi_class_indices[i] = valid_classes[i]
                            if len(multi_class_indices) >= max_samples:
                                break
                    
                    # Si aún falta, usar otros índices
                    if len(multi_class_indices) < max_samples:
                        all_indices = set(range(len(valid_classes)))
                        used_indices = set(multi_class_indices.keys())
                        remaining_indices = list(all_indices - used_indices)
                        
                        needed = max_samples - len(multi_class_indices)
                        if remaining_indices:
                            sampled_indices = np.random.choice(remaining_indices, size=needed, replace=False)
                            for i in sampled_indices:
                                multi_class_indices[i] = valid_classes[i]
                
                return multi_class_indices

        # Comportamiento normal (sin caché)
        multi_class_indices = {}
        for i, pred_set in enumerate(valid_classes):
            if pred_set.sum() > 1:  # Múltiples clases predichas
                multi_class_indices[i] = pred_set
                if len(multi_class_indices) >= max_samples:
                    break

        # Si no hay suficientes casos multi-clase, agregar casos normales
        if len(multi_class_indices) < max_samples:
            all_indices = set(range(len(valid_classes)))
            multi_class_indices_set = set(multi_class_indices.keys())
            remaining_indices = list(all_indices - multi_class_indices_set)
            
            needed = max_samples - len(multi_class_indices)
            if remaining_indices:
                sampled_indices = np.random.choice(remaining_indices, size=needed, replace=False)
                for i in sampled_indices:
                    multi_class_indices[i] = valid_classes[i]
        
        # Si es la primera ejecución (sin caché), guardar los índices usados
        if not use_cache and multi_class_indices:
            cache_manager.cache_visualization_indices(list(multi_class_indices.keys()))
        
        return multi_class_indices
    
    def plot_predictions(self, results: PredictionResults, max_samples: int = 3) -> Optional[plt.Figure]:
        """
        Grafica las predicciones - siempre muestra max_samples gráficos
        
        Args:
            results: Resultados de la predicción
            max_samples: Máximo de ejemplos a mostrar (siempre muestra esta cantidad)
            
        Returns:
            Figure de matplotlib
        """
        try:
            # Siempre mostrar max_samples gráficos
            num_samples = max_samples
            
            # Crear subplots - siempre max_samples
            fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            # Casos multi-clase encontrados
            multi_class_items = list(results.multi_class_indices.items())
            
            # Llenar primeros slots con casos multi-clase
            for idx, (i, valid_classes) in enumerate(multi_class_items):
                # Imagen original
                axes[idx, 0].imshow(
                    self.data['X_test'][i].reshape(28, 28).numpy(), 
                    cmap='gray'
                )
                axes[idx, 0].set_title(f"Test image: {self.data['y_test'][i]}")
                axes[idx, 0].set_xticks([])
                axes[idx, 0].set_yticks([])
                
                # Scores de clases
                axes[idx, 1].bar(
                    range(len(self.cp_classifier.classes_)), 
                    results.probabilities[i], 
                    label="Class probabilities", 
                    color='#90CDF4FF'
                )
                axes[idx, 1].set_xticks(range(len(self.cp_classifier.classes_)))
                axes[idx, 1].set_xticklabels([str(i) for i in self.cp_classifier.classes_])
                axes[idx, 1].axhline(
                    y=1 - self.cp_classifier.q_hat, 
                    label='Umbral (1 - q)', 
                    color="#FC766AFF", 
                    linestyle='dashed'
                )
                axes[idx, 1].legend(loc=1)
                idx_pred = str(list(map(int, self.cp_classifier.classes_[valid_classes])))
                axes[idx, 1].set_title(rf"$C(X_{{test}})$: { {idx_pred} }")
            
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creando gráfico: {e}")
            return None


class StreamlitFigureManager:
    """
    Gestiona múltiples figuras para Streamlit
    Resuelve el problema de que Streamlit solo muestra la última figura
    """
    
    def __init__(self):
        self.figures: Dict[str, plt.Figure] = {}
    
    def add_figure(self, name: str, figure: plt.Figure):
        """Agrega una figura al gestor"""
        self.figures[name] = figure
    
    def get_figure(self, name: str) -> Optional[plt.Figure]:
        """Obtiene una figura específica"""
        return self.figures.get(name)
    
    def get_all_figures(self) -> Dict[str, plt.Figure]:
        """Obtiene todas las figuras"""
        return self.figures.copy()
    
    def clear(self):
        """Limpia todas las figuras"""
        for fig in self.figures.values():
            plt.close(fig)
        self.figures.clear()


def run(model_type: str = "mlp", alpha: float = 0.05, epochs: int = 1, 
        max_samples: int = 3, noise_level: float = 0.0, use_cache: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[StreamlitFigureManager]]:
    """
    Función principal que ejecuta la demostración completa
    
    Args:
        model_type: Tipo de modelo ('mlp' o 'cnn')
        alpha: Nivel de significancia
        epochs: Número de épocas
        max_samples: Máximo de muestras para visualizar
        noise_level: Nivel de ruido gaussiano (0.0 = sin ruido, 1.0 = máximo ruido)
        use_cache: Si usar índices cacheados para consistencia
        
    Returns:
        Tuple: (resultados_dict, figure_manager) para uso con Streamlit
    """
    # Inicializar gestor de figuras
    figure_manager = StreamlitFigureManager()
    
    # Crear demo
    demo = ConformalPredictionDemo(alpha=alpha, model_type=model_type, epochs=epochs)
    
    # 1. Cargar datos (con ruido si corresponde)
    if not demo.load_and_process_data(noise_level=noise_level):
        return None, figure_manager
    
#    # 2. Crear modelo
#    if not demo.create_model():
#        return None, figure_manager
    
    # 3. Entrenar modelo (solo la primera vez)
    if not demo.train_model():
        return None, figure_manager
    
    # 4. Calibrar clasificador conformal (cada vez que cambia alpha o ruido)
    if not demo.calibrate():
        return None, figure_manager
    
    # 5. Evaluar modelo
    results = demo.evaluate_model(use_cache=use_cache)
    if results is None:
        return None, figure_manager
    
    # 6. Crear visualización
    fig = demo.plot_predictions(results, max_samples=max_samples)
    if fig is not None:
        figure_manager.add_figure("predictions", fig)
    
    # Preparar resultados para retorno
    results_dict = {
        'accuracy': results.accuracy,
        'multi_class_count': len(results.multi_class_indices),
        'alpha': alpha,
        'model_type': model_type,
        'threshold': 1 - demo.cp_classifier.q_hat if demo.cp_classifier else None,
        'noise_level': noise_level,
        'use_cache': use_cache,
        'cache_status': '🔒 Cacheado' if use_cache else '🔄 Dinámico',
        'model_trained': cache_manager.is_model_trained(),
        'cached_indices': cache_manager.get_visualization_indices() if use_cache else []
    }
    
    return results_dict, figure_manager


# Función de compatibilidad con el código original
def get_fig_cp():
    """
    Función de compatibilidad que mantiene la interfaz original
    Devuelve la última figura para compatibilidad con código existente
    """
    results_dict, figure_manager = run()
    
    if figure_manager and figure_manager.figures:
        # Devolver la última figura (compatibilidad con código original)
        last_figure = list(figure_manager.figures.values())[-1]
        return last_figure
    
    return None


if __name__ == "__main__":
    # Ejemplo de uso
    results, figures = run(model_type="mlp", alpha=0.05, epochs=1)
    
    if results:
        print("Resultados:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        if figures.figures:
            print(f"Figuras generadas: {list(figures.figures.keys())}")
            plt.show()
    else:
        print("Error en la ejecución")