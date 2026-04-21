"""
Módulo refactorizado para demostración de Conformal Prediction
Utiliza clases y buenas prácticas de programación
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import streamlit as st

class StreamlitCache:
    """Cache que persiste entre ejecuciones usando st.session_state"""
    
    def __init__(self):
        if 'streamlit_cache' not in st.session_state:
            st.session_state.streamlit_cache = {}
    
    @property
    def _cache(self):
        return st.session_state.streamlit_cache
    
    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value
        print(f"✅ Cacheado: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)
    
    def has(self, key: str) -> bool:
        return key in self._cache
    
    def keys(self) -> List[str]:
        return list(self._cache.keys())
    
    def clear(self):
        self._cache.clear()
        print("🧹 Caché limpiado")



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
    accuracy:            float
    predictions:         int | float             #float o enteros
    prediction_sets:     list[int] | list[float] #lista de enteros o floats
    probabilities:       float
    scores:              float                   #para el caso de clasificacion seran las "1 - probabilidades"
    multi_class_indices: Dict[int, Any]


class ConformalPredictionDemo:
    """
    Clase principal para demostrar Conformal Prediction
    Encapsula toda la lógica de carga de datos, entrenamiento y visualización
    """
    
    def __init__(self, model_type: str = "mlp", problem: str = "clf", epochs: int = 1):
        """
        Inicializa la demostración
        
        Args:
            alpha: Nivel de significancia para conformal prediction
            model_type: Tipo de modelo ('mlp' o 'cnn')
            epochs: Número de épocas de entrenamiento
        """
        self.alpha       = None
        self.model_type  = model_type
        self.problem     = problem
        self.epochs      = epochs
        self.model       = None
        self.cp_model    = None #cls/reg.
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

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model
    
    def train_model(self) -> bool:
        """
        Entrena el modelo base (solo una vez)
        
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        if self.problem == "clf":
            self.cp_model= SplitConformalClassifier(self.get_model())
        elif self.problem == "reg":
            self.cp_model= SplitConformalRegressor(self.get_model())

        try:
            if self.get_model() is None or self.data is None:
                raise ValueError(f"Modelo o datos no inicializados: \n model: {self.get_model()}\n data: {self.data}")
            
            print("🚀 Entrenando modelo base...")
            # Entrenar el modelo (sin epochs en fit())
            self.cp_model.fit(self.data['X_train'], self.data['y_train'])
            print(f"✅ Modelo entrenado")
            return True
            
        except Exception as e:
            print(f"Error entrenando modelo: {e}")
            return False
    
    def calibrate(self, alpha_=None) -> bool:
        """
        Calibra el clasificador conformal (se puede ejecutar múltiples veces)
        
        Returns:
            bool: True si la calibración fue exitosa
        """
        try:
            if SplitConformalClassifier is None:
                raise ImportError("SplitConformalClassifier no está disponible")
                
            if self.model is None or self.data is None or self.cp_model is None:
                raise ValueError(f"Modelo, datos o modelo conformal no inicializados: \n model: {self.model}\n data: {self.data}\n cp_model: {self.cp_model}")

            # Entrenar y calibrar
            print(f"🎯 Calibrando clasificador conformal con alpha={alpha_}")

            if alpha_ is None:
                raise ValueError("Alpha no indicado")
            self.alpha = alpha_
            self.cp_model.calibrate(self.data['X_cal'], self.data['y_cal'], alpha=self.alpha)
            
            print(f"✅ Clasificador conformal calibrado")
            return True
            
        except Exception as e:
            print(f"Error calibrando clasificador conformal: {e}")
            return False

    def evaluate_model(self) -> Optional[PredictionResults]:
        """
        Evalúa el modelo y genera predicciones conformales
        
        Args:
            use_cache: Si usar índices de referencia cacheados
            
        Returns:
            PredictionResults con todos los resultados
        """
        try:
            if self.model is None or self.data is None or self.cp_model is None:
                raise ValueError(f"Modelo, datos o clasificador conformal no inicializados: \n model: {self.model}\n data: {self.data}\n cp_model: {self.cp_model}")
            
            # Predicciones del modelo
            y_pred = self.model.predict(self.data['X_test'])
            accuracy = (y_pred == self.data['y_test']).float().mean().item()
            
            # Predicciones conjuntos y probabilidades
            pred_sets      = self.cp_model.predict_set(self.data['X_test'])
            probabilidades = self.cp_model.predict_proba(self.data['X_test'])
            scores         = 1 - probabilidades
            
            # Encontrar casos con múltiples clases (usando caché si corresponde)
            multi_class_indices = self._find_multi_class_predictions(scores)
            
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
    
    def _find_multi_class_predictions(self, scores: Any, max_samples: int = 3) -> Dict[int, Any]:
        """
        Encuentra predicciones con múltiples clases
        
        Args:
            scores: Scores del modelo
            max_samples: Máximo de muestras a retornar
            
        Returns:
            Dict con índices y clases válidas
        """
        threshold = self.cp_model.q_hat
        valid_classes = scores <= threshold
        
        multi_class_indices = {}
        
        # 1. Buscar casos con múltiples clases
        for i, pred_set in enumerate(valid_classes):
            if pred_set.sum() > 1:
                multi_class_indices[i] = pred_set
                if len(multi_class_indices) >= max_samples:
                    return multi_class_indices
        
        # 2. Si no hay suficientes multi-clase, completar con casos normales
        needed = max_samples - len(multi_class_indices)
        if needed > 0:
            # Encontrar índices que no son multi-clase
            normal_indices = [i for i in range(len(valid_classes)) 
                            if i not in multi_class_indices]
            
            # Tomar aleatoriamente los que faltan
            if normal_indices:
                selected = np.random.choice(normal_indices, size=needed, replace=False)
                for i in selected:
                    multi_class_indices[i] = valid_classes[i]
        
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
                    range(len(self.cp_model.classes_)), 
                    results.probabilities[i], 
                    label="Class probabilities", 
                    color='#90CDF4FF'
                )
                axes[idx, 1].set_xticks(range(len(self.cp_model.classes_)))
                axes[idx, 1].set_xticklabels([str(i) for i in self.cp_model.classes_])
                axes[idx, 1].axhline(
                    y=1 - self.cp_model.q_hat, 
                    label='Umbral (1 - q)', 
                    color="#FC766AFF", 
                    linestyle='dashed'
                )
                axes[idx, 1].legend(loc=1)
                idx_pred = str(list(map(int, self.cp_model.classes_[valid_classes])))
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
        max_samples: int = 3, noise_level: float = 0.0, cache: StreamlitCache = None) -> Tuple[Optional[Dict[str, Any]], Optional[StreamlitFigureManager]]:
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

    use_cache = cache.activate if cache is not None else False

    # Inicializar gestor de figuras
    figure_manager = StreamlitFigureManager()
    
    # Crear demo
    demo = ConformalPredictionDemo(model_type=model_type, epochs=epochs)
    
    # 1. Cargar datos (con ruido si corresponde)
    if not demo.load_and_process_data(noise_level=noise_level):
        return None, figure_manager
    
     # 2. Crear modelo (siempre se crea)
    if not demo.create_model():
        return None, figure_manager

    # 3. Entrenar modelo
    if use_cache and cache.has("model"):
        # Usar modelo cacheado
        demo.set_model(cache.get("model"))
        demo.cp_model = cache.get("cp_model")
        print("Usando modelo cacheado")
    else:
        # Entrenar nuevo modelo
        if not demo.train_model():
            return None, figure_manager
        if use_cache:
            cache.set("model", demo.get_model())
            cache.set("cp_model", demo.cp_model)
            print("Modelo cacheado")
    
    # 4. Calibrar clasificador conformal (cada vez que cambia alpha o ruido)
    if not demo.calibrate(alpha_=alpha):
        return None, figure_manager
    
    # 5. Evaluar modelo
    results = demo.evaluate_model()
    if results is None:
        return None, figure_manager
    
    # 6. Manejar índices con/sin cache
    if use_cache:
        if cache.has("indices"):
            # Usar índices cacheados PERO forzar a que aparezcan aunque no sean multi-clase
            cached_indices = cache.get("indices")
            new_multi_class = {}
            
            # Recalcular predicciones para índices cacheados (más seguro)
            scores = 1 - demo.cp_model.predict_proba(demo.data['X_test'])
            threshold = demo.cp_model.q_hat
            valid_classes = scores <= threshold
            
            for i in cached_indices:
                if i < len(valid_classes):
                    new_multi_class[i] = valid_classes[i]
            
            results.multi_class_indices = new_multi_class
            print("Usando índices cacheados (predicciones recalcualdas)")
        else:
            # Primera vez: guardar índices
            cache.set("indices", list(results.multi_class_indices.keys()))
            print("Índices cacheados por primera vez")
        
    # 7. Crear visualización
    fig = demo.plot_predictions(results, max_samples=max_samples)
    if fig is not None:
        figure_manager.add_figure("predictions", fig)
    
    # Preparar resultados para retorno
    results_dict = {
        'accuracy': results.accuracy,
        'multi_class_count': len(results.multi_class_indices),
        'alpha': alpha,
        'model_type': model_type,
        'threshold': 1 - demo.cp_model.q_hat if demo.cp_model else None,
        'noise_level': noise_level,
        'use_cache': use_cache,
        'cache_status': '🔒 Cacheado' if use_cache else '🔄 Dinámico',
        'model_trained': cache.has("model"),
        'cached_indices': cache.get("indices", []) if use_cache and cache.has("indices") else []
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