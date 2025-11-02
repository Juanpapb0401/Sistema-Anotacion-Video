"""
Configuración para la integración de etiquetas
"""

# Mapeo de etiquetas de Label Studio a etiquetas estandarizadas
LABEL_MAPPING = {
    # Caminar hacia cámara
    "Caminar acercandose": "caminar_hacia_camara",
    "Caminando hacia adelante de lado": "caminar_hacia_camara",
    
    # Caminar de regreso
    "Caminar alejandose (espalda)": "caminar_de_regreso",
    "Caminar alejandose (espaldas)": "caminar_de_regreso",  # Variante del otro grupo
    "Devolviendose caminando de lado": "caminar_de_regreso",
    
    # Girar (todas las variantes)
    "Giro 180 desde la izquierda": "girar",
    "Giro 180 desde la derecha": "girar",
    "Giro 180 izquierda": "girar",  # Variante del otro grupo
    "Giro 180 derecha": "girar",    # Variante del otro grupo
    "Giro 360": "girar",
    
    # Sentarse
    "Sentarse": "sentarse",
    
    # Ponerse de pie
    "Ponerse de pie": "ponerse_de_pie",
    "Pararse": "ponerse_de_pie",
    
    # Actividades adicionales (excluidas del modelo principal)
    "Sentadilla": "sentadilla",
    "Sentadillas": "sentadilla",  # Variante del otro grupo
    "Inclinarse a la derecha": "inclinacion_lateral",
    "Inclinarse a la izquierda": "inclinacion_lateral",
    "Inclinarse hacia la derecha": "inclinacion_lateral",
    "Inclinarse hacia la izquierda": "inclinacion_lateral",
    "Inclinarse derecha": "inclinacion_lateral",  # Variante del otro grupo
    "Inclinarse izquierda": "inclinacion_lateral",  # Variante del otro grupo
    "Parado sin movimiento": "sin_movimiento",
    "Sentado sin movimiento": "sin_movimiento",
}

# Etiquetas principales para el modelo (las 5 requeridas)
MAIN_LABELS = [
    "caminar_hacia_camara",
    "caminar_de_regreso",
    "girar",
    "sentarse",
    "ponerse_de_pie"
]

# Etiquetas a excluir del dataset final
EXCLUDED_LABELS = [
    "sentadilla",
    "inclinacion_lateral",
    "sin_movimiento"
]

# Rutas
FEATURES_PATH = "../../Entrega1/data/03_features"
VIDEOS_JSON_PATH = "../../Videos/JSON"
VIDEOS_JSON_PATH_OTHER_GROUP = "../../Videos/JSON_otro_grupo"
FEATURES_PATH_OTHER_GROUP = "../../Entrega1/data/raw_videos_otro_grupo"  # Path where features for other group videos will be
OUTPUT_PATH = "../data"

# Personas y sus videos
PERSONS = {
    "Joshua": {
        "normal": list(range(1, 11)),  # 1-10
        "lento": [5, 6]  # Lento5, Lento6
    },
    "Juan": {
        "normal": list(range(1, 11)),  # 1-10
        "lento": [1, 2]  # Lento1, Lento2
    },
    "Santiago": {
        "normal": list(range(1, 12)),  # 1-11
        "lento": []
    },
    "Thomas": {
        "normal": list(range(1, 11)),  # 1-10
        "lento": [3, 4]  # Lento3, Lento4
    }
}

# Configuración específica para videos del otro grupo
OTHER_GROUP_VIDEOS = {
    "OtroGrupo": {
        "videos": [
            "VIDEO_01", "VIDEO_03", "VIDEO_04", "VIDEO_05", "VIDEO_06", "VIDEO_07",
            "VIDEO_08", "VIDEO_09", "VIDEO_010", "VIDEO_011", "VIDEO_012", "VIDEO_013",
            "VIDEO_014", "VIDEO_015", "VIDEO_016", "VIDEO_017", "VIDEO_018",
            "VIDEO_020", "VIDEO_021", "VIDEO_022", "VIDEO_023", "VIDEO_024"
        ]
    }
}

VIDEOS_JSON_PATH_OTHER_GROUP = "../../Videos/JSON_otro_grupo"
