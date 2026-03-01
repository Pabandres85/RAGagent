"""agents/specialists — Paquete de agentes especialistas."""
from agents.specialists.talento_humano import TalentoHumanoAgent
from agents.specialists.infraestructura import InfraestructuraAgent
from agents.specialists.dotacion import DotacionAgent
from agents.specialists.medicamentos_dispositivos import MedicamentosDispositivosAgent
from agents.specialists.procesos_prioritarios import ProcesosPrioritariosAgent
from agents.specialists.historia_clinica import HistoriaClinicaAgent
from agents.specialists.interdependencia import InterdependenciaAgent

__all__ = [
    "TalentoHumanoAgent",
    "InfraestructuraAgent",
    "DotacionAgent",
    "MedicamentosDispositivosAgent",
    "ProcesosPrioritariosAgent",
    "HistoriaClinicaAgent",
    "InterdependenciaAgent",
]
