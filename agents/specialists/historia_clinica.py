"""
Agente especialista: Historia Clínica y Registros (Resolución 3100 de 2019).
"""
from agents.base_specialist import BaseSpecialist


class HistoriaClinicaAgent(BaseSpecialist):
    MODULE = "historia_clinica"
    MODULE_NAME = "Historia Clínica y Registros"
