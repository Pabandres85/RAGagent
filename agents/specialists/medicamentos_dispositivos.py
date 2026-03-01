"""
Agente especialista: Medicamentos y Dispositivos Médicos (Resolución 3100 de 2019).
"""
from agents.base_specialist import BaseSpecialist


class MedicamentosDispositivosAgent(BaseSpecialist):
    MODULE = "medicamentos_dispositivos"
    MODULE_NAME = "Medicamentos y Dispositivos Médicos"
