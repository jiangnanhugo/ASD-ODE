import numpy as np
from scibench.data.base import KnownEquation, register_eq_class
from scibench.symbolic_data_generator import LogUniformSampling
@register_eq_class
class BIOMD0000000936(KnownEquation):
    _eq_name = 'vars1_prog1'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "ferrel2011 - autonomous biochemical oscillator in cell cycle in Xenopus laevis v2"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x_1**9.0/(x_1**8.0 + 0.003906) + 0.1']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x_1**9.0/(x_1**8.0 + 0.003906) + 0.1])

@register_eq_class
class BIOMD0000000425(KnownEquation):
    _eq_name = 'vars1_prog2'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Tan2012 - Antibiotic Treatment Inoculum Effect"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['nan']
    
    def np_eq(self, t, x):
        return np.array([nan])

@register_eq_class
class BIOMD0000000414(KnownEquation):
    _eq_name = 'vars1_prog3'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Band2012-DII-Venus-ReducedModel"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=1, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.0026*x_1 - 0.005*x_1/(0.056*x_1 + 0.16) + 0.005']
    
    def np_eq(self, t, x):
        return np.array([-0.0026*x_1 - 0.005*x_1/(0.056*x_1 + 0.16) + 0.005])

@register_eq_class
class BIOMD0000000728(KnownEquation):
    _eq_name = 'vars2_prog1'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Norel1990 - MPF and Cyclin Oscillations"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['1.0*x[0]**2*x_2 - 10.0*x[0]/(x[0] + 1.0) + 3.466*x_2', '1.2 - 1.0*x[0]']
    
    def np_eq(self, t, x):
        return np.array([1.0*x[0]**2*x_2 - 10.0*x[0]/(x[0] + 1.0) + 3.466*x_2, 1.2 - 1.0*x[0]])

@register_eq_class
class BIOMD0000000815(KnownEquation):
    _eq_name = 'vars2_prog2'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Chrobak2011 - A mathematical model of induced cancer-adaptive immune system competition"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.03125*x[0]**2 - 0.125*x[0]*x_2 + 0.0625*x[0]', '-0.08594*x[0]*x_2 - 0.03125*x_2**2 + 0.03125*x_2']
    
    def np_eq(self, t, x):
        return np.array([-0.03125*x[0]**2 - 0.125*x[0]*x_2 + 0.0625*x[0], -0.08594*x[0]*x_2 - 0.03125*x_2**2 + 0.03125*x_2])

@register_eq_class
class BIOMD0000000346(KnownEquation):
    _eq_name = 'vars2_prog3'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "FitzHugh1961-NerveMembrane"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0]**3 + 3.0*x[0] + 3.0*x_2 - 1.2', '-0.3333*x[0] - 0.2667*x_2 + 0.2333']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0]**3 + 3.0*x[0] + 3.0*x_2 - 1.2, -0.3333*x[0] - 0.2667*x_2 + 0.2333])

@register_eq_class
class BIOMD0000000678(KnownEquation):
    _eq_name = 'vars2_prog4'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Tomida2003 - Calcium Oscillatory-induced translocation of nuclear factor of activated T cells"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.359*x[0]*x_4 + 0.147*x_2 + 0.035*x_3', '0.359*x[0]*x_4 - 0.207*x_2']
    
    def np_eq(self, t, x):
        return np.array([-0.359*x[0]*x_4 + 0.147*x_2 + 0.035*x_3, 0.359*x[0]*x_4 - 0.207*x_2])

@register_eq_class
class BIOMD0000000062(KnownEquation):
    _eq_name = 'vars2_prog5'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Bhartiya2003-Tryptophan-operon"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.01*x[0] + 2.197/(x_3**1.92 + 11.26)', '2.025e+4*x[0]/(x_3 + 810.0) - 0.01*x_2 - 25.0*x_2/(x_2 + 0.2)']
    
    def np_eq(self, t, x):
        return np.array([-0.01*x[0] + 2.197/(x_3**1.92 + 11.26), 2.025e+4*x[0]/(x_3 + 810.0) - 0.01*x_2 - 25.0*x_2/(x_2 + 0.2)])

@register_eq_class
class BIOMD0000000776(KnownEquation):
    _eq_name = 'vars2_prog6'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'log']
    _description = "Monro2008 - chemotherapy resistance"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.006*x[0]*np.log(5.0e-13*x[0] + 5.0e-13*x_2)', '-0.006*x_2*np.log(5.0e-13*x[0] + 5.0e-13*x_2)']
    
    def np_eq(self, t, x):
        return np.array([0.006*x[0]*np.log(5.0e-13*x[0] + 5.0e-13*x_2), -0.006*x_2*np.log(5.0e-13*x[0] + 5.0e-13*x_2)])

@register_eq_class
class BIOMD0000000919(KnownEquation):
    _eq_name = 'vars2_prog7'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Ledzewicz2013 - On optimal chemotherapy with a strongly targeted agent for a model of tumor immune system interactions with generalized logistic growth"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.5e-5*x[0]*x_2**2 + 0.005*x[0]*x_2 - 0.375*x[0] + 0.118', '-1.0*x[0]*x_2 + 0.56*x_2 - 0.0007179*x_2**2.0']
    
    def np_eq(self, t, x):
        return np.array([-1.5e-5*x[0]*x_2**2 + 0.005*x[0]*x_2 - 0.375*x[0] + 0.118, -1.0*x[0]*x_2 + 0.56*x_2 - 0.0007179*x_2**2.0])

@register_eq_class
class BIOMD0000000760(KnownEquation):
    _eq_name = 'vars2_prog8'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Feizabadi2011/1 - immunodeficiency in cancer core model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-2.5e-7*x[0]**2 + 0.3*x[0] - 1.0*x_2/(x_2 + 1000.0)', '-9.333e-8*x[0]**2 + 0.028*x[0] - 4.0e-7*x_2**2 + 0.4*x_2']
    
    def np_eq(self, t, x):
        return np.array([-2.5e-7*x[0]**2 + 0.3*x[0] - 1.0*x_2/(x_2 + 1000.0), -9.333e-8*x[0]**2 + 0.028*x[0] - 4.0e-7*x_2**2 + 0.4*x_2])

@register_eq_class
class BIOMD0000000751(KnownEquation):
    _eq_name = 'vars2_prog9'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'exp', 'pow']
    _description = "Wilkie2013b - immune-induced cancer dormancy and immune evasion-basic"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-2.0e-11*x[0]**2 - 0.000105*x[0]*x_2*np.exp(-10*t/1681) + 0.2*x[0]', '-0.2*x_2**2/(0.001*x[0]*x_2 + 100.0) + 0.2*x_2']
    
    def np_eq(self, t, x):
        return np.array([-2.0e-11*x[0]**2 - 0.000105*x[0]*x_2*np.exp(-10*t/1681) + 0.2*x[0], -0.2*x_2**2/(0.001*x[0]*x_2 + 100.0) + 0.2*x_2])

@register_eq_class
class BIOMD0000000538(KnownEquation):
    _eq_name = 'vars2_prog10'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Clarke2000 - One-hit model of cell death in neuronal degenerations"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.278*x[0]', '-0.223*x_2']
    
    def np_eq(self, t, x):
        return np.array([-0.278*x[0], -0.223*x_2])

@register_eq_class
class BIOMD0000000774(KnownEquation):
    _eq_name = 'vars2_prog11'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Wodarz2018/1 - simple model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.004*x[0] + 0.004*x_2/(0.01*x[0]**1.0 + 1.0)', '0.006*x[0] - 0.003*x_2 - 0.004*x_2/(0.01*x[0]**1.0 + 1.0)']
    
    def np_eq(self, t, x):
        return np.array([0.004*x[0] + 0.004*x_2/(0.01*x[0]**1.0 + 1.0), 0.006*x[0] - 0.003*x_2 - 0.004*x_2/(0.01*x[0]**1.0 + 1.0)])

@register_eq_class
class BIOMD0000001037(KnownEquation):
    _eq_name = 'vars2_prog12'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Alharbi2019 - Tumor-normal model (TNM) of the development of tumor cells and their impact on normal cell dynamics"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.931*x[0]*x_2 + 0.431*x[0]', '1.189*x[0]*x_2 - 0.1772*x_2**2 + 0.443*x_2']
    
    def np_eq(self, t, x):
        return np.array([-0.931*x[0]*x_2 + 0.431*x[0], 1.189*x[0]*x_2 - 0.1772*x_2**2 + 0.443*x_2])

@register_eq_class
class BIOMD0000000552(KnownEquation):
    _eq_name = 'vars2_prog13'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Ehrenstein2000 - Positive-Feedback model for the loss of acetylcholine in Alzheimer's disease"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.007*x[0]*x_2', '-0.004*x[0] - 0.01*x_2 + 0.33']
    
    def np_eq(self, t, x):
        return np.array([-0.007*x[0]*x_2, -0.004*x[0] - 0.01*x_2 + 0.33])

@register_eq_class
class BIOMD0000000485(KnownEquation):
    _eq_name = 'vars2_prog14'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Cao2013 - Application of ABSIS method in the bistable SchlÃ¶gl model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.00096*x[0]**3 + 0.1229*x[0]**2 - 3.072*x[0] + 12.5', '0.00096*x[0]**3 - 0.1229*x[0]**2 + 3.072*x[0] - 12.5']
    
    def np_eq(self, t, x):
        return np.array([-0.00096*x[0]**3 + 0.1229*x[0]**2 - 3.072*x[0] + 12.5, 0.00096*x[0]**3 - 0.1229*x[0]**2 + 3.072*x[0] - 12.5])

@register_eq_class
class BIOMD0000001013(KnownEquation):
    _eq_name = 'vars2_prog15'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Leon-Triana2021 - Competition between tumour cells and single-target CAR T-cells"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.9*x[0]*x_2/(x_2 + 1.0e+10) - 0.04*x[0]*x_2/(x[0] + 2.0e+9) - 0.1429*x[0]', '0.02*x_2']
    
    def np_eq(self, t, x):
        return np.array([0.9*x[0]*x_2/(x_2 + 1.0e+10) - 0.04*x[0]*x_2/(x[0] + 2.0e+9) - 0.1429*x[0], 0.02*x_2])

@register_eq_class
class BIOMD0000001024(KnownEquation):
    _eq_name = 'vars2_prog16'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Chaudhury2020 - Lotka-Volterra mathematical model of CAR-T cell and tumour kinetics"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.002*x[0]*x_2 - 0.16*x[0]', '0.15*x_2']
    
    def np_eq(self, t, x):
        return np.array([0.002*x[0]*x_2 - 0.16*x[0], 0.15*x_2])

@register_eq_class
class BIOMD0000000549(KnownEquation):
    _eq_name = 'vars2_prog17'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Baker2013 - Cytokine Mediated Inflammation in Rheumatoid Arthritis - Age Dependent"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-x[0] + 7.0*x_2**2/(x_2**2 + 0.25)', '49.0*t**2*x_2**2/(t**2*x[0]**2*x_2**2 + 1.0*t**2*x[0]**2 + 1.0*t**2*x_2**2 + 1.0*t**2 + 225.0*x[0]**2*x_2**2 + 225.0*x[0]**2 + 225.0*x_2**2 + 225.0) + 1.0*x_2**2/(x[0]**2*x_2**2 + 1.0*x[0]**2 + 1.0*x_2**2 + 1.0) - 1.25*x_2 + 0.025/(x[0]**2 + 1.0)']
    
    def np_eq(self, t, x):
        return np.array([-x[0] + 7.0*x_2**2/(x_2**2 + 0.25), 49.0*t**2*x_2**2/(t**2*x[0]**2*x_2**2 + 1.0*t**2*x[0]**2 + 1.0*t**2*x_2**2 + 1.0*t**2 + 225.0*x[0]**2*x_2**2 + 225.0*x[0]**2 + 225.0*x_2**2 + 225.0) + 1.0*x_2**2/(x[0]**2*x_2**2 + 1.0*x[0]**2 + 1.0*x_2**2 + 1.0) - 1.25*x_2 + 0.025/(x[0]**2 + 1.0)])

@register_eq_class
class BIOMD0000000550(KnownEquation):
    _eq_name = 'vars2_prog18'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Baker2013 - Cytokine Mediated Inflammation in Rheumatoid Arthritis"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-x[0] + 3.5*x_2**2/(x_2**2 + 0.25)', '1.0*x_2**2/(x[0]**2*x_2**2 + 1.0*x[0]**2 + 1.0*x_2**2 + 1.0) - 1.25*x_2 + 0.025/(x[0]**2 + 1.0)']
    
    def np_eq(self, t, x):
        return np.array([-x[0] + 3.5*x_2**2/(x_2**2 + 0.25), 1.0*x_2**2/(x[0]**2*x_2**2 + 1.0*x[0]**2 + 1.0*x_2**2 + 1.0) - 1.25*x_2 + 0.025/(x[0]**2 + 1.0)])

@register_eq_class
class BIOMD0000000114(KnownEquation):
    _eq_name = 'vars2_prog19'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Somogyi1990-CaOscillations"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-5.0*x[0]*x_2**4.0/(x_2**4.0 + 81.0) - 0.01*x[0] + 2.0*x_2', '5.0*x[0]*x_2**4.0/(x_2**4.0 + 81.0) + 0.01*x[0] - 3.0*x_2 + 1.0']
    
    def np_eq(self, t, x):
        return np.array([-5.0*x[0]*x_2**4.0/(x_2**4.0 + 81.0) - 0.01*x[0] + 2.0*x_2, 5.0*x[0]*x_2**4.0/(x_2**4.0 + 81.0) + 0.01*x[0] - 3.0*x_2 + 1.0])

@register_eq_class
class BIOMD0000000799(KnownEquation):
    _eq_name = 'vars2_prog20'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Cucuianu2010 - A hypothetical-mathematical model of acute myeloid leukaemia pathogenesis"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.1*x[0] + 0.3*x[0]/(0.5*x[0] + 0.5*x_2 + 1.0)', '-0.1*x_2 + 0.3*x_2/(0.5*x[0] + 0.5*x_2 + 1.0)']
    
    def np_eq(self, t, x):
        return np.array([-0.1*x[0] + 0.3*x[0]/(0.5*x[0] + 0.5*x_2 + 1.0), -0.1*x_2 + 0.3*x_2/(0.5*x[0] + 0.5*x_2 + 1.0)])

@register_eq_class
class BIOMD0000000897(KnownEquation):
    _eq_name = 'vars2_prog21'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Khajanchi2015 - The combined effects of optimal control in cancer remission"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.124*x[0]*x_2/(x_2 + 2.019e+7) - 0.041*x[0] + 1.3e+4', '0.18*x_2']
    
    def np_eq(self, t, x):
        return np.array([0.124*x[0]*x_2/(x_2 + 2.019e+7) - 0.041*x[0] + 1.3e+4, 0.18*x_2])

@register_eq_class
class BIOMD0000000782(KnownEquation):
    _eq_name = 'vars2_prog22'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Wang2016/3 - oncolytic efficacy of M1 virus-SN model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.2*x[0]*x_2 - 0.02*x[0] + 0.02', '0.16*x[0]*x_2 - 0.03*x_2']
    
    def np_eq(self, t, x):
        return np.array([-0.2*x[0]*x_2 - 0.02*x[0] + 0.02, 0.16*x[0]*x_2 - 0.03*x_2])

@register_eq_class
class BIOMD0000000098(KnownEquation):
    _eq_name = 'vars2_prog23'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Goldbeter1990-CalciumSpike-CICR"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-10.0*x[0] - 65.0*x[0]**2.0/(x[0]**2.0 + 1.0) + 500.0*x[0]**4.0*x_2**2.0/(x[0]**4.0*x_2**2.0 + 4.0*x[0]**4.0 + 0.6561*x_2**2.0 + 2.624) + 1.0*x_2 + 3.197', '65.0*x[0]**2.0/(x[0]**2.0 + 1.0) - 500.0*x[0]**4.0*x_2**2.0/(x[0]**4.0*x_2**2.0 + 4.0*x[0]**4.0 + 0.6561*x_2**2.0 + 2.624) - 1.0*x_2']
    
    def np_eq(self, t, x):
        return np.array([-10.0*x[0] - 65.0*x[0]**2.0/(x[0]**2.0 + 1.0) + 500.0*x[0]**4.0*x_2**2.0/(x[0]**4.0*x_2**2.0 + 4.0*x[0]**4.0 + 0.6561*x_2**2.0 + 2.624) + 1.0*x_2 + 3.197, 65.0*x[0]**2.0/(x[0]**2.0 + 1.0) - 500.0*x[0]**4.0*x_2**2.0/(x[0]**4.0*x_2**2.0 + 4.0*x[0]**4.0 + 0.6561*x_2**2.0 + 2.624) - 1.0*x_2])

@register_eq_class
class BIOMD0000000793(KnownEquation):
    _eq_name = 'vars2_prog24'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Chen2011/1 - bone marrow invasion absolute model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.2*x[0]**2 + 0.1*x[0]', '-1.0*x[0]*x_2 - 0.8*x_2**2 + 0.7*x_2']
    
    def np_eq(self, t, x):
        return np.array([-0.2*x[0]**2 + 0.1*x[0], -1.0*x[0]*x_2 - 0.8*x_2**2 + 0.7*x_2])

@register_eq_class
class BIOMD0000000486(KnownEquation):
    _eq_name = 'vars2_prog25'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Cao2013 - Application of ABSIS method in the reversible isomerization model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.12*x[0] + 1.0*x_2', '0.12*x[0] - 1.0*x_2']
    
    def np_eq(self, t, x):
        return np.array([-0.12*x[0] + 1.0*x_2, 0.12*x[0] - 1.0*x_2])

@register_eq_class
class BIOMD0000000785(KnownEquation):
    _eq_name = 'vars2_prog26'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Sotolongo-Costa2003 - Behavior of tumors under nonstationary therapy"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0]*x_2 + 2.0*x[0]', '1.0*x[0]*x_2 - 0.2*x[0] - 0.5*x_2 + 0.25']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0]*x_2 + 2.0*x[0], 1.0*x[0]*x_2 - 0.2*x[0] - 0.5*x_2 + 0.25])

@register_eq_class
class BIOMD0000000553(KnownEquation):
    _eq_name = 'vars2_prog27'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Ehrenstein1997 - The choline-leakage hypothesis in Alzheimer's disease"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.007*x[0]*x_2', '-0.004*x[0] - 0.01*x_2 + 0.33']
    
    def np_eq(self, t, x):
        return np.array([-0.007*x[0]*x_2, -0.004*x[0] - 0.01*x_2 + 0.33])

@register_eq_class
class BIOMD0000000484(KnownEquation):
    _eq_name = 'vars2_prog28'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Cao2013 - Application of ABSIS method in birth-death process"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['1.0 - 0.025*x[0]', '0.025*x[0] - 1.0']
    
    def np_eq(self, t, x):
        return np.array([1.0 - 0.025*x[0], 0.025*x[0] - 1.0])

@register_eq_class
class BIOMD0000000762(KnownEquation):
    _eq_name = 'vars2_prog29'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Kuznetsov1994 - Nonlinear dynamics of immunogenic tumors"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.124*x[0]*x_2/(x_2 + 2.019e+7) - 0.041*x[0] + 1.3e+4', '0.18*x_2']
    
    def np_eq(self, t, x):
        return np.array([0.124*x[0]*x_2/(x_2 + 2.019e+7) - 0.041*x[0] + 1.3e+4, 0.18*x_2])

@register_eq_class
class BIOMD0000000573(KnownEquation):
    _eq_name = 'vars2_prog30'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Aguilera 2014 - HIV latency. Interaction between HIV proteins and immune response"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.029*x[0]*x_2 + 0.134*x[0]/(x[0] + 380.0) + 0.001', '-0.927*x[0]*x_2 + 0.07']
    
    def np_eq(self, t, x):
        return np.array([-0.029*x[0]*x_2 + 0.134*x[0]/(x[0] + 380.0) + 0.001, -0.927*x[0]*x_2 + 0.07])

@register_eq_class
class BIOMD0000000795(KnownEquation):
    _eq_name = 'vars2_prog31'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Chen2011/2 - bone marrow invasion relative model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.8*x[0]**2 - 0.9*x[0]*x_2 + 0.7*x[0]', '-0.1*x[0]*x_2 - 0.2*x_2**2 + 0.1*x_2']
    
    def np_eq(self, t, x):
        return np.array([-0.8*x[0]**2 - 0.9*x[0]*x_2 + 0.7*x[0], -0.1*x[0]*x_2 - 0.2*x_2**2 + 0.1*x_2])

@register_eq_class
class BIOMD0000000836(KnownEquation):
    _eq_name = 'vars2_prog32'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Radosavljevic2009-BioterroristAttack-PanicProtection-1"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.6*x[0]**2 - 2.8*x[0]*x_2 + 6.0*x[0]', '1.0*x[0]*x_2']
    
    def np_eq(self, t, x):
        return np.array([-0.6*x[0]**2 - 2.8*x[0]*x_2 + 6.0*x[0], 1.0*x[0]*x_2])

@register_eq_class
class BIOMD0000000758(KnownEquation):
    _eq_name = 'vars2_prog33'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Babbs2012 - immunotherapy"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.004*x[0] - 4.0*x_2', '0.09*x[0]*x_2 - 0.1*x_2']
    
    def np_eq(self, t, x):
        return np.array([0.004*x[0] - 4.0*x_2, 0.09*x[0]*x_2 - 0.1*x_2])

@register_eq_class
class BIOMD0000000935(KnownEquation):
    _eq_name = 'vars2_prog34'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Ferrel2011 - Cdk1 and APC regulation in cell cycle in Xenopus laevis"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-3.0*x[0]*x_2**8.0/(x_2**8.0 + 0.003906) + 0.1', '-3.0*x[0]**8.0*x_2/(x[0]**8.0 + 0.003906) + 3.0*x[0]**8.0/(x[0]**8.0 + 0.003906) - 1.0*x_2']
    
    def np_eq(self, t, x):
        return np.array([-3.0*x[0]*x_2**8.0/(x_2**8.0 + 0.003906) + 0.1, -3.0*x[0]**8.0*x_2/(x[0]**8.0 + 0.003906) + 3.0*x[0]**8.0/(x[0]**8.0 + 0.003906) - 1.0*x_2])

@register_eq_class
class BIOMD0000000115(KnownEquation):
    _eq_name = 'vars2_prog35'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Somogyi1990-CaOscillations-SingleCaSpike"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-10.0*x[0]*x_2**4.0/(x_2**4.0 + 3.842) - 0.01*x[0] + 2.01*x_2 + 10.0*x_2**5.0/(x_2**4.0 + 3.842)', '10.0*x[0]*x_2**4.0/(x_2**4.0 + 3.842) + 0.01*x[0] - 3.01*x_2 - 10.0*x_2**5.0/(x_2**4.0 + 3.842) + 1.0']
    
    def np_eq(self, t, x):
        return np.array([-10.0*x[0]*x_2**4.0/(x_2**4.0 + 3.842) - 0.01*x[0] + 2.01*x_2 + 10.0*x_2**5.0/(x_2**4.0 + 3.842), 10.0*x[0]*x_2**4.0/(x_2**4.0 + 3.842) + 0.01*x[0] - 3.01*x_2 - 10.0*x_2**5.0/(x_2**4.0 + 3.842) + 1.0])

@register_eq_class
class BIOMD0000000742(KnownEquation):
    _eq_name = 'vars2_prog36'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Garcia2018basic - cancer and immune cell count basic model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.514*x[0]', '10.0 - 0.02*x_2']
    
    def np_eq(self, t, x):
        return np.array([0.514*x[0], 10.0 - 0.02*x_2])

@register_eq_class
class BIOMD0000000753(KnownEquation):
    _eq_name = 'vars2_prog37'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Figueredo2013/1 - immunointeraction base model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=2, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.006544*x[0]**2 - 1.0*x[0]*x_2 + 1.636*x[0]', '-0.003*x[0]*x_2 + 1.131*x[0]*x_2/(x[0] + 20.19) - 2.0*x_2 + 0.318']
    
    def np_eq(self, t, x):
        return np.array([-0.006544*x[0]**2 - 1.0*x[0]*x_2 + 1.636*x[0], -0.003*x[0]*x_2 + 1.131*x[0]*x_2/(x[0] + 20.19) - 2.0*x_2 + 0.318])

@register_eq_class
class BIOMD0000000922(KnownEquation):
    _eq_name = 'vars3_prog1'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Turner2015-Human/Mosquito ELP Model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['600.0 - 0.411*x[0]', '0.361*x[0] - 0.184*x[1]', '0.134*x[1] - 0.345*x_3']
    
    def np_eq(self, t, x):
        return np.array([600.0 - 0.411*x[0], 0.361*x[0] - 0.184*x[1], 0.134*x[1] - 0.345*x_3])

@register_eq_class
class BIOMD0000001031(KnownEquation):
    _eq_name = 'vars3_prog2'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Al-Tuwairqi2020 - Dynamics of cancer virotherapy - Phase I treatment"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0]*x_3', '1.0*x[0]*x_3 - 1.0*x[1]', '-0.02*x[0]*x_3 + 1.0*x[1] - 0.15*x_3']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0]*x_3, 1.0*x[0]*x_3 - 1.0*x[1], -0.02*x[0]*x_3 + 1.0*x[1] - 0.15*x_3])

@register_eq_class
class BIOMD0000000807(KnownEquation):
    _eq_name = 'vars3_prog3'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Fassoni2019 - Oncogenesis encompassing mutations and genetic instability"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.01 - 0.01*x[0]', '0.03*x[1]', '-0.5*x_3**2 + 0.034*x_3']
    
    def np_eq(self, t, x):
        return np.array([0.01 - 0.01*x[0], 0.03*x[1], -0.5*x_3**2 + 0.034*x_3])

@register_eq_class
class BIOMD0000000156(KnownEquation):
    _eq_name = 'vars3_prog4'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Zatorsky2006-p53-Model5"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-3.7*x[0]*x[1] + 2.0*x[0]', '-0.9*x[1] + 1.1*x_3', '1.5*x[0] - 1.1*x_3']
    
    def np_eq(self, t, x):
        return np.array([-3.7*x[0]*x[1] + 2.0*x[0], -0.9*x[1] + 1.1*x_3, 1.5*x[0] - 1.1*x_3])

@register_eq_class
class BIOMD0000000878(KnownEquation):
    _eq_name = 'vars3_prog5'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Lenbury2001-InsulinKineticsModel-A"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.1*x[0]*x_3 + 0.2*x[1]*x_3 + 0.1*x_3', '-0.01*x[0] + 0.01 + 0.01/x_3', '-0.1*x[1]*x_3 + 0.257*x[1] - 0.1*x_3**2 + 0.331*x_3 - 0.3187']
    
    def np_eq(self, t, x):
        return np.array([-0.1*x[0]*x_3 + 0.2*x[1]*x_3 + 0.1*x_3, -0.01*x[0] + 0.01 + 0.01/x_3, -0.1*x[1]*x_3 + 0.257*x[1] - 0.1*x_3**2 + 0.331*x_3 - 0.3187])

@register_eq_class
class BIOMD0000000159(KnownEquation):
    _eq_name = 'vars3_prog6'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Zatorsky2006-p53-Model1"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-3.2*x[0]*x[1] + 0.3', '-0.1*x[1] + 0.1*x_3', '0.4*x[0] - 0.1*x_3']
    
    def np_eq(self, t, x):
        return np.array([-3.2*x[0]*x[1] + 0.3, -0.1*x[1] + 0.1*x_3, 0.4*x[0] - 0.1*x_3])

@register_eq_class
class BIOMD0000000982(KnownEquation):
    _eq_name = 'vars3_prog7'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Law2020 - SIR model of COVID-19 transmission in Malyasia with time-varying parameters"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.1663*0.9534**t*x[0]*x[1]/(x[0] + x[1] + x_3)', '0.1663*0.9534**t*x[0]*x[1]/(x[0] + x[1] + x_3) - 0.05*x[1]', '0.05*x[1]']
    
    def np_eq(self, t, x):
        return np.array([-0.1663*0.9534**t*x[0]*x[1]/(x[0] + x[1] + x_3), 0.1663*0.9534**t*x[0]*x[1]/(x[0] + x[1] + x_3) - 0.05*x[1], 0.05*x[1]])

@register_eq_class
class BIOMD0000000800(KnownEquation):
    _eq_name = 'vars3_prog8'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Precup2012 - Mathematical modeling of cell dynamics after allogeneic bone marrow transplantation"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.23*x[0]**2/(x[0] + x[1] + 2.0*x_3 + 1.0) + 0.23*x[0]*x[1]/(x[0] + x[1] + 2.0*x_3 + 1.0) - 0.01*x[0] + 0.23*x[0]/(x[0] + x[1] + 2.0*x_3 + 1.0)', '0.45*x[0]*x[1]/(x[0] + x[1] + 2.0*x_3 + 1.0) + 0.45*x[1]**2/(x[0] + x[1] + 2.0*x_3 + 1.0) - 0.01*x[1] + 0.45*x[1]/(x[0] + x[1] + 2.0*x_3 + 1.0)', '-0.46*x[0]*x_3/(2.0*x[0] + 2.0*x[1] + x_3 + 1.0) - 0.46*x[1]*x_3/(2.0*x[0] + 2.0*x[1] + x_3 + 1.0) + 0.22*x_3']
    
    def np_eq(self, t, x):
        return np.array([0.23*x[0]**2/(x[0] + x[1] + 2.0*x_3 + 1.0) + 0.23*x[0]*x[1]/(x[0] + x[1] + 2.0*x_3 + 1.0) - 0.01*x[0] + 0.23*x[0]/(x[0] + x[1] + 2.0*x_3 + 1.0), 0.45*x[0]*x[1]/(x[0] + x[1] + 2.0*x_3 + 1.0) + 0.45*x[1]**2/(x[0] + x[1] + 2.0*x_3 + 1.0) - 0.01*x[1] + 0.45*x[1]/(x[0] + x[1] + 2.0*x_3 + 1.0), -0.46*x[0]*x_3/(2.0*x[0] + 2.0*x[1] + x_3 + 1.0) - 0.46*x[1]*x_3/(2.0*x[0] + 2.0*x[1] + x_3 + 1.0) + 0.22*x_3])

@register_eq_class
class BIOMD0000000778(KnownEquation):
    _eq_name = 'vars3_prog9'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Wei2017 - tumor T cell and cytokine interaction"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0e-5*x[0]**2 - 0.1*x[0]*x_3/(x[0] + 50.0) + 0.01*x[0]', '0.1*x[0]*x[1]/(x[0] + 1000.0) - 0.03*x[1]', '0.01*x[0]*x[1]/(x[0] + 100.0) - 50.0*x_3']
    
    def np_eq(self, t, x):
        return np.array([-1.0e-5*x[0]**2 - 0.1*x[0]*x_3/(x[0] + 50.0) + 0.01*x[0], 0.1*x[0]*x[1]/(x[0] + 1000.0) - 0.03*x[1], 0.01*x[0]*x[1]/(x[0] + 100.0) - 50.0*x_3])

@register_eq_class
class BIOMD0000000278(KnownEquation):
    _eq_name = 'vars3_prog10'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Lemaire2004 - Role of RANK/RANKL/OPG pathway in bone remodelling process"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.035*x[0]*x_3/(x_3 + 0.00025) - 0.000175*x[0]/(x_3 + 0.00025) + 0.001*x_3/(x_3 + 0.005) + 2.5e-7/(x_3 + 0.005)', '0.035*x[0]*x_3/(x_3 + 0.00025) + 0.000175*x[0]/(x_3 + 0.00025) - 0.189*x[1]', '6.84*x[1]/(2.949e+4*x[0] + 1.588) - 0.7*x_3**2/(x_3 + 0.005) - 0.000175*x_3/(x_3 + 0.005)']
    
    def np_eq(self, t, x):
        return np.array([-0.035*x[0]*x_3/(x_3 + 0.00025) - 0.000175*x[0]/(x_3 + 0.00025) + 0.001*x_3/(x_3 + 0.005) + 2.5e-7/(x_3 + 0.005), 0.035*x[0]*x_3/(x_3 + 0.00025) + 0.000175*x[0]/(x_3 + 0.00025) - 0.189*x[1], 6.84*x[1]/(2.949e+4*x[0] + 1.588) - 0.7*x_3**2/(x_3 + 0.005) - 0.000175*x_3/(x_3 + 0.005)])

@register_eq_class
class BIOMD0000000519(KnownEquation):
    _eq_name = 'vars3_prog11'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Smallbone2013 - Colon Crypt cycle - Version 1"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.002207*x[0]**2 - 0.002207*x[0]*x[1] - 0.002207*x[0]*x_3 + 0.1648*x[0]', '-0.01312*x[0]**2 - 0.0216*x[0]*x[1] - 0.01312*x[0]*x_3 + 1.574*x[0] - 0.008477*x[1]**2 - 0.008477*x[1]*x_3 + 0.5972*x[1]', '-0.04052*x[0]*x[1] - 0.04052*x[1]**2 - 0.04052*x[1]*x_3 + 4.863*x[1] - 1.101*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.002207*x[0]**2 - 0.002207*x[0]*x[1] - 0.002207*x[0]*x_3 + 0.1648*x[0], -0.01312*x[0]**2 - 0.0216*x[0]*x[1] - 0.01312*x[0]*x_3 + 1.574*x[0] - 0.008477*x[1]**2 - 0.008477*x[1]*x_3 + 0.5972*x[1], -0.04052*x[0]*x[1] - 0.04052*x[1]**2 - 0.04052*x[1]*x_3 + 4.863*x[1] - 1.101*x_3])

@register_eq_class
class BIOMD0000000884(KnownEquation):
    _eq_name = 'vars3_prog12'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Cortes2019 - Optimality of the spontaneous prophage induction rate."
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.99*x[0]**2/(x[0] + x[1]) - 1.0*x[0]*x[1]/(x[0] + x[1]) + 0.99*x[0]', '-0.99*x[0]*x[1]/(x[0] + x[1]) - 1.0*x[1]**2/(x[0] + x[1]) + 1.0*x[1]', '-0.001*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.99*x[0]**2/(x[0] + x[1]) - 1.0*x[0]*x[1]/(x[0] + x[1]) + 0.99*x[0], -0.99*x[0]*x[1]/(x[0] + x[1]) - 1.0*x[1]**2/(x[0] + x[1]) + 1.0*x[1], -0.001*x_3])

@register_eq_class
class BIOMD0000000754(KnownEquation):
    _eq_name = 'vars3_prog13'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Figueredo2013/2 - immunointeraction model with IL2"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0]*x[1]/(x[0] + 1.0) + 0.18*x[0]', '0.05*x[0] + 0.124*x[1]*x_3/(x_3 + 20.0) - 0.03*x[1]', '5.0*x[0]*x[1]/(x[0] + 10.0) - 10.0*x_3']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0]*x[1]/(x[0] + 1.0) + 0.18*x[0], 0.05*x[0] + 0.124*x[1]*x_3/(x_3 + 20.0) - 0.03*x[1], 5.0*x[0]*x[1]/(x[0] + 10.0) - 10.0*x_3])

@register_eq_class
class BIOMD0000000763(KnownEquation):
    _eq_name = 'vars3_prog14'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Dritschel2018 - A mathematical model of cytotoxic and helper T cell interactions in a tumour microenvironment"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-10.0*x[0]**2 - 2.075*x[0]*x_3 + 10.0*x[0]', '0.19*x[0]*x[1]/(x[0]**2 + 0.0016) - 1.0*x[1] + 0.5', '-2.075*x[0]*x_3 + 1.0*x[1]*x_3 - 1.0*x_3 + 2.0']
    
    def np_eq(self, t, x):
        return np.array([-10.0*x[0]**2 - 2.075*x[0]*x_3 + 10.0*x[0], 0.19*x[0]*x[1]/(x[0]**2 + 0.0016) - 1.0*x[1] + 0.5, -2.075*x[0]*x_3 + 1.0*x[1]*x_3 - 1.0*x_3 + 2.0])

@register_eq_class
class BIOMD0000000733(KnownEquation):
    _eq_name = 'vars3_prog15'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'log']
    _description = "Moore-2004-Mathematical model for CML and T cell interaction"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.062*x[0]*x_3/(x_3 + 720.0) - 0.23*x[0] + 0.37', '0.00868*x[0]*x_3/(x_3 + 720.0) - 0.057*x[1]*x_3 + 0.98*x[1]*x_3/(x_3 + 720.0) - 0.3*x[1]', '0.003*x[1]*x_3 + 0.006*x_3*np.log(1/x_3) + 0.0500748602820822*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.062*x[0]*x_3/(x_3 + 720.0) - 0.23*x[0] + 0.37, 0.00868*x[0]*x_3/(x_3 + 720.0) - 0.057*x[1]*x_3 + 0.98*x[1]*x_3/(x_3 + 720.0) - 0.3*x[1], 0.003*x[1]*x_3 + 0.006*x_3*np.log(1/x_3) + 0.0500748602820822*x_3])

@register_eq_class
class BIOMD0000000036(KnownEquation):
    _eq_name = 'vars3_prog16'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Tyson1999-CircClock"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '-0.1*x[1] + 1.0/(25.0*x_3**2 + 0.0625*x_3**2/(0.05*(x_3 + 0.000625)**0.5 + (x_3 + 0.000625)**1.0 + 0.000625) - 100.0*x_3**2/(40.0*(x_3 + 0.000625)**0.5 + 1.0) + 1.0)', '0.5*x[1] - 0.1*x_3 - 20.0*x_3/(40.0*x_3*(x_3 + 0.000625)**0.5 + 1.0*x_3 + 2.0*(x_3 + 0.000625)**0.5 + 0.05) - 0.03*x_3/(x_3 + 0.05)']
    
    def np_eq(self, t, x):
        return np.array([0, -0.1*x[1] + 1.0/(25.0*x_3**2 + 0.0625*x_3**2/(0.05*(x_3 + 0.000625)**0.5 + (x_3 + 0.000625)**1.0 + 0.000625) - 100.0*x_3**2/(40.0*(x_3 + 0.000625)**0.5 + 1.0) + 1.0), 0.5*x[1] - 0.1*x_3 - 20.0*x_3/(40.0*x_3*(x_3 + 0.000625)**0.5 + 1.0*x_3 + 2.0*(x_3 + 0.000625)**0.5 + 0.05) - 0.03*x_3/(x_3 + 0.05)])

@register_eq_class
class BIOMD0000000729(KnownEquation):
    _eq_name = 'vars3_prog17'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Goldbeter1996 - Cyclin Cdc2 kinase Oscillations"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.25*x[0]*x_3/(x[0] + 0.02) - 0.01*x[0] + 0.05', '-3.0*x[0]*x[1]/(-x[0]*x[1] + 1.01*x[0] - 0.5*x[1] + 0.505) + 3.0*x[0]/(-x[0]*x[1] + 1.01*x[0] - 0.5*x[1] + 0.505) - 1.5*x[1]/(x[1] + 0.01)', '-1.0*x[1]*x_3/(1.01 - x_3) + 1.0*x[1]/(1.01 - x_3) - 0.5*x_3/(x_3 + 0.01)']
    
    def np_eq(self, t, x):
        return np.array([-0.25*x[0]*x_3/(x[0] + 0.02) - 0.01*x[0] + 0.05, -3.0*x[0]*x[1]/(-x[0]*x[1] + 1.01*x[0] - 0.5*x[1] + 0.505) + 3.0*x[0]/(-x[0]*x[1] + 1.01*x[0] - 0.5*x[1] + 0.505) - 1.5*x[1]/(x[1] + 0.01), -1.0*x[1]*x_3/(1.01 - x_3) + 1.0*x[1]/(1.01 - x_3) - 0.5*x_3/(x_3 + 0.01)])

@register_eq_class
class BIOMD0000000882(KnownEquation):
    _eq_name = 'vars3_prog18'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Munz2009 - Zombie SIZRC"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.009*x[0]*x[1] + 0.05', '0.004*x[0]*x[1]', '0.005*x[0]*x[1]']
    
    def np_eq(self, t, x):
        return np.array([-0.009*x[0]*x[1] + 0.05, 0.004*x[0]*x[1], 0.005*x[0]*x[1]])

@register_eq_class
class BIOMD0000000937(KnownEquation):
    _eq_name = 'vars3_prog19'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Ferrel2011 - Autonomous biochemical oscillator in regulation of CDK1 Plk1 and APC in Xenopus Laevis cell cycle"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-3.0*x[0]*x[1]**8.0/(x[1]**8.0 + 0.003906) + 0.1', '-3.0*x[1]*x_3**8.0/(x_3**8.0 + 0.003906) - 1.0*x[1] + 3.0*x_3**8.0/(x_3**8.0 + 0.003906)', '-3.0*x[0]**8.0*x_3/(x[0]**8.0 + 0.003906) + 3.0*x[0]**8.0/(x[0]**8.0 + 0.003906) - 1.0*x_3']
    
    def np_eq(self, t, x):
        return np.array([-3.0*x[0]*x[1]**8.0/(x[1]**8.0 + 0.003906) + 0.1, -3.0*x[1]*x_3**8.0/(x_3**8.0 + 0.003906) - 1.0*x[1] + 3.0*x_3**8.0/(x_3**8.0 + 0.003906), -3.0*x[0]**8.0*x_3/(x[0]**8.0 + 0.003906) + 3.0*x[0]**8.0/(x[0]**8.0 + 0.003906) - 1.0*x_3])

@register_eq_class
class BIOMD0000000893(KnownEquation):
    _eq_name = 'vars3_prog20'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "GonzalezMiranda2013 - The effect of circadian oscillations on biochemical cell signaling by NF-κB"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-954.5*x[0]*x_3/(x[0] + 0.029) - 0.007*x[0]/x_3 + 0.007/x_3', '1.0*x[0]**2 - 1.0*x[1]', '0.035*x[0] + 1.0*x[1] - 0.035']
    
    def np_eq(self, t, x):
        return np.array([-954.5*x[0]*x_3/(x[0] + 0.029) - 0.007*x[0]/x_3 + 0.007/x_3, 1.0*x[0]**2 - 1.0*x[1], 0.035*x[0] + 1.0*x[1] - 0.035])

@register_eq_class
class BIOMD0000000003(KnownEquation):
    _eq_name = 'vars3_prog21'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Goldbeter1991 - Min Mit Oscil"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.25*x[0]*x_3/(x[0] + 0.02) - 0.01*x[0] + 0.025', '-3.0*x[0]*x[1]/(-x[0]*x[1] + 1.005*x[0] - 0.5*x[1] + 0.5025) + 3.0*x[0]/(-x[0]*x[1] + 1.005*x[0] - 0.5*x[1] + 0.5025) - 1.5*x[1]/(x[1] + 0.005)', '-1.0*x[1]*x_3/(1.005 - x_3) + 1.0*x[1]/(1.005 - x_3) - 0.5*x_3/(x_3 + 0.005)']
    
    def np_eq(self, t, x):
        return np.array([-0.25*x[0]*x_3/(x[0] + 0.02) - 0.01*x[0] + 0.025, -3.0*x[0]*x[1]/(-x[0]*x[1] + 1.005*x[0] - 0.5*x[1] + 0.5025) + 3.0*x[0]/(-x[0]*x[1] + 1.005*x[0] - 0.5*x[1] + 0.5025) - 1.5*x[1]/(x[1] + 0.005), -1.0*x[1]*x_3/(1.005 - x_3) + 1.0*x[1]/(1.005 - x_3) - 0.5*x_3/(x_3 + 0.005)])

@register_eq_class
class BIOMD0000000891(KnownEquation):
    _eq_name = 'vars3_prog22'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Khajanchi2019 - Stability Analysis of a Mathematical Model forGlioma-Immune Interaction under OptimalTherapy"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.482*x[0]**2 - 0.07*x[0]*x[1]/(x[0] + 0.903) - 2.745*x[0]*x_3/(x[0] + 0.903) + 0.482*x[0]', '-0.019*x[0]*x[1]/(x[0] + 0.031) - 0.331*x[1]**2 + 0.331*x[1]', '0.124*x[0]*x_3/(x[0] + 2.874) - 0.017*x[0]*x_3/(x[0] + 0.379) - 0.007*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.482*x[0]**2 - 0.07*x[0]*x[1]/(x[0] + 0.903) - 2.745*x[0]*x_3/(x[0] + 0.903) + 0.482*x[0], -0.019*x[0]*x[1]/(x[0] + 0.031) - 0.331*x[1]**2 + 0.331*x[1], 0.124*x[0]*x_3/(x[0] + 2.874) - 0.017*x[0]*x_3/(x[0] + 0.379) - 0.007*x_3])

@register_eq_class
class BIOMD0000000713(KnownEquation):
    _eq_name = 'vars3_prog23'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Aston2018 - Dynamics of Hepatitis C Infection"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.002*x[0] + 1.065e+4*x[0]/(x[0] + x[1]) + 0.118*x[1]', '-0.118*x[1] + 342.5*x[1]/(x[0] + x[1])', '204.0*x[1] - 17.91*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.002*x[0] + 1.065e+4*x[0]/(x[0] + x[1]) + 0.118*x[1], -0.118*x[1] + 342.5*x[1]/(x[0] + x[1]), 204.0*x[1] - 17.91*x_3])

@register_eq_class
class BIOMD0000001023(KnownEquation):
    _eq_name = 'vars3_prog24'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Alharbi2020 - An ODE-based model of the dynamics of tumor cell progression and its effects on normal cell growth and immune system functionality"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.931*x[0]*x[1] - 0.138*x[0]*x_3 + 0.431*x[0]', '1.189*x[0]*x[1] - 0.1772*x[1]**2 - 0.147*x[1]*x_3 + 0.443*x[1]', '-0.813*x[0]*x_3 + 0.271*x[0]*x_3/(x[0] + 0.813) - 0.363*x[1]*x_3 + 0.783*x[1]*x_3/(x[1] + 0.862) - 0.57*x_3 + 0.7']
    
    def np_eq(self, t, x):
        return np.array([-0.931*x[0]*x[1] - 0.138*x[0]*x_3 + 0.431*x[0], 1.189*x[0]*x[1] - 0.1772*x[1]**2 - 0.147*x[1]*x_3 + 0.443*x[1], -0.813*x[0]*x_3 + 0.271*x[0]*x_3/(x[0] + 0.813) - 0.363*x[1]*x_3 + 0.783*x[1]*x_3/(x[1] + 0.862) - 0.57*x_3 + 0.7])

@register_eq_class
class BIOMD0000000321(KnownEquation):
    _eq_name = 'vars3_prog25'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Grange2001 - L Dopa PK model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-2.11*x[0]', '0.889*x[0] - 1.659*x[1]', '0.4199*x[1] - 0.06122*x_3']
    
    def np_eq(self, t, x):
        return np.array([-2.11*x[0], 0.889*x[0] - 1.659*x[1], 0.4199*x[1] - 0.06122*x_3])

@register_eq_class
class BIOMD0000000191(KnownEquation):
    _eq_name = 'vars3_prog26'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Montañez2008-Arginine-catabolism"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '-28.09*x[1]/(x[1] + 5.143*x_3 + 360.0) - 302.2*x[1]/(x[1] + 1.0*x_3 + 847.0) - 0.013*x[1]/(x[1] + 90.0) + 110.0*x_3/(1.5*x[1] + x_3 + 1500.0)', '-110.0*x_3/(1.5*x[1] + x_3 + 1500.0) - 1.33*x_3/(x_3 + 16.0) + 132.4/(0.002778*x[1] + 0.01429*x_3 + 1.0) + 117.8/(0.002778*x[1] + 0.001181*x_3 + 1.0)']
    
    def np_eq(self, t, x):
        return np.array([0, -28.09*x[1]/(x[1] + 5.143*x_3 + 360.0) - 302.2*x[1]/(x[1] + 1.0*x_3 + 847.0) - 0.013*x[1]/(x[1] + 90.0) + 110.0*x_3/(1.5*x[1] + x_3 + 1500.0), -110.0*x_3/(1.5*x[1] + x_3 + 1500.0) - 1.33*x_3/(x_3 + 16.0) + 132.4/(0.002778*x[1] + 0.01429*x_3 + 1.0) + 117.8/(0.002778*x[1] + 0.001181*x_3 + 1.0)])

@register_eq_class
class BIOMD0000000157(KnownEquation):
    _eq_name = 'vars3_prog27'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Zatorsky2006-p53-Model4"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.9 - 1.7*x[1]', '-0.8*x[1] + 0.8*x_3', '1.1*x[0] - 0.8*x_3']
    
    def np_eq(self, t, x):
        return np.array([0.9 - 1.7*x[1], -0.8*x[1] + 0.8*x_3, 1.1*x[0] - 0.8*x_3])

@register_eq_class
class BIOMD0000000911(KnownEquation):
    _eq_name = 'vars3_prog28'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Merola2008 - An insight into tumor dormancy equilibrium via the analysis of its domain of attraction"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.125*x[0]**2 - 0.3*x[0]*x[1] + 0.9*x[0] + 10.0', '0.1*x[1]*x_3 - 0.02*x[1]', '-0.1*x[1]*x_3 - 1.143*x_3**2 + 0.77*x_3']
    
    def np_eq(self, t, x):
        return np.array([-1.125*x[0]**2 - 0.3*x[0]*x[1] + 0.9*x[0] + 10.0, 0.1*x[1]*x_3 - 0.02*x[1], -0.1*x[1]*x_3 - 1.143*x_3**2 + 0.77*x_3])

@register_eq_class
class BIOMD0000000894(KnownEquation):
    _eq_name = 'vars3_prog29'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Bose2011 - Noise-assisted interactions of tumor and immune cells"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0]**2 - 1.0*x[0]*x[1] + 1.748*x[0] + 2.73*x_3', '1.0*x[1]*x_3 - 0.05*x[1]', '1.126*x[0] - 15.89*x_3']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0]**2 - 1.0*x[0]*x[1] + 1.748*x[0] + 2.73*x_3, 1.0*x[1]*x_3 - 0.05*x[1], 1.126*x[0] - 15.89*x_3])

@register_eq_class
class BIOMD0000000777(KnownEquation):
    _eq_name = 'vars3_prog30'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Chakrabarty2010 - A control theory approach to cancer remission aided by an optimal therapy"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-3.6e-8*x[0]**2 + 0.13*x[0]', '-0.041*x[1]', '-2.5e-9*x_3**2 + 0.025*x_3']
    
    def np_eq(self, t, x):
        return np.array([-3.6e-8*x[0]**2 + 0.13*x[0], -0.041*x[1], -2.5e-9*x_3**2 + 0.025*x_3])

@register_eq_class
class BIOMD0000000933(KnownEquation):
    _eq_name = 'vars3_prog31'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Kosiuk2015-Geometric analysis of the Goldbeter minimal model for the embryonic cell cycle"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.25*x[0] - 0.25*x_3 + 0.25', '-6.0*x[0]*x[1]/(-2.0*x[0]*x[1] + 2.002*x[0] - 1.0*x[1] + 1.001) + 6.0*x[0]/(-2.0*x[0]*x[1] + 2.002*x[0] - 1.0*x[1] + 1.001) - 1.5*x[1]/(x[1] + 0.001)', '-1.0*x[1]*x_3/(1.001 - x_3) + 1.0*x[1]/(1.001 - x_3) - 0.7*x_3/(x_3 + 0.001)']
    
    def np_eq(self, t, x):
        return np.array([-0.25*x[0] - 0.25*x_3 + 0.25, -6.0*x[0]*x[1]/(-2.0*x[0]*x[1] + 2.002*x[0] - 1.0*x[1] + 1.001) + 6.0*x[0]/(-2.0*x[0]*x[1] + 2.002*x[0] - 1.0*x[1] + 1.001) - 1.5*x[1]/(x[1] + 0.001), -1.0*x[1]*x_3/(1.001 - x_3) + 1.0*x[1]/(1.001 - x_3) - 0.7*x_3/(x_3 + 0.001)])

@register_eq_class
class BIOMD0000000783(KnownEquation):
    _eq_name = 'vars3_prog32'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Dong2014 - Mathematical modeling on helper t cells in a tumor immune system"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.003272*x[0]**2 - 1.0*x[0]*x[1] + 1.636*x[0]', '0.04*x[0]*x[1] + 0.01*x[1]*x_3 - 0.374*x[1]', '0.002*x[0]*x_3 - 0.055*x_3 + 0.38']
    
    def np_eq(self, t, x):
        return np.array([-0.003272*x[0]**2 - 1.0*x[0]*x[1] + 1.636*x[0], 0.04*x[0]*x[1] + 0.01*x[1]*x_3 - 0.374*x[1], 0.002*x[0]*x_3 - 0.055*x_3 + 0.38])

@register_eq_class
class BIOMD0000000890(KnownEquation):
    _eq_name = 'vars3_prog33'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Bhattacharya2014 - A mathematical model of the sterol regulatory element binding protein 2 cholesterol biosynthesis pathway"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.001*x[0]', '1.0*x[0] - 0.002*x[1]', '0.462*x[1] - 0.004*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.001*x[0], 1.0*x[0] - 0.002*x[1], 0.462*x[1] - 0.004*x_3])

@register_eq_class
class BIOMD0000000663(KnownEquation):
    _eq_name = 'vars3_prog34'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Wodarz2007 - HIV/CD4 T-cell interaction"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.1*x[0]**2*x_3 - 0.1*x[0]*x[1]*x_3 + 0.8*x[0]*x_3 - 0.1*x[0]', '-0.1*x[0]*x[1]*x_3 + 0.2*x[0]*x_3 - 0.1*x[1]**2*x_3 + 1.0*x[1]*x_3 - 0.2*x[1]', '1.0*x[1] - 0.5*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.1*x[0]**2*x_3 - 0.1*x[0]*x[1]*x_3 + 0.8*x[0]*x_3 - 0.1*x[0], -0.1*x[0]*x[1]*x_3 + 0.2*x[0]*x_3 - 0.1*x[1]**2*x_3 + 1.0*x[1]*x_3 - 0.2*x[1], 1.0*x[1] - 0.5*x_3])

@register_eq_class
class BIOMD0000000944(KnownEquation):
    _eq_name = 'vars3_prog35'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Goldbeter2013-Oscillatory activity of cyclin-dependent kinases in the cell cycle"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.25*x[0]*x_3/(x[0] + 0.001) - 0.046*x[0] + 0.06', '-4.0*x[0]*x[1]/(-x[0]*x[1] + 1.002*x[0] - 0.5*x[1] + 0.501) + 4.0*x[0]/(-x[0]*x[1] + 1.002*x[0] - 0.5*x[1] + 0.501) - 2.0*x[1]/(x[1] + 0.002)', '-1.0*x[1]*x_3/(1.01 - x_3) + 1.0*x[1]/(1.01 - x_3) - 0.7*x_3/(x_3 + 0.01)']
    
    def np_eq(self, t, x):
        return np.array([-0.25*x[0]*x_3/(x[0] + 0.001) - 0.046*x[0] + 0.06, -4.0*x[0]*x[1]/(-x[0]*x[1] + 1.002*x[0] - 0.5*x[1] + 0.501) + 4.0*x[0]/(-x[0]*x[1] + 1.002*x[0] - 0.5*x[1] + 0.501) - 2.0*x[1]/(x[1] + 0.002), -1.0*x[1]*x_3/(1.01 - x_3) + 1.0*x[1]/(1.01 - x_3) - 0.7*x_3/(x_3 + 0.01)])

@register_eq_class
class BIOMD0000000054(KnownEquation):
    _eq_name = 'vars3_prog36'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Ataullahkhanov1996-Adenylate"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.3*x[0]*x[1] - 0.1*x[0]*x_3 + 0.2449*x[0]*(-0.5*x[1]**2 + x[1]*x_3 + 0.1667*x_3**2)**0.5 + 12.1', '-0.1*x[0]*x[1] - 0.03333*x[0]*x_3 + 0.08165*x[0]*(-0.5*x[1]**2 + x[1]*x_3 + 0.1667*x_3**2)**0.5 + 10.01*(-0.4286*x[1] + x_3 - 0.3499*(-0.5*x[1]**2 + x[1]*x_3 + 0.1667*x_3**2)**0.5)**0.41*(x[1] + 0.3333*x_3 - 0.8165*(-0.5*x[1]**2 + x[1]*x_3 + 0.1667*x_3**2)**0.5)**0.52 - 0.04', '-8.706e-5*(x[1] + 0.3333*x_3 - 0.8165*(-0.5*x[1]**2 + x[1]*x_3 + 0.1667*x_3**2)**0.5)**1.2/(-0.5*x[1] + 1.167*x_3 - 0.4082*(-0.5*x[1]**2 + x[1]*x_3 + 0.1667*x_3**2)**0.5)**1.0 + 0.02']
    
    def np_eq(self, t, x):
        return np.array([-0.3*x[0]*x[1] - 0.1*x[0]*x_3 + 0.2449*x[0]*(-0.5*x[1]**2 + x[1]*x_3 + 0.1667*x_3**2)**0.5 + 12.1, -0.1*x[0]*x[1] - 0.03333*x[0]*x_3 + 0.08165*x[0]*(-0.5*x[1]**2 + x[1]*x_3 + 0.1667*x_3**2)**0.5 + 10.01*(-0.4286*x[1] + x_3 - 0.3499*(-0.5*x[1]**2 + x[1]*x_3 + 0.1667*x_3**2)**0.5)**0.41*(x[1] + 0.3333*x_3 - 0.8165*(-0.5*x[1]**2 + x[1]*x_3 + 0.1667*x_3**2)**0.5)**0.52 - 0.04, -8.706e-5*(x[1] + 0.3333*x_3 - 0.8165*(-0.5*x[1]**2 + x[1]*x_3 + 0.1667*x_3**2)**0.5)**1.2/(-0.5*x[1] + 1.167*x_3 - 0.4082*(-0.5*x[1]**2 + x[1]*x_3 + 0.1667*x_3**2)**0.5)**1.0 + 0.02])

@register_eq_class
class BIOMD0000000850(KnownEquation):
    _eq_name = 'vars3_prog37'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'log']
    _description = "Jenner2019 - Oncolytic virotherapy for tumours following a Gompertz growth law"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0]*x_3/(x[0] + x[1]) + 0.1*x[0]*np.log(1/x[0]) + 0.460518775331821*x[0]', '1.0*x[0]*x_3/(x[0] + x[1]) - 0.01*x[1]', '0.01*x[1] - 0.1*x_3']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0]*x_3/(x[0] + x[1]) + 0.1*x[0]*np.log(1/x[0]) + 0.460518775331821*x[0], 1.0*x[0]*x_3/(x[0] + x[1]) - 0.01*x[1], 0.01*x[1] - 0.1*x_3])

@register_eq_class
class BIOMD0000000166(KnownEquation):
    _eq_name = 'vars3_prog38'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Zhu2007-TF-modulated-by-Calcium"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['54.0*x[0]**2*x_3**4/(x[0]**2*x_3**4 + 0.0625*x[0]**2 + 10.0*x_3**4/(16.0*x_3**4 + 1.0) + 0.625/(16.0*x_3**4 + 1.0)) + 6.0*x[0]**2/(x[0]**2 + 10.0/(16.0*x_3**4 + 1.0)) - 1.0*x[0] + 0.1', '-0.7*x[1] - 325.0*x[1]**2.0*x_3**4.0/(x[1]**2.0*x_3**4.0 + 0.04477*x[1]**2.0 + 2.89*x_3**4.0 + 0.1294) + 30.0*x_3**2.0/(x_3**2.0 + 0.25)', '0.7*x[1] + 325.0*x[1]**2.0*x_3**4.0/(x[1]**2.0*x_3**4.0 + 0.04477*x[1]**2.0 + 2.89*x_3**4.0 + 0.1294) - 10.0*x_3 - 30.0*x_3**2.0/(x_3**2.0 + 0.25) + 2.71']
    
    def np_eq(self, t, x):
        return np.array([54.0*x[0]**2*x_3**4/(x[0]**2*x_3**4 + 0.0625*x[0]**2 + 10.0*x_3**4/(16.0*x_3**4 + 1.0) + 0.625/(16.0*x_3**4 + 1.0)) + 6.0*x[0]**2/(x[0]**2 + 10.0/(16.0*x_3**4 + 1.0)) - 1.0*x[0] + 0.1, -0.7*x[1] - 325.0*x[1]**2.0*x_3**4.0/(x[1]**2.0*x_3**4.0 + 0.04477*x[1]**2.0 + 2.89*x_3**4.0 + 0.1294) + 30.0*x_3**2.0/(x_3**2.0 + 0.25), 0.7*x[1] + 325.0*x[1]**2.0*x_3**4.0/(x[1]**2.0*x_3**4.0 + 0.04477*x[1]**2.0 + 2.89*x_3**4.0 + 0.1294) - 10.0*x_3 - 30.0*x_3**2.0/(x_3**2.0 + 0.25) + 2.71])

@register_eq_class
class BIOMD0000000274(KnownEquation):
    _eq_name = 'vars3_prog39'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Rattanakul2003-BoneFormationModel"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.1*x[0] + 0.05/(x[1] + 0.1)', '0.0675*x[0]*x[1]*x_3/(x[0]**2 + 0.5) + 0.0009*x[1]*x_3/(x[0]**2 + 0.5) - 0.03*x[1]', '-0.00045*x[0]*x_3/(x[0] + 0.025) + 0.0009*x[0] - 0.0009*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.1*x[0] + 0.05/(x[1] + 0.1), 0.0675*x[0]*x[1]*x_3/(x[0]**2 + 0.5) + 0.0009*x[1]*x_3/(x[0]**2 + 0.5) - 0.03*x[1], -0.00045*x[0]*x_3/(x[0] + 0.025) + 0.0009*x[0] - 0.0009*x_3])

@register_eq_class
class BIOMD0000000821(KnownEquation):
    _eq_name = 'vars3_prog40'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'log', 'pow']
    _description = "Yazdjer2019 - reinforcement learning-based control of tumor growth under anti-angiogenic therapy"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.192*x[0]*np.log(x[0]/x[1])', '-0.009*x[0]**0.6667*x[1] + 5.85*x[0] - 0.66*x[1]*x_3', '-1.3*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.192*x[0]*np.log(x[0]/x[1]), -0.009*x[0]**0.6667*x[1] + 5.85*x[0] - 0.66*x[1]*x_3, -1.3*x_3])

@register_eq_class
class BIOMD0000001038(KnownEquation):
    _eq_name = 'vars3_prog41'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Alharbi2019 - Tumor-normal-vitamins model (TNVM) of the effects of vitamins on delaying the growth of tumor cells"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.982*x[0]*x[1] + 0.222*x[0]*x_3 + 0.431*x[0]', '0.229*x[0]*x[1] - 0.1772*x[1]**2 - 0.497*x[1]*x_3 + 0.443*x[1]', '0.898 - 0.961*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.982*x[0]*x[1] + 0.222*x[0]*x_3 + 0.431*x[0], 0.229*x[0]*x[1] - 0.1772*x[1]**2 - 0.497*x[1]*x_3 + 0.443*x[1], 0.898 - 0.961*x_3])

@register_eq_class
class BIOMD0000000520(KnownEquation):
    _eq_name = 'vars3_prog42'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Smallbone2013 - Colon Crypt cycle - Version 0"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '1.0*x[0]**2/(x[0] + 2.924) + 0.218*x[0] - 0.024*x[1]', '1.0*x[1]**2/(x[1] + 29.24) + 0.547*x[1] - 1.83*x_3']
    
    def np_eq(self, t, x):
        return np.array([0, 1.0*x[0]**2/(x[0] + 2.924) + 0.218*x[0] - 0.024*x[1], 1.0*x[1]**2/(x[1] + 29.24) + 0.547*x[1] - 1.83*x_3])

@register_eq_class
class BIOMD0000000299(KnownEquation):
    _eq_name = 'vars3_prog43'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Leloup1999-CircadianRhythms-Neurospora"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.505*x[0]/(x[0] + 0.5) + 1.6/(x_3**4.0 + 1.0)', '0.5*x[0] - 0.5*x[1] - 1.4*x[1]/(x[1] + 0.13) + 0.6*x_3', '0.5*x[1] - 0.6*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.505*x[0]/(x[0] + 0.5) + 1.6/(x_3**4.0 + 1.0), 0.5*x[0] - 0.5*x[1] - 1.4*x[1]/(x[1] + 0.13) + 0.6*x_3, 0.5*x[1] - 0.6*x_3])

@register_eq_class
class BIOMD0000000906(KnownEquation):
    _eq_name = 'vars3_prog44'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Dubey2007 - A mathematical model for the effect of toxicant on the immune system (without toxicant effect) Model1"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.2*x[0]**2 - 0.05*x[0]*x[1] + 0.9*x[0]', '0.295*x[0]*x[1] - 0.8*x[1] + 0.04', '2.4*x[0] - 0.1*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.2*x[0]**2 - 0.05*x[0]*x[1] + 0.9*x[0], 0.295*x[0]*x[1] - 0.8*x[1] + 0.04, 2.4*x[0] - 0.1*x_3])

@register_eq_class
class BIOMD0000000813(KnownEquation):
    _eq_name = 'vars3_prog45'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Anderson2015 - Qualitative behavior of systems of tumor-CD4+-cytokine interactions with treatments"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-3.0e-5*x[0]**2 - 0.1*x[0]*x_3/(x[0] + 1.0) + 0.03*x[0]', '0.02*x[0]*x[1]/(x[0] + 10.0) - 0.02*x[1] + 10.0', '0.1*x[0]*x[1]/(x[0] + 0.1) - 47.0*x_3']
    
    def np_eq(self, t, x):
        return np.array([-3.0e-5*x[0]**2 - 0.1*x[0]*x_3/(x[0] + 1.0) + 0.03*x[0], 0.02*x[0]*x[1]/(x[0] + 10.0) - 0.02*x[1] + 10.0, 0.1*x[0]*x[1]/(x[0] + 0.1) - 47.0*x_3])

@register_eq_class
class BIOMD0000000548(KnownEquation):
    _eq_name = 'vars3_prog46'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Sneppen2009 - Modeling proteasome dynamics in Parkinson's disease"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0]*x[1] + 25.0/(x[1] + 1.0)', '-1.0*x[0]*x[1] - x[1] + 1.0*x_3 + 1.0', '1.0*x[0]*x[1] - 1.0*x_3']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0]*x[1] + 25.0/(x[1] + 1.0), -1.0*x[0]*x[1] - x[1] + 1.0*x_3 + 1.0, 1.0*x[0]*x[1] - 1.0*x_3])

@register_eq_class
class BIOMD0000000323(KnownEquation):
    _eq_name = 'vars3_prog47'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Kim2011-Oscillator-SimpleIII"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-3.333*x[0]/(3.333*x[0] + 1.0) + 1.0/(x[1]**5.0 + 1.0)', '-3.333*x[1]/(3.333*x[1] + 1.0) + 1.0/(x_3**5.0 + 1.0)', '-3.333*x_3/(3.333*x_3 + 1.0) + 1.0/(x[0]**5.0 + 1.0)']
    
    def np_eq(self, t, x):
        return np.array([-3.333*x[0]/(3.333*x[0] + 1.0) + 1.0/(x[1]**5.0 + 1.0), -3.333*x[1]/(3.333*x[1] + 1.0) + 1.0/(x_3**5.0 + 1.0), -3.333*x_3/(3.333*x_3 + 1.0) + 1.0/(x[0]**5.0 + 1.0)])

@register_eq_class
class BIOMD0000000079(KnownEquation):
    _eq_name = 'vars3_prog48'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Goldbeter2006-weightCycling"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.1*x[0]/(x[0] + 0.2) + 0.1*x[1]', '-1.5*x[1]*x_3/(x[1] + 0.01) - 1.0*x[1]/(1.01 - x[1]) + 1.0/(1.01 - x[1])', '-6.0*x[0]*x_3/(1.01 - x_3) + 6.0*x[0]/(1.01 - x_3) - 2.5*x_3/(x_3 + 0.01)']
    
    def np_eq(self, t, x):
        return np.array([-0.1*x[0]/(x[0] + 0.2) + 0.1*x[1], -1.5*x[1]*x_3/(x[1] + 0.01) - 1.0*x[1]/(1.01 - x[1]) + 1.0/(1.01 - x[1]), -6.0*x[0]*x_3/(1.01 - x_3) + 6.0*x[0]/(1.01 - x_3) - 2.5*x_3/(x_3 + 0.01)])

@register_eq_class
class BIOMD0000000773(KnownEquation):
    _eq_name = 'vars3_prog49'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Wodarz2018/2 - model with transit amplifying cells"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.004*x[0] + 0.0096*x[1]/(0.01*x[0]**1.0 + 1.0)', '0.006*x[0] - 0.004*x[1]', '0.024*x[1] - 0.0096*x[1]/(0.01*x[0]**1.0 + 1.0) - 0.003*x_3']
    
    def np_eq(self, t, x):
        return np.array([0.004*x[0] + 0.0096*x[1]/(0.01*x[0]**1.0 + 1.0), 0.006*x[0] - 0.004*x[1], 0.024*x[1] - 0.0096*x[1]/(0.01*x[0]**1.0 + 1.0) - 0.003*x_3])

@register_eq_class
class BIOMD0000001036(KnownEquation):
    _eq_name = 'vars3_prog50'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Cappuccio2007 - Tumor-immune system interactions and determination of the optimal therapeutic protocol in immunotherapy"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.044*x[0]*x_3/(x_3 + 0.02) - 0.038*x[0] + 1.009*x[1]', '-0.018*x[0] - 0.123*x[1]**2 + 0.123*x[1]', '0.9*x[0] - 1.8*x_3']
    
    def np_eq(self, t, x):
        return np.array([0.044*x[0]*x_3/(x_3 + 0.02) - 0.038*x[0] + 1.009*x[1], -0.018*x[0] - 0.123*x[1]**2 + 0.123*x[1], 0.9*x[0] - 1.8*x_3])

@register_eq_class
class BIOMD0000000670(KnownEquation):
    _eq_name = 'vars3_prog51'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Owen1998 - tumour growth model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-17.86*x[0]*x_3**2 + 0.05*x[0]*x_3/(x[0] + x[1] + x_3 + 1.0) - 0.1*x[0] + 0.625*x_3 + 0.01', '-x[1] + 2.0*x[1]/(x[0] + x[1] + x_3 + 1.0)', '-25.0*x[0]*x_3**2 - x_3 + 4.0*x_3/(x[0] + x[1] + x_3 + 1.0)']
    
    def np_eq(self, t, x):
        return np.array([-17.86*x[0]*x_3**2 + 0.05*x[0]*x_3/(x[0] + x[1] + x_3 + 1.0) - 0.1*x[0] + 0.625*x_3 + 0.01, -x[1] + 2.0*x[1]/(x[0] + x[1] + x_3 + 1.0), -25.0*x[0]*x_3**2 - x_3 + 4.0*x_3/(x[0] + x[1] + x_3 + 1.0)])

@register_eq_class
class BIOMD0000000752(KnownEquation):
    _eq_name = 'vars3_prog52'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Wilkie2013r - immune-induced cancer dormancy and immune evasion-resistance"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-2.0e-11*x[0]**2 - 2.0e-11*x[0]*x_3 + 0.2*x[0]', '-0.2*x[1]**2/(0.001*x[0]*x[1] + 1.0e-6*x[1]*x_3 + 100.0) + 0.2*x[1]', '-2.0e-11*x[0]*x_3 - 2.0e-11*x_3**2 + 0.2*x_3']
    
    def np_eq(self, t, x):
        return np.array([-2.0e-11*x[0]**2 - 2.0e-11*x[0]*x_3 + 0.2*x[0], -0.2*x[1]**2/(0.001*x[0]*x[1] + 1.0e-6*x[1]*x_3 + 100.0) + 0.2*x[1], -2.0e-11*x[0]*x_3 - 2.0e-11*x_3**2 + 0.2*x_3])

@register_eq_class
class BIOMD0000000781(KnownEquation):
    _eq_name = 'vars3_prog53'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Wang2016/2 - oncolytic efficacy of M1 virus-SNT model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.2*x[0]*x[1] - 0.5*x[0]*x_3 - 0.02*x[0] + 0.02', '0.16*x[0]*x[1] - 0.03*x[1]', '0.4*x[0]*x_3 - 0.028*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.2*x[0]*x[1] - 0.5*x[0]*x_3 - 0.02*x[0] + 0.02, 0.16*x[0]*x[1] - 0.03*x[1], 0.4*x[0]*x_3 - 0.028*x_3])

@register_eq_class
class BIOMD0000000184(KnownEquation):
    _eq_name = 'vars3_prog54'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Lavrentovich2008-Ca-Oscillations"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-15.0*x[0]**2/(x[0]**2 + 0.01) - 1.0*x[0] + 3.466*x[0]**2.02*x[1]*x_3**2.2/(0.04332*x[0]**2.02*x_3**2.2 + 0.0002734*x[0]**2.02 + x[0]**4.04*x_3**2.2 + 0.00631*x[0]**4.04 + 0.0004693*x_3**2.2 + 2.961e-6) - 3.466*x[0]**3.02*x_3**2.2/(0.04332*x[0]**2.02*x_3**2.2 + 0.0002734*x[0]**2.02 + x[0]**4.04*x_3**2.2 + 0.00631*x[0]**4.04 + 0.0004693*x_3**2.2 + 2.961e-6) + 0.5*x[1] + 0.05', '15.0*x[0]**2/(x[0]**2 + 0.01) + 0.5*x[0] - 3.466*x[0]**2.02*x[1]*x_3**2.2/(0.04332*x[0]**2.02*x_3**2.2 + 0.0002734*x[0]**2.02 + x[0]**4.04*x_3**2.2 + 0.00631*x[0]**4.04 + 0.0004693*x_3**2.2 + 2.961e-6) + 3.466*x[0]**3.02*x_3**2.2/(0.04332*x[0]**2.02*x_3**2.2 + 0.0002734*x[0]**2.02 + x[0]**4.04*x_3**2.2 + 0.00631*x[0]**4.04 + 0.0004693*x_3**2.2 + 2.961e-6) - 0.5*x[1]', '0.05*x[0]**2/(x[0]**2 + 0.09) - 0.08*x_3']
    
    def np_eq(self, t, x):
        return np.array([-15.0*x[0]**2/(x[0]**2 + 0.01) - 1.0*x[0] + 3.466*x[0]**2.02*x[1]*x_3**2.2/(0.04332*x[0]**2.02*x_3**2.2 + 0.0002734*x[0]**2.02 + x[0]**4.04*x_3**2.2 + 0.00631*x[0]**4.04 + 0.0004693*x_3**2.2 + 2.961e-6) - 3.466*x[0]**3.02*x_3**2.2/(0.04332*x[0]**2.02*x_3**2.2 + 0.0002734*x[0]**2.02 + x[0]**4.04*x_3**2.2 + 0.00631*x[0]**4.04 + 0.0004693*x_3**2.2 + 2.961e-6) + 0.5*x[1] + 0.05, 15.0*x[0]**2/(x[0]**2 + 0.01) + 0.5*x[0] - 3.466*x[0]**2.02*x[1]*x_3**2.2/(0.04332*x[0]**2.02*x_3**2.2 + 0.0002734*x[0]**2.02 + x[0]**4.04*x_3**2.2 + 0.00631*x[0]**4.04 + 0.0004693*x_3**2.2 + 2.961e-6) + 3.466*x[0]**3.02*x_3**2.2/(0.04332*x[0]**2.02*x_3**2.2 + 0.0002734*x[0]**2.02 + x[0]**4.04*x_3**2.2 + 0.00631*x[0]**4.04 + 0.0004693*x_3**2.2 + 2.961e-6) - 0.5*x[1], 0.05*x[0]**2/(x[0]**2 + 0.09) - 0.08*x_3])

@register_eq_class
class BIOMD0000001011(KnownEquation):
    _eq_name = 'vars3_prog55'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Leon-Triana2020 - CAR T-cell therapy in B-cell acute lymphoblastic leukaemia"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.07143*x[0]', '0.033*x[1]', '-0.01667*x_3']
    
    def np_eq(self, t, x):
        return np.array([-0.07143*x[0], 0.033*x[1], -0.01667*x_3])

@register_eq_class
class BIOMD0000000258(KnownEquation):
    _eq_name = 'vars3_prog56'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Ortega2006 - bistability from double phosphorylation in signal transduction"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=3, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-100.0*x[0]/(100.0*x[0] + 100.0*x[1] + 1.0) + 90.91*x[1]/(100.0*x[1] + 100.0*x_3 + 1.0)', '100.0*x[0]/(100.0*x[0] + 100.0*x[1] + 1.0) - 90.91*x[1]/(100.0*x[1] + 100.0*x_3 + 1.0) - 100.0*x[1]/(100.0*x[0] + 100.0*x[1] + 1.0) + 90.91*x_3/(100.0*x[1] + 100.0*x_3 + 1.0)', '100.0*x[1]/(100.0*x[0] + 100.0*x[1] + 1.0) - 90.91*x_3/(100.0*x[1] + 100.0*x_3 + 1.0)']
    
    def np_eq(self, t, x):
        return np.array([-100.0*x[0]/(100.0*x[0] + 100.0*x[1] + 1.0) + 90.91*x[1]/(100.0*x[1] + 100.0*x_3 + 1.0), 100.0*x[0]/(100.0*x[0] + 100.0*x[1] + 1.0) - 90.91*x[1]/(100.0*x[1] + 100.0*x_3 + 1.0) - 100.0*x[1]/(100.0*x[0] + 100.0*x[1] + 1.0) + 90.91*x_3/(100.0*x[1] + 100.0*x_3 + 1.0), 100.0*x[1]/(100.0*x[0] + 100.0*x[1] + 1.0) - 90.91*x_3/(100.0*x[1] + 100.0*x_3 + 1.0)])

@register_eq_class
class BIOMD0000000174(KnownEquation):
    _eq_name = 'vars4_prog1'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'exp', 'pow']
    _description = "Del-Conte-Zerial2008-Rab5-Rab7-cut-out-switch"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.3*t*x[0]/(1.0*t + 1.284*t*np.exp(-2.5*x[1]) + 100.0 + 128.4*np.exp(-2.5*x[1])) - 1.0*x[0] + 0.06*x[1] + 0.31*x[1]/(1.0 + 2.46*np.exp(-3.0*x_4)) + 1.0', '0.3*t*x[0]/(1.0*t + 1.284*t*np.exp(-2.5*x[1]) + 100.0 + 128.4*np.exp(-2.5*x[1])) - 0.06*x[1] - 0.31*x[1]/(1.0 + 2.46*np.exp(-3.0*x_4))', '-0.21*x[2]*x_4**3.0/(x_4**3.0 + 0.1) - 0.483*x[2] - 0.021*x[2]/(1.0 + 20.09*np.exp(-3.0*x[1])) + 0.15*x_4 + 0.483', '0.21*x[2]*x_4**3.0/(x_4**3.0 + 0.1) + 0.021*x[2]/(1.0 + 20.09*np.exp(-3.0*x[1])) - 0.15*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.3*t*x[0]/(1.0*t + 1.284*t*np.exp(-2.5*x[1]) + 100.0 + 128.4*np.exp(-2.5*x[1])) - 1.0*x[0] + 0.06*x[1] + 0.31*x[1]/(1.0 + 2.46*np.exp(-3.0*x_4)) + 1.0, 0.3*t*x[0]/(1.0*t + 1.284*t*np.exp(-2.5*x[1]) + 100.0 + 128.4*np.exp(-2.5*x[1])) - 0.06*x[1] - 0.31*x[1]/(1.0 + 2.46*np.exp(-3.0*x_4)), -0.21*x[2]*x_4**3.0/(x_4**3.0 + 0.1) - 0.483*x[2] - 0.021*x[2]/(1.0 + 20.09*np.exp(-3.0*x[1])) + 0.15*x_4 + 0.483, 0.21*x[2]*x_4**3.0/(x_4**3.0 + 0.1) + 0.021*x[2]/(1.0 + 20.09*np.exp(-3.0*x[1])) - 0.15*x_4])

@register_eq_class
class BIOMD0000001022(KnownEquation):
    _eq_name = 'vars4_prog2'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Creemers2021 - Tumor-immune dynamics and implications on immunotherapy responses"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['6.0*x[0]**0.8 - 0.035*x[0]*x[1]/(0.001751*x[0] + 0.001751*x[1] + 1.0)', '-0.019*x[1] + 1.0*x[2]', '0.003*x[0]*x_4/(x[0] + 1.0e+7)', '-0.003*x[0]*x_4/(x[0] + 1.0e+7)']
    
    def np_eq(self, t, x):
        return np.array([6.0*x[0]**0.8 - 0.035*x[0]*x[1]/(0.001751*x[0] + 0.001751*x[1] + 1.0), -0.019*x[1] + 1.0*x[2], 0.003*x[0]*x_4/(x[0] + 1.0e+7), -0.003*x[0]*x_4/(x[0] + 1.0e+7)])

@register_eq_class
class BIOMD0000000932(KnownEquation):
    _eq_name = 'vars4_prog3'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Garde2020-Minimal model describing metabolic oscillations in Bacillus subtilis biofilms"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-5.3*x[0]*x[2] + 8.29*x[0]', '2.0*x[0] - 2.3*x[1]', '2.3*x[1] - 4.0*x[2]', '0.1*x[0]*x[2]*x_4']
    
    def np_eq(self, t, x):
        return np.array([-5.3*x[0]*x[2] + 8.29*x[0], 2.0*x[0] - 2.3*x[1], 2.3*x[1] - 4.0*x[2], 0.1*x[0]*x[2]*x_4])

@register_eq_class
class BIOMD0000000881(KnownEquation):
    _eq_name = 'vars4_prog4'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Kogan2013 - A mathematical model for the immunotherapeutic control of the TH1 TH2 imbalance in melanoma"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.001*x[0] + 350.0/(x_4 + 0.35)', '-0.001*x[1] + 180.0/(x[2] + 0.18)', '0.016 - 0.6*x[2]', '-0.36*x_4 + 0.06 + 0.011/(x[2] + 0.025)']
    
    def np_eq(self, t, x):
        return np.array([-0.001*x[0] + 350.0/(x_4 + 0.35), -0.001*x[1] + 180.0/(x[2] + 0.18), 0.016 - 0.6*x[2], -0.36*x_4 + 0.06 + 0.011/(x[2] + 0.025)])

@register_eq_class
class BIOMD0000000521(KnownEquation):
    _eq_name = 'vars4_prog5'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Ribba2012 - Low-grade gliomas tumour growth inhibition model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.24*x[0]', '-0.175*x[0]*x[1] - 0.00121*x[1]**2 - 0.00121*x[1]*x[2] - 0.00121*x[1]*x_4 + 0.118*x[1] + 0.003*x_4', '-0.175*x[0]*x[2] + 0.003', '0.175*x[0]*x[2] - 0.012*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.24*x[0], -0.175*x[0]*x[1] - 0.00121*x[1]**2 - 0.00121*x[1]*x[2] - 0.00121*x[1]*x_4 + 0.118*x[1] + 0.003*x_4, -0.175*x[0]*x[2] + 0.003, 0.175*x[0]*x[2] - 0.012*x_4])

@register_eq_class
class BIOMD0000000957(KnownEquation):
    _eq_name = 'vars4_prog6'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Roda2020 - SIR model of COVID-19 spread in Wuhan"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '-1.009*x[1]', '0.1*x[1]', '0.909*x[1]']
    
    def np_eq(self, t, x):
        return np.array([0, -1.009*x[1], 0.1*x[1], 0.909*x[1]])

@register_eq_class
class BIOMD0000000267(KnownEquation):
    _eq_name = 'vars4_prog7'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Lebeda2008 - BoNT paralysis (3 step model)"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.058*x[0]', '0.058*x[0] - 0.141*x[1]', '0.141*x[1] - 0.013*x[2]', '0.013*x[2]']
    
    def np_eq(self, t, x):
        return np.array([-0.058*x[0], 0.058*x[0] - 0.141*x[1], 0.141*x[1] - 0.013*x[2], 0.013*x[2]])

@register_eq_class
class BIOMD0000000252(KnownEquation):
    _eq_name = 'vars4_prog8'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Hunziker2010-p53-StressSpecificResponse"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-5000.0*x[0]*x[2] - 0.1*x[0] + 7200.0*x_4 + 1000.0', '0.03*x[0]**2 - 0.6*x[1]', '-5000.0*x[0]*x[2] + 1.4*x[1] - 0.2*x[2] + 7211.0*x_4', '5000.0*x[0]*x[2] - 7211.0*x_4']
    
    def np_eq(self, t, x):
        return np.array([-5000.0*x[0]*x[2] - 0.1*x[0] + 7200.0*x_4 + 1000.0, 0.03*x[0]**2 - 0.6*x[1], -5000.0*x[0]*x[2] + 1.4*x[1] - 0.2*x[2] + 7211.0*x_4, 5000.0*x[0]*x[2] - 7211.0*x_4])

@register_eq_class
class BIOMD0000000801(KnownEquation):
    _eq_name = 'vars4_prog9'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'sin', 'pow']
    _description = "Sturrock2015 - glioma growth"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.7875*x[0]**2*x[1] + 1.575*x[0]*x[1] - 0.072*x[0]*x[2]', '-1.0*x[0]*x[1] - 0.7*x[1]*x[2] - 20.5*x[1] + 20.0*x_4', '-0.72*x[0]*x[2] + 0.7*x[1]*x[2] + 0.49*x[1] - 0.01*x[2]', '20.0*x[1] - 20.01*x_4 + 1.0*piecewise(0.001 < 0.002*np.sin(6*Pi*t), 0.002*np.sin(6*Pi*t), 0.001)']
    
    def np_eq(self, t, x):
        return np.array([-0.7875*x[0]**2*x[1] + 1.575*x[0]*x[1] - 0.072*x[0]*x[2], -1.0*x[0]*x[1] - 0.7*x[1]*x[2] - 20.5*x[1] + 20.0*x_4, -0.72*x[0]*x[2] + 0.7*x[1]*x[2] + 0.49*x[1] - 0.01*x[2], 20.0*x[1] - 20.01*x_4 + 1.0*piecewise(0.001 < 0.002*np.sin(6*Pi*t), 0.002*np.sin(6*Pi*t), 0.001)])

@register_eq_class
class BIOMD0000000885(KnownEquation):
    _eq_name = 'vars4_prog10'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Sumana2018 - Mathematical modeling of cancer-immune system considering the role of antibodies."
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0e-8*x[0]**2 + 0.001*x[0]', '0.009*x[0] - 0.01*x[1]', '100.0*x[0] + 1000.0*x[1] - 6.884*x[2]', '-3.022e+7*x[2]*x_4 - 4.398e-10*x_4**2 + 0.431*x_4']
    
    def np_eq(self, t, x):
        return np.array([-1.0e-8*x[0]**2 + 0.001*x[0], 0.009*x[0] - 0.01*x[1], 100.0*x[0] + 1000.0*x[1] - 6.884*x[2], -3.022e+7*x[2]*x_4 - 4.398e-10*x_4**2 + 0.431*x_4])

@register_eq_class
class BIOMD0000000254(KnownEquation):
    _eq_name = 'vars4_prog11'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Bier2000-GlycolyticOscillation"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.02*x[0]*x[1] + 0.36', '0.04*x[0]*x[1] - 0.01*x[1] - 6.0*x[1]/(x[1] + 13.0) + 0.01*x_4', '-0.02*x[2]*x_4 + 0.36', '0.01*x[1] + 0.04*x[2]*x_4 - 0.01*x_4 - 6.0*x_4/(x_4 + 13.0)']
    
    def np_eq(self, t, x):
        return np.array([-0.02*x[0]*x[1] + 0.36, 0.04*x[0]*x[1] - 0.01*x[1] - 6.0*x[1]/(x[1] + 13.0) + 0.01*x_4, -0.02*x[2]*x_4 + 0.36, 0.01*x[1] + 0.04*x[2]*x_4 - 0.01*x_4 - 6.0*x_4/(x_4 + 13.0)])

@register_eq_class
class BIOMD0000000877(KnownEquation):
    _eq_name = 'vars4_prog12'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Ontah2019 - Dynamic analysis of a tumor treatment model using oncolytic virus and chemotherapy with saturated infection rate"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-4.675e-5*x[0]**2 - 4.675e-5*x[0]*x[1] - 0.01*x[0]*x[2]/(x[0] + x[1] + 0.5) - 50.0*x[0]*x_4/(x_4 + 1.0e+4) + 0.1*x[0]', '0.01*x[0]*x[2]/(x[0] + x[1] + 0.5) - 60.0*x[1]*x_4/(x_4 + 1.0e+4) - 0.5*x[1]', '-0.01*x[0]*x[2]/(x[0] + x[1] + 0.5) + 0.25*x[1] - 0.1*x[2]', '150.0 - 4.17*x_4']
    
    def np_eq(self, t, x):
        return np.array([-4.675e-5*x[0]**2 - 4.675e-5*x[0]*x[1] - 0.01*x[0]*x[2]/(x[0] + x[1] + 0.5) - 50.0*x[0]*x_4/(x_4 + 1.0e+4) + 0.1*x[0], 0.01*x[0]*x[2]/(x[0] + x[1] + 0.5) - 60.0*x[1]*x_4/(x_4 + 1.0e+4) - 0.5*x[1], -0.01*x[0]*x[2]/(x[0] + x[1] + 0.5) + 0.25*x[1] - 0.1*x[2], 150.0 - 4.17*x_4])

@register_eq_class
class BIOMD0000000092(KnownEquation):
    _eq_name = 'vars4_prog13'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Fuentes2005-ZymogenActivation"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1000.0*x[0]*x[1] - 0.004*x[0]', '-1000.0*x[0]*x[1] + 0.004*x[0] + 0.002*x_4', '0.004*x[0] + 0.001*x_4', '1000.0*x[0]*x[1] - 0.001*x_4']
    
    def np_eq(self, t, x):
        return np.array([-1000.0*x[0]*x[1] - 0.004*x[0], -1000.0*x[0]*x[1] + 0.004*x[0] + 0.002*x_4, 0.004*x[0] + 0.001*x_4, 1000.0*x[0]*x[1] - 0.001*x_4])

@register_eq_class
class BIOMD0000000746(KnownEquation):
    _eq_name = 'vars4_prog14'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Saad2017 - immune checkpoint and BCG in superficial bladder cancer"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.003*x[0]', '0.25*x[0]*x[1]/(x_4 + 2000.0) + 0.052*x[1]*x[2]/(x_4 + 2000.0) - 0.041*x[1]', '6.5e+5 - 0.1*x[2]', '1.519e+5 - 166.3*x_4']
    
    def np_eq(self, t, x):
        return np.array([0.003*x[0], 0.25*x[0]*x[1]/(x_4 + 2000.0) + 0.052*x[1]*x[2]/(x_4 + 2000.0) - 0.041*x[1], 6.5e+5 - 0.1*x[2], 1.519e+5 - 166.3*x_4])

@register_eq_class
class BIOMD0000000765(KnownEquation):
    _eq_name = 'vars4_prog15'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Mager2005 - Quasi-equilibrium pharmacokinetic model for drugs exhibiting target-mediated drug disposition"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.1*x[0]*x[2] - 1.0*x[0] + 0.1*x_4', '0', '-0.1*x[0]*x[2] - 0.566*x[2] + 0.1*x_4', '0.1*x[0]*x[2] - 0.1*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.1*x[0]*x[2] - 1.0*x[0] + 0.1*x_4, 0, -0.1*x[0]*x[2] - 0.566*x[2] + 0.1*x_4, 0.1*x[0]*x[2] - 0.1*x_4])

@register_eq_class
class BIOMD0000000743(KnownEquation):
    _eq_name = 'vars4_prog16'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Gallaher2018 - Tumor–Immune dynamics in multiple myeloma"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.008*x[0]**2*x[1]*x[2]/(x[0]*x[1]*x[2] + 150.0*x[0]*x[1] + 375.0*x[0]*x[2] + 5.625e+4*x[0] + 3.0*x[1]*x[2] + 450.0*x[1] + 1125.0*x[2] + 1.688e+5) + 0.005*x[0]**2*x[1]/(x[0]*x[1] + 375.0*x[0] + 3.0*x[1] + 1125.0) + 0.005*x[0]**2*x[2]/(x[0]*x[2] + 150.0*x[0] + 3.0*x[2] + 450.0) - 0.0018*x[0]**2 + 0.008*x[0]*x[1]*x[2]*x_4/(x[1]*x[2]*x_4 + 25.0*x[1]*x[2] + 150.0*x[1]*x_4 + 3750.0*x[1] + 375.0*x[2]*x_4 + 9375.0*x[2] + 5.625e+4*x_4 + 1.406e+6) - 0.016*x[0]*x[1]*x[2]/(x[1]*x[2] + 150.0*x[1] + 375.0*x[2] + 5.625e+4) + 0.005*x[0]*x[1]*x_4/(x[1]*x_4 + 25.0*x[1] + 375.0*x_4 + 9375.0) - 0.01*x[0]*x[1]/(x[1] + 375.0) + 0.005*x[0]*x[2]*x_4/(x[2]*x_4 + 25.0*x[2] + 150.0*x_4 + 3750.0) - 0.01*x[0]*x[2]/(x[2] + 150.0) + 0.016*x[0] + 0.001', '-1.625e-5*x[0]*x[1]**2/(x[0] + 150.0) + 0.013*x[0]*x[1]/(x[0] + 150.0) - 8.125e-5*x[1]**2*x[2]/(x[2] + 3.0) - 1.625e-5*x[1]**2 + 0.065*x[1]*x[2]/(x[2] + 3.0) - 0.007*x[1]', '-8.889e-5*x[1]*x[2]**2/(x[1] + 375.0) + 0.04*x[1]*x[2]/(x[1] + 375.0) - 8.889e-5*x[2]**2 + 0.015*x[2] + 0.03', '-0.002075*x[0]*x_4**2/(x[0] + 3.0) + 0.166*x[0]*x_4/(x[0] + 3.0) - 0.001037*x_4**2 + 0.007*x_4']
    
    def np_eq(self, t, x):
        return np.array([0.008*x[0]**2*x[1]*x[2]/(x[0]*x[1]*x[2] + 150.0*x[0]*x[1] + 375.0*x[0]*x[2] + 5.625e+4*x[0] + 3.0*x[1]*x[2] + 450.0*x[1] + 1125.0*x[2] + 1.688e+5) + 0.005*x[0]**2*x[1]/(x[0]*x[1] + 375.0*x[0] + 3.0*x[1] + 1125.0) + 0.005*x[0]**2*x[2]/(x[0]*x[2] + 150.0*x[0] + 3.0*x[2] + 450.0) - 0.0018*x[0]**2 + 0.008*x[0]*x[1]*x[2]*x_4/(x[1]*x[2]*x_4 + 25.0*x[1]*x[2] + 150.0*x[1]*x_4 + 3750.0*x[1] + 375.0*x[2]*x_4 + 9375.0*x[2] + 5.625e+4*x_4 + 1.406e+6) - 0.016*x[0]*x[1]*x[2]/(x[1]*x[2] + 150.0*x[1] + 375.0*x[2] + 5.625e+4) + 0.005*x[0]*x[1]*x_4/(x[1]*x_4 + 25.0*x[1] + 375.0*x_4 + 9375.0) - 0.01*x[0]*x[1]/(x[1] + 375.0) + 0.005*x[0]*x[2]*x_4/(x[2]*x_4 + 25.0*x[2] + 150.0*x_4 + 3750.0) - 0.01*x[0]*x[2]/(x[2] + 150.0) + 0.016*x[0] + 0.001, -1.625e-5*x[0]*x[1]**2/(x[0] + 150.0) + 0.013*x[0]*x[1]/(x[0] + 150.0) - 8.125e-5*x[1]**2*x[2]/(x[2] + 3.0) - 1.625e-5*x[1]**2 + 0.065*x[1]*x[2]/(x[2] + 3.0) - 0.007*x[1], -8.889e-5*x[1]*x[2]**2/(x[1] + 375.0) + 0.04*x[1]*x[2]/(x[1] + 375.0) - 8.889e-5*x[2]**2 + 0.015*x[2] + 0.03, -0.002075*x[0]*x_4**2/(x[0] + 3.0) + 0.166*x[0]*x_4/(x[0] + 3.0) - 0.001037*x_4**2 + 0.007*x_4])

@register_eq_class
class BIOMD0000000888(KnownEquation):
    _eq_name = 'vars4_prog17'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Unni2019 - Mathematical Modeling Analysis and Simulation of Tumor Dynamics with Drug Interventions"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.431*x[0]', '1.3e+4 - 0.041*x[1]', '0.024*x[2] + 480.0', '0.01*x[0]*x[2] - 0.02*x_4']
    
    def np_eq(self, t, x):
        return np.array([0.431*x[0], 1.3e+4 - 0.041*x[1], 0.024*x[2] + 480.0, 0.01*x[0]*x[2] - 0.02*x_4])

@register_eq_class
class BIOMD0000000790(KnownEquation):
    _eq_name = 'vars4_prog18'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Alvarez2019 - A nonlinear mathematical model of cell-mediated immune response for tumor phenotypic heterogeneity"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.514*x[0]', '0.1799*x[1]', '0.124*x[0]*x[2]/(x[0] + x[1] + 2.019e+7) + 0.124*x[1]*x[2]/(x[0] + x[1] + 2.019e+7) - 0.041*x[2] + 1.3e+4', '-0.02*x_4']
    
    def np_eq(self, t, x):
        return np.array([0.514*x[0], 0.1799*x[1], 0.124*x[0]*x[2]/(x[0] + x[1] + 2.019e+7) + 0.124*x[1]*x[2]/(x[0] + x[1] + 2.019e+7) - 0.041*x[2] + 1.3e+4, -0.02*x_4])

@register_eq_class
class BIOMD0000000616(KnownEquation):
    _eq_name = 'vars4_prog19'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'sin', 'pow']
    _description = "Dunster2014 - WBC Interactions (Model1)"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.1*x[0] + 1.0*x[1]', '-1.0*x[1] + 1.0*x[2]**2/(x[2]**2 + 0.01) + 0.05*piecewise(t < 1.0*Pi, np.sin(t)**2, 0)*piecewise(t < 10.0, 1, 0)', '0.1*x[0] - 0.001*x[2]*x_4 - 1.0*x[2]', '1.0*x[1] - 0.01*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.1*x[0] + 1.0*x[1], -1.0*x[1] + 1.0*x[2]**2/(x[2]**2 + 0.01) + 0.05*piecewise(t < 1.0*Pi, np.sin(t)**2, 0)*piecewise(t < 10.0, 1, 0), 0.1*x[0] - 0.001*x[2]*x_4 - 1.0*x[2], 1.0*x[1] - 0.01*x_4])

@register_eq_class
class BIOMD0000000459(KnownEquation):
    _eq_name = 'vars4_prog20'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Liebal2012 - B.subtilis post-transcriptional instability model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '-0.016*x[1] + 0.003*x[2]*x_4 + 1.0e+4', '-0.003*x[2]*x_4', '0']
    
    def np_eq(self, t, x):
        return np.array([0, -0.016*x[1] + 0.003*x[2]*x_4 + 1.0e+4, -0.003*x[2]*x_4, 0])

@register_eq_class
class BIOMD0000000797(KnownEquation):
    _eq_name = 'vars4_prog21'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Hu2018 - Dynamics of tumor-CD4+-cytokine-host cells interactions with treatments"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.2*x[0]*x[2]/(x[0] + 1.0e+5) + 0.514*x[0]', '0.835*x[0]*x[2]/(x[0] + 1000.0) - 0.1*x[1]', '5.4*x[0]*x[1]/(x[0] + 1000.0) - 34.0*x[2]', '0.282*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.2*x[0]*x[2]/(x[0] + 1.0e+5) + 0.514*x[0], 0.835*x[0]*x[2]/(x[0] + 1000.0) - 0.1*x[1], 5.4*x[0]*x[1]/(x[0] + 1000.0) - 34.0*x[2], 0.282*x_4])

@register_eq_class
class BIOMD0000000517(KnownEquation):
    _eq_name = 'vars4_prog22'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Smallbone2013 - Colon Crypt cycle - Version 3"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.006953*x[0]**2 + 0.01381*x[0]**2/(x_4 + 0.784) - 0.006953*x[0]*x[1] + 0.01381*x[0]*x[1]/(x_4 + 0.784) - 0.006953*x[0]*x[2] + 0.01381*x[0]*x[2]/(x_4 + 0.784) - 0.006953*x[0]*x_4 + 0.01381*x[0]*x_4/(x_4 + 0.784) + 0.7376*x[0] - 1.465*x[0]/(x_4 + 0.784) - 0.00308*x[0]/(x_4 + 0.154)', '-0.01189*x[0]**2 - 0.03135*x[0]*x[1] - 0.01189*x[0]*x[2] - 0.01189*x[0]*x_4 + 1.261*x[0] - 0.01946*x[1]**2 - 0.01946*x[1]*x[2] - 0.01946*x[1]*x_4 + 2.064*x[1] - 8.42*x[1]/(x_4 + 15.36)', '-0.08033*x[0]*x[1] - 0.08033*x[1]**2 - 0.08033*x[1]*x[2] - 0.08033*x[1]*x_4 + 8.522*x[1] - 5.108*x[2]/(x_4 + 2.704)', '-0.01381*x[0]**2/(x_4 + 0.784) - 0.01381*x[0]*x[1]/(x_4 + 0.784) - 0.01381*x[0]*x[2]/(x_4 + 0.784) - 0.01381*x[0]*x_4/(x_4 + 0.784) + 1.465*x[0]/(x_4 + 0.784) - 0.168*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.006953*x[0]**2 + 0.01381*x[0]**2/(x_4 + 0.784) - 0.006953*x[0]*x[1] + 0.01381*x[0]*x[1]/(x_4 + 0.784) - 0.006953*x[0]*x[2] + 0.01381*x[0]*x[2]/(x_4 + 0.784) - 0.006953*x[0]*x_4 + 0.01381*x[0]*x_4/(x_4 + 0.784) + 0.7376*x[0] - 1.465*x[0]/(x_4 + 0.784) - 0.00308*x[0]/(x_4 + 0.154), -0.01189*x[0]**2 - 0.03135*x[0]*x[1] - 0.01189*x[0]*x[2] - 0.01189*x[0]*x_4 + 1.261*x[0] - 0.01946*x[1]**2 - 0.01946*x[1]*x[2] - 0.01946*x[1]*x_4 + 2.064*x[1] - 8.42*x[1]/(x_4 + 15.36), -0.08033*x[0]*x[1] - 0.08033*x[1]**2 - 0.08033*x[1]*x[2] - 0.08033*x[1]*x_4 + 8.522*x[1] - 5.108*x[2]/(x_4 + 2.704), -0.01381*x[0]**2/(x_4 + 0.784) - 0.01381*x[0]*x[1]/(x_4 + 0.784) - 0.01381*x[0]*x[2]/(x_4 + 0.784) - 0.01381*x[0]*x_4/(x_4 + 0.784) + 1.465*x[0]/(x_4 + 0.784) - 0.168*x_4])

@register_eq_class
class BIOMD0000000460(KnownEquation):
    _eq_name = 'vars4_prog23'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Liebal2012 - B.subtilis sigB proteolysis model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '0.052*x[2] + 1.0e+4', '-0.052*x[2]', '0']
    
    def np_eq(self, t, x):
        return np.array([0, 0.052*x[2] + 1.0e+4, -0.052*x[2], 0])

@register_eq_class
class BIOMD0000000461(KnownEquation):
    _eq_name = 'vars4_prog24'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Liebal2012 - B.subtilis transcription inhibition model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '-0.044*x[1] - 0.76*x[1]/(x_4 + 1.0) + 0.041*x[2] + 9.0*x_4 + 1.0e+4', '-0.041*x[2]', '0.76*x[1]/(x_4 + 1.0) - 9.0*x_4']
    
    def np_eq(self, t, x):
        return np.array([0, -0.044*x[1] - 0.76*x[1]/(x_4 + 1.0) + 0.041*x[2] + 9.0*x_4 + 1.0e+4, -0.041*x[2], 0.76*x[1]/(x_4 + 1.0) - 9.0*x_4])

@register_eq_class
class BIOMD0000000880(KnownEquation):
    _eq_name = 'vars4_prog25'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Trisilowati2018 - Optimal control of tumor-immune system interaction with treatment"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.431*x[0]', '0', '0.234*x[2]', '-1.7e-5*x_4**2 + 0.017*x_4']
    
    def np_eq(self, t, x):
        return np.array([0.431*x[0], 0, 0.234*x[2], -1.7e-5*x_4**2 + 0.017*x_4])

@register_eq_class
class BIOMD0000000683(KnownEquation):
    _eq_name = 'vars4_prog26'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Wodarz1999 CTL memory response HIV"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.1*x[0]*x[1]*x[2] - 0.05*x[0]*x[2] - 0.01*x[0]', '-1.5*x[1]*x[2]*piecewise(t < 1, 1, piecewise(t > 15, 1, 21/5000)) - 0.1*x[1] + 1.0', '1.5*x[1]*x[2]*piecewise(t < 1, 1, piecewise(t > 15, 1, 21/5000)) - 1.0*x[2]*x_4 - 0.2*x[2]', '0.05*x[0]*x[2] - 0.1*x_4']
    
    def np_eq(self, t, x):
        return np.array([0.1*x[0]*x[1]*x[2] - 0.05*x[0]*x[2] - 0.01*x[0], -1.5*x[1]*x[2]*piecewise(t < 1, 1, piecewise(t > 15, 1, 21/5000)) - 0.1*x[1] + 1.0, 1.5*x[1]*x[2]*piecewise(t < 1, 1, piecewise(t > 15, 1, 21/5000)) - 1.0*x[2]*x_4 - 0.2*x[2], 0.05*x[0]*x[2] - 0.1*x_4])

@register_eq_class
class BIOMD0000000764(KnownEquation):
    _eq_name = 'vars4_prog27'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'exp', 'pow']
    _description = "Malinzi2019 - chemovirotherapy"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.4031*x[0]**2 - 0.4031*x[0]*x[1] - 1957.0*x[0]*x[2] - 0.0009785*x[0]*x_4 + 0.4031*x[0]', '1957.0*x[0]*x[2] - 0.001174*x[1]*x_4 - 1.0*x[1]', '-1957.0*x[0]*x[2] + 2.0*x[1] - 0.001957*x[2]', '-8.141*x_4 + 97.85*np.exp(-0.403131115459883*t)']
    
    def np_eq(self, t, x):
        return np.array([-0.4031*x[0]**2 - 0.4031*x[0]*x[1] - 1957.0*x[0]*x[2] - 0.0009785*x[0]*x_4 + 0.4031*x[0], 1957.0*x[0]*x[2] - 0.001174*x[1]*x_4 - 1.0*x[1], -1957.0*x[0]*x[2] + 2.0*x[1] - 0.001957*x[2], -8.141*x_4 + 97.85*np.exp(-0.403131115459883*t)])

@register_eq_class
class BIOMD0000000741(KnownEquation):
    _eq_name = 'vars4_prog28'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Eftimie2018 - Cancer and Immune biomarkers"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.006*x[0]', '-2.079e-9*x[0]*x[1] + 2.079*x[0] - 0.4*x[1]', '4560.0 - 0.11*x[2]', '1.955e+4 - 2.14*x_4']
    
    def np_eq(self, t, x):
        return np.array([0.006*x[0], -2.079e-9*x[0]*x[1] + 2.079*x[0] - 0.4*x[1], 4560.0 - 0.11*x[2], 1.955e+4 - 2.14*x_4])

@register_eq_class
class BIOMD0000000802(KnownEquation):
    _eq_name = 'vars4_prog29'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Hoffman2018- ADCC against cancer"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.001*x[0]*x[1]*x[2] - 0.001*x[0]*x[2] + 1.44*x[1]*x[2]', '-0.001*x[0]*x[1] + 0.001*x[0] - 1.44*x[1]', '-1.0*x[1]*x_4/(x[1] + 0.5)', '-1.0*x[1]*x_4/(x[1] + 0.5) - 120.0*x[2]*x_4 + 2400.0*x[2] + 120.0*x_4**2 - 2414.0*x_4']
    
    def np_eq(self, t, x):
        return np.array([0.001*x[0]*x[1]*x[2] - 0.001*x[0]*x[2] + 1.44*x[1]*x[2], -0.001*x[0]*x[1] + 0.001*x[0] - 1.44*x[1], -1.0*x[1]*x_4/(x[1] + 0.5), -1.0*x[1]*x_4/(x[1] + 0.5) - 120.0*x[2]*x_4 + 2400.0*x[2] + 120.0*x_4**2 - 2414.0*x_4])

@register_eq_class
class BIOMD0000000962(KnownEquation):
    _eq_name = 'vars4_prog30'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan Hubei and China"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-3.293e-8*x[0]*x[1]', '3.293e-8*x[0]*x[1] - 0.063*x[1]', '0.063*x[1] - 0.05*x[2]', '0.05*x[2]']
    
    def np_eq(self, t, x):
        return np.array([-3.293e-8*x[0]*x[1], 3.293e-8*x[0]*x[1] - 0.063*x[1], 0.063*x[1] - 0.05*x[2], 0.05*x[2]])

@register_eq_class
class BIOMD0000000044(KnownEquation):
    _eq_name = 'vars4_prog31'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Borghans1997 - Calcium Oscillation - Model 2"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['10.0*x[1] - 2.5', '19.5*x[1]**4*x[2]**4*x_4**2/(x[1]**4*x[2]**4*x_4**2 + 0.04*x[1]**4*x[2]**4 + 0.0016*x[1]**4*x_4**2 + 6.4e-5*x[1]**4 + 0.0081*x[2]**4*x_4**2 + 0.000324*x[2]**4 + 1.296e-5*x_4**2 + 5.184e-7) - 6.5*x[1]**2/(x[1]**2 + 0.01) - 10.0*x[1] + 1.0*x_4 + 2.5', '-80.0*x[1]**4.0*x[2]**2/(x[1]**4.0*x[2]**2 + 1.0*x[1]**4.0 + 0.0256*x[2]**2 + 0.0256) - 0.1*x[2] + 1.25', '-19.5*x[1]**4*x[2]**4*x_4**2/(x[1]**4*x[2]**4*x_4**2 + 0.04*x[1]**4*x[2]**4 + 0.0016*x[1]**4*x_4**2 + 6.4e-5*x[1]**4 + 0.0081*x[2]**4*x_4**2 + 0.000324*x[2]**4 + 1.296e-5*x_4**2 + 5.184e-7) + 6.5*x[1]**2/(x[1]**2 + 0.01) - 1.0*x_4']
    
    def np_eq(self, t, x):
        return np.array([10.0*x[1] - 2.5, 19.5*x[1]**4*x[2]**4*x_4**2/(x[1]**4*x[2]**4*x_4**2 + 0.04*x[1]**4*x[2]**4 + 0.0016*x[1]**4*x_4**2 + 6.4e-5*x[1]**4 + 0.0081*x[2]**4*x_4**2 + 0.000324*x[2]**4 + 1.296e-5*x_4**2 + 5.184e-7) - 6.5*x[1]**2/(x[1]**2 + 0.01) - 10.0*x[1] + 1.0*x_4 + 2.5, -80.0*x[1]**4.0*x[2]**2/(x[1]**4.0*x[2]**2 + 1.0*x[1]**4.0 + 0.0256*x[2]**2 + 0.0256) - 0.1*x[2] + 1.25, -19.5*x[1]**4*x[2]**4*x_4**2/(x[1]**4*x[2]**4*x_4**2 + 0.04*x[1]**4*x[2]**4 + 0.0016*x[1]**4*x_4**2 + 6.4e-5*x[1]**4 + 0.0081*x[2]**4*x_4**2 + 0.000324*x[2]**4 + 1.296e-5*x_4**2 + 5.184e-7) + 6.5*x[1]**2/(x[1]**2 + 0.01) - 1.0*x_4])

@register_eq_class
class BIOMD0000000780(KnownEquation):
    _eq_name = 'vars4_prog32'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Wang2016/1 - oncolytic efficacy of M1 virus-SNTM model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.2*x[0]*x[1] - 0.4*x[0]*x[2] - 0.02*x[0] + 0.02', '0.16*x[0]*x[1] - 0.03*x[1]', '0.32*x[0]*x[2] - 0.1*x[2]*x_4 - 0.06*x[2]', '0.05*x[2]*x_4 - 0.03*x_4 + 0.001']
    
    def np_eq(self, t, x):
        return np.array([-0.2*x[0]*x[1] - 0.4*x[0]*x[2] - 0.02*x[0] + 0.02, 0.16*x[0]*x[1] - 0.03*x[1], 0.32*x[0]*x[2] - 0.1*x[2]*x_4 - 0.06*x[2], 0.05*x[2]*x_4 - 0.03*x_4 + 0.001])

@register_eq_class
class BIOMD0000000716(KnownEquation):
    _eq_name = 'vars4_prog33'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Lee2018 - Avian human bilinear incidence (BI) model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.003467*x[0]*x[1] - 0.00137*x[0] + 5.644', '0.003467*x[0]*x[1] - 0.004837*x[1]', '-1.624e-10*x[1]*x[2] - 3.959e-5*x[2] + 11.17', '1.624e-10*x[1]*x[2] - 3.959e-5*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.003467*x[0]*x[1] - 0.00137*x[0] + 5.644, 0.003467*x[0]*x[1] - 0.004837*x[1], -1.624e-10*x[1]*x[2] - 3.959e-5*x[2] + 11.17, 1.624e-10*x[1]*x[2] - 3.959e-5*x_4])

@register_eq_class
class BIOMD0000000748(KnownEquation):
    _eq_name = 'vars4_prog34'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Phan2017 - innate immune in oncolytic virotherapy"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.36*x[0]**2 - 0.36*x[0]*x[1] - 0.11*x[0]*x[2] + 0.36*x[0]', '0.11*x[0]*x[2] - 0.48*x[1]*x_4 - 1.0*x[1]', '-0.11*x[0]*x[2] + 9.0*x[1] - 0.16*x[2]*x_4 - 0.2*x[2]', '0.6*x[1]*x_4 - 0.036*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.36*x[0]**2 - 0.36*x[0]*x[1] - 0.11*x[0]*x[2] + 0.36*x[0], 0.11*x[0]*x[2] - 0.48*x[1]*x_4 - 1.0*x[1], -0.11*x[0]*x[2] + 9.0*x[1] - 0.16*x[2]*x_4 - 0.2*x[2], 0.6*x[1]*x_4 - 0.036*x_4])

@register_eq_class
class BIOMD0000000920(KnownEquation):
    _eq_name = 'vars4_prog35'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'exp', 'pow']
    _description = "Jarrett2015 - Modelling the interaction between immune response bacterial dynamics and inflammatory damage"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.135*x[0]*x[1] + 0.25*x[0]*x[2] - 0.25*x[0] + 0.11*x_4', '-2.0*x[0]*x[1] - 0.017*x[1] + 0.45*x[2] + 1.05*x_4', '1.5*x[1]*x[2] - 0.9*x[2]**2 - 5.0*x[2]*x_4 + 0.9*x[2] + 1.0*np.exp(-0.01*t)', '-0.01*x[0]*x_4 - 0.27*x[1]*x_4 + 0.27*x[1] - 0.08*x[2]*x_4 + 0.2*x[2] - 0.12*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.135*x[0]*x[1] + 0.25*x[0]*x[2] - 0.25*x[0] + 0.11*x_4, -2.0*x[0]*x[1] - 0.017*x[1] + 0.45*x[2] + 1.05*x_4, 1.5*x[1]*x[2] - 0.9*x[2]**2 - 5.0*x[2]*x_4 + 0.9*x[2] + 1.0*np.exp(-0.01*t), -0.01*x[0]*x_4 - 0.27*x[1]*x_4 + 0.27*x[1] - 0.08*x[2]*x_4 + 0.2*x[2] - 0.12*x_4])

@register_eq_class
class BIOMD0000000904(KnownEquation):
    _eq_name = 'vars4_prog36'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Admon2017 - Modelling tumor growth with immune response and drug using ordinary differential equations"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.9*x[0]*x[2] - 1.11*x[0] + 1.6*x[1]', '1.0*x[0] - 0.9*x[1]*x[2] - 1.2*x[1]', '-0.085*x[0]*x[2] - 0.085*x[1]*x[2] + 0.1*x[2]*(x[0] + x[1])**3.0/((x[0] + x[1])**3.0 + 0.2) - 0.29*x[2] + 0.029', '0']
    
    def np_eq(self, t, x):
        return np.array([-0.9*x[0]*x[2] - 1.11*x[0] + 1.6*x[1], 1.0*x[0] - 0.9*x[1]*x[2] - 1.2*x[1], -0.085*x[0]*x[2] - 0.085*x[1]*x[2] + 0.1*x[2]*(x[0] + x[1])**3.0/((x[0] + x[1])**3.0 + 0.2) - 0.29*x[2] + 0.029, 0])

@register_eq_class
class BIOMD0000000518(KnownEquation):
    _eq_name = 'vars4_prog37'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Smallbone2013 - Colon Crypt cycle - Version 2"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.003065*x[0]**2 + 0.00246*x[0]**2/(x_4 + 1.571) - 0.003065*x[0]*x[1] + 0.00246*x[0]*x[1]/(x_4 + 1.571) - 0.003065*x[0]*x[2] + 0.00246*x[0]*x[2]/(x_4 + 1.571) - 0.003065*x[0]*x_4 + 0.00246*x[0]*x_4/(x_4 + 1.571) + 0.3678*x[0] - 0.6094*x[0]/(x_4 + 1.571)', '-0.01359*x[0]**2 - 0.02238*x[0]*x[1] - 0.01359*x[0]*x[2] - 0.01359*x[0]*x_4 + 1.631*x[0] - 0.008783*x[1]**2 - 0.008783*x[1]*x[2] - 0.008783*x[1]*x_4 + 1.054*x[1] - 1.321*x[1]/(x_4 + 1.571)', '-0.04198*x[0]*x[1] - 0.04198*x[1]**2 - 0.04198*x[1]*x[2] - 0.04198*x[1]*x_4 + 5.038*x[1] - 3.461*x[2]/(x_4 + 1.571)', '-0.00246*x[0]**2/(x_4 + 1.571) - 0.00246*x[0]*x[1]/(x_4 + 1.571) - 0.00246*x[0]*x[2]/(x_4 + 1.571) - 0.00246*x[0]*x_4/(x_4 + 1.571) + 0.2952*x[0]/(x_4 + 1.571) - 0.038*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.003065*x[0]**2 + 0.00246*x[0]**2/(x_4 + 1.571) - 0.003065*x[0]*x[1] + 0.00246*x[0]*x[1]/(x_4 + 1.571) - 0.003065*x[0]*x[2] + 0.00246*x[0]*x[2]/(x_4 + 1.571) - 0.003065*x[0]*x_4 + 0.00246*x[0]*x_4/(x_4 + 1.571) + 0.3678*x[0] - 0.6094*x[0]/(x_4 + 1.571), -0.01359*x[0]**2 - 0.02238*x[0]*x[1] - 0.01359*x[0]*x[2] - 0.01359*x[0]*x_4 + 1.631*x[0] - 0.008783*x[1]**2 - 0.008783*x[1]*x[2] - 0.008783*x[1]*x_4 + 1.054*x[1] - 1.321*x[1]/(x_4 + 1.571), -0.04198*x[0]*x[1] - 0.04198*x[1]**2 - 0.04198*x[1]*x[2] - 0.04198*x[1]*x_4 + 5.038*x[1] - 3.461*x[2]/(x_4 + 1.571), -0.00246*x[0]**2/(x_4 + 1.571) - 0.00246*x[0]*x[1]/(x_4 + 1.571) - 0.00246*x[0]*x[2]/(x_4 + 1.571) - 0.00246*x[0]*x_4/(x_4 + 1.571) + 0.2952*x[0]/(x_4 + 1.571) - 0.038*x_4])

@register_eq_class
class BIOMD0000000717(KnownEquation):
    _eq_name = 'vars4_prog38'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Lee2018 - Avian human half-saturated incidence (HSI) model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.003467*x[0]*x[1]/(x[1] + 1.8e+5) - 0.00137*x[0] + 5.644', '0.003467*x[0]*x[1]/(x[1] + 1.8e+5) - 0.004837*x[1]', '-1.624e-10*x[1]*x[2]/(x[1] + 1.2e+5) - 3.959e-5*x[2] + 11.17', '1.624e-10*x[1]*x[2]/(x[1] + 1.2e+5) - 3.959e-5*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.003467*x[0]*x[1]/(x[1] + 1.8e+5) - 0.00137*x[0] + 5.644, 0.003467*x[0]*x[1]/(x[1] + 1.8e+5) - 0.004837*x[1], -1.624e-10*x[1]*x[2]/(x[1] + 1.2e+5) - 3.959e-5*x[2] + 11.17, 1.624e-10*x[1]*x[2]/(x[1] + 1.2e+5) - 3.959e-5*x_4])

@register_eq_class
class BIOMD0000000113(KnownEquation):
    _eq_name = 'vars4_prog39'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Dupont1992-Ca-dpt-protein-phospho"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-10.0*x[0] - 65.0*x[0]**2.0/(x[0]**2.0 + 1.0) + 500.0*x[0]**4.0*x[1]**2.0/(x[0]**4.0*x[1]**2.0 + 4.0*x[0]**4.0 + 0.6561*x[1]**2.0 + 2.624) + 1.0*x[1] + 3.7', '65.0*x[0]**2.0/(x[0]**2.0 + 1.0) - 500.0*x[0]**4.0*x[1]**2.0/(x[0]**4.0*x[1]**2.0 + 4.0*x[0]**4.0 + 0.6561*x[1]**2.0 + 2.624) - 1.0*x[1]', '0', '-20.0*x[0]**1.0*x_4/(-x[0]**1.0*x[2]*x_4 + 1.01*x[0]**1.0*x[2] - 2.5*x[2]*x_4 + 2.525*x[2]) + 20.0*x[0]**1.0/(-x[0]**1.0*x[2]*x_4 + 1.01*x[0]**1.0*x[2] - 2.5*x[2]*x_4 + 2.525*x[2]) - 2.5*x_4/(x[2]*x_4 + 0.01*x[2])']
    
    def np_eq(self, t, x):
        return np.array([-10.0*x[0] - 65.0*x[0]**2.0/(x[0]**2.0 + 1.0) + 500.0*x[0]**4.0*x[1]**2.0/(x[0]**4.0*x[1]**2.0 + 4.0*x[0]**4.0 + 0.6561*x[1]**2.0 + 2.624) + 1.0*x[1] + 3.7, 65.0*x[0]**2.0/(x[0]**2.0 + 1.0) - 500.0*x[0]**4.0*x[1]**2.0/(x[0]**4.0*x[1]**2.0 + 4.0*x[0]**4.0 + 0.6561*x[1]**2.0 + 2.624) - 1.0*x[1], 0, -20.0*x[0]**1.0*x_4/(-x[0]**1.0*x[2]*x_4 + 1.01*x[0]**1.0*x[2] - 2.5*x[2]*x_4 + 2.525*x[2]) + 20.0*x[0]**1.0/(-x[0]**1.0*x[2]*x_4 + 1.01*x[0]**1.0*x[2] - 2.5*x[2]*x_4 + 2.525*x[2]) - 2.5*x_4/(x[2]*x_4 + 0.01*x[2])])

@register_eq_class
class BIOMD0000000909(KnownEquation):
    _eq_name = 'vars4_prog40'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'exp', 'pow']
    _description = "dePillis2003 - The dynamics of an optimally controlled tumor model: A case study"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0]**2 - 1.0*x[0]*x[1] + 0.9*x[0] + 0.1*x[0]*np.exp(-x_4)', '-1.0*x[0]*x[1] - 1.5*x[1]**2 - 0.5*x[1]*x[2] + 1.2*x[1] + 0.3*x[1]*np.exp(-x_4)', '-1.0*x[1]*x[2] + 0.01*x[1]*x[2]/(x[1] + 0.3) - 0.4*x[2] + 0.2*x[2]*np.exp(-x_4) + 0.33', '-1.0*x_4']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0]**2 - 1.0*x[0]*x[1] + 0.9*x[0] + 0.1*x[0]*np.exp(-x_4), -1.0*x[0]*x[1] - 1.5*x[1]**2 - 0.5*x[1]*x[2] + 1.2*x[1] + 0.3*x[1]*np.exp(-x_4), -1.0*x[1]*x[2] + 0.01*x[1]*x[2]/(x[1] + 0.3) - 0.4*x[2] + 0.2*x[2]*np.exp(-x_4) + 0.33, -1.0*x_4])

@register_eq_class
class BIOMD0000000615(KnownEquation):
    _eq_name = 'vars4_prog41'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Kuznetsov2016(II) - α-syn aggregation kinetics in Parkinson's Disease"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['nan', '-9.627e-7*x[1]', 'nan', '-9.627e-7*x_4']
    
    def np_eq(self, t, x):
        return np.array([nan, -9.627e-7*x[1], nan, -9.627e-7*x_4])

@register_eq_class
class BIOMD0000000435(KnownEquation):
    _eq_name = 'vars4_prog42'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "deBack2012 - Lineage Specification in Pancreas Development"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0]', '-1.0*x[1] + 21.0*(x[1]*x_4)**4.0/(1.0*x[0]**4.0 + 21.0*(x[1]*x_4)**4.0)', '-1.0*x[2]', '-1.0*x_4 + 21.0*(x[1]*x_4)**4.0/(1.0*x[2]**4.0 + 21.0*(x[1]*x_4)**4.0)']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0], -1.0*x[1] + 21.0*(x[1]*x_4)**4.0/(1.0*x[0]**4.0 + 21.0*(x[1]*x_4)**4.0), -1.0*x[2], -1.0*x_4 + 21.0*(x[1]*x_4)**4.0/(1.0*x[2]**4.0 + 21.0*(x[1]*x_4)**4.0)])

@register_eq_class
class BIOMD0000000866(KnownEquation):
    _eq_name = 'vars4_prog43'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Simon2019 - NIK-dependent p100 processing into p52 Michaelis-Menten SBML 2v4"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '-0.05*x[1]*x_4/(x[1] + 10.0) + 0.5', '0.05*x[1]*x_4/(x[1] + 10.0)', '0']
    
    def np_eq(self, t, x):
        return np.array([0, -0.05*x[1]*x_4/(x[1] + 10.0) + 0.5, 0.05*x[1]*x_4/(x[1] + 10.0), 0])

@register_eq_class
class BIOMD0000000756(KnownEquation):
    _eq_name = 'vars4_prog44'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Figueredo2013/3 - immunointeraction full model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.8e-11*x[0]**2 - 1.0*x[0]*x[1]/(x[0] + 1.0e+5) + 0.27*x[0]*x_4/(x_4 + 2.0e+7) + 0.18*x[0]', '0.035*x[0]/(10.0*x_4 + 1.0) - 1.24*x[1]*x[2]*x_4/(x[2]*x_4 + 0.112*x[2] + 2.0e+7*x_4 + 2.24e+6) + 0.01538*x[1]*x[2]/(x[2] + 2.0e+7) - 0.03*x[1]', '5.0*x[0]*x[1]/(0.001*x[0]*x_4 + 1.0*x[0] + 1.0*x_4 + 1000.0) - 10.0*x[2]', '2.84*x[0]**2/(x[0]**2 + 1.0e+12) - 10.0*x_4']
    
    def np_eq(self, t, x):
        return np.array([-1.8e-11*x[0]**2 - 1.0*x[0]*x[1]/(x[0] + 1.0e+5) + 0.27*x[0]*x_4/(x_4 + 2.0e+7) + 0.18*x[0], 0.035*x[0]/(10.0*x_4 + 1.0) - 1.24*x[1]*x[2]*x_4/(x[2]*x_4 + 0.112*x[2] + 2.0e+7*x_4 + 2.24e+6) + 0.01538*x[1]*x[2]/(x[2] + 2.0e+7) - 0.03*x[1], 5.0*x[0]*x[1]/(0.001*x[0]*x_4 + 1.0*x[0] + 1.0*x_4 + 1000.0) - 10.0*x[2], 2.84*x[0]**2/(x[0]**2 + 1.0e+12) - 10.0*x_4])

@register_eq_class
class BIOMD0000000874(KnownEquation):
    _eq_name = 'vars4_prog45'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Perelson1993 - HIVinfection-CD4Tcells-ModelA"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-2.0e-5*x[0]**2 - 2.0e-5*x[0]*x[1] - 2.0e-5*x[0]*x[2] + 0.01*x[0] + 10.0', '-0.023*x[1]', '0.003*x[1] - 0.24*x[2]', '240.0*x[2] - 2.4*x_4']
    
    def np_eq(self, t, x):
        return np.array([-2.0e-5*x[0]**2 - 2.0e-5*x[0]*x[1] - 2.0e-5*x[0]*x[2] + 0.01*x[0] + 10.0, -0.023*x[1], 0.003*x[1] - 0.24*x[2], 240.0*x[2] - 2.4*x_4])

@register_eq_class
class BIOMD0000000770(KnownEquation):
    _eq_name = 'vars4_prog46'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Eftimie2017/1 - interaction of Th and macrophage"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-9.0e-10*x[0]**2*x[2] - 9.0e-10*x[0]*x[1]*x[2] + 0.09*x[0]*x[2] - 0.03*x[0] + 0.008*x[2]', '-9.0e-10*x[0]*x[1]*x_4 - 9.0e-10*x[1]**2*x_4 + 0.09*x[1]*x_4 - 0.03*x[1] + 0.001*x_4', '0.001*x[0] - 2.0e-11*x[2]**2 - 2.0e-11*x[2]*x_4 - 0.05*x[2] + 0.09*x_4', '-2.0e-11*x[1]*x[2]*x_4 - 2.0e-11*x[1]*x_4**2 + 0.02*x[1]*x_4 + 0.001*x[1] + 0.05*x[2] - 0.11*x_4']
    
    def np_eq(self, t, x):
        return np.array([-9.0e-10*x[0]**2*x[2] - 9.0e-10*x[0]*x[1]*x[2] + 0.09*x[0]*x[2] - 0.03*x[0] + 0.008*x[2], -9.0e-10*x[0]*x[1]*x_4 - 9.0e-10*x[1]**2*x_4 + 0.09*x[1]*x_4 - 0.03*x[1] + 0.001*x_4, 0.001*x[0] - 2.0e-11*x[2]**2 - 2.0e-11*x[2]*x_4 - 0.05*x[2] + 0.09*x_4, -2.0e-11*x[1]*x[2]*x_4 - 2.0e-11*x[1]*x_4**2 + 0.02*x[1]*x_4 + 0.001*x[1] + 0.05*x[2] - 0.11*x_4])

@register_eq_class
class BIOMD0000000887(KnownEquation):
    _eq_name = 'vars4_prog47'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Lim2014 - HTLV-I infection A dynamic struggle between viral persistence and host immunity"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.001*x[0]*x[2] - 0.012*x[0] + 10.0', '0.001*x[0]*x[2] - 0.033*x[1] + 0.011*x[2]', '0.003*x[1] - 0.029*x[2]*x_4 - 0.03*x[2]', '0.036*x[2] - 0.03*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.001*x[0]*x[2] - 0.012*x[0] + 10.0, 0.001*x[0]*x[2] - 0.033*x[1] + 0.011*x[2], 0.003*x[1] - 0.029*x[2]*x_4 - 0.03*x[2], 0.036*x[2] - 0.03*x_4])

@register_eq_class
class BIOMD0000000233(KnownEquation):
    _eq_name = 'vars4_prog48'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Wilhelm2009-BistableReaction"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '0', '-1.0*x[2]**2 - 1.0*x[2]*x_4 - 1.5*x[2] + 16.0*x_4', '1.0*x[2]**2 - 8.0*x_4']
    
    def np_eq(self, t, x):
        return np.array([0, 0, -1.0*x[2]**2 - 1.0*x[2]*x_4 - 1.5*x[2] + 16.0*x_4, 1.0*x[2]**2 - 8.0*x_4])

@register_eq_class
class BIOMD0000000714(KnownEquation):
    _eq_name = 'vars4_prog49'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Reynolds2006 - Reduced model of the acute inflammatory response"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.015*x[0]**2 - 1.8*x[0]*x[1]/(12.76*x_4**2 + 1.0) + 0.3*x[0] - 0.003*x[0]/(0.01*x[0] + 0.002)', '0.008*x[0]/(1.276*x[0]*x_4**2/(12.76*x_4**2 + 1.0) + 0.1*x[0]/(12.76*x_4**2 + 1.0) + 0.1276*x[1]*x_4**2/(12.76*x_4**2 + 1.0) + 0.01*x[1]/(12.76*x_4**2 + 1.0) + 0.2551*x[2]*x_4**2/(12.76*x_4**2 + 1.0) + 0.02*x[2]/(12.76*x_4**2 + 1.0) + 1.531*x_4**2 + 0.12) - 0.05*x[1] + 0.0008*x[1]/(1.276*x[0]*x_4**2/(12.76*x_4**2 + 1.0) + 0.1*x[0]/(12.76*x_4**2 + 1.0) + 0.1276*x[1]*x_4**2/(12.76*x_4**2 + 1.0) + 0.01*x[1]/(12.76*x_4**2 + 1.0) + 0.2551*x[2]*x_4**2/(12.76*x_4**2 + 1.0) + 0.02*x[2]/(12.76*x_4**2 + 1.0) + 1.531*x_4**2 + 0.12) + 0.0016*x[2]/(1.276*x[0]*x_4**2/(12.76*x_4**2 + 1.0) + 0.1*x[0]/(12.76*x_4**2 + 1.0) + 0.1276*x[1]*x_4**2/(12.76*x_4**2 + 1.0) + 0.01*x[1]/(12.76*x_4**2 + 1.0) + 0.2551*x[2]*x_4**2/(12.76*x_4**2 + 1.0) + 0.02*x[2]/(12.76*x_4**2 + 1.0) + 1.531*x_4**2 + 0.12)', '8.128e-8*x[1]**6/(2.322e-7*x[1]**6*x_4**12/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 1.092e-7*x[1]**6*x_4**10/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 2.141e-8*x[1]**6*x_4**8/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 2.238e-9*x[1]**6*x_4**6/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 1.316e-10*x[1]**6*x_4**4/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 4.127e-12*x[1]**6*x_4**2/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 5.392e-14*x[1]**6/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 4.666e-8*x_4**12 + 2.195e-8*x_4**10 + 4.302e-9*x_4**8 + 4.497e-10*x_4**6 + 2.644e-11*x_4**4 + 8.291e-13*x_4**2 + 1.083e-14) - 0.02*x[2]', '0.04*x[1]/(12.76*x[1]*x_4**2/(12.76*x_4**2 + 1.0) + 1.0*x[1]/(12.76*x_4**2 + 1.0) + 612.3*x[2]*x_4**2/(12.76*x_4**2 + 1.0) + 48.0*x[2]/(12.76*x_4**2 + 1.0) + 12.76*x_4**2 + 1.0) + 1.92*x[2]/(12.76*x[1]*x_4**2/(12.76*x_4**2 + 1.0) + 1.0*x[1]/(12.76*x_4**2 + 1.0) + 612.3*x[2]*x_4**2/(12.76*x_4**2 + 1.0) + 48.0*x[2]/(12.76*x_4**2 + 1.0) + 12.76*x_4**2 + 1.0) - 0.1*x_4 + 0.013']
    
    def np_eq(self, t, x):
        return np.array([-0.015*x[0]**2 - 1.8*x[0]*x[1]/(12.76*x_4**2 + 1.0) + 0.3*x[0] - 0.003*x[0]/(0.01*x[0] + 0.002), 0.008*x[0]/(1.276*x[0]*x_4**2/(12.76*x_4**2 + 1.0) + 0.1*x[0]/(12.76*x_4**2 + 1.0) + 0.1276*x[1]*x_4**2/(12.76*x_4**2 + 1.0) + 0.01*x[1]/(12.76*x_4**2 + 1.0) + 0.2551*x[2]*x_4**2/(12.76*x_4**2 + 1.0) + 0.02*x[2]/(12.76*x_4**2 + 1.0) + 1.531*x_4**2 + 0.12) - 0.05*x[1] + 0.0008*x[1]/(1.276*x[0]*x_4**2/(12.76*x_4**2 + 1.0) + 0.1*x[0]/(12.76*x_4**2 + 1.0) + 0.1276*x[1]*x_4**2/(12.76*x_4**2 + 1.0) + 0.01*x[1]/(12.76*x_4**2 + 1.0) + 0.2551*x[2]*x_4**2/(12.76*x_4**2 + 1.0) + 0.02*x[2]/(12.76*x_4**2 + 1.0) + 1.531*x_4**2 + 0.12) + 0.0016*x[2]/(1.276*x[0]*x_4**2/(12.76*x_4**2 + 1.0) + 0.1*x[0]/(12.76*x_4**2 + 1.0) + 0.1276*x[1]*x_4**2/(12.76*x_4**2 + 1.0) + 0.01*x[1]/(12.76*x_4**2 + 1.0) + 0.2551*x[2]*x_4**2/(12.76*x_4**2 + 1.0) + 0.02*x[2]/(12.76*x_4**2 + 1.0) + 1.531*x_4**2 + 0.12), 8.128e-8*x[1]**6/(2.322e-7*x[1]**6*x_4**12/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 1.092e-7*x[1]**6*x_4**10/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 2.141e-8*x[1]**6*x_4**8/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 2.238e-9*x[1]**6*x_4**6/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 1.316e-10*x[1]**6*x_4**4/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 4.127e-12*x[1]**6*x_4**2/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 5.392e-14*x[1]**6/(x_4**12 + 0.4704*x_4**10 + 0.0922*x_4**8 + 0.009638*x_4**6 + 0.0005667*x_4**4 + 1.777e-5*x_4**2 + 2.322e-7) + 4.666e-8*x_4**12 + 2.195e-8*x_4**10 + 4.302e-9*x_4**8 + 4.497e-10*x_4**6 + 2.644e-11*x_4**4 + 8.291e-13*x_4**2 + 1.083e-14) - 0.02*x[2], 0.04*x[1]/(12.76*x[1]*x_4**2/(12.76*x_4**2 + 1.0) + 1.0*x[1]/(12.76*x_4**2 + 1.0) + 612.3*x[2]*x_4**2/(12.76*x_4**2 + 1.0) + 48.0*x[2]/(12.76*x_4**2 + 1.0) + 12.76*x_4**2 + 1.0) + 1.92*x[2]/(12.76*x[1]*x_4**2/(12.76*x_4**2 + 1.0) + 1.0*x[1]/(12.76*x_4**2 + 1.0) + 612.3*x[2]*x_4**2/(12.76*x_4**2 + 1.0) + 48.0*x[2]/(12.76*x_4**2 + 1.0) + 12.76*x_4**2 + 1.0) - 0.1*x_4 + 0.013])

@register_eq_class
class BIOMD0000000855(KnownEquation):
    _eq_name = 'vars4_prog50'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Cooper2015 - Modeling the effects of systemic mediators on the inflammatory phase of wound healing"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.72*x[0]**2 - 34.8*x[0]*x[2]/(82.64*x_4**2 + 1.0) - 35.03*x[0]*x_4 + 14.4*x[0] - 1.728*x[0]/(0.2*x[0] + 0.048)', '-3.16*x[1]*x[2]/(82.64*x_4**2 + 1.0) - 10.9*x[1]*x_4**2 - 2.03*x[1]*x_4 - 0.37*x[1] + 1.02*x_4', '3.148*x[0]/(18.52*x[0] + 41.5*x[1] + 0.06*x[2] + 10.58*x_4 + 0.54) + 7.055*x[1]/(18.52*x[0] + 41.5*x[1] + 0.06*x[2] + 10.58*x_4 + 0.54) - 0.5*x[2] + 0.0102*x[2]/(18.52*x[0] + 41.5*x[1] + 0.06*x[2] + 10.58*x_4 + 0.54) + 1.799*x_4/(18.52*x[0] + 41.5*x[1] + 0.06*x[2] + 10.58*x_4 + 0.54)', '6.68*x[0]/(17.58*x[0] + 2.97*x[1] + 3.3) + 1.129*x[1]/(17.58*x[0] + 2.97*x[1] + 3.3) - 6.42*x[2]*x_4/(82.64*x_4**2 + 1.0) - 1.02*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.72*x[0]**2 - 34.8*x[0]*x[2]/(82.64*x_4**2 + 1.0) - 35.03*x[0]*x_4 + 14.4*x[0] - 1.728*x[0]/(0.2*x[0] + 0.048), -3.16*x[1]*x[2]/(82.64*x_4**2 + 1.0) - 10.9*x[1]*x_4**2 - 2.03*x[1]*x_4 - 0.37*x[1] + 1.02*x_4, 3.148*x[0]/(18.52*x[0] + 41.5*x[1] + 0.06*x[2] + 10.58*x_4 + 0.54) + 7.055*x[1]/(18.52*x[0] + 41.5*x[1] + 0.06*x[2] + 10.58*x_4 + 0.54) - 0.5*x[2] + 0.0102*x[2]/(18.52*x[0] + 41.5*x[1] + 0.06*x[2] + 10.58*x_4 + 0.54) + 1.799*x_4/(18.52*x[0] + 41.5*x[1] + 0.06*x[2] + 10.58*x_4 + 0.54), 6.68*x[0]/(17.58*x[0] + 2.97*x[1] + 3.3) + 1.129*x[1]/(17.58*x[0] + 2.97*x[1] + 3.3) - 6.42*x[2]*x_4/(82.64*x_4**2 + 1.0) - 1.02*x_4])

@register_eq_class
class BIOMD0000000642(KnownEquation):
    _eq_name = 'vars4_prog51'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Mufudza2012 - Estrogen effect on the dynamics of breast cancer"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.3*x[0]**2 - 1.0*x[0]*x[1] + 0.1*x[0]', '0.47*x[0] - 0.4*x[1]**2 - 0.9*x[1]*x[2] + 1.0*x[1]', '-0.085*x[1]*x[2] + 0.2*x[1]*x[2]/(x[1] + 0.3) - 0.4567*x[2] + 0.4', '0']
    
    def np_eq(self, t, x):
        return np.array([-0.3*x[0]**2 - 1.0*x[0]*x[1] + 0.1*x[0], 0.47*x[0] - 0.4*x[1]**2 - 0.9*x[1]*x[2] + 1.0*x[1], -0.085*x[1]*x[2] + 0.2*x[1]*x[2]/(x[1] + 0.3) - 0.4567*x[2] + 0.4, 0])

@register_eq_class
class BIOMD0000000058(KnownEquation):
    _eq_name = 'vars4_prog52'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Bindschadler2001-coupled-Ca-oscillators"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.003443*x[0]**4*x[1]**4/(9.488e-8*x[0]**8/(0.0007716*x[0]**4 + 0.01852*x[0]**3 + 0.1667*x[0]**2 + 0.6667*x[0] + 1) + 3.604e-6*x[0]**7/(0.00463*x[0]**4 + 0.3148*x[0]**3 + 4.667*x[0]**2 + 26.0*x[0] + 50.0) + 2.277e-6*x[0]**7/(0.0007716*x[0]**4 + 0.01852*x[0]**3 + 0.1667*x[0]**2 + 0.6667*x[0] + 1) + 8.649e-5*x[0]**6/(0.00463*x[0]**4 + 0.3148*x[0]**3 + 4.667*x[0]**2 + 26.0*x[0] + 50.0) + 2.049e-5*x[0]**6/(0.0007716*x[0]**4 + 0.01852*x[0]**3 + 0.1667*x[0]**2 + 0.6667*x[0] + 1) + 2.053e-8*x[0]**6/(1.111e-5*x[0]**4 + 0.001244*x[0]**3 + 0.04151*x[0]**2 + 0.3733*x[0] + 1) + 0.0007785*x[0]**5/(0.00463*x[0]**4 + 0.3148*x[0]**3 + 4.667*x[0]**2 + 26.0*x[0] + 50.0) + 8.198e-5*x[0]**5/(0.0007716*x[0]**4 + 0.01852*x[0]**3 + 0.1667*x[0]**2 + 0.6667*x[0] + 1) + 4.928e-7*x[0]**5/(1.111e-5*x[0]**4 + 0.001244*x[0]**3 + 0.04151*x[0]**2 + 0.3733*x[0] + 1) + 1.56e-8*x[0]**5/(8.0e-6*x[0]**4 + 0.001248*x[0]**3 + 0.0672*x[0]**2 + 1.36*x[0] + 6.0) + 0.003114*x[0]**4/(0.00463*x[0]**4 + 0.3148*x[0]**3 + 4.667*x[0]**2 + 26.0*x[0] + 50.0) + 0.000123*x[0]**4/(0.0007716*x[0]**4 + 0.01852*x[0]**3 + 0.1667*x[0]**2 + 0.6667*x[0] + 1) + 4.435e-6*x[0]**4/(1.111e-5*x[0]**4 + 0.001244*x[0]**3 + 0.04151*x[0]**2 + 0.3733*x[0] + 1) + 3.744e-7*x[0]**4/(8.0e-6*x[0]**4 + 0.001248*x[0]**3 + 0.0672*x[0]**2 + 1.36*x[0] + 6.0) + 1.235e-10*x[0]**4/(1.6e-7*x[0]**4 + 3.2e-5*x[0]**3 + 0.0024*x[0]**2 + 0.08*x[0] + 1) + 0.004671*x[0]**3/(0.00463*x[0]**4 + 0.3148*x[0]**3 + 4.667*x[0]**2 + 26.0*x[0] + 50.0) + 1.774e-5*x[0]**3/(1.111e-5*x[0]**4 + 0.001244*x[0]**3 + 0.04151*x[0]**2 + 0.3733*x[0] + 1) + 3.37e-6*x[0]**3/(8.0e-6*x[0]**4 + 0.001248*x[0]**3 + 0.0672*x[0]**2 + 1.36*x[0] + 6.0) + 2.963e-9*x[0]**3/(1.6e-7*x[0]**4 + 3.2e-5*x[0]**3 + 0.0024*x[0]**2 + 0.08*x[0] + 1) + 2.661e-5*x[0]**2/(1.111e-5*x[0]**4 + 0.001244*x[0]**3 + 0.04151*x[0]**2 + 0.3733*x[0] + 1) + 1.348e-5*x[0]**2/(8.0e-6*x[0]**4 + 0.001248*x[0]**3 + 0.0672*x[0]**2 + 1.36*x[0] + 6.0) + 2.667e-8*x[0]**2/(1.6e-7*x[0]**4 + 3.2e-5*x[0]**3 + 0.0024*x[0]**2 + 0.08*x[0] + 1) + 2.022e-5*x[0]/(8.0e-6*x[0]**4 + 0.001248*x[0]**3 + 0.0672*x[0]**2 + 1.36*x[0] + 6.0) + 1.067e-7*x[0]/(1.6e-7*x[0]**4 + 3.2e-5*x[0]**3 + 0.0024*x[0]**2 + 0.08*x[0] + 1) + 1.6e-7/(1.6e-7*x[0]**4 + 3.2e-5*x[0]**3 + 0.0024*x[0]**2 + 0.08*x[0] + 1)) - 1.2*x[0]**2/(x[0]**2 + 0.0324) - 0.01*x[0] + 0.01*x[2] + 0.2', '-556.0*x[0]**2*x[1]/(27.8*x[0]**3/(x[0] + 6.0) + 44.0*x[0]**2/(x[0] + 50.0) + 1557.0*x[0]**2/(x[0] + 6.0) + 2464.0*x[0]/(x[0] + 50.0) + 8340.0*x[0]/(x[0] + 6.0) + 1.32e+4/(x[0] + 50.0)) - 736.7*x[0]*x[1]/(27.8*x[0]**3/(x[0] + 6.0) + 44.0*x[0]**2/(x[0] + 50.0) + 1557.0*x[0]**2/(x[0] + 6.0) + 2464.0*x[0]/(x[0] + 50.0) + 8340.0*x[0]/(x[0] + 6.0) + 1.32e+4/(x[0] + 50.0)) - 1.6*x[1]/(x[0] + 1.6) + 1.6/(x[0] + 1.6)', '0.01*x[0] + 0.003443*x[2]**4*x_4**4/(9.488e-8*x[2]**8/(0.0007716*x[2]**4 + 0.01852*x[2]**3 + 0.1667*x[2]**2 + 0.6667*x[2] + 1) + 3.604e-6*x[2]**7/(0.00463*x[2]**4 + 0.3148*x[2]**3 + 4.667*x[2]**2 + 26.0*x[2] + 50.0) + 2.277e-6*x[2]**7/(0.0007716*x[2]**4 + 0.01852*x[2]**3 + 0.1667*x[2]**2 + 0.6667*x[2] + 1) + 8.649e-5*x[2]**6/(0.00463*x[2]**4 + 0.3148*x[2]**3 + 4.667*x[2]**2 + 26.0*x[2] + 50.0) + 2.049e-5*x[2]**6/(0.0007716*x[2]**4 + 0.01852*x[2]**3 + 0.1667*x[2]**2 + 0.6667*x[2] + 1) + 2.053e-8*x[2]**6/(1.111e-5*x[2]**4 + 0.001244*x[2]**3 + 0.04151*x[2]**2 + 0.3733*x[2] + 1) + 0.0007785*x[2]**5/(0.00463*x[2]**4 + 0.3148*x[2]**3 + 4.667*x[2]**2 + 26.0*x[2] + 50.0) + 8.198e-5*x[2]**5/(0.0007716*x[2]**4 + 0.01852*x[2]**3 + 0.1667*x[2]**2 + 0.6667*x[2] + 1) + 4.928e-7*x[2]**5/(1.111e-5*x[2]**4 + 0.001244*x[2]**3 + 0.04151*x[2]**2 + 0.3733*x[2] + 1) + 1.56e-8*x[2]**5/(8.0e-6*x[2]**4 + 0.001248*x[2]**3 + 0.0672*x[2]**2 + 1.36*x[2] + 6.0) + 0.003114*x[2]**4/(0.00463*x[2]**4 + 0.3148*x[2]**3 + 4.667*x[2]**2 + 26.0*x[2] + 50.0) + 0.000123*x[2]**4/(0.0007716*x[2]**4 + 0.01852*x[2]**3 + 0.1667*x[2]**2 + 0.6667*x[2] + 1) + 4.435e-6*x[2]**4/(1.111e-5*x[2]**4 + 0.001244*x[2]**3 + 0.04151*x[2]**2 + 0.3733*x[2] + 1) + 3.744e-7*x[2]**4/(8.0e-6*x[2]**4 + 0.001248*x[2]**3 + 0.0672*x[2]**2 + 1.36*x[2] + 6.0) + 1.235e-10*x[2]**4/(1.6e-7*x[2]**4 + 3.2e-5*x[2]**3 + 0.0024*x[2]**2 + 0.08*x[2] + 1) + 0.004671*x[2]**3/(0.00463*x[2]**4 + 0.3148*x[2]**3 + 4.667*x[2]**2 + 26.0*x[2] + 50.0) + 1.774e-5*x[2]**3/(1.111e-5*x[2]**4 + 0.001244*x[2]**3 + 0.04151*x[2]**2 + 0.3733*x[2] + 1) + 3.37e-6*x[2]**3/(8.0e-6*x[2]**4 + 0.001248*x[2]**3 + 0.0672*x[2]**2 + 1.36*x[2] + 6.0) + 2.963e-9*x[2]**3/(1.6e-7*x[2]**4 + 3.2e-5*x[2]**3 + 0.0024*x[2]**2 + 0.08*x[2] + 1) + 2.661e-5*x[2]**2/(1.111e-5*x[2]**4 + 0.001244*x[2]**3 + 0.04151*x[2]**2 + 0.3733*x[2] + 1) + 1.348e-5*x[2]**2/(8.0e-6*x[2]**4 + 0.001248*x[2]**3 + 0.0672*x[2]**2 + 1.36*x[2] + 6.0) + 2.667e-8*x[2]**2/(1.6e-7*x[2]**4 + 3.2e-5*x[2]**3 + 0.0024*x[2]**2 + 0.08*x[2] + 1) + 2.022e-5*x[2]/(8.0e-6*x[2]**4 + 0.001248*x[2]**3 + 0.0672*x[2]**2 + 1.36*x[2] + 6.0) + 1.067e-7*x[2]/(1.6e-7*x[2]**4 + 3.2e-5*x[2]**3 + 0.0024*x[2]**2 + 0.08*x[2] + 1) + 1.6e-7/(1.6e-7*x[2]**4 + 3.2e-5*x[2]**3 + 0.0024*x[2]**2 + 0.08*x[2] + 1)) - 1.2*x[2]**2/(x[2]**2 + 0.0324) - 0.01*x[2] + 0.2', '-556.0*x[2]**2*x_4/(27.8*x[2]**3/(x[2] + 6.0) + 44.0*x[2]**2/(x[2] + 50.0) + 1557.0*x[2]**2/(x[2] + 6.0) + 2464.0*x[2]/(x[2] + 50.0) + 8340.0*x[2]/(x[2] + 6.0) + 1.32e+4/(x[2] + 50.0)) - 736.7*x[2]*x_4/(27.8*x[2]**3/(x[2] + 6.0) + 44.0*x[2]**2/(x[2] + 50.0) + 1557.0*x[2]**2/(x[2] + 6.0) + 2464.0*x[2]/(x[2] + 50.0) + 8340.0*x[2]/(x[2] + 6.0) + 1.32e+4/(x[2] + 50.0)) - 1.6*x_4/(x[2] + 1.6) + 1.6/(x[2] + 1.6)']
    
    def np_eq(self, t, x):
        return np.array([0.003443*x[0]**4*x[1]**4/(9.488e-8*x[0]**8/(0.0007716*x[0]**4 + 0.01852*x[0]**3 + 0.1667*x[0]**2 + 0.6667*x[0] + 1) + 3.604e-6*x[0]**7/(0.00463*x[0]**4 + 0.3148*x[0]**3 + 4.667*x[0]**2 + 26.0*x[0] + 50.0) + 2.277e-6*x[0]**7/(0.0007716*x[0]**4 + 0.01852*x[0]**3 + 0.1667*x[0]**2 + 0.6667*x[0] + 1) + 8.649e-5*x[0]**6/(0.00463*x[0]**4 + 0.3148*x[0]**3 + 4.667*x[0]**2 + 26.0*x[0] + 50.0) + 2.049e-5*x[0]**6/(0.0007716*x[0]**4 + 0.01852*x[0]**3 + 0.1667*x[0]**2 + 0.6667*x[0] + 1) + 2.053e-8*x[0]**6/(1.111e-5*x[0]**4 + 0.001244*x[0]**3 + 0.04151*x[0]**2 + 0.3733*x[0] + 1) + 0.0007785*x[0]**5/(0.00463*x[0]**4 + 0.3148*x[0]**3 + 4.667*x[0]**2 + 26.0*x[0] + 50.0) + 8.198e-5*x[0]**5/(0.0007716*x[0]**4 + 0.01852*x[0]**3 + 0.1667*x[0]**2 + 0.6667*x[0] + 1) + 4.928e-7*x[0]**5/(1.111e-5*x[0]**4 + 0.001244*x[0]**3 + 0.04151*x[0]**2 + 0.3733*x[0] + 1) + 1.56e-8*x[0]**5/(8.0e-6*x[0]**4 + 0.001248*x[0]**3 + 0.0672*x[0]**2 + 1.36*x[0] + 6.0) + 0.003114*x[0]**4/(0.00463*x[0]**4 + 0.3148*x[0]**3 + 4.667*x[0]**2 + 26.0*x[0] + 50.0) + 0.000123*x[0]**4/(0.0007716*x[0]**4 + 0.01852*x[0]**3 + 0.1667*x[0]**2 + 0.6667*x[0] + 1) + 4.435e-6*x[0]**4/(1.111e-5*x[0]**4 + 0.001244*x[0]**3 + 0.04151*x[0]**2 + 0.3733*x[0] + 1) + 3.744e-7*x[0]**4/(8.0e-6*x[0]**4 + 0.001248*x[0]**3 + 0.0672*x[0]**2 + 1.36*x[0] + 6.0) + 1.235e-10*x[0]**4/(1.6e-7*x[0]**4 + 3.2e-5*x[0]**3 + 0.0024*x[0]**2 + 0.08*x[0] + 1) + 0.004671*x[0]**3/(0.00463*x[0]**4 + 0.3148*x[0]**3 + 4.667*x[0]**2 + 26.0*x[0] + 50.0) + 1.774e-5*x[0]**3/(1.111e-5*x[0]**4 + 0.001244*x[0]**3 + 0.04151*x[0]**2 + 0.3733*x[0] + 1) + 3.37e-6*x[0]**3/(8.0e-6*x[0]**4 + 0.001248*x[0]**3 + 0.0672*x[0]**2 + 1.36*x[0] + 6.0) + 2.963e-9*x[0]**3/(1.6e-7*x[0]**4 + 3.2e-5*x[0]**3 + 0.0024*x[0]**2 + 0.08*x[0] + 1) + 2.661e-5*x[0]**2/(1.111e-5*x[0]**4 + 0.001244*x[0]**3 + 0.04151*x[0]**2 + 0.3733*x[0] + 1) + 1.348e-5*x[0]**2/(8.0e-6*x[0]**4 + 0.001248*x[0]**3 + 0.0672*x[0]**2 + 1.36*x[0] + 6.0) + 2.667e-8*x[0]**2/(1.6e-7*x[0]**4 + 3.2e-5*x[0]**3 + 0.0024*x[0]**2 + 0.08*x[0] + 1) + 2.022e-5*x[0]/(8.0e-6*x[0]**4 + 0.001248*x[0]**3 + 0.0672*x[0]**2 + 1.36*x[0] + 6.0) + 1.067e-7*x[0]/(1.6e-7*x[0]**4 + 3.2e-5*x[0]**3 + 0.0024*x[0]**2 + 0.08*x[0] + 1) + 1.6e-7/(1.6e-7*x[0]**4 + 3.2e-5*x[0]**3 + 0.0024*x[0]**2 + 0.08*x[0] + 1)) - 1.2*x[0]**2/(x[0]**2 + 0.0324) - 0.01*x[0] + 0.01*x[2] + 0.2, -556.0*x[0]**2*x[1]/(27.8*x[0]**3/(x[0] + 6.0) + 44.0*x[0]**2/(x[0] + 50.0) + 1557.0*x[0]**2/(x[0] + 6.0) + 2464.0*x[0]/(x[0] + 50.0) + 8340.0*x[0]/(x[0] + 6.0) + 1.32e+4/(x[0] + 50.0)) - 736.7*x[0]*x[1]/(27.8*x[0]**3/(x[0] + 6.0) + 44.0*x[0]**2/(x[0] + 50.0) + 1557.0*x[0]**2/(x[0] + 6.0) + 2464.0*x[0]/(x[0] + 50.0) + 8340.0*x[0]/(x[0] + 6.0) + 1.32e+4/(x[0] + 50.0)) - 1.6*x[1]/(x[0] + 1.6) + 1.6/(x[0] + 1.6), 0.01*x[0] + 0.003443*x[2]**4*x_4**4/(9.488e-8*x[2]**8/(0.0007716*x[2]**4 + 0.01852*x[2]**3 + 0.1667*x[2]**2 + 0.6667*x[2] + 1) + 3.604e-6*x[2]**7/(0.00463*x[2]**4 + 0.3148*x[2]**3 + 4.667*x[2]**2 + 26.0*x[2] + 50.0) + 2.277e-6*x[2]**7/(0.0007716*x[2]**4 + 0.01852*x[2]**3 + 0.1667*x[2]**2 + 0.6667*x[2] + 1) + 8.649e-5*x[2]**6/(0.00463*x[2]**4 + 0.3148*x[2]**3 + 4.667*x[2]**2 + 26.0*x[2] + 50.0) + 2.049e-5*x[2]**6/(0.0007716*x[2]**4 + 0.01852*x[2]**3 + 0.1667*x[2]**2 + 0.6667*x[2] + 1) + 2.053e-8*x[2]**6/(1.111e-5*x[2]**4 + 0.001244*x[2]**3 + 0.04151*x[2]**2 + 0.3733*x[2] + 1) + 0.0007785*x[2]**5/(0.00463*x[2]**4 + 0.3148*x[2]**3 + 4.667*x[2]**2 + 26.0*x[2] + 50.0) + 8.198e-5*x[2]**5/(0.0007716*x[2]**4 + 0.01852*x[2]**3 + 0.1667*x[2]**2 + 0.6667*x[2] + 1) + 4.928e-7*x[2]**5/(1.111e-5*x[2]**4 + 0.001244*x[2]**3 + 0.04151*x[2]**2 + 0.3733*x[2] + 1) + 1.56e-8*x[2]**5/(8.0e-6*x[2]**4 + 0.001248*x[2]**3 + 0.0672*x[2]**2 + 1.36*x[2] + 6.0) + 0.003114*x[2]**4/(0.00463*x[2]**4 + 0.3148*x[2]**3 + 4.667*x[2]**2 + 26.0*x[2] + 50.0) + 0.000123*x[2]**4/(0.0007716*x[2]**4 + 0.01852*x[2]**3 + 0.1667*x[2]**2 + 0.6667*x[2] + 1) + 4.435e-6*x[2]**4/(1.111e-5*x[2]**4 + 0.001244*x[2]**3 + 0.04151*x[2]**2 + 0.3733*x[2] + 1) + 3.744e-7*x[2]**4/(8.0e-6*x[2]**4 + 0.001248*x[2]**3 + 0.0672*x[2]**2 + 1.36*x[2] + 6.0) + 1.235e-10*x[2]**4/(1.6e-7*x[2]**4 + 3.2e-5*x[2]**3 + 0.0024*x[2]**2 + 0.08*x[2] + 1) + 0.004671*x[2]**3/(0.00463*x[2]**4 + 0.3148*x[2]**3 + 4.667*x[2]**2 + 26.0*x[2] + 50.0) + 1.774e-5*x[2]**3/(1.111e-5*x[2]**4 + 0.001244*x[2]**3 + 0.04151*x[2]**2 + 0.3733*x[2] + 1) + 3.37e-6*x[2]**3/(8.0e-6*x[2]**4 + 0.001248*x[2]**3 + 0.0672*x[2]**2 + 1.36*x[2] + 6.0) + 2.963e-9*x[2]**3/(1.6e-7*x[2]**4 + 3.2e-5*x[2]**3 + 0.0024*x[2]**2 + 0.08*x[2] + 1) + 2.661e-5*x[2]**2/(1.111e-5*x[2]**4 + 0.001244*x[2]**3 + 0.04151*x[2]**2 + 0.3733*x[2] + 1) + 1.348e-5*x[2]**2/(8.0e-6*x[2]**4 + 0.001248*x[2]**3 + 0.0672*x[2]**2 + 1.36*x[2] + 6.0) + 2.667e-8*x[2]**2/(1.6e-7*x[2]**4 + 3.2e-5*x[2]**3 + 0.0024*x[2]**2 + 0.08*x[2] + 1) + 2.022e-5*x[2]/(8.0e-6*x[2]**4 + 0.001248*x[2]**3 + 0.0672*x[2]**2 + 1.36*x[2] + 6.0) + 1.067e-7*x[2]/(1.6e-7*x[2]**4 + 3.2e-5*x[2]**3 + 0.0024*x[2]**2 + 0.08*x[2] + 1) + 1.6e-7/(1.6e-7*x[2]**4 + 3.2e-5*x[2]**3 + 0.0024*x[2]**2 + 0.08*x[2] + 1)) - 1.2*x[2]**2/(x[2]**2 + 0.0324) - 0.01*x[2] + 0.2, -556.0*x[2]**2*x_4/(27.8*x[2]**3/(x[2] + 6.0) + 44.0*x[2]**2/(x[2] + 50.0) + 1557.0*x[2]**2/(x[2] + 6.0) + 2464.0*x[2]/(x[2] + 50.0) + 8340.0*x[2]/(x[2] + 6.0) + 1.32e+4/(x[2] + 50.0)) - 736.7*x[2]*x_4/(27.8*x[2]**3/(x[2] + 6.0) + 44.0*x[2]**2/(x[2] + 50.0) + 1557.0*x[2]**2/(x[2] + 6.0) + 2464.0*x[2]/(x[2] + 50.0) + 8340.0*x[2]/(x[2] + 6.0) + 1.32e+4/(x[2] + 50.0)) - 1.6*x_4/(x[2] + 1.6) + 1.6/(x[2] + 1.6)])

@register_eq_class
class BIOMD0000000902(KnownEquation):
    _eq_name = 'vars4_prog53'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Wang2019 - A mathematical model of oncolytic virotherapy with time delay"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-9.631e-5*x[0]**2 - 9.631e-5*x[0]*x[2] + 0.206*x[0]', '0', '-0.01*x[2]*x_4 - 1.0*x[2]', '0.02*x[2]*x_4 - 0.5*x_4']
    
    def np_eq(self, t, x):
        return np.array([-9.631e-5*x[0]**2 - 9.631e-5*x[0]*x[2] + 0.206*x[0], 0, -0.01*x[2]*x_4 - 1.0*x[2], 0.02*x[2]*x_4 - 0.5*x_4])

@register_eq_class
class BIOMD0000001034(KnownEquation):
    _eq_name = 'vars4_prog54'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Bunimovich-Mendrazitsky2007 - Mathematical model of BCG immunotherapy"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.25*x[0]*x[1] - 0.285*x[0]*x_4 - 1.0*x[0] + 1.9', '0.085*x[0]*x[1] - 0.003*x[1]*x[2] - 0.41*x[1] + 0.52*x[2]', '0.285*x[0]*x_4 - 1.1*x[1]*x[2]', '-0.285*x[0]*x_4 - 0.0018*x_4**2 + 0.12*x_4']
    
    def np_eq(self, t, x):
        return np.array([-1.25*x[0]*x[1] - 0.285*x[0]*x_4 - 1.0*x[0] + 1.9, 0.085*x[0]*x[1] - 0.003*x[1]*x[2] - 0.41*x[1] + 0.52*x[2], 0.285*x[0]*x_4 - 1.1*x[1]*x[2], -0.285*x[0]*x_4 - 0.0018*x_4**2 + 0.12*x_4])

@register_eq_class
class BIOMD0000000892(KnownEquation):
    _eq_name = 'vars4_prog55'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Sandip2013 - Modeling the dynamics of hepatitis C virus with combined antiviral drug therapy: interferon and ribavirin."
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-5.528e-8*x[0]**2 + 1.98*x[0] + 1.0', '-1.0*x[1]', '1.74*x[1] - 6.0*x[2]', '1.16*x[1] - 6.0*x_4']
    
    def np_eq(self, t, x):
        return np.array([-5.528e-8*x[0]**2 + 1.98*x[0] + 1.0, -1.0*x[1], 1.74*x[1] - 6.0*x[2], 1.16*x[1] - 6.0*x_4])

@register_eq_class
class BIOMD0000000275(KnownEquation):
    _eq_name = 'vars4_prog56'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Goldbeter2007-Somitogenesis-Switch"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0]*x[2] + 4.97', '-1.0*x[1] + 7.1*x_4**2.0/(x_4**2.0 + 0.04) + 0.365', '1.0*x[1] - 0.28*x[2]', '-1.0*x_4 + 0.04/(x[0]**2.0 + 0.04)']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0]*x[2] + 4.97, -1.0*x[1] + 7.1*x_4**2.0/(x_4**2.0 + 0.04) + 0.365, 1.0*x[1] - 0.28*x[2], -1.0*x_4 + 0.04/(x[0]**2.0 + 0.04)])

@register_eq_class
class BIOMD0000000322(KnownEquation):
    _eq_name = 'vars4_prog57'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Kim2011-Oscillator-SimpleI"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0] + 0.57*x[2] + 1.5*x_4', '-1.0*x[1] + 2.5*x_4', '-1.0*x[2] + 1.0/(x[1]**6.5 + 1.0)', '1.0*x[0]**6.5/(x[0]**6.5 + 1.0) - 1.0*x_4']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0] + 0.57*x[2] + 1.5*x_4, -1.0*x[1] + 2.5*x_4, -1.0*x[2] + 1.0/(x[1]**6.5 + 1.0), 1.0*x[0]**6.5/(x[0]**6.5 + 1.0) - 1.0*x_4])

@register_eq_class
class BIOMD0000000715(KnownEquation):
    _eq_name = 'vars4_prog58'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'exp']
    _description = "Huo2017 - SEIS epidemic model with the impact of media"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.8*x[0]*x[2]*np.exp(-0.08*x_4) - 0.6*x[0] + 0.7*x[2] + 0.8', '0.8*x[0]*x[2]*np.exp(-0.08*x_4) - 0.69*x[1]', '0.09*x[1] - 1.32*x[2]', '0.99*x[0] + 0.4*x[1] + 0.8*x[2] - 0.6*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.8*x[0]*x[2]*np.exp(-0.08*x_4) - 0.6*x[0] + 0.7*x[2] + 0.8, 0.8*x[0]*x[2]*np.exp(-0.08*x_4) - 0.69*x[1], 0.09*x[1] - 1.32*x[2], 0.99*x[0] + 0.4*x[1] + 0.8*x[2] - 0.6*x_4])

@register_eq_class
class BIOMD0000000930(KnownEquation):
    _eq_name = 'vars4_prog59'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Liu2017 - chemotherapy targeted model of tumor immune system"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.8*x[0]*x_4 + 0.431*x[0]', '0.015*x[0]*x[1]/(x[0] + 20.2) - 0.6*x[1]*x_4 - 0.041*x[1] + 1.2e+4', '-0.6*x[2]*x_4 - 0.012*x[2] + 7.5e+8', '0.45 - 0.9*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.8*x[0]*x_4 + 0.431*x[0], 0.015*x[0]*x[1]/(x[0] + 20.2) - 0.6*x[1]*x_4 - 0.041*x[1] + 1.2e+4, -0.6*x[2]*x_4 - 0.012*x[2] + 7.5e+8, 0.45 - 0.9*x_4])

@register_eq_class
class BIOMD0000001014(KnownEquation):
    _eq_name = 'vars4_prog60'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Leon-Triana2021 - Competition between tumour cells and dual-target CAR T-cells"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0.9*x[0]*x[1]/(x[1] + 1.0e+10) - 0.02*x[0]*x[1]/(x[0] + 2.0e+9) - 0.1429*x[0] + 0.2*x[2]', '0.02*x[1]', '0.9*x[2]*x_4/(x_4 + 1.0e+10) - 0.3429*x[2]', '-0.01667*x_4']
    
    def np_eq(self, t, x):
        return np.array([0.9*x[0]*x[1]/(x[1] + 1.0e+10) - 0.02*x[0]*x[1]/(x[0] + 2.0e+9) - 0.1429*x[0] + 0.2*x[2], 0.02*x[1], 0.9*x[2]*x_4/(x_4 + 1.0e+10) - 0.3429*x[2], -0.01667*x_4])

@register_eq_class
class BIOMD0000000876(KnownEquation):
    _eq_name = 'vars4_prog61'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Aavani2019 - The role of CD4 T cells in immune system activation and viral reproduction in a simple model for HIV infection"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['10.0 - 0.01*x[0]', '-1.0*x[1]*x[2] - 1.0*x[1]', '0.001*x[0]*x[2]*x_4/(x[2] + 1000.0) - 0.1*x[2]', '2000.0*x[1] - 23.0*x_4']
    
    def np_eq(self, t, x):
        return np.array([10.0 - 0.01*x[0], -1.0*x[1]*x[2] - 1.0*x[1], 0.001*x[0]*x[2]*x_4/(x[2] + 1000.0) - 0.1*x[2], 2000.0*x[1] - 23.0*x_4])

@register_eq_class
class BIOMD0000000875(KnownEquation):
    _eq_name = 'vars4_prog62'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Nelson2000- HIV-1 general model 1"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['10.0 - 0.03*x[0]', '-0.5*x[1]', '120.0*x[1] - 3.0*x[2]', '120.0*x[1] - 3.0*x_4']
    
    def np_eq(self, t, x):
        return np.array([10.0 - 0.03*x[0], -0.5*x[1], 120.0*x[1] - 3.0*x[2], 120.0*x[1] - 3.0*x_4])

@register_eq_class
class BIOMD0000000283(KnownEquation):
    _eq_name = 'vars4_prog63'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Chance1943-Peroxidase-ES-Kinetics"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0]*x[1]', '-1.0*x[0]*x[1] + 0.5*x[2]', '1.0*x[0]*x[1] - 0.5*x[2]', '0.5*x[2]']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0]*x[1], -1.0*x[0]*x[1] + 0.5*x[2], 1.0*x[0]*x[1] - 0.5*x[2], 0.5*x[2]])

@register_eq_class
class BIOMD0000000224(KnownEquation):
    _eq_name = 'vars4_prog64'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Meyer1991-CalciumSpike-ICC"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.004*x[0]**2/(0.0001*x[0]**2 + 0.0225) - 20.0*x[1]**4*x[2]*x_4/(x[1]**4 + 4.0*x[1]**3 + 6.0*x[1]**2 + 4.0*x[1] + 1.0) + 20.0*x[1]**4*x[2]/(x[1]**4 + 4.0*x[1]**3 + 6.0*x[1]**2 + 4.0*x[1] + 1.0) - 0.01*x[2]*x_4 + 0.01*x[2]', '-1.0*x[1] + 1.1 - 1.009/(0.01*x[0] + 1.0)', '0.004*x[0]**2/(0.0001*x[0]**2 + 0.0225) + 20.0*x[1]**4*x[2]*x_4/(x[1]**4 + 4.0*x[1]**3 + 6.0*x[1]**2 + 4.0*x[1] + 1.0) - 20.0*x[1]**4*x[2]/(x[1]**4 + 4.0*x[1]**3 + 6.0*x[1]**2 + 4.0*x[1] + 1.0) + 0.01*x[2]*x_4 - 0.01*x[2]', '-1.0e-8*x[0]**4*x_4 + 1.0e-8*x[0]**4 - 0.02']
    
    def np_eq(self, t, x):
        return np.array([-0.004*x[0]**2/(0.0001*x[0]**2 + 0.0225) - 20.0*x[1]**4*x[2]*x_4/(x[1]**4 + 4.0*x[1]**3 + 6.0*x[1]**2 + 4.0*x[1] + 1.0) + 20.0*x[1]**4*x[2]/(x[1]**4 + 4.0*x[1]**3 + 6.0*x[1]**2 + 4.0*x[1] + 1.0) - 0.01*x[2]*x_4 + 0.01*x[2], -1.0*x[1] + 1.1 - 1.009/(0.01*x[0] + 1.0), 0.004*x[0]**2/(0.0001*x[0]**2 + 0.0225) + 20.0*x[1]**4*x[2]*x_4/(x[1]**4 + 4.0*x[1]**3 + 6.0*x[1]**2 + 4.0*x[1] + 1.0) - 20.0*x[1]**4*x[2]/(x[1]**4 + 4.0*x[1]**3 + 6.0*x[1]**2 + 4.0*x[1] + 1.0) + 0.01*x[2]*x_4 - 0.01*x[2], -1.0e-8*x[0]**4*x_4 + 1.0e-8*x[0]**4 - 0.02])

@register_eq_class
class BIOMD0000000854(KnownEquation):
    _eq_name = 'vars4_prog65'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Gray2016 - The Akt switch model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.0077*x[0] + 0.35*x[1] + 0.55*x[2]', '-0.3577*x[1] + 0.55*x_4', '0.0077*x[0] - 1.32*x[2] + 0.35*x_4', '0.0077*x[1] + 0.77*x[2] - 0.9*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.0077*x[0] + 0.35*x[1] + 0.55*x[2], -0.3577*x[1] + 0.55*x_4, 0.0077*x[0] - 1.32*x[2] + 0.35*x_4, 0.0077*x[1] + 0.77*x[2] - 0.9*x_4])

@register_eq_class
class BIOMD0000000847(KnownEquation):
    _eq_name = 'vars4_prog66'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Adams2019 - The regulatory role of shikimate in plant phenylalanine metabolism"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-100.0*x[0]/(x[0] + 0.1*x[1] + 0.1) - 2.0*x[0]/(x[0] + 100.0) + 1.5*x[2]/(x[2] + 100.0) + 25.0', '100.0*x[0]/(x[0] + 0.1*x[1] + 0.1) - 75.0*x[1]*x[2]/(x[1]*x[2] + 0.1*x[1] + 1.0*x[2] + 0.1) - 5.0*x[1]/(x[1] + 1.0)', '2.0*x[0]/(x[0] + 100.0) - 75.0*x[1]*x[2]/(x[1]*x[2] + 0.1*x[1] + 1.0*x[2] + 0.1) - 1.5*x[2]/(x[2] + 100.0) + 75.0*x_4/(x_4 + 1.0)', '75.0*x[1]*x[2]/(x[1]*x[2] + 0.1*x[1] + 1.0*x[2] + 0.1) - 75.0*x_4/(x_4 + 1.0)']
    
    def np_eq(self, t, x):
        return np.array([-100.0*x[0]/(x[0] + 0.1*x[1] + 0.1) - 2.0*x[0]/(x[0] + 100.0) + 1.5*x[2]/(x[2] + 100.0) + 25.0, 100.0*x[0]/(x[0] + 0.1*x[1] + 0.1) - 75.0*x[1]*x[2]/(x[1]*x[2] + 0.1*x[1] + 1.0*x[2] + 0.1) - 5.0*x[1]/(x[1] + 1.0), 2.0*x[0]/(x[0] + 100.0) - 75.0*x[1]*x[2]/(x[1]*x[2] + 0.1*x[1] + 1.0*x[2] + 0.1) - 1.5*x[2]/(x[2] + 100.0) + 75.0*x_4/(x_4 + 1.0), 75.0*x[1]*x[2]/(x[1]*x[2] + 0.1*x[1] + 1.0*x[2] + 0.1) - 75.0*x_4/(x_4 + 1.0)])

@register_eq_class
class BIOMD0000000060(KnownEquation):
    _eq_name = 'vars4_prog67'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Keizer1996-Ryanodine-receptor-adaptation"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-984.1*x[0] + 28.8*x[2]', '-385.9*x[1] + 1094.0*x[2]', '984.1*x[0] + 385.9*x[1] - 1124.0*x[2] + 0.1*x_4', '1.75*x[2] - 0.1*x_4']
    
    def np_eq(self, t, x):
        return np.array([-984.1*x[0] + 28.8*x[2], -385.9*x[1] + 1094.0*x[2], 984.1*x[0] + 385.9*x[1] - 1124.0*x[2] + 0.1*x_4, 1.75*x[2] - 0.1*x_4])

@register_eq_class
class BIOMD0000000363(KnownEquation):
    _eq_name = 'vars4_prog68'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Lee2010-ThrombinActivation-OneForm-minimal"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.005*x[0]', '0.005*x[0] - 0.01*x[1]', '0.01*x[1]', '0']
    
    def np_eq(self, t, x):
        return np.array([-0.005*x[0], 0.005*x[0] - 0.01*x[1], 0.01*x[1], 0])

@register_eq_class
class BIOMD0000001035(KnownEquation):
    _eq_name = 'vars4_prog69'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Al-Tuwairqi2020 - Dynamics of cancer virotherapy with immune response"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.36*x[0]**2 - 0.36*x[0]*x[1] - 0.5*x[0]*x[2] - 0.36*x[0]*x_4 + 0.232*x[0]', '0.5*x[0]*x[2] - 0.48*x[1]*x_4 - 1.0*x[1]', '-0.5*x[0]*x[2] + 2.0*x[1] - 0.16*x[2]*x_4 - 0.2*x[2]', '0.29*x[0]*x_4 + 0.6*x[1]*x_4 - 0.16*x_4']
    
    def np_eq(self, t, x):
        return np.array([-0.36*x[0]**2 - 0.36*x[0]*x[1] - 0.5*x[0]*x[2] - 0.36*x[0]*x_4 + 0.232*x[0], 0.5*x[0]*x[2] - 0.48*x[1]*x_4 - 1.0*x[1], -0.5*x[0]*x[2] + 2.0*x[1] - 0.16*x[2]*x_4 - 0.2*x[2], 0.29*x[0]*x_4 + 0.6*x[1]*x_4 - 0.16*x_4])

@register_eq_class
class BIOMD0000000772(KnownEquation):
    _eq_name = 'vars4_prog70'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Wang2019 - A mathematical model of oncolytic virotherapy with time delay"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-9.631e-5*x[0]**2 - 9.631e-5*x[0]*x[2] + 0.206*x[0]', '0', '-0.01*x[2]*x_4 - 1.0*x[2]', '0.02*x[2]*x_4 - 0.5*x_4']
    
    def np_eq(self, t, x):
        return np.array([-9.631e-5*x[0]**2 - 9.631e-5*x[0]*x[2] + 0.206*x[0], 0, -0.01*x[2]*x_4 - 1.0*x[2], 0.02*x[2]*x_4 - 0.5*x_4])

@register_eq_class
class BIOMD0000000045(KnownEquation):
    _eq_name = 'vars4_prog71'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Borghans1997 - Calcium Oscillation - Model 3"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['1.0*x[1] - 0.027', '25.0*x[1]**2*x[2]**2/(x[1]**2*x[2]**2 + 0.004225*x[1]**2 + 0.000484*x[2]**2 + 2.045e-6) - 1.5*x[1]**2/(x[1]**2 + 0.000676) - 3.1*x[1]**2/(x[1]**2 + 2.5e-5) - 1.0*x[1] + 0.5*x[2] + 0.169*x_4**2/(x_4**2 + 0.01) + 0.5*x_4 + 0.027', '-25.0*x[1]**2*x[2]**2/(x[1]**2*x[2]**2 + 0.004225*x[1]**2 + 0.000484*x[2]**2 + 2.045e-6) + 3.1*x[1]**2/(x[1]**2 + 2.5e-5) - 0.5*x[2]', '1.5*x[1]**2/(x[1]**2 + 0.000676) - 0.169*x_4**2/(x_4**2 + 0.01) - 0.5*x_4']
    
    def np_eq(self, t, x):
        return np.array([1.0*x[1] - 0.027, 25.0*x[1]**2*x[2]**2/(x[1]**2*x[2]**2 + 0.004225*x[1]**2 + 0.000484*x[2]**2 + 2.045e-6) - 1.5*x[1]**2/(x[1]**2 + 0.000676) - 3.1*x[1]**2/(x[1]**2 + 2.5e-5) - 1.0*x[1] + 0.5*x[2] + 0.169*x_4**2/(x_4**2 + 0.01) + 0.5*x_4 + 0.027, -25.0*x[1]**2*x[2]**2/(x[1]**2*x[2]**2 + 0.004225*x[1]**2 + 0.000484*x[2]**2 + 2.045e-6) + 3.1*x[1]**2/(x[1]**2 + 2.5e-5) - 0.5*x[2], 1.5*x[1]**2/(x[1]**2 + 0.000676) - 0.169*x_4**2/(x_4**2 + 0.01) - 0.5*x_4])

@register_eq_class
class BIOMD0000001012(KnownEquation):
    _eq_name = 'vars4_prog72'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Leon-Triana2020 - CAR T-cell therapy in B-cell acute lymphoblastic leukaemia with contribution from immature B cells"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.07143*x[0]', '0.033*x[1]', '-0.01667*x[2] + 1.667e+6/(1.0e-9*x[0] + 1.0)', '1.0e+7 - 1.667e+6/(1.0e-9*x[0] + 1.0)']
    
    def np_eq(self, t, x):
        return np.array([-0.07143*x[0], 0.033*x[1], -0.01667*x[2] + 1.667e+6/(1.0e-9*x[0] + 1.0), 1.0e+7 - 1.667e+6/(1.0e-9*x[0] + 1.0)])

@register_eq_class
class BIOMD0000000841(KnownEquation):
    _eq_name = 'vars4_prog73'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Dhawan2019 - Endogenous miRNA sponges mediate the generation of oscillatory dynamics for a non-coding RNA network"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=4, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-10.0*x[0]*x[1] - 0.01*x[0] + 1.0', '-10.0*x[0]*x[1] - 0.1*x[1]*x[2] - 1.0*x[1] + 1.0 + 200.0/(1.0e+16*(1/delay(x_4, 0.5))**8.0 + 1.0)', '-0.1*x[1]*x[2] - 0.1*x[2] + 1.0', '-0.1*x_4 + 10.0*delay(x[2], 0.5)']
    
    def np_eq(self, t, x):
        return np.array([-10.0*x[0]*x[1] - 0.01*x[0] + 1.0, -10.0*x[0]*x[1] - 0.1*x[1]*x[2] - 1.0*x[1] + 1.0 + 200.0/(1.0e+16*(1/delay(x_4, 0.5))**8.0 + 1.0), -0.1*x[1]*x[2] - 0.1*x[2] + 1.0, -0.1*x_4 + 10.0*delay(x[2], 0.5)])

@register_eq_class
class BIOMD0000000709(KnownEquation):
    _eq_name = 'vars5_prog1'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Liu2017 - Dynamics of Avian Influenza with Allee Growth Effect"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.25e-10*x[0]**3 + 6.35e-6*x[0]**2 - 0.005*x[0]', '0', '30.00', '-0.444*x[3]', '0.1*x[3]']
    
    def np_eq(self, t, x):
        return np.array([-1.25e-10*x[0]**3 + 6.35e-6*x[0]**2 - 0.005*x[0], 0, 30.00, -0.444*x[3], 0.1*x[3]])

@register_eq_class
class BIOMD0000000039(KnownEquation):
    _eq_name = 'vars5_prog2'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Marhl2000-CaOscillations"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-300.0*x[0]**8/(x[0]**8 + 0.1678) - 4100.0*x[0]**3/(x[0]**2 + 25.0) + 4100.0*x[0]**2*x[1]/(x[0]**2 + 25.0) + 125.0*x[0]**2*x[2]/(x[0]**2 + 25.0) - 0.1*x[0]*x_5 - 20.05*x[0] + 0.05*x[1] + 0.006*x[2] + 0.01*x[3]', '1025.0*x[0]**3/(x[0]**2 + 25.0) - 1025.0*x[0]**2*x[1]/(x[0]**2 + 25.0) + 5.013*x[0] - 0.0125*x[1]', '75.0*x[0]**8/(x[0]**8 + 0.1678) - 31.25*x[0]**2*x[2]/(x[0]**2 + 25.0) - 0.0015*x[2]', '0.1*x[0]*x_5 - 0.01*x[3]', '-0.1*x[0]*x_5 + 0.01*x[3]']
    
    def np_eq(self, t, x):
        return np.array([-300.0*x[0]**8/(x[0]**8 + 0.1678) - 4100.0*x[0]**3/(x[0]**2 + 25.0) + 4100.0*x[0]**2*x[1]/(x[0]**2 + 25.0) + 125.0*x[0]**2*x[2]/(x[0]**2 + 25.0) - 0.1*x[0]*x_5 - 20.05*x[0] + 0.05*x[1] + 0.006*x[2] + 0.01*x[3], 1025.0*x[0]**3/(x[0]**2 + 25.0) - 1025.0*x[0]**2*x[1]/(x[0]**2 + 25.0) + 5.013*x[0] - 0.0125*x[1], 75.0*x[0]**8/(x[0]**8 + 0.1678) - 31.25*x[0]**2*x[2]/(x[0]**2 + 25.0) - 0.0015*x[2], 0.1*x[0]*x_5 - 0.01*x[3], -0.1*x[0]*x_5 + 0.01*x[3]])

@register_eq_class
class BIOMD0000000289(KnownEquation):
    _eq_name = 'vars5_prog3'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Alexander2010-Tcell-Regulation-Sys1"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.5*x[0]', '0.016*x[0]*x[2] + 200.0*x[0] - 0.25*x[1]', '1000.0*x[0] - 0.25*x[2]', '2000.0*x[2] - 5.003*x[3]', '0']
    
    def np_eq(self, t, x):
        return np.array([-0.5*x[0], 0.016*x[0]*x[2] + 200.0*x[0] - 0.25*x[1], 1000.0*x[0] - 0.25*x[2], 2000.0*x[2] - 5.003*x[3], 0])

@register_eq_class
class BIOMD0000000290(KnownEquation):
    _eq_name = 'vars5_prog4'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Alexander2010-Tcell-Regulation-Sys2"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.5*x[0]', '0.016*x[0]*x[2] + 200.0*x[0] - 0.25*x[1]', '1000.0*x[0] - 0.25*x[2]', '2000.0*x[2] - 5.0*x[3] - 1.25e+5*x[3]/(x[3] + 5.0e+7)', '0']
    
    def np_eq(self, t, x):
        return np.array([-0.5*x[0], 0.016*x[0]*x[2] + 200.0*x[0] - 0.25*x[1], 1000.0*x[0] - 0.25*x[2], 2000.0*x[2] - 5.0*x[3] - 1.25e+5*x[3]/(x[3] + 5.0e+7), 0])

@register_eq_class
class BIOMD0000000100(KnownEquation):
    _eq_name = 'vars5_prog5'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Rozi2003-GlycogenPhosphorylase-Activation"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '-6.0*x[1]**2/(x[1]**2 + 0.01) - 10.0*x[1] + 20.0*x[1]**2.0*x[2]**4*x[3]**2/(x[1]**2.0*x[2]**4*x[3]**2 + 0.04*x[1]**2.0*x[2]**4 + 0.0016*x[1]**2.0*x[3]**2 + 6.4e-5*x[1]**2.0 + 0.25*x[2]**4*x[3]**2 + 0.01*x[2]**4 + 0.0004*x[3]**2 + 1.6e-5) + 1.0*x[3] + 3.0', '-30.0*x[1]**4.0*x[2]**2.0/(x[1]**4.0*x[2]**2.0 + 1.0*x[1]**4.0 + 0.0256*x[2]**2.0 + 0.0256) - 0.1*x[2] + 1.0', '6.0*x[1]**2/(x[1]**2 + 0.01) - 20.0*x[1]**2.0*x[2]**4*x[3]**2/(x[1]**2.0*x[2]**4*x[3]**2 + 0.04*x[1]**2.0*x[2]**4 + 0.0016*x[1]**2.0*x[3]**2 + 6.4e-5*x[1]**2.0 + 0.25*x[2]**4*x[3]**2 + 0.01*x[2]**4 + 0.0004*x[3]**2 + 1.6e-5) - 1.0*x[3]', '-13.5*x[1]**4*x_5/(-x[1]**4*x_5 + 1.0*x[1]**4 + 0.1*x[1]**4/(16.0*x[1]**4 + 1.0) - 0.0625*x_5 + 0.0625 + 0.00625/(16.0*x[1]**4 + 1.0)) + 13.5*x[1]**4/(-x[1]**4*x_5 + 1.0*x[1]**4 + 0.1*x[1]**4/(16.0*x[1]**4 + 1.0) - 0.0625*x_5 + 0.0625 + 0.00625/(16.0*x[1]**4 + 1.0)) - 1.5*x_5/(-x_5 + 1.0 + 0.1/(16.0*x[1]**4 + 1.0)) - 3.3*x_5/(x_5 + 0.1) + 1.5/(-x_5 + 1.0 + 0.1/(16.0*x[1]**4 + 1.0))']
    
    def np_eq(self, t, x):
        return np.array([0, -6.0*x[1]**2/(x[1]**2 + 0.01) - 10.0*x[1] + 20.0*x[1]**2.0*x[2]**4*x[3]**2/(x[1]**2.0*x[2]**4*x[3]**2 + 0.04*x[1]**2.0*x[2]**4 + 0.0016*x[1]**2.0*x[3]**2 + 6.4e-5*x[1]**2.0 + 0.25*x[2]**4*x[3]**2 + 0.01*x[2]**4 + 0.0004*x[3]**2 + 1.6e-5) + 1.0*x[3] + 3.0, -30.0*x[1]**4.0*x[2]**2.0/(x[1]**4.0*x[2]**2.0 + 1.0*x[1]**4.0 + 0.0256*x[2]**2.0 + 0.0256) - 0.1*x[2] + 1.0, 6.0*x[1]**2/(x[1]**2 + 0.01) - 20.0*x[1]**2.0*x[2]**4*x[3]**2/(x[1]**2.0*x[2]**4*x[3]**2 + 0.04*x[1]**2.0*x[2]**4 + 0.0016*x[1]**2.0*x[3]**2 + 6.4e-5*x[1]**2.0 + 0.25*x[2]**4*x[3]**2 + 0.01*x[2]**4 + 0.0004*x[3]**2 + 1.6e-5) - 1.0*x[3], -13.5*x[1]**4*x_5/(-x[1]**4*x_5 + 1.0*x[1]**4 + 0.1*x[1]**4/(16.0*x[1]**4 + 1.0) - 0.0625*x_5 + 0.0625 + 0.00625/(16.0*x[1]**4 + 1.0)) + 13.5*x[1]**4/(-x[1]**4*x_5 + 1.0*x[1]**4 + 0.1*x[1]**4/(16.0*x[1]**4 + 1.0) - 0.0625*x_5 + 0.0625 + 0.00625/(16.0*x[1]**4 + 1.0)) - 1.5*x_5/(-x_5 + 1.0 + 0.1/(16.0*x[1]**4 + 1.0)) - 3.3*x_5/(x_5 + 0.1) + 1.5/(-x_5 + 1.0 + 0.1/(16.0*x[1]**4 + 1.0))])

@register_eq_class
class BIOMD0000000970(KnownEquation):
    _eq_name = 'vars5_prog6'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Hou2020 - SEIR model of COVID-19 transmission in Wuhan"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-3.249e-8*x[0]*x[1] - 2.166e-8*x[0]*x[2]', '3.249e-8*x[0]*x[1] + 2.166e-8*x[0]*x[2] - 0.14*x[1]', '0.14*x[1] - 0.048*x[2]', '0.048*x[2]', '0']
    
    def np_eq(self, t, x):
        return np.array([-3.249e-8*x[0]*x[1] - 2.166e-8*x[0]*x[2], 3.249e-8*x[0]*x[1] + 2.166e-8*x[0]*x[2] - 0.14*x[1], 0.14*x[1] - 0.048*x[2], 0.048*x[2], 0])

@register_eq_class
class BIOMD0000000980(KnownEquation):
    _eq_name = 'vars5_prog7'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Malkov2020 - SEIRS model of COVID-19 transmission with time-varying R values and reinfection"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.168*x[0]*x[2]/x_5 + 0.017*x[3]', '0.168*x[0]*x[2]/x_5 - 0.192*x[1]', '0.192*x[1] - 0.056*x[2]', '0.056*x[2] - 0.017*x[3]', '0']
    
    def np_eq(self, t, x):
        return np.array([-0.168*x[0]*x[2]/x_5 + 0.017*x[3], 0.168*x[0]*x[2]/x_5 - 0.192*x[1], 0.192*x[1] - 0.056*x[2], 0.056*x[2] - 0.017*x[3], 0])

@register_eq_class
class BIOMD0000000708(KnownEquation):
    _eq_name = 'vars5_prog8'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Liu2017 - Dynamics of Avian Influenza with Logistic Growth"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0e-7*x[0]**2 + 0.005*x[0]', '0', '30.00', '-0.444*x[3]', '0.1*x[3]']
    
    def np_eq(self, t, x):
        return np.array([-1.0e-7*x[0]**2 + 0.005*x[0], 0, 30.00, -0.444*x[3], 0.1*x[3]])

@register_eq_class
class BIOMD0000000921(KnownEquation):
    _eq_name = 'vars5_prog9'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Khajanchi2017 - Uniform Persistence and Global Stability for a Brain Tumor and Immune System Interaction"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.133e-8*x[0]**2 - 1.5*x[0]*x[1]/(x[0]*x[3] + 1.0e+4*x[0] + 2.7e+4*x[3] + 2.7e+8) - 0.12*x[0]*x[2]/(x[0]*x[3] + 1.0e+4*x[0] + 2.7e+4*x[3] + 2.7e+8) + 0.01*x[0]', '-0.019*x[0]*x[1]/(x[0] + 2.7e+4) - 0.331*x[1]**2 + 0.331*x[1] + 0.116*x_5/(x[3]*x_5 + 1.05e+4*x[3] + 1.0e+4*x_5 + 1.05e+8)', '-0.169*x[0]*x[2]/(x[0] + 3.344e+5) - 0.007*x[2]', '6.331e+4 - 6.93*x[3]', '-0.102*x_5']
    
    def np_eq(self, t, x):
        return np.array([-1.133e-8*x[0]**2 - 1.5*x[0]*x[1]/(x[0]*x[3] + 1.0e+4*x[0] + 2.7e+4*x[3] + 2.7e+8) - 0.12*x[0]*x[2]/(x[0]*x[3] + 1.0e+4*x[0] + 2.7e+4*x[3] + 2.7e+8) + 0.01*x[0], -0.019*x[0]*x[1]/(x[0] + 2.7e+4) - 0.331*x[1]**2 + 0.331*x[1] + 0.116*x_5/(x[3]*x_5 + 1.05e+4*x[3] + 1.0e+4*x_5 + 1.05e+8), -0.169*x[0]*x[2]/(x[0] + 3.344e+5) - 0.007*x[2], 6.331e+4 - 6.93*x[3], -0.102*x_5])

@register_eq_class
class BIOMD0000000869(KnownEquation):
    _eq_name = 'vars5_prog10'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Simon2019 - NIK-dependent p100 processing into p52 and IkBd degradation Michaelis-Menten SBML 2v4"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '-0.05*x[1]*x[3]/(x[1] + 10.0) + 0.5', '0.05*x[1]*x[3]/(x[1] + 10.0)', '0', '-0.05*x[3]*x_5/(x_5 + 10.0)']
    
    def np_eq(self, t, x):
        return np.array([0, -0.05*x[1]*x[3]/(x[1] + 10.0) + 0.5, 0.05*x[1]*x[3]/(x[1] + 10.0), 0, -0.05*x[3]*x_5/(x_5 + 10.0)])

@register_eq_class
class BIOMD0000000027(KnownEquation):
    _eq_name = 'vars5_prog11'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Markevich2004 - MAPK double phosphorylation ordered Michaelis-Menton"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.0002*x[0]*x[3]/(0.02*x[0] + 0.002*x[1] + 1.0) + 0.003333*x[1]*x_5/(0.01282*x[0] + 0.05556*x[1] + 0.04545*x[2] + 1.0)', '0.0002*x[0]*x[3]/(0.02*x[0] + 0.002*x[1] + 1.0) - 0.03*x[1]*x[3]/(0.02*x[0] + 0.002*x[1] + 1.0) - 0.003333*x[1]*x_5/(0.01282*x[0] + 0.05556*x[1] + 0.04545*x[2] + 1.0) + 0.003818*x[2]*x_5/(0.01282*x[0] + 0.05556*x[1] + 0.04545*x[2] + 1.0)', '0.03*x[1]*x[3]/(0.02*x[0] + 0.002*x[1] + 1.0) - 0.003818*x[2]*x_5/(0.01282*x[0] + 0.05556*x[1] + 0.04545*x[2] + 1.0)', '0', '0']
    
    def np_eq(self, t, x):
        return np.array([-0.0002*x[0]*x[3]/(0.02*x[0] + 0.002*x[1] + 1.0) + 0.003333*x[1]*x_5/(0.01282*x[0] + 0.05556*x[1] + 0.04545*x[2] + 1.0), 0.0002*x[0]*x[3]/(0.02*x[0] + 0.002*x[1] + 1.0) - 0.03*x[1]*x[3]/(0.02*x[0] + 0.002*x[1] + 1.0) - 0.003333*x[1]*x_5/(0.01282*x[0] + 0.05556*x[1] + 0.04545*x[2] + 1.0) + 0.003818*x[2]*x_5/(0.01282*x[0] + 0.05556*x[1] + 0.04545*x[2] + 1.0), 0.03*x[1]*x[3]/(0.02*x[0] + 0.002*x[1] + 1.0) - 0.003818*x[2]*x_5/(0.01282*x[0] + 0.05556*x[1] + 0.04545*x[2] + 1.0), 0, 0])

@register_eq_class
class BIOMD0000000868(KnownEquation):
    _eq_name = 'vars5_prog12'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Simon2019 - NIK-dependent p100 processing into p52 Mass Action SBML 2v4"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '-0.005*x[1]*x[3] + 0.5', '0.05*x_5', '-0.005*x[1]*x[3] + 0.05*x_5', '0.005*x[1]*x[3] - 0.05*x_5']
    
    def np_eq(self, t, x):
        return np.array([0, -0.005*x[1]*x[3] + 0.5, 0.05*x_5, -0.005*x[1]*x[3] + 0.05*x_5, 0.005*x[1]*x[3] - 0.05*x_5])

@register_eq_class
class BIOMD0000000851(KnownEquation):
    _eq_name = 'vars5_prog13'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Ho2019 - Mathematical models of transmission dynamics and vaccine strategies in Hong Kong during the 2017-2018 winter influenza season (Simple)"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-2.752*x[0]*x[3] + 0.1293*x[2] - 0.015', '-1.514*x[1]*x[3] - 0.1293*x[2] + 0.015', '0.015 - 0.1293*x[2]', '2.752*x[0]*x[3] + 1.514*x[1]*x[3] - 2.127*x[3]', '2.127*x[3]']
    
    def np_eq(self, t, x):
        return np.array([-2.752*x[0]*x[3] + 0.1293*x[2] - 0.015, -1.514*x[1]*x[3] - 0.1293*x[2] + 0.015, 0.015 - 0.1293*x[2], 2.752*x[0]*x[3] + 1.514*x[1]*x[3] - 2.127*x[3], 2.127*x[3]])

@register_eq_class
class BIOMD0000000929(KnownEquation):
    _eq_name = 'vars5_prog14'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Li2016 - Model for pancreatic cancer patients receiving immunotherapy"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.005*x[0]*(x[2]/x[0])**0.6667/((x[2]/x[0])**0.6667 + 0.3) + 0.019*x[0]', '0.125*x[0]*x[1]/(x[0] + 5.6e+7) - 0.0131*x[1]', '0.125*x[2]*x_5/(x_5 + 6.745e+10) - 0.02*x[2] + 3500.0', '0.125*x[3]*x_5/(x_5 + 6.745e+10) - 0.015*x[3] + 1.3e+5', '0.125*x_5**2/(x_5 + 6.745e+10) - 0.001*x_5 + 9600.0']
    
    def np_eq(self, t, x):
        return np.array([-0.005*x[0]*(x[2]/x[0])**0.6667/((x[2]/x[0])**0.6667 + 0.3) + 0.019*x[0], 0.125*x[0]*x[1]/(x[0] + 5.6e+7) - 0.0131*x[1], 0.125*x[2]*x_5/(x_5 + 6.745e+10) - 0.02*x[2] + 3500.0, 0.125*x[3]*x_5/(x_5 + 6.745e+10) - 0.015*x[3] + 1.3e+5, 0.125*x_5**2/(x_5 + 6.745e+10) - 0.001*x_5 + 9600.0])

@register_eq_class
class BIOMD0000000796(KnownEquation):
    _eq_name = 'vars5_prog15'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Yang2012 - cancer growth with angiogenesis"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.01*x[0]**2 - 0.01*x[0]*x[2] + 0.09*x[0]', '-0.005*x[1]**2 - 0.01*x[1]*x[2] + 0.05*x[1]', '-0.01*x[0]*x[2] - 0.04*x[2]**2*x_5 + 0.2*x[2]*x_5 - 0.05*x[2]', '0.01*x[1]*x[2] - 0.11*x[3]', '-0.01*x[2]*x_5**2 + 0.01*x[2]*x_5 + 0.1*x[3] - 0.01*x_5']
    
    def np_eq(self, t, x):
        return np.array([-0.01*x[0]**2 - 0.01*x[0]*x[2] + 0.09*x[0], -0.005*x[1]**2 - 0.01*x[1]*x[2] + 0.05*x[1], -0.01*x[0]*x[2] - 0.04*x[2]**2*x_5 + 0.2*x[2]*x_5 - 0.05*x[2], 0.01*x[1]*x[2] - 0.11*x[3], -0.01*x[2]*x_5**2 + 0.01*x[2]*x_5 + 0.1*x[3] - 0.01*x_5])

@register_eq_class
class BIOMD0000000008(KnownEquation):
    _eq_name = 'vars5_prog16'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Gardner1998 - Cell Cycle Goldbeter"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.5*x[0]*x[1]/(x[0] + 0.02) - 0.05*x[0]*x[3] - 0.02*x[0] + 0.055*x_5 + 0.1', '-0.3*x[1]*x[2]/(1.2 - x[1]) - 0.1*x[1]/(x[1] + 0.1) + 0.3*x[2]/(1.2 - x[1])', '-0.75*x[0]*x[2]/(-x[0]*x[2] + 1.1*x[0] - 0.3*x[2] + 0.33) + 0.75*x[0]/(-x[0]*x[2] + 1.1*x[0] - 0.3*x[2] + 0.33) - 0.25*x[2]/(x[2] + 0.1)', '-0.05*x[0]*x[3] - 0.05*x[3] + 0.052*x_5 + 0.2', '0.05*x[0]*x[3] - 0.057*x_5']
    
    def np_eq(self, t, x):
        return np.array([-0.5*x[0]*x[1]/(x[0] + 0.02) - 0.05*x[0]*x[3] - 0.02*x[0] + 0.055*x_5 + 0.1, -0.3*x[1]*x[2]/(1.2 - x[1]) - 0.1*x[1]/(x[1] + 0.1) + 0.3*x[2]/(1.2 - x[1]), -0.75*x[0]*x[2]/(-x[0]*x[2] + 1.1*x[0] - 0.3*x[2] + 0.33) + 0.75*x[0]/(-x[0]*x[2] + 1.1*x[0] - 0.3*x[2] + 0.33) - 0.25*x[2]/(x[2] + 0.1), -0.05*x[0]*x[3] - 0.05*x[3] + 0.052*x_5 + 0.2, 0.05*x[0]*x[3] - 0.057*x_5])

@register_eq_class
class BIOMD0000000629(KnownEquation):
    _eq_name = 'vars5_prog17'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Haffez2017 - RAR interaction with synthetic analogues"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.6*x[0]*x[2] + 0.1*x[1]', '0.6*x[0]*x[2] - 0.014*x[1]*x[3] - 0.1*x[1] + 0.2*x_5', '-0.6*x[0]*x[2] + 0.1*x[1]', '-0.014*x[1]*x[3] + 0.2*x_5', '0.014*x[1]*x[3] - 0.2*x_5']
    
    def np_eq(self, t, x):
        return np.array([-0.6*x[0]*x[2] + 0.1*x[1], 0.6*x[0]*x[2] - 0.014*x[1]*x[3] - 0.1*x[1] + 0.2*x_5, -0.6*x[0]*x[2] + 0.1*x[1], -0.014*x[1]*x[3] + 0.2*x_5, 0.014*x[1]*x[3] - 0.2*x_5])

@register_eq_class
class BIOMD0000000043(KnownEquation):
    _eq_name = 'vars5_prog18'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Borghans1997 - Calcium Oscillation - Model 1"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['10.0*x[1] - 2.0', '2.0e+4*x[1]**4*x[2]*x[3]**2/(400.0*x[1]**4*x[3]**2 + 16.0*x[1]**4 + 1.0*x[3]**2 + 0.04) - 6.5*x[1]**2/(x[1]**2 + 0.01) - 10.0*x[1] + 1.0*x[3] + 2.0', '-5000.0*x[1]**4*x[2] - 5.0*x[2] + 5.0', '-2.0e+4*x[1]**4*x[2]*x[3]**2/(400.0*x[1]**4*x[3]**2 + 16.0*x[1]**4 + 1.0*x[3]**2 + 0.04) + 6.5*x[1]**2/(x[1]**2 + 0.01) - 1.0*x[3]', '5000.0*x[1]**4*x[2] + 5.0*x[2] - 5.0']
    
    def np_eq(self, t, x):
        return np.array([10.0*x[1] - 2.0, 2.0e+4*x[1]**4*x[2]*x[3]**2/(400.0*x[1]**4*x[3]**2 + 16.0*x[1]**4 + 1.0*x[3]**2 + 0.04) - 6.5*x[1]**2/(x[1]**2 + 0.01) - 10.0*x[1] + 1.0*x[3] + 2.0, -5000.0*x[1]**4*x[2] - 5.0*x[2] + 5.0, -2.0e+4*x[1]**4*x[2]*x[3]**2/(400.0*x[1]**4*x[3]**2 + 16.0*x[1]**4 + 1.0*x[3]**2 + 0.04) + 6.5*x[1]**2/(x[1]**2 + 0.01) - 1.0*x[3], 5000.0*x[1]**4*x[2] + 5.0*x[2] - 5.0])

@register_eq_class
class BIOMD0000000295(KnownEquation):
    _eq_name = 'vars5_prog19'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Akman2008-Circadian-Clock-Model1"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.384*x[0] + 0.331*x[2] + 0.314*x_5', '-0.412*x[1] + 0.295*x[3] + 0.295*x_5', '0.223*x[0] - 0.331*x[2]', '0.272*x[1] - 0.295*x[3]', '-0.885*x_5/(x_5 + 0.085) + 3.831e+4/((x[2] + x[3])**6.396 + 3.13e+4)']
    
    def np_eq(self, t, x):
        return np.array([-0.384*x[0] + 0.331*x[2] + 0.314*x_5, -0.412*x[1] + 0.295*x[3] + 0.295*x_5, 0.223*x[0] - 0.331*x[2], 0.272*x[1] - 0.295*x[3], -0.885*x_5/(x_5 + 0.085) + 3.831e+4/((x[2] + x[3])**6.396 + 3.13e+4)])

@register_eq_class
class BIOMD0000000769(KnownEquation):
    _eq_name = 'vars5_prog20'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Eftimie2017/2 - interaction of Th and macrophage in melanoma"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-6.9e-10*x[0]**2 + 0.69*x[0]', '-9.0e-10*x[1]**2*x[3] - 9.0e-10*x[1]*x[2]*x[3] + 0.09*x[1]*x[3] - 0.03*x[1] + 0.008*x[3]', '-9.0e-10*x[1]*x[2]*x_5 - 9.0e-10*x[2]**2*x_5 + 0.09*x[2]*x_5 - 0.03*x[2] + 0.001*x_5', '0.001*x[1] - 2.0e-11*x[3]**2 - 2.0e-11*x[3]*x_5 - 0.05*x[3] + 0.09*x_5', '-2.0e-11*x[2]*x[3]*x_5 - 2.0e-11*x[2]*x_5**2 + 0.02*x[2]*x_5 + 0.001*x[2] + 0.05*x[3] - 0.11*x_5']
    
    def np_eq(self, t, x):
        return np.array([-6.9e-10*x[0]**2 + 0.69*x[0], -9.0e-10*x[1]**2*x[3] - 9.0e-10*x[1]*x[2]*x[3] + 0.09*x[1]*x[3] - 0.03*x[1] + 0.008*x[3], -9.0e-10*x[1]*x[2]*x_5 - 9.0e-10*x[2]**2*x_5 + 0.09*x[2]*x_5 - 0.03*x[2] + 0.001*x_5, 0.001*x[1] - 2.0e-11*x[3]**2 - 2.0e-11*x[3]*x_5 - 0.05*x[3] + 0.09*x_5, -2.0e-11*x[2]*x[3]*x_5 - 2.0e-11*x[2]*x_5**2 + 0.02*x[2]*x_5 + 0.001*x[2] + 0.05*x[3] - 0.11*x_5])

@register_eq_class
class BIOMD0000000766(KnownEquation):
    _eq_name = 'vars5_prog21'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Macnamara2015/1 - virotherapy full model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-5.098e-9*x[0]**2 - 5.098e-9*x[0]*x[1] - 2.0*x[0]*x[3]/(x[3] + 1000.0) - 0.004*x[0]*x_5/(x[0] + 1.0) + 0.927*x[0]', '0.004*x[0]*x_5/(x[0] + 1.0) - 2.0*x[1]*x[3]/(x[3] + 1000.0) - 1.0*x[1]', '-0.00025*x[2]**2*x_5/(x_5 + 1.0e+4) + 2.5*x[2]*x_5/(x_5 + 1.0e+4)', '0.4*x[0]*x[2]/(x[0] + x_5 + 1.0e+4) + 0.4*x[2]*x_5/(x[0] + x_5 + 1.0e+4) - 0.1*x[3]', '1000.0*x[1] - 2.042*x_5']
    
    def np_eq(self, t, x):
        return np.array([-5.098e-9*x[0]**2 - 5.098e-9*x[0]*x[1] - 2.0*x[0]*x[3]/(x[3] + 1000.0) - 0.004*x[0]*x_5/(x[0] + 1.0) + 0.927*x[0], 0.004*x[0]*x_5/(x[0] + 1.0) - 2.0*x[1]*x[3]/(x[3] + 1000.0) - 1.0*x[1], -0.00025*x[2]**2*x_5/(x_5 + 1.0e+4) + 2.5*x[2]*x_5/(x_5 + 1.0e+4), 0.4*x[0]*x[2]/(x[0] + x_5 + 1.0e+4) + 0.4*x[2]*x_5/(x[0] + x_5 + 1.0e+4) - 0.1*x[3], 1000.0*x[1] - 2.042*x_5])

@register_eq_class
class BIOMD0000000744(KnownEquation):
    _eq_name = 'vars5_prog22'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Hu2019 - Pancreatic cancer dynamics"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.96*x[0]*x[2]/(x[3] + 1.0e+8) + 0.019*x[0]', '0.125*x[1]*x[3]/(x[3] + 5.6e+10) - 0.013*x[1]', '124.5*x[2]*x_5/(x[3]*x_5 + 2.0e+10*x[3] + 1.0e+6*x_5 + 2.0e+16) - 0.02*x[2] + 3500.0', '1.25e+4*x[0]*x[1]/(x_5 + 8.9e+10) + 5.85*x[0]*x[2]/(x[0] + 1.0e+6) - 0.034*x[3]', '7.3*x[0]*x[2]/(x[0] + 1.0e+6) - 0.034*x_5']
    
    def np_eq(self, t, x):
        return np.array([-0.96*x[0]*x[2]/(x[3] + 1.0e+8) + 0.019*x[0], 0.125*x[1]*x[3]/(x[3] + 5.6e+10) - 0.013*x[1], 124.5*x[2]*x_5/(x[3]*x_5 + 2.0e+10*x[3] + 1.0e+6*x_5 + 2.0e+16) - 0.02*x[2] + 3500.0, 1.25e+4*x[0]*x[1]/(x_5 + 8.9e+10) + 5.85*x[0]*x[2]/(x[0] + 1.0e+6) - 0.034*x[3], 7.3*x[0]*x[2]/(x[0] + 1.0e+6) - 0.034*x_5])

@register_eq_class
class BIOMD0000000413(KnownEquation):
    _eq_name = 'vars5_prog23'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Band2012-DII-Venus-FullModel"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.001*x[0]*x[1] - 0.79*x[0] + 0.334*x[2] + 30.5', '-0.001*x[0]*x[1] + 0.334*x[2]', '0.001*x[0]*x[1] - 1.15*x[2]*x_5 - 0.334*x[2] + 4.665*x[3]', '1.15*x[2]*x_5 - 4.665*x[3]', '-1.15*x[2]*x_5 + 4.49*x[3] - 0.003*x_5 + 0.486']
    
    def np_eq(self, t, x):
        return np.array([-0.001*x[0]*x[1] - 0.79*x[0] + 0.334*x[2] + 30.5, -0.001*x[0]*x[1] + 0.334*x[2], 0.001*x[0]*x[1] - 1.15*x[2]*x_5 - 0.334*x[2] + 4.665*x[3], 1.15*x[2]*x_5 - 4.665*x[3], -1.15*x[2]*x_5 + 4.49*x[3] - 0.003*x_5 + 0.486])

@register_eq_class
class BIOMD0000000745(KnownEquation):
    _eq_name = 'vars5_prog24'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Jarrett2018 - trastuzumab-induced immune response in murine HER2+ breast cancer model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.187*x[0]*x[1] + 0.06701*x[0]*x_5 + 0.044*x[0]', '-0.722*x[0]*x[1] - 0.199*x[1]*x[2] - 0.2*x[1]*x[3] + 0.199*x[2] + 0.2*x[3]', '-1.824*x[0]*x[2] + 0.101*x[0] - 0.045*x[1]*x[2] + 0.045*x[1]', '-0.911*x[1]*x[3] + 0.027*x[2]*x[3] - 0.027*x[2] - 0.027*x[3] + 0.027', '0.743*x[2]*x_5**2 - 0.743*x[2]*x_5 - 0.211*x_5**2 + 0.211*x_5']
    
    def np_eq(self, t, x):
        return np.array([-0.187*x[0]*x[1] + 0.06701*x[0]*x_5 + 0.044*x[0], -0.722*x[0]*x[1] - 0.199*x[1]*x[2] - 0.2*x[1]*x[3] + 0.199*x[2] + 0.2*x[3], -1.824*x[0]*x[2] + 0.101*x[0] - 0.045*x[1]*x[2] + 0.045*x[1], -0.911*x[1]*x[3] + 0.027*x[2]*x[3] - 0.027*x[2] - 0.027*x[3] + 0.027, 0.743*x[2]*x_5**2 - 0.743*x[2]*x_5 - 0.211*x_5**2 + 0.211*x_5])

@register_eq_class
class BIOMD0000000609(KnownEquation):
    _eq_name = 'vars5_prog25'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Reddyhoff2015 - Acetaminophen metabolism and toxicity"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-2.26e+14*x[0]*x[3] - 2.0*x[0]', '-1.6e+18*x[1]*x[2] - 2.0*x[1]', '-1.6e+18*x[1]*x[2] - 110.0*x[2] + 0.315*x[3]', '-2.26e+14*x[0]*x[3] + 0.032*x[2] - 3.305*x[3]', '110.0*x[2]']
    
    def np_eq(self, t, x):
        return np.array([-2.26e+14*x[0]*x[3] - 2.0*x[0], -1.6e+18*x[1]*x[2] - 2.0*x[1], -1.6e+18*x[1]*x[2] - 110.0*x[2] + 0.315*x[3], -2.26e+14*x[0]*x[3] + 0.032*x[2] - 3.305*x[3], 110.0*x[2]])

@register_eq_class
class BIOMD0000000979(KnownEquation):
    _eq_name = 'vars5_prog26'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Malkov2020 - SEIRS model of COVID-19 transmission with reinfection"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.168*x[0]*x[2]/x_5 + 0.017*x[3]', '0.168*x[0]*x[2]/x_5 - 0.192*x[1]', '0.192*x[1] - 0.056*x[2]', '0.056*x[2] - 0.017*x[3]', '0']
    
    def np_eq(self, t, x):
        return np.array([-0.168*x[0]*x[2]/x_5 + 0.017*x[3], 0.168*x[0]*x[2]/x_5 - 0.192*x[1], 0.192*x[1] - 0.056*x[2], 0.056*x[2] - 0.017*x[3], 0])

@register_eq_class
class BIOMD0000000905(KnownEquation):
    _eq_name = 'vars5_prog27'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Dubey2007 - A mathematical model for the effect of toxicant on the immune system (with toxicant effect) Model2"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.2*x[0]**2 - 0.05*x[0]*x[1] + 0.5*x[0]', '0.295*x[0]*x[1] - 0.3*x[1]*x_5 - 0.8*x[1] + 0.04', '2.4*x[0] - 0.1*x[2]', '5.0 - 0.4*x[3]', '-0.6*x[1]*x_5 + 1.2*x[3] - 0.02*x_5']
    
    def np_eq(self, t, x):
        return np.array([-0.2*x[0]**2 - 0.05*x[0]*x[1] + 0.5*x[0], 0.295*x[0]*x[1] - 0.3*x[1]*x_5 - 0.8*x[1] + 0.04, 2.4*x[0] - 0.1*x[2], 5.0 - 0.4*x[3], -0.6*x[1]*x_5 + 1.2*x[3] - 0.02*x_5])

@register_eq_class
class BIOMD0000000898(KnownEquation):
    _eq_name = 'vars5_prog28'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Jiao2018 - Feedback regulation in a stem cell model with acute myeloid leukaemia"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.225*x[0]**2/(0.23*x[2]**1.0 + 0.006*x[2]**2.0 + 1.0) + 0.45*x[0]/(0.23*x[2]**1.0 + 0.006*x[2]**2.0 + 1.0) - 0.5*x[0]/(0.2*x[2]**1.0 + 1.0)', '-0.45*x[0]/(0.23*x[2]**1.0 + 0.006*x[2]**2.0 + 1.0) + 1.0*x[0]/(0.2*x[2]**1.0 + 1.0) - 0.4896*x[1]**2/(0.135*x[2]**1.0 + 0.00275*x[2]**2.0 + 1.0) - 0.4896*x[1]*x[3]/(0.135*x[2]**1.0 + 0.00275*x[2]**2.0 + 1.0) + 0.9792*x[1]/(0.135*x[2]**1.0 + 0.00275*x[2]**2.0 + 1.0) - 0.72*x[1]/(0.11*x[2]**1.0 + 1.0)', '-0.9792*x[1]/(0.135*x[2]**1.0 + 0.00275*x[2]**2.0 + 1.0) + 1.44*x[1]/(0.11*x[2]**1.0 + 1.0) - 0.275*x[2]', '-0.56*x[1]*x[3] - 0.56*x[3]**2 + 0.42*x[3]', '0.28*x[3] - 0.3*x_5']
    
    def np_eq(self, t, x):
        return np.array([-0.225*x[0]**2/(0.23*x[2]**1.0 + 0.006*x[2]**2.0 + 1.0) + 0.45*x[0]/(0.23*x[2]**1.0 + 0.006*x[2]**2.0 + 1.0) - 0.5*x[0]/(0.2*x[2]**1.0 + 1.0), -0.45*x[0]/(0.23*x[2]**1.0 + 0.006*x[2]**2.0 + 1.0) + 1.0*x[0]/(0.2*x[2]**1.0 + 1.0) - 0.4896*x[1]**2/(0.135*x[2]**1.0 + 0.00275*x[2]**2.0 + 1.0) - 0.4896*x[1]*x[3]/(0.135*x[2]**1.0 + 0.00275*x[2]**2.0 + 1.0) + 0.9792*x[1]/(0.135*x[2]**1.0 + 0.00275*x[2]**2.0 + 1.0) - 0.72*x[1]/(0.11*x[2]**1.0 + 1.0), -0.9792*x[1]/(0.135*x[2]**1.0 + 0.00275*x[2]**2.0 + 1.0) + 1.44*x[1]/(0.11*x[2]**1.0 + 1.0) - 0.275*x[2], -0.56*x[1]*x[3] - 0.56*x[3]**2 + 0.42*x[3], 0.28*x[3] - 0.3*x_5])

@register_eq_class
class BIOMD0000000916(KnownEquation):
    _eq_name = 'vars5_prog29'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Kraan199-Kinetics of Cortisol Metabolism and Excretion."
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-26.6*x[0]', '26.6*x[0] - 4.8*x[1] + 1.2*x[3]', '1.2*x[3]', '1.2*x[1] - 2.4*x[3]', '3.6*x[1]']
    
    def np_eq(self, t, x):
        return np.array([-26.6*x[0], 26.6*x[0] - 4.8*x[1] + 1.2*x[3], 1.2*x[3], 1.2*x[1] - 2.4*x[3], 3.6*x[1]])

@register_eq_class
class BIOMD0000000732(KnownEquation):
    _eq_name = 'vars5_prog30'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Kirschner1998-Immunotherapy-Tumour"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['0', '-1.0*x[1]*x_5/(x[1] + 1.0e+5) + 0.18*x[1]', '0', '5.0*x[1]*x_5/(x[1] + 1000.0) - 10.0*x[3]', '0.035*x[1] + 6.2e-9*x[3]*x_5 - 0.03*x_5']
    
    def np_eq(self, t, x):
        return np.array([0, -1.0*x[1]*x_5/(x[1] + 1.0e+5) + 0.18*x[1], 0, 5.0*x[1]*x_5/(x[1] + 1000.0) - 10.0*x[3], 0.035*x[1] + 6.2e-9*x[3]*x_5 - 0.03*x_5])

@register_eq_class
class BIOMD0000000768(KnownEquation):
    _eq_name = 'vars5_prog31'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Eftimie2010 - immunity to melanoma"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.1*x[0] + 0.09*x[0]/(1.0*x[3] + 1.0) + 0.008*x[1]*x_5/(1.0*x[1]*x[3] + 1.0*x[1] + 1000.0*x[3] + 1000.0)', '-0.2*x[1]*x[2]/(x[1] + 1.0e+5) + 0.514*x[1]*x[3]/(1.0*x[2] + 1.0) + 0.514*x[1]/(1.0*x[2] + 1.0)', '8.6*x[0]*x[1]/(1.0*x[1]*x[3] + 1.0*x[1] + 1000.0*x[3] + 1000.0) + 1.0*x[1]/(1.0*x[1]*x[3] + 1.0*x[1] + 1000.0*x[3] + 1000.0) - 34.0*x[2]', '10.0*x[1]**2/(x[1]**2 + 1.0e+12) + 1.0*x[1]/(x[1] + 1000.0) - 34.0*x[3]', '5.4*x[0]*x[1]/(1.0*x[1]*x[3] + 1.0*x[1] + 1000.0*x[3] + 1000.0) + 1.0*x[1]/(1.0*x[1]*x[3] + 1.0*x[1] + 1000.0*x[3] + 1000.0) - 34.0*x_5']
    
    def np_eq(self, t, x):
        return np.array([-0.1*x[0] + 0.09*x[0]/(1.0*x[3] + 1.0) + 0.008*x[1]*x_5/(1.0*x[1]*x[3] + 1.0*x[1] + 1000.0*x[3] + 1000.0), -0.2*x[1]*x[2]/(x[1] + 1.0e+5) + 0.514*x[1]*x[3]/(1.0*x[2] + 1.0) + 0.514*x[1]/(1.0*x[2] + 1.0), 8.6*x[0]*x[1]/(1.0*x[1]*x[3] + 1.0*x[1] + 1000.0*x[3] + 1000.0) + 1.0*x[1]/(1.0*x[1]*x[3] + 1.0*x[1] + 1000.0*x[3] + 1000.0) - 34.0*x[2], 10.0*x[1]**2/(x[1]**2 + 1.0e+12) + 1.0*x[1]/(x[1] + 1000.0) - 34.0*x[3], 5.4*x[0]*x[1]/(1.0*x[1]*x[3] + 1.0*x[1] + 1000.0*x[3] + 1000.0) + 1.0*x[1]/(1.0*x[1]*x[3] + 1.0*x[1] + 1000.0*x[3] + 1000.0) - 34.0*x_5])

@register_eq_class
class BIOMD0000000886(KnownEquation):
    _eq_name = 'vars5_prog32'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Dubey2008 - Modeling the interaction between avascular cancerous cells and acquired immune response"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-4.6*x[0]**2 - 0.101*x[0]*x[2] - 0.008*x[0]*x_5 + 0.18*x[0]', '0.3*x[0]*x[1] + 1.5*x[0] - 0.2*x[1]', '0.3*x[0]*x[2] + 1.4*x[0] + 0.05*x[1]*x[2] - 0.041*x[2]', '0.4*x[0]*x[3] + 0.45*x[0] + 0.3*x[1]*x[3] - 0.03*x[3]', '-0.5*x[0]*x_5 + 0.35*x[3] - 0.3*x_5']
    
    def np_eq(self, t, x):
        return np.array([-4.6*x[0]**2 - 0.101*x[0]*x[2] - 0.008*x[0]*x_5 + 0.18*x[0], 0.3*x[0]*x[1] + 1.5*x[0] - 0.2*x[1], 0.3*x[0]*x[2] + 1.4*x[0] + 0.05*x[1]*x[2] - 0.041*x[2], 0.4*x[0]*x[3] + 0.45*x[0] + 0.3*x[1]*x[3] - 0.03*x[3], -0.5*x[0]*x_5 + 0.35*x[3] - 0.3*x_5])

@register_eq_class
class BIOMD0000000721(KnownEquation):
    _eq_name = 'vars5_prog33'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Graham2013 - Role of osteocytes in targeted bone remodeling"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.0025*x[0]*x[2]**1.0 + 0.5*x[2]**1.0', '-0.0005*x[0]*x[1]**1.0 + 0.1*x[0]**2.0 - 0.0005*x[0]**3.0 - 0.1*x[1] - 0.1*x[1]**1.0*x[3]**1.0 + 0.1*x[1]**1.0', '0.0025*x[0]*x[2]**1.0 + 0.1*x[1]**1.0*x[3]**1.0 - 0.6*x[2]**1.0', '0.1*x[0]**1.0*x[1]**1.0*(1.0 - 0.005*x[0])**1.0/(x[2] + 1.0)**1.0 - 0.1*x[3]**1.0', '0.015*x[2] - 0.7*x[3]']
    
    def np_eq(self, t, x):
        return np.array([-0.0025*x[0]*x[2]**1.0 + 0.5*x[2]**1.0, -0.0005*x[0]*x[1]**1.0 + 0.1*x[0]**2.0 - 0.0005*x[0]**3.0 - 0.1*x[1] - 0.1*x[1]**1.0*x[3]**1.0 + 0.1*x[1]**1.0, 0.0025*x[0]*x[2]**1.0 + 0.1*x[1]**1.0*x[3]**1.0 - 0.6*x[2]**1.0, 0.1*x[0]**1.0*x[1]**1.0*(1.0 - 0.005*x[0])**1.0/(x[2] + 1.0)**1.0 - 0.1*x[3]**1.0, 0.015*x[2] - 0.7*x[3]])

@register_eq_class
class BIOMD0000000840(KnownEquation):
    _eq_name = 'vars5_prog34'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Caldwell2019 - The Vicodin abuse problem"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['3.0e+6 - 0.67*x[0]', '0.22*x[0] - 0.19*x[1]', '0.05*x[1] - 0.19*x[2]', '0.05*x[2] - 0.03*x[3] + 0.24*x_5', '0.03*x[3] - 0.533*x_5']
    
    def np_eq(self, t, x):
        return np.array([3.0e+6 - 0.67*x[0], 0.22*x[0] - 0.19*x[1], 0.05*x[1] - 0.19*x[2], 0.05*x[2] - 0.03*x[3] + 0.24*x_5, 0.03*x[3] - 0.533*x_5])

@register_eq_class
class BIOMD0000000903(KnownEquation):
    _eq_name = 'vars5_prog35'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Solis-perez2019 - A fractional mathematical model of breast cancer competition model"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-3.304e-7*x[0]**2 + 600.0*x[0]*x_5/(x[0] + 1.135e+6) + 0.75*x[0]', '-9.975e-15*x[0]**2*x[1] + 2.264e-7*x[0]**2 - 0.01*x[1]', '-2.8e-8*x[2]**2 - 100.0*x[2]*x_5/(x[2] + 1.25e+6) + 0.7*x[2]', '0.2*x[1]*x[3]/(x[1] + 3.0e+5) - 0.2*x[3]*x_5/(x_5 + 400.0) - 0.29*x[3] + 1.3e+4', '-0.01*x[0]*x_5/(x[0] + 1.135e+6) - 0.01*x[1]*x_5/(x[1] + 1.135e+7) - 0.01*x[2]*x_5/(x[2] + 1.25e+6) - 0.97*x_5 + 2700.0']
    
    def np_eq(self, t, x):
        return np.array([-3.304e-7*x[0]**2 + 600.0*x[0]*x_5/(x[0] + 1.135e+6) + 0.75*x[0], -9.975e-15*x[0]**2*x[1] + 2.264e-7*x[0]**2 - 0.01*x[1], -2.8e-8*x[2]**2 - 100.0*x[2]*x_5/(x[2] + 1.25e+6) + 0.7*x[2], 0.2*x[1]*x[3]/(x[1] + 3.0e+5) - 0.2*x[3]*x_5/(x_5 + 400.0) - 0.29*x[3] + 1.3e+4, -0.01*x[0]*x_5/(x[0] + 1.135e+6) - 0.01*x[1]*x_5/(x[1] + 1.135e+7) - 0.01*x[2]*x_5/(x[2] + 1.25e+6) - 0.97*x_5 + 2700.0])

@register_eq_class
class BIOMD0000000722(KnownEquation):
    _eq_name = 'vars5_prog36'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'exp', 'pow']
    _description = "Bianchi2015 -Model for lymphangiogenesis in normal and diabetic wounds"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-500.0*x[0] + 8.1*x[1] + 1.305e+5*np.exp(-0.522*t)', '4.0e+8*x[0]**2/(x[0]**4 + 8.1e+9) - 1.017e-8*x[1]**2 - 0.1939*x[1] + 542.0', '0.002*x[1] - 0.001*x[2]*x[3] - 11.0*x[2] + 1.9', '-2.123e-6*x[1]*x[3] + 1.0e+7*x[2]**2*piecewise(x_5 < 10000.0, 1 - 0.0001*x_5, 0)/(x[2]**4 + 8.1e+9) - 0.001*x[2]*x[3]*piecewise(x[3] + x_5 < 10000.0, 0, 1) + 1.0*x[2]*x[3]/(0.984*x[0]*x[2] + 10.08*x[0] + 4.1*x[2] + 42.0) - 2.123e-6*x[3]**2 - 2.123e-6*x[3]*x_5 - 0.05*x[3]*piecewise(x[3] + x_5 < 10000.0, 0, 1) + 0.42*x[3]/(0.24*x[0] + 1.0) + 500.0*piecewise(x_5 < 10000.0, 1 - 0.0001*x_5, 0)', '0.001*x[2]*x[3]*piecewise(x[3] + x_5 < 10000.0, 0, 1) + 0.05*x[3]*piecewise(x[3] + x_5 < 10000.0, 0, 1)']
    
    def np_eq(self, t, x):
        return np.array([-500.0*x[0] + 8.1*x[1] + 1.305e+5*np.exp(-0.522*t), 4.0e+8*x[0]**2/(x[0]**4 + 8.1e+9) - 1.017e-8*x[1]**2 - 0.1939*x[1] + 542.0, 0.002*x[1] - 0.001*x[2]*x[3] - 11.0*x[2] + 1.9, -2.123e-6*x[1]*x[3] + 1.0e+7*x[2]**2*piecewise(x_5 < 10000.0, 1 - 0.0001*x_5, 0)/(x[2]**4 + 8.1e+9) - 0.001*x[2]*x[3]*piecewise(x[3] + x_5 < 10000.0, 0, 1) + 1.0*x[2]*x[3]/(0.984*x[0]*x[2] + 10.08*x[0] + 4.1*x[2] + 42.0) - 2.123e-6*x[3]**2 - 2.123e-6*x[3]*x_5 - 0.05*x[3]*piecewise(x[3] + x_5 < 10000.0, 0, 1) + 0.42*x[3]/(0.24*x[0] + 1.0) + 500.0*piecewise(x_5 < 10000.0, 1 - 0.0001*x_5, 0), 0.001*x[2]*x[3]*piecewise(x[3] + x_5 < 10000.0, 0, 1) + 0.05*x[3]*piecewise(x[3] + x_5 < 10000.0, 0, 1)])

@register_eq_class
class BIOMD0000000707(KnownEquation):
    _eq_name = 'vars5_prog37'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Revilla2003 - Controlling HIV infection using recombinant viruses"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.004*x[0]*x[1] - 0.01*x[0] + 2.0', '-2.0*x[1] + 50.0*x[2]', '0.004*x[0]*x[1] - 0.004*x[2]*x[3] - 0.33*x[2]', '-2.0*x[3] + 2000.0*x_5', '0.004*x[2]*x[3] - 2.0*x_5']
    
    def np_eq(self, t, x):
        return np.array([-0.004*x[0]*x[1] - 0.01*x[0] + 2.0, -2.0*x[1] + 50.0*x[2], 0.004*x[0]*x[1] - 0.004*x[2]*x[3] - 0.33*x[2], -2.0*x[3] + 2000.0*x_5, 0.004*x[2]*x[3] - 2.0*x_5])

@register_eq_class
class BIOMD0000000798(KnownEquation):
    _eq_name = 'vars5_prog38'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Sharp2019 - AML"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.5*x[0]**2 + 0.36*x[0]', '0.14*x[0] - 0.43*x[1]**2 - 0.43*x[1]*x[3] - 0.009998*x[1]', '0.44*x[1] - 0.275*x[2]', '-0.27*x[1]*x[3] - 0.27*x[3]**2 + 0.22*x[3] - 0.015*x[3]/(x[3] + 0.01)', '0.05*x[3] - 0.3*x_5']
    
    def np_eq(self, t, x):
        return np.array([-0.5*x[0]**2 + 0.36*x[0], 0.14*x[0] - 0.43*x[1]**2 - 0.43*x[1]*x[3] - 0.009998*x[1], 0.44*x[1] - 0.275*x[2], -0.27*x[1]*x[3] - 0.27*x[3]**2 + 0.22*x[3] - 0.015*x[3]/(x[3] + 0.01), 0.05*x[3] - 0.3*x_5])

@register_eq_class
class BIOMD0000000945(KnownEquation):
    _eq_name = 'vars5_prog39'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Evans2004 - Cell based mathematical model of topotecan"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.029*x[0] + 6.116*x[2]', '0.029*x[0]', '0.001*x[2]*x_5 - 1.07*x[2] + 0.186*x[3] + 1.75*x_5', '0.027*x[2] - 0.186*x[3]', '-0.0003932*x[2]*x_5 + 0.01136*x[2] - 4.449*x_5']
    
    def np_eq(self, t, x):
        return np.array([-0.029*x[0] + 6.116*x[2], 0.029*x[0], 0.001*x[2]*x_5 - 1.07*x[2] + 0.186*x[3] + 1.75*x_5, 0.027*x[2] - 0.186*x[3], -0.0003932*x[2]*x_5 + 0.01136*x[2] - 4.449*x_5])

@register_eq_class
class BIOMD0000000984(KnownEquation):
    _eq_name = 'vars5_prog40'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Fang2020 - SEIR model of COVID-19 transmission considering government interventions in Wuhan"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.0*x[0]*x[2]/x_5', '1.0*x[0]*x[2]/x_5 - 0.143*x[1]', '0.143*x[1] - 0.098*x[2]', '0.098*x[2]', '0']
    
    def np_eq(self, t, x):
        return np.array([-1.0*x[0]*x[2]/x_5, 1.0*x[0]*x[2]/x_5 - 0.143*x[1], 0.143*x[1] - 0.098*x[2], 0.098*x[2], 0])

@register_eq_class
class BIOMD0000000040(KnownEquation):
    _eq_name = 'vars5_prog41'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Field1974-Oregonator"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-1.6e+9*x[0]*x[3] - 0.0804*x[0] + 1.0*x[2]', '0', '-1.0*x[2] + 480.0*x[3]', '-1.6e+9*x[0]*x[3] + 0.0804*x[0] - 8.0e+7*x[3]**2 + 480.0*x[3]', '0']
    
    def np_eq(self, t, x):
        return np.array([-1.6e+9*x[0]*x[3] - 0.0804*x[0] + 1.0*x[2], 0, -1.0*x[2] + 480.0*x[3], -1.6e+9*x[0]*x[3] + 0.0804*x[0] - 8.0e+7*x[3]**2 + 480.0*x[3], 0])

@register_eq_class
class BIOMD0000000662(KnownEquation):
    _eq_name = 'vars5_prog42'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'log']
    _description = "Moore2004 - Chronic Myeloid Leukemic cells and T-lymphocyte interaction"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.047*x[0]*x[2] + 0.23*x[0]*np.log(1/x[0]) + 2.1155933815706*x[0]', '-0.063*x[0]*x[1]/(x[0] + 43.0) - 0.05*x[1] + 0.071', '0.03528*x[0]*x[1]/(x[0] + 43.0) - 0.008*x[0]*x[2] + 0.53*x[0]*x[2]/(x[0] + 43.0) - 0.12*x[2]', '0', '0']
    
    def np_eq(self, t, x):
        return np.array([-0.047*x[0]*x[2] + 0.23*x[0]*np.log(1/x[0]) + 2.1155933815706*x[0], -0.063*x[0]*x[1]/(x[0] + 43.0) - 0.05*x[1] + 0.071, 0.03528*x[0]*x[1]/(x[0] + 43.0) - 0.008*x[0]*x[2] + 0.53*x[0]*x[2]/(x[0] + 43.0) - 0.12*x[2], 0, 0])

@register_eq_class
class BIOMD0000000004(KnownEquation):
    _eq_name = 'vars5_prog43'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const']
    _description = "Goldbeter1991 - Min Mit Oscil Expl Inact"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.25*x[0]*x[2]/(x[0] + 0.02) - 0.01*x[0] + 0.025', '3.0*x[0]*x[3]/(x[0]*x[3] + 0.005*x[0] + 0.5*x[3] + 0.0025) - 1.5*x[1]/(x[1] + 0.005)', '1.0*x[1]*x_5/(x_5 + 0.005) - 0.5*x[2]/(x[2] + 0.005)', '-3.0*x[0]*x[3]/(x[0]*x[3] + 0.005*x[0] + 0.5*x[3] + 0.0025) + 1.5*x[1]/(x[1] + 0.005)', '-1.0*x[1]*x_5/(x_5 + 0.005) + 0.5*x[2]/(x[2] + 0.005)']
    
    def np_eq(self, t, x):
        return np.array([-0.25*x[0]*x[2]/(x[0] + 0.02) - 0.01*x[0] + 0.025, 3.0*x[0]*x[3]/(x[0]*x[3] + 0.005*x[0] + 0.5*x[3] + 0.0025) - 1.5*x[1]/(x[1] + 0.005), 1.0*x[1]*x_5/(x_5 + 0.005) - 0.5*x[2]/(x[2] + 0.005), -3.0*x[0]*x[3]/(x[0]*x[3] + 0.005*x[0] + 0.5*x[3] + 0.0025) + 1.5*x[1]/(x[1] + 0.005), -1.0*x[1]*x_5/(x_5 + 0.005) + 0.5*x[2]/(x[2] + 0.005)])

@register_eq_class
class BIOMD0000000914(KnownEquation):
    _eq_name = 'vars5_prog44'
    _operator_set = ['add', 'sub', 'mul', 'div', 'const', 'pow']
    _description = "Parra-Guillen2013 - Mathematical model approach to describe tumour response in mice after vaccine administration-model1"
    def __init__(self):
        
        self.vars_range_and_types = [LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True), LogUniformSampling((1e-2, 10.0), only_positive=True)]
        super().__init__(num_vars=5, vars_range_and_types=self.vars_range_and_types)
        x= self.x
        self.sympy_eq = ['-0.091*x[0]', '0.091*x[0] - 0.091*x[1]', '0.091*x[1] - 0.091*x[2]', '-463.6*x[2]*x[3]/(x_5**5.24 + 429.3) + 5.24', '0.039*x[3] - 0.039*x_5']
    
    def np_eq(self, t, x):
        return np.array([-0.091*x[0], 0.091*x[0] - 0.091*x[1], 0.091*x[1] - 0.091*x[2], -463.6*x[2]*x[3]/(x_5**5.24 + 429.3) + 5.24, 0.039*x[3] - 0.039*x_5])

